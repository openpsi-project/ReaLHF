from typing import Tuple
import logging
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import transformers

import api.model
import api.utils

logger = logging.getLogger("Model Utils")


@torch.no_grad()
def masked_normalization(
    x,
    mask=None,
    dim=None,
    inplace=False,
    unbiased=False,
    eps=1e-5,
    high_precision=True,
):
    """Normalize x with a mask. Typically used in advantage normalization.

    Args:
        x (torch.Tensor):
            Tensor to be normalized.
        mask (torch.Tensor, optional):
            A mask with the same shape as x. Defaults to None.
        dim (int or tuple of ints, optional):
            Dimensions to be normalized. Defaults to None.
        inplace (bool, optional):
            Whether to perform in-place operation. Defaults to False.
        eps (torch.Tensor, optional):
            Minimal denominator. Defaults to 1e-5.

    Returns:
        torch.Tensor:
            Normalized x, with the same shape as x.
    """
    dtype = torch.float64 if high_precision else torch.float32
    x = x.to(dtype)
    if not inplace:
        x = x.clone()
    if dim is None:
        dim = tuple(range(len(x.shape)))
    if mask is None:
        factor = torch.tensor(np.prod([x.shape[d] for d in dim]), dtype=dtype, device=x.device)
    else:
        mask = mask.to(dtype)
        assert len(mask.shape) == len(x.shape), (mask.shape, x.shape, dim)
        for i in range(len(x.shape)):
            if i in dim:
                assert mask.shape[i] == x.shape[i], (mask.shape, x.shape, dim)
            else:
                assert mask.shape[i] == 1, (mask.shape, x.shape, dim)
        x = x * mask
        factor = mask.sum(dim, keepdim=True)
    x_sum = x.sum(dim=dim, keepdim=True)
    x_sum_sq = x.square().sum(dim=dim, keepdim=True)
    if dist.is_initialized():
        dist.all_reduce(factor, op=dist.ReduceOp.SUM)
        dist.all_reduce(x_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(x_sum_sq, op=dist.ReduceOp.SUM)
    mean = x_sum / factor
    meansq = x_sum_sq / factor
    var = meansq - mean**2
    if unbiased:
        var *= factor / (factor - 1)
    return ((x - mean) / (var.sqrt() + eps)).float()


class ExponentialRunningMeanStd(nn.Module):

    def __init__(self, input_shape, beta=0.999, epsilon=1e-5, high_precision=True):
        super().__init__()
        self.__beta = beta
        self.__eps = epsilon
        self.__input_shape = input_shape

        self.__dtype = torch.float64 if high_precision else torch.float32

        self.__mean = nn.Parameter(torch.zeros(input_shape, dtype=self.__dtype), requires_grad=False)
        self.__mean_sq = nn.Parameter(torch.zeros(input_shape, dtype=self.__dtype), requires_grad=False)
        self.__debiasing_term = nn.Parameter(torch.zeros(1, dtype=self.__dtype), requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.__mean.zero_()
        self.__mean_sq.zero_()
        self.__debiasing_term.zero_()

    def forward(self, *args, **kwargs):
        # we don't implement the forward function because its meaning
        # is somewhat ambiguous
        raise NotImplementedError()

    def __check(self, x, mask):
        assert isinstance(x, torch.Tensor)
        trailing_shape = x.shape[-len(self.__input_shape):]
        assert trailing_shape == self.__input_shape, (
            'Trailing shape of input tensor'
            f'{x.shape} does not equal to configured input shape {self.__input_shape}')
        if mask is not None:
            assert mask.shape == (*x.shape[:-len(self.__input_shape)],
                                  *((1,) * len(self.__input_shape))), (mask.shape, x.shape)

    @torch.no_grad()
    def update(self, x, mask=None):
        self.__check(x, mask)
        x = x.to(self.__dtype)
        if mask is not None:
            mask = mask.to(self.__dtype)
        norm_dims = tuple(range(len(x.shape) - len(self.__input_shape)))
        if mask is None:
            factor = torch.tensor(np.prod(x.shape[:-len(self.__input_shape)]),
                                  dtype=self.__dtype,
                                  device=x.device)
        else:
            x = x * mask
            factor = mask.sum()

        x_sum = x.sum(dim=norm_dims)
        x_sum_sq = x.square().sum(dim=norm_dims)
        if dist.is_initialized():
            dist.all_reduce(factor, op=dist.ReduceOp.SUM)
            dist.all_reduce(x_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(x_sum_sq, op=dist.ReduceOp.SUM)
        batch_mean = x_sum / factor
        batch_sq_mean = x_sum_sq / factor

        self.__mean.data[:] = self.__beta * self.__mean.data[:] + batch_mean * (1.0 - self.__beta)
        self.__mean_sq.data[:] = self.__beta * self.__mean_sq.data[:] + batch_sq_mean * (1.0 - self.__beta)
        self.__debiasing_term.data[:] = self.__beta * self.__debiasing_term.data[:] + 1.0 - self.__beta

    @torch.no_grad()
    def mean_std(self):
        debiased_mean = self.__mean / self.__debiasing_term.clamp(min=self.__eps)
        debiased_mean_sq = self.__mean_sq / self.__debiasing_term.clamp(min=self.__eps)
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var.sqrt()

    @torch.no_grad()
    def normalize(self, x):
        self.__check(x, None)
        x = x.to(self.__dtype)
        mean, std = self.mean_std()
        return ((x - mean) / std).clip(-5, 5).float()  # clipping is a trick from hide and seek

    @torch.no_grad()
    def denormalize(self, x):
        self.__check(x, None)
        x = x.to(self.__dtype)
        mean, std = self.mean_std()
        return (x * std + mean).float()


class MovingAverageRunningMeanStd(nn.Module):

    def __init__(self, input_shape, high_precision=True):
        super().__init__()
        self.__input_shape = input_shape

        self.__dtype = torch.float64 if high_precision else torch.float32

        self.__mean = nn.Parameter(torch.zeros(input_shape, dtype=self.__dtype), requires_grad=False)
        self.__mean_sq = nn.Parameter(torch.zeros(input_shape, dtype=self.__dtype), requires_grad=False)
        self.__accum_denominator = 0

        self.reset_parameters()

    def reset_parameters(self):
        self.__mean.zero_()
        self.__mean_sq.zero_()
        self.__accum_denominator = 0

    def forward(self, *args, **kwargs):
        # we don't implement the forward function because its meaning
        # is somewhat ambiguous
        raise NotImplementedError()

    def __check(self, x, mask):
        assert isinstance(x, torch.Tensor)
        trailing_shape = x.shape[-len(self.__input_shape):]
        assert trailing_shape == self.__input_shape, (
            'Trailing shape of input tensor'
            f'{x.shape} does not equal to configured input shape {self.__input_shape}')
        if mask is not None:
            assert mask.shape == (*x.shape[:-len(self.__input_shape)],
                                  *((1,) * len(self.__input_shape))), (mask.shape, x.shape)

    @torch.no_grad()
    def update(self, x, mask=None):
        self.__check(x, mask)
        x = x.to(self.__dtype)
        if mask is not None:
            mask = mask.to(self.__dtype)
        norm_dims = tuple(range(len(x.shape) - len(self.__input_shape)))
        if mask is None:
            factor = torch.tensor(np.prod(x.shape[:-len(self.__input_shape)]),
                                  dtype=self.__dtype,
                                  device=x.device)
        else:
            x = x * mask
            factor = mask.sum()

        x_sum = x.sum(dim=norm_dims)
        x_sum_sq = x.square().sum(dim=norm_dims)
        if dist.is_initialized():
            dist.all_reduce(factor, op=dist.ReduceOp.SUM)
            dist.all_reduce(x_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(x_sum_sq, op=dist.ReduceOp.SUM)

        self.__mean.data[:] = (self.__accum_denominator * self.__mean.data[:] +
                               x_sum) / (self.__accum_denominator + factor)
        self.__mean_sq.data[:] = (self.__accum_denominator * self.__mean_sq.data[:] +
                                  x_sum_sq) / (self.__accum_denominator + factor)
        self.__accum_denominator += factor

    @torch.no_grad()
    def mean_std(self):
        return self.__mean.clone(), (self.__mean_sq - self.__mean**2).clamp(min=1e-2).sqrt()

    @torch.no_grad()
    def normalize(self, x):
        self.__check(x, None)
        x = x.to(self.__dtype)
        mean, std = self.mean_std()
        return ((x - mean) / std).clip(-5, 5).float()  # clipping is a trick from hide and seek

    @torch.no_grad()
    def denormalize(self, x):
        self.__check(x, None)
        x = x.to(self.__dtype)
        mean, std = self.mean_std()
        return (x * std + mean).float()


def save_hf_or_lora_model(model: api.model.Model, output_dir: str):
    from impl.model.lora import is_lora_model, get_lora_state_dict
    module = model.module
    tokenizer = model.tokenizer
    logger.info(f'saving the model for epoch {model.version.epoch} step {model.version.epoch_step}...')
    os.makedirs(os.path.abspath(
        os.path.join(
            output_dir,
            f"epoch{model.version.epoch}step{model.version.epoch_step}",
        )),
                exist_ok=True)
    if not is_lora_model(module):
        api.utils.save_hf_format(
            module,
            tokenizer,
            output_dir,
            sub_folder=f"epoch{model.version.epoch}step{model.version.epoch_step}",
        )
        return
    lora_sd = get_lora_state_dict(module)
    torch.save(
        lora_sd,
        os.path.join(
            output_dir,
            f"epoch{model.version.epoch}step{model.version.epoch_step}",
            "lora.bin",
        ),
    )


def get_eos_indices(
    input_ids: torch.LongTensor,
    tokenizer: transformers.PreTrainedTokenizerFast,
) -> Tuple[torch.LongTensor, torch.FloatTensor]:
    if torch.any(input_ids[:, 0] == tokenizer.eos_token_id):
        indices = (input_ids[:, 0] == tokenizer.eos_token_id).nonzero().flatten()
        bad_input_ids = input_ids[indices]
        bad_strs = tokenizer.batch_decode(bad_input_ids,
                                          skip_special_tokens=True,
                                          clean_up_tokenization_spaces=True)
        raise RuntimeError(f"Generated sequence terminates unexpectedly early: {bad_strs}")
    seq_len = input_ids.shape[1]
    eos_mask = (input_ids == tokenizer.eos_token_id).float()
    seq_no_eos_mask = (eos_mask.sum(1) == 0).float()
    eos_indices = eos_mask.argmax(1)
    eos_indices = (eos_indices * (1 - seq_no_eos_mask) + seq_no_eos_mask * (seq_len - 1)).long()
    return eos_indices, seq_no_eos_mask