from typing import Tuple, Union, Type
import logging
import dataclasses
import numpy as np
import torch
import torch.distributed as dist
import transformers

logger = logging.getLogger("Data Manipulation")


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


# TODO: temp solution, all data going through pp models must be non-boolean tensors
# before input to pipe model, convert all data to tensors (# input of pipe model should be tensors)-> convert back to original type
# after output from pipe -> convert all data to tensors
def to_tensor(x: Union[int, bool, torch.Tensor, None]):
    device = torch.cuda.current_device()
    if isinstance(x, int) or isinstance(x, bool):
        assert x >= 0
        return torch.tensor(x, dtype=torch.long, device=device)
    elif x is None:
        return torch.tensor(-1, dtype=torch.long, device=device)
    elif torch.is_tensor(x):
        # if x.dtype != torch.bool:
        #     return x.to(device=device)
        # else:
        #     # convert bool tensor to long tensor
        #     return x.to(dtype=torch.long, device=device)
        return x.to(device=device)
    else:
        raise NotImplementedError(f"Cannot convert {x} to tensor")


def from_tensor(x: torch.Tensor, _type: Type):
    try:
        if int(x) < 0:
            return None
    except:
        pass
    if _type == int:
        return int(x)
    elif _type == bool:
        return bool(x)
    elif _type == torch.Tensor:
        return x
    else:
        raise NotImplementedError(f"Cannot convert tensor to {_type}")


class TensorDataclassToTupleInterface:

    def to_tuple(self):
        t = []
        for v in dataclasses.asdict(self).values():
            t.append(to_tensor(v))
        return tuple(t)

    @classmethod
    def from_tuple(cls, t):
        x = cls()
        for i, f in enumerate(dataclasses.fields(x)):
            setattr(x, f.name, from_tensor(t[i], f.type))
        return x