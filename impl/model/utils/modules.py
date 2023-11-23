from typing import Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

import base.logging as logging

logger = logging.getLogger("Modules")


class LayerNormLinear(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layer_norm_epsilon: float,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if dtype is None:
            dtype = torch.float16
        self.ln = nn.LayerNorm(input_dim, eps=layer_norm_epsilon, dtype=dtype, device=device)
        self.linear = nn.Linear(input_dim, output_dim, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.ln(x))


class LayerNormMLP(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        resid_pdrop: float,
        activation_function: str,
        layer_norm_epsilon: float,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        if dtype is None:
            dtype = torch.float16

        self.ln = nn.LayerNorm(hidden_dim, eps=layer_norm_epsilon, dtype=dtype, device=device)
        self.c_fc = nn.Linear(hidden_dim, intermediate_dim, dtype=dtype, device=device)
        self.c_proj = nn.Linear(intermediate_dim, hidden_dim, dtype=dtype, device=device)
        if activation_function == "gelu":
            self.act = nn.functional.gelu
        elif activation_function == 'gelu_new':
            from .activations import new_gelu_activation
            self.act = new_gelu_activation
        else:
            raise NotImplementedError("Only \"gelu\" activation function is available.")
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.ln(hidden_states)
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return self.dropout(hidden_states)


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