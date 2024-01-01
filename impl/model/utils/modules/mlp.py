from typing import Callable, Optional, Union
import functools
import math

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

import base.logging as logging

logger = logging.getLogger("Modules")


def get_activation_fn(activation_function: str) -> Callable:
    if activation_function == "gelu":
        return nn.functional.gelu
    elif activation_function == "gelu_new":
        from .activations import new_gelu_activation

        return new_gelu_activation
    elif activation_function == "silu":
        return nn.SiLU()
    else:
        raise NotImplementedError('Only "gelu" activation function is available.')


class LayerNormLinear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layer_norm_epsilon: float,
        use_attention_bias: bool,
        layer_norm_type: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        layer_index=None,
    ):
        super().__init__()
        if dtype is None:
            dtype = torch.float16
        if layer_norm_type is None:
            layer_norm_fn = nn.LayerNorm
        elif layer_norm_type == "rms":
            layer_norm_fn = LlamaRMSNorm
        self.ln = layer_norm_fn(input_dim, eps=layer_norm_epsilon, dtype=dtype, device=device)
        self.linear = nn.Linear(input_dim, output_dim, bias=use_attention_bias, dtype=dtype, device=device)
        self.layer_index = layer_index

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
        self.act = get_activation_fn(activation_function)
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.ln(hidden_states)
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return self.dropout(hidden_states)


class LlamaLayerNormMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        activation_function: str,
        layer_norm_epsilon: float,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        if dtype is None:
            dtype = torch.float16
        self.hidden_size = hidden_dim
        self.intermediate_size = intermediate_dim
        self.ln = LlamaRMSNorm(hidden_dim, eps=layer_norm_epsilon, dtype=dtype, device=device)
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False, dtype=dtype, device=device
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False, dtype=dtype, device=device
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False, dtype=dtype, device=device
        )
        self.act_fn = get_activation_fn(activation_function)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaRMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
