from typing import Callable, Optional, Union
import functools
import math
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from realhf.impl.model.parallelism.model_parallel.modules import (
    ColumnParallelLinear,
    merged_linear_with_grad_accumulation_and_async_allreduce,
    RowParallelLinear,
)
import realhf.base.constants as constants
import realhf.base.logging as logging

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
        raise NotImplementedError(
            'Only "gelu" activation function is available.'
        )


SEQUENCE_PARALLEL_WARNED = False


class LayerNormQKVLinear(nn.Module):

    def __init__(
        self,
        input_dim: int,
        head_dim: int,
        n_q_heads: int,
        n_kv_heads: int,
        layer_norm_epsilon: float,
        use_attention_bias: bool,
        layer_norm_type: Optional[str] = None,
        # parallelism
        model_parallel: bool = False,  # We set this as an option for replacing this module with layers in transformer engine
        gradient_accumulation_fusion: bool = False,
        # dtype and device
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        layer_index=None,
    ):
        super().__init__()
        sequence_parallel = constants.sequence_parallel()
        if not model_parallel and (
            sequence_parallel or gradient_accumulation_fusion
        ):
            global SEQUENCE_PARALLEL_WARNED
            if not SEQUENCE_PARALLEL_WARNED:
                logger.warning(
                    "sequence_parallel and gradient_accumulation_fusion are only available in model parallel mode"
                )
                SEQUENCE_PARALLEL_WARNED = True
            sequence_parallel = False
            gradient_accumulation_fusion = False
        if dtype is None:
            dtype = torch.float16
        if layer_norm_type is None:
            layer_norm_fn = nn.LayerNorm
        elif layer_norm_type == "rms":
            layer_norm_fn = LlamaRMSNorm
        self.ln = layer_norm_fn(
            input_dim, eps=layer_norm_epsilon, dtype=dtype, device=device
        )

        self.model_parallel = model_parallel
        self.layer_index = layer_index
        if not model_parallel:
            self.q_attn = nn.Linear(
                input_dim,
                head_dim * n_q_heads,
                bias=use_attention_bias,
                dtype=dtype,
                device=device,
            )
            self.k_attn = nn.Linear(
                input_dim,
                head_dim * n_kv_heads,
                bias=use_attention_bias,
                dtype=dtype,
                device=device,
            )
            self.v_attn = nn.Linear(
                input_dim,
                head_dim * n_kv_heads,
                bias=use_attention_bias,
                dtype=dtype,
                device=device,
            )
        else:
            self.mp_worldsize = constants.model_parallel_world_size()
            assert n_q_heads % self.mp_worldsize == 0, (
                f"n_q_heads {n_q_heads} must be divisible by "
                f"mp_worldsize {self.mp_worldsize}"
            )
            hidden_dim = input_dim
            self.q_attn = ColumnParallelLinear(
                hidden_dim,
                head_dim * n_q_heads,
                bias=use_attention_bias,
                async_tensor_model_parallel_allreduce=not sequence_parallel,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
                dtype=dtype,
                device=device,
            )
            if n_kv_heads > 1 and n_kv_heads % self.mp_worldsize == 0:
                # split model parallel among heads if possible
                self.k_attn = ColumnParallelLinear(
                    hidden_dim,
                    head_dim * n_kv_heads,
                    bias=use_attention_bias,
                    async_tensor_model_parallel_allreduce=not sequence_parallel,
                    gradient_accumulation_fusion=gradient_accumulation_fusion,
                    dtype=dtype,
                    device=device,
                )
                self.v_attn = ColumnParallelLinear(
                    hidden_dim,
                    head_dim * n_kv_heads,
                    bias=use_attention_bias,
                    async_tensor_model_parallel_allreduce=not sequence_parallel,
                    gradient_accumulation_fusion=gradient_accumulation_fusion,
                    dtype=dtype,
                    device=device,
                )
            else:
                if n_kv_heads > 1:
                    logger.warning(
                        f"Cannot split {n_kv_heads} kv heads evenly among "
                        f"{self.mp_worldsize} model parallel ranks, "
                        f"use unsplitted linear for kv heads instead"
                    )
                self.k_attn = nn.Linear(
                    hidden_dim,
                    head_dim * n_kv_heads,
                    bias=use_attention_bias,
                    dtype=dtype,
                    device=device,
                )
                self.v_attn = nn.Linear(
                    hidden_dim,
                    head_dim * n_kv_heads,
                    bias=use_attention_bias,
                    dtype=dtype,
                    device=device,
                )
                dist.all_reduce(
                    self.k_attn.weight.data,
                    op=dist.ReduceOp.SUM,
                    group=constants.model_parallel_group(),
                )
                if use_attention_bias:
                    dist.all_reduce(
                        self.k_attn.bias.data,
                        op=dist.ReduceOp.SUM,
                        group=constants.model_parallel_group(),
                    )
                dist.all_reduce(
                    self.v_attn.weight.data,
                    op=dist.ReduceOp.SUM,
                    group=constants.model_parallel_group(),
                )
                if use_attention_bias:
                    dist.all_reduce(
                        self.v_attn.bias.data,
                        op=dist.ReduceOp.SUM,
                        group=constants.model_parallel_group(),
                    )

        self.d = head_dim
        self.nq = n_q_heads
        self.nkv = n_kv_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.ln(x)
        if not self.model_parallel:
            q = self.q_attn(hidden_states)
            k = self.k_attn(hidden_states)
            v = self.v_attn(hidden_states)
            # qkv = self.linear(hidden_states)
            # q, k, v = torch.split(qkv, (self.d * self.nq, self.d * self.nkv, self.d * self.nkv), dim=-1)
            q = q.view(*q.shape[:-1], self.nq, self.d)
            k = k.view(*k.shape[:-1], self.nkv, self.d)
            v = v.view(*v.shape[:-1], self.nkv, self.d)
        else:
            _gradient_accumulation_fusion = (
                self.q_attn.gradient_accumulation_fusion
            )
            _async_grad_allreduce = (
                self.q_attn.async_tensor_model_parallel_allreduce
            )
            _sequence_parallel = constants.sequence_parallel()
            _is_w_parallel = [
                True,
                isinstance(self.k_attn, ColumnParallelLinear),
                isinstance(self.v_attn, ColumnParallelLinear),
            ]
            q, k, v = merged_linear_with_grad_accumulation_and_async_allreduce(
                hidden_states,
                _gradient_accumulation_fusion,
                _async_grad_allreduce,
                _sequence_parallel,
                _is_w_parallel,
                self.q_attn.weight,
                self.q_attn.bias,
                self.k_attn.weight,
                self.k_attn.bias,
                self.v_attn.weight,
                self.v_attn.bias,
            )
            q = q.view(*q.shape[:-1], self.nq // self.mp_worldsize, self.d)
            if isinstance(self.k_attn, ColumnParallelLinear):
                k = k.view(*k.shape[:-1], self.nkv // self.mp_worldsize, self.d)
                v = v.view(*v.shape[:-1], self.nkv // self.mp_worldsize, self.d)
            else:
                k = k.view(*k.shape[:-1], self.nkv, self.d)
                v = v.view(*v.shape[:-1], self.nkv, self.d)
        return q, k, v


class LayerNormMLP(nn.Module):
    SEQUENCE_PARALLEL_WARNED = False

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        resid_pdrop: float,
        activation_function: str,
        layer_norm_epsilon: float,
        # parallelism
        model_parallel: bool = False,  # We set this as an option for replacing this module with layers in transformer engine
        gradient_accumulation_fusion: bool = False,
        # dtype and device
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        sequence_parallel = constants.sequence_parallel()
        if not model_parallel and (
            sequence_parallel or gradient_accumulation_fusion
        ):
            global SEQUENCE_PARALLEL_WARNED
            if not SEQUENCE_PARALLEL_WARNED:
                logger.warning(
                    "sequence_parallel and gradient_accumulation_fusion are only available in model parallel mode"
                )
                SEQUENCE_PARALLEL_WARNED = True
            sequence_parallel = False
            gradient_accumulation_fusion = False
        if dtype is None:
            dtype = torch.float16

        self.ln = nn.LayerNorm(
            hidden_dim, eps=layer_norm_epsilon, dtype=dtype, device=device
        )
        self.model_parallel = model_parallel
        if not model_parallel:
            self.c_fc = nn.Linear(
                hidden_dim, intermediate_dim, dtype=dtype, device=device
            )
            self.c_proj = nn.Linear(
                intermediate_dim, hidden_dim, dtype=dtype, device=device
            )
        else:
            self.c_fc = ColumnParallelLinear(
                hidden_dim,
                intermediate_dim,
                async_tensor_model_parallel_allreduce=not sequence_parallel,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
                dtype=dtype,
                device=device,
            )
            self.c_proj = RowParallelLinear(
                intermediate_dim,
                hidden_dim,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
                dtype=dtype,
                device=device,
            )
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
        # parallelism
        model_parallel: bool = False,  # We set this as an option for replacing this module with layers in transformer engine
        gradient_accumulation_fusion: bool = False,
        # dtype and device
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        sequence_parallel = constants.sequence_parallel()
        if not model_parallel and (
            sequence_parallel or gradient_accumulation_fusion
        ):
            global SEQUENCE_PARALLEL_WARNED
            if not SEQUENCE_PARALLEL_WARNED:
                logger.warning(
                    "sequence_parallel and gradient_accumulation_fusion are only available in model parallel mode"
                )
                SEQUENCE_PARALLEL_WARNED = True
            sequence_parallel = False
            gradient_accumulation_fusion = False

        if dtype is None:
            dtype = torch.float16
        self.hidden_size = hidden_dim
        self.intermediate_size = intermediate_dim
        self.ln = LlamaRMSNorm(
            hidden_dim, eps=layer_norm_epsilon, dtype=dtype, device=device
        )

        self.model_parallel = model_parallel
        if not model_parallel:
            self.gate_proj = nn.Linear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                dtype=dtype,
                device=device,
            )
            self.up_proj = nn.Linear(
                self.hidden_size,
                self.intermediate_size,
                bias=False,
                dtype=dtype,
                device=device,
            )
            self.down_proj = nn.Linear(
                self.intermediate_size,
                self.hidden_size,
                bias=False,
                dtype=dtype,
                device=device,
            )
        else:
            self.gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                async_tensor_model_parallel_allreduce=not sequence_parallel,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
                bias=False,
                dtype=dtype,
                device=device,
            )
            self.up_proj = ColumnParallelLinear(
                self.hidden_size,
                self.intermediate_size,
                async_tensor_model_parallel_allreduce=not sequence_parallel,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
                bias=False,
                dtype=dtype,
                device=device,
            )
            self.down_proj = RowParallelLinear(
                self.intermediate_size,
                self.hidden_size,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
                bias=False,
                dtype=dtype,
                device=device,
            )
        self.act_fn = get_activation_fn(activation_function)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        if not self.model_parallel:
            return self.down_proj(
                self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            )
        else:
            _gradient_accumulation_fusion = (
                self.gate_proj.gradient_accumulation_fusion
            )
            _async_grad_allreduce = (
                self.gate_proj.async_tensor_model_parallel_allreduce
            )
            _sequence_parallel = constants.sequence_parallel()
            _is_w_parallel = [True, True]
            gate, upproj = (
                merged_linear_with_grad_accumulation_and_async_allreduce(
                    x,
                    _gradient_accumulation_fusion,
                    _async_grad_allreduce,
                    _sequence_parallel,
                    _is_w_parallel,
                    self.gate_proj.weight,
                    self.gate_proj.bias,
                    self.up_proj.weight,
                    self.up_proj.bias,
                )
            )
            return self.down_proj(self.act_fn(gate) * upproj)


class _LlamaRMSNorm(nn.Module):

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
        self.weight = nn.Parameter(
            torch.ones(hidden_size, dtype=dtype, device=device)
        )
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon
        )
        return self.weight * hidden_states.to(input_dtype)


if constants.use_te_impl():
    try:
        # HACK: we use transformer engine's rms norm as long as we can find the transformer engine package
        import transformer_engine.pytorch as te

        def _TELlamaRMSNorm(
            hidden_size: int,
            eps: float = 1e-6,
            dtype: Optional[torch.dtype] = None,
            device: Optional[Union[str, torch.device]] = None,
        ):
            return te.module.rmsnorm.RMSNorm(
                hidden_size=hidden_size,
                eps=eps,
                sequence_parallel=constants.sequence_parallel(),
                params_dtype=dtype,
                device=device,
            )

        LlamaRMSNorm = _TELlamaRMSNorm
    except ModuleNotFoundError:
        LlamaRMSNorm = _LlamaRMSNorm
    except ImportError:
        LlamaRMSNorm = _LlamaRMSNorm
else:
    LlamaRMSNorm = _LlamaRMSNorm

if constants.use_te_impl():
    from transformer_engine.pytorch.module.layernorm_mlp import (
        LayerNormMLP as _TELayerNormMLP,
    )

    # The same signature as LlamaLayerNormMLP
    def LlamaLayerNormMLP(
        hidden_dim: int,
        intermediate_dim: int,
        activation_function: str,
        layer_norm_epsilon: float,
        # parallelism
        model_parallel: bool = False,  # We set this as an option for replacing this module with layers in transformer engine
        gradient_accumulation_fusion: bool = False,
        # dtype and device
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        assert activation_function == "silu"
        return _TELayerNormMLP(
            hidden_size=hidden_dim,
            ffn_hidden_size=intermediate_dim,
            eps=layer_norm_epsilon,
            sequence_parallel=constants.sequence_parallel(),
            return_bias=False,
            tp_group=constants.model_parallel_group(),
            tp_size=constants.model_parallel_world_size(),
            bias=False,
            normalization="RMSNorm",
            activation="swiglu",
            fuse_wgrad_accumulation=gradient_accumulation_fusion,
            params_dtype=dtype,
            set_parallel_mode=model_parallel,
            device=device,
        )
