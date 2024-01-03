from typing import Callable, Dict, List, Optional, Tuple, Union
import functools
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.checkpoint

from impl.model.nn.flash_mqat.flash_mqat_api import (DeepSpeedChatLikeFlashMQATCriticModel,
                                                     HuggingfaceLikeFlashMQATForCausalLM)
from impl.model.nn.flash_mqat.flash_mqat_base import FlashMQATConfig, FlashMQATModel
from impl.model.utils.data import DuckGenerationOutput, PipeCacheData, PipeTransferData
from impl.model.utils.model_parallel.modules import (ColumnParallelLinear, LayerNormParallelMLP,
                                                     LlamaLayerNormParallelMLP,
                                                     merged_linear_with_grad_accumulation_and_async_allreduce,
                                                     parallel_lm_logits, ParallelEmbedding, RowParallelLinear)
from impl.model.utils.modules import LlamaRMSNorm, RotaryEmbedding
from impl.model.utils.save_load import load_from_disk, save_to_disk
from impl.model.utils.tensor import pad_sequence_parallel_input
import base.constants
import base.logging as logging
import impl.model.utils.model_parallel.mappings as tensor_parallel

try:
    from flash_attn import (flash_attn_func, flash_attn_varlen_func, flash_attn_varlen_func_with_kvcache,
                            flash_attn_with_kvcache)
except ModuleNotFoundError:
    pass

logger = logging.getLogger("ParallelFlashMQAT")


class ParallelVocabPositionEmbedding(nn.Module):

    def __init__(
        self,
        config: FlashMQATConfig,
        sequence_parallel: Optional[bool] = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if dtype is None:
            self.dtype = dtype = torch.float16
        super().__init__()
        self.n_positions = config.n_positions
        self.sequence_parallel = sequence_parallel
        self.wte = ParallelEmbedding(config.vocab_size, config.hidden_dim, dtype=dtype, device=device)

        self.apply_abs_pos_embed = not config.apply_rotary
        if self.apply_abs_pos_embed:
            self.wpe = ParallelEmbedding(config.n_positions, config.hidden_dim, dtype=dtype, device=device)

        self.embed_drop = nn.Dropout(config.embd_pdrop)

        self.self_attention_mask = torch.tril(
            torch.ones((config.n_positions, config.n_positions), dtype=torch.bool, device=device))
        self.fixed_abs_position_ids = config.fixed_abs_position_ids

    def forward(self, x: PipeTransferData, y: PipeCacheData) -> PipeTransferData:
        # Initial sanity check.
        with_cache = y.cache_seqlens is not None
        if with_cache and len(y.input_ids.shape) == 2:
            assert y.input_ids.shape[1] == 1
        elif with_cache and len(y.input_ids.shape) == 1:
            if x.cu_seqlens is None:
                y.input_ids = y.input_ids.unsqueeze(-1)
        packed = len(y.input_ids.shape) == 1
        if packed and ((x.cu_seqlens is None) or (x.max_seqlen is None)):
            raise ValueError("cu_seqlens and max_seqlen must be both provided for packed input.")

        # Set position ids.
        if not y.position_ids is None:
            raise ValueError("In our use cases, position_ids must be None.")
        if not packed and y.position_ids is None:
            # input_ids is given
            batch_size, input_length = y.input_ids.shape
            device = y.input_ids.device
            y.position_ids = torch.arange(input_length, dtype=torch.long, device=device)
            if y.cache_seqlens is not None:  # during generation
                y.position_ids = y.position_ids + y.cache_seqlens.unsqueeze(1)
            else:
                y.position_ids = y.position_ids.repeat(batch_size, 1)
        elif y.position_ids is None:
            # packed_input_ids is given
            lengths = x.cu_seqlens[1:] - x.cu_seqlens[:-1]
            if y.cache_seqlens is None:
                y.position_ids = torch.cat(
                    [torch.arange(int(l), dtype=torch.int32, device=y.input_ids.device) for l in lengths])
                assert (y.position_ids < x.max_seqlen).all() and y.position_ids.max() == x.max_seqlen - 1
            else:
                y.position_ids = torch.cat([
                    torch.arange(int(l), dtype=torch.int32, device=y.input_ids.device) + cache_len
                    for l, cache_len in zip(lengths, y.cache_seqlens)
                ])
            if x.max_seqlen > self.n_positions:
                raise ValueError(f"max_seqlen ({x.max_seqlen}) must be <= n_positions ({self.n_positions}).")
            assert y.position_ids.shape == y.input_ids.shape, (
                y.position_ids.shape,
                y.input_ids.shape,
                lengths,
                x.cu_seqlens,
            )

        if x.attention_mask is not None:
            # For debugging only.
            attention_mask = x.attention_mask
            if self.fixed_abs_position_ids:
                y.position_ids = torch.arange(y.input_ids.shape[-1],
                                              dtype=torch.long,
                                              device=y.input_ids.device).unsqueeze(0)
            else:
                y.position_ids = attention_mask.long().cumsum(-1) - 1
                y.position_ids.masked_fill_(attention_mask == 0, 1)
            seqlen = y.input_ids.shape[-1]
            self_attention_mask = self.self_attention_mask[None, :seqlen, :seqlen]
            self_attention_mask = self_attention_mask * attention_mask.view(batch_size, 1, -1).to(
                dtype=torch.bool, device=self_attention_mask.device)
            x.attention_mask = self_attention_mask.unsqueeze(1)

        inputs_embeds = self.wte(y.input_ids)
        if self.apply_abs_pos_embed:
            inputs_embeds = inputs_embeds + self.wpe(y.position_ids)

        if self.sequence_parallel:
            inputs_embeds = tensor_parallel.scatter_to_sequence_parallel_region(inputs_embeds)
            # `scatter_to_sequence_parallel_region` returns a view, which prevents
            # the original tensor from being garbage collected. Clone to facilitate GC.
            # Has a small runtime cost (~0.5%).
            # if self.config.clone_scatter_output_in_embedding:
            #     embeddings = embeddings.clone()
            # with tensor_parallel.get_cuda_rng_tracker().fork():
            x.pp_output = self.embed_drop(inputs_embeds)
        else:
            x.pp_output = self.embed_drop(inputs_embeds)
        return x


class ParallelCausalSelfAttentionLayer(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        n_kv_heads: int,
        head_dim: int,
        resid_pdrop: float,
        attn_pdrop: float,
        layer_index: int,
        layer_norm_epsilon: float,
        # gpt2 does not scale attn by inverse layer idx, in contrast to starcoder
        scale_attn_by_inverse_layer_idx: bool,
        # llama does not require attention bias
        use_attention_bias: bool,
        # layer norm type is special for llama
        layer_norm_type: Optional[str] = None,
        # rotary embedding
        apply_rotary: bool = False,
        rotary_base: float = 10000.0,
        rotary_interleaved: bool = False,  # False for LLaMA, GPT-neoX; True for GPT-J
        rotary_scaling: Optional[float] = None,
        rotary_scaling_type: Optional[str] = None,
        # parallel settings
        sequence_parallel: Optional[bool] = False,
        gradient_accumulation_fusion: bool = True,
        # device and dtype
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        if dtype is None:
            dtype = torch.float16
        assert device != "cpu", "Tensor Parallel FlashMQAT does not support CPU."
        assert hidden_dim % head_dim == 0
        n_q_heads = hidden_dim // head_dim

        if layer_norm_type is None:
            layer_norm_fn = nn.LayerNorm
        elif layer_norm_type == "rms":
            layer_norm_fn = LlamaRMSNorm

        self.ln = layer_norm_fn(hidden_dim, eps=layer_norm_epsilon, dtype=dtype, device=device)

        self.q_attn = ColumnParallelLinear(
            hidden_dim,
            head_dim * n_q_heads,
            bias=use_attention_bias,
            async_tensor_model_parallel_allreduce=not sequence_parallel,
            sequence_parallel=sequence_parallel,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
            dtype=dtype,
            device=device,
        )
        self.mp_worldsize = base.constants.model_parallel_world_size()
        assert n_q_heads % self.mp_worldsize == 0, (f"n_q_heads {n_q_heads} must be divisible by "
                                                    f"mp_worldsize {self.mp_worldsize}")
        if n_kv_heads > 1 and n_kv_heads % self.mp_worldsize == 0:
            # split model parallel among heads if possible
            self.k_attn = ColumnParallelLinear(
                hidden_dim,
                head_dim * n_kv_heads,
                bias=use_attention_bias,
                async_tensor_model_parallel_allreduce=not sequence_parallel,
                sequence_parallel=sequence_parallel,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
                dtype=dtype,
                device=device,
            )
            self.v_attn = ColumnParallelLinear(
                hidden_dim,
                head_dim * n_kv_heads,
                bias=use_attention_bias,
                async_tensor_model_parallel_allreduce=not sequence_parallel,
                sequence_parallel=sequence_parallel,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
                dtype=dtype,
                device=device,
            )
        else:
            if n_kv_heads > 1:
                logger.warning(f"Cannot split {n_kv_heads} kv heads evenly among "
                               f"{self.mp_worldsize} model parallel ranks, "
                               f"use unsplitted linear for kv heads instead")
            self.k_attn = nn.Linear(hidden_dim,
                                    head_dim * n_kv_heads,
                                    bias=use_attention_bias,
                                    dtype=dtype,
                                    device=device)
            self.v_attn = nn.Linear(hidden_dim,
                                    head_dim * n_kv_heads,
                                    bias=use_attention_bias,
                                    dtype=dtype,
                                    device=device)
            dist.all_reduce(self.k_attn.weight.data,
                            op=dist.ReduceOp.SUM,
                            group=base.constants.model_parallel_group())
            if use_attention_bias:
                dist.all_reduce(self.k_attn.bias.data,
                                op=dist.ReduceOp.SUM,
                                group=base.constants.model_parallel_group())
            dist.all_reduce(self.v_attn.weight.data,
                            op=dist.ReduceOp.SUM,
                            group=base.constants.model_parallel_group())
            if use_attention_bias:
                dist.all_reduce(self.v_attn.bias.data,
                                op=dist.ReduceOp.SUM,
                                group=base.constants.model_parallel_group())

        self.c_proj = RowParallelLinear(
            hidden_dim,
            hidden_dim,
            bias=use_attention_bias,
            sequence_parallel=sequence_parallel,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
            dtype=dtype,
            device=device,
        )
        self.resid_dropout = nn.Dropout(resid_pdrop)

        self.attn_pdrop = attn_pdrop

        self.applied_attn_pdrop = attn_pdrop

        self.apply_rotary = apply_rotary
        self.rotary_interleaved = rotary_interleaved
        if self.apply_rotary:
            # Will layzily update the cache sequence length of cache.,
            # so we don't need to pass in max_positions.
            self.rotary_emb = RotaryEmbedding(
                head_dim,
                base=rotary_base,
                scale_factor=rotary_scaling,
                scale_type=rotary_scaling_type,
                interleaved=rotary_interleaved,
                device=device,
            )

        # constant
        self.h = hidden_dim
        self.nq = n_q_heads
        self.nkv = n_kv_heads
        if self.nq % self.nkv != 0:
            raise ValueError("n_kv_heads must divide n_q_heads")
        self.d = head_dim

        self.layer_index = layer_index

        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx

    def train(self, mode: bool):
        if not mode:
            self.applied_attn_pdrop = 0.0
        else:
            self.applied_attn_pdrop = self.attn_pdrop
        super().train(mode)
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
        cache_seqlens: Optional[Union[int, torch.Tensor]] = None,
        attention_mask: Optional[torch.BoolTensor] = None,  # only used for debugging
        max_seqlen: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # input hidden_states shape: [bs, seq, hidden_dim]/[total_seqlen, hidden_dim]

        # NOTE: we must ensure the passed-in argument is an interger
        # if we convert the argument to implicitly when calling rotary embedding or flash-attn,
        # aten::item will be called, which will cause a device-host sync and slow down performance.
        assert max_seqlen is None or isinstance(max_seqlen, int), type(max_seqlen)
        assert cu_seqlens is None or cu_seqlens.dtype == torch.int32

        # default upcast, scale
        if self.scale_attn_by_inverse_layer_idx:
            unscale = self.layer_index + 1
            scale_factor = unscale**-1
        else:
            unscale = 1.0
            scale_factor = 1
        scale_factor /= self.d**0.5

        hidden_states = self.ln(hidden_states)

        _gradient_accumulation_fusion = self.q_attn.gradient_accumulation_fusion
        _async_grad_allreduce = self.q_attn.async_tensor_model_parallel_allreduce
        _sequence_parallel = self.q_attn.sequence_parallel
        _is_w_parallel = [
            True,
            isinstance(self.k_attn, ColumnParallelLinear),
            isinstance(self.v_attn, ColumnParallelLinear)
        ]
        q, k, v = merged_linear_with_grad_accumulation_and_async_allreduce(
            hidden_states, _gradient_accumulation_fusion, _async_grad_allreduce, _sequence_parallel,
            _is_w_parallel, self.q_attn.weight, self.q_attn.bias, self.k_attn.weight, self.k_attn.bias,
            self.v_attn.weight, self.v_attn.bias)
        qk = torch.cat([q, k], dim=-1)

        if isinstance(self.k_attn, ColumnParallelLinear):
            qk = qk.view(*qk.shape[:-1], (self.nq + self.nkv) // self.mp_worldsize, self.d)
            v = v.view(*v.shape[:-1], self.nkv // self.mp_worldsize, self.d)
        else:
            qk = qk.view(*qk.shape[:-1], self.nq // self.mp_worldsize + self.nkv, self.d)
            v = v.view(*v.shape[:-1], self.nkv, self.d)

        if self.apply_rotary and k_cache is None:
            # otherwise, we input rotary cos/sin directly into flash_attn_with_kvcache
            qk = self.rotary_emb(
                qk,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        elif self.apply_rotary:
            self.rotary_emb._update_cos_sin_cache(k_cache.shape[1], device=q.device, dtype=q.dtype)
            # Rotary cos/sin will be automatically offset by cache_seqlens in flash_attn.
            rotary_cos, rotary_sin = self.rotary_emb._cos_cached, self.rotary_emb._sin_cached
        else:
            rotary_cos = rotary_sin = None

        if isinstance(self.k_attn, ColumnParallelLinear):
            q, k = qk.split((self.nq // self.mp_worldsize, self.nkv // self.mp_worldsize), dim=-2)
        else:
            q, k = qk.split((self.nq // self.mp_worldsize, self.nkv), dim=-2)

        if k_cache is not None and len(q.shape) == 4:
            # k_cache/v_cache shape: [bs, max_seq, n_kv_heads, head_dim]
            if cache_seqlens is None:
                raise RuntimeError("cache_seqlens must be provided if kv_cache is not None.")
            if not (q.shape[1] == k.shape[1] == v.shape[1] == 1):
                raise RuntimeError(
                    "Can only generate one token at a time, "
                    f"while seqence length (q={q.shape[1]}, k={k.shape[1]}, v={v.shape[1]}) is larger than 1."
                )
            # k_cache and v_cache will be modified in-place.
            hidden_states = flash_attn_with_kvcache(
                q,
                k_cache,
                v_cache,
                k=k,
                v=v,
                cache_seqlens=cache_seqlens,
                softmax_scale=scale_factor,
                causal=False,  # True or False doesn't matter because seqlen=1
                rotary_cos=rotary_cos,
                rotary_sin=rotary_sin,
                rotary_interleaved=self.rotary_interleaved,
            )
        elif k_cache is not None and len(q.shape) == 3:
            hidden_states = flash_attn_varlen_func_with_kvcache(
                q=q,
                cu_seqlens_q=cu_seqlens,
                max_seqlen_q=max_seqlen,
                k_cache=k_cache,
                v_cache=v_cache,
                cache_seqlens=cache_seqlens,
                k=k,
                v=v,
                cu_seqlens_k=cu_seqlens,
                softmax_scale=scale_factor,
                causal=True,
                rotary_cos=rotary_cos,
                rotary_sin=rotary_sin,
                rotary_interleaved=self.rotary_interleaved,
            )
        elif cu_seqlens is not None:
            assert max_seqlen is not None
            assert len(q.shape) == 3
            hidden_states = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                dropout_p=self.applied_attn_pdrop,
                softmax_scale=scale_factor,
                causal=True,
            )
        else:
            hidden_states = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.applied_attn_pdrop,
                softmax_scale=scale_factor,
                causal=True,
            )

        hidden_states = self.c_proj(hidden_states.flatten(start_dim=-2))
        hidden_states = self.resid_dropout(hidden_states)
        return hidden_states, k, v


class ParallelFlashMQATBlock(nn.Module):

    def __init__(
        self,
        config: FlashMQATConfig,
        layer_index: int,
        output_layernorm: bool = False,
        sequence_parallel: Optional[bool] = False,
        gradient_accumulation_fusion: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        if dtype is None:
            dtype = torch.float16
        self.layer_index = layer_index
        self.attn = ParallelCausalSelfAttentionLayer(
            hidden_dim=config.hidden_dim,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            resid_pdrop=config.resid_pdrop,
            attn_pdrop=config.attn_pdrop,
            layer_index=layer_index,
            layer_norm_epsilon=config.layer_norm_epsilon,
            scale_attn_by_inverse_layer_idx=config.scale_attn_by_inverse_layer_idx,
            layer_norm_type=config.layer_norm_type,
            use_attention_bias=config.use_attention_bias,
            apply_rotary=config.apply_rotary,
            rotary_base=config.rotary_base,
            rotary_interleaved=config.rotary_interleaved,
            rotary_scaling=config.rotary_scaling,
            rotary_scaling_type=config.rotary_scaling_type,
            sequence_parallel=sequence_parallel,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
            dtype=dtype,
            device=device,
        )
        if config.mlp_type is None:
            self.mlp = LayerNormParallelMLP(
                hidden_dim=config.hidden_dim,
                intermediate_dim=config.intermediate_dim,
                resid_pdrop=config.resid_pdrop,
                activation_function=config.activation_function,
                layer_norm_epsilon=config.layer_norm_epsilon,
                sequence_parallel=sequence_parallel,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
                dtype=dtype,
                device=device,
            )
        elif config.mlp_type == "llama":
            self.mlp = LlamaLayerNormParallelMLP(
                hidden_dim=config.hidden_dim,
                intermediate_dim=config.intermediate_dim,
                activation_function=config.activation_function,
                layer_norm_epsilon=config.layer_norm_epsilon,
                sequence_parallel=sequence_parallel,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
                dtype=dtype,
                device=device,
            )
        self.output_layernorm = output_layernorm
        if output_layernorm:
            if config.layer_norm_type is None:
                layer_norm_fn = nn.LayerNorm
            elif config.layer_norm_type == "rms":
                layer_norm_fn = LlamaRMSNorm
            self.ln_f = layer_norm_fn(config.hidden_dim,
                                      eps=config.layer_norm_epsilon,
                                      dtype=dtype,
                                      device=device)

        self.ckpt_attn = False
        self.ckpt_mlp = False
        self.ckpt_full = False

    def gradient_checkpointing_enable(self, attn: bool = False, mlp: bool = False):
        """Called by backend"""
        if attn or mlp:
            self.ckpt_attn = attn
            self.ckpt_mlp = mlp
        else:
            self.ckpt_full = True

    def forward(self, x: PipeTransferData, y: PipeCacheData) -> PipeTransferData:
        pp_input = x.pp_input
        cu_seqlens = x.cu_seqlens
        k_cache = y.k_cache
        v_cache = y.v_cache
        cache_seqlens = y.cache_seqlens
        max_seqlen = x.max_seqlen
        attention_mask = x.attention_mask
        if self.ckpt_full:
            pp_output, k, v = torch.utils.checkpoint.checkpoint(
                self._forward,
                pp_input,
                cu_seqlens,
                k_cache,
                v_cache,
                cache_seqlens,
                max_seqlen,
                attention_mask,
                False,
                False,
            )
        else:
            pp_output, k, v = self._forward(
                pp_input,
                cu_seqlens,
                k_cache,
                v_cache,
                cache_seqlens,
                max_seqlen,
                attention_mask,
                ckpt_attn=self.ckpt_attn,
                ckpt_mlp=self.ckpt_mlp,
            )

        x.pp_output = pp_output
        if x.store_kv_cache:
            if y.k_cache is None:
                y.k_cache = k.detach()
            if y.v_cache is None:
                y.v_cache = v.detach()
            if y.cache_seqlens is None and x.cu_seqlens is not None:
                y.cache_seqlens = x.cu_seqlens[1:] - x.cu_seqlens[:-1]
        return x

    def _forward(
        self,
        pp_input: torch.Tensor,
        cu_seqlens: torch.Tensor,
        k_cache: Optional[torch.Tensor],
        v_cache: Optional[torch.Tensor],
        cache_seqlens: Optional[torch.Tensor],
        max_seqlen: int,
        attention_mask: Optional[torch.Tensor],
        ckpt_attn: Optional[bool] = False,
        ckpt_mlp: Optional[bool] = False,
    ) -> PipeTransferData:
        h = pp_input
        if ckpt_attn:
            attn_out, k, v = torch.utils.checkpoint.checkpoint(self.attn, h, cu_seqlens, k_cache, v_cache,
                                                               cache_seqlens, attention_mask, max_seqlen)
        else:
            attn_out, k, v = self.attn(
                hidden_states=h,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                k_cache=k_cache,
                v_cache=v_cache,
                cache_seqlens=cache_seqlens,
                attention_mask=attention_mask,
            )
        h = h + attn_out
        if ckpt_mlp:
            h = torch.utils.checkpoint.checkpoint(self.mlp, h) + h
        else:
            h = self.mlp(h) + h
        if self.output_layernorm:
            h = self.ln_f(h)
        return h, k, v


class SequenceParallelCriticHead(nn.Linear):

    def forward(self, x: PipeTransferData, y: PipeCacheData) -> PipeTransferData:
        all_gather_buffer = tensor_parallel.gather_from_sequence_parallel_region(x.pp_input)
        x.pp_output = nn.functional.linear(all_gather_buffer, self.weight, self.bias)
        return x


class SequenceParallelActorHead(ColumnParallelLinear):

    def forward(self, x: PipeTransferData, y: PipeCacheData) -> PipeTransferData:
        x.pp_output = parallel_lm_logits(
            x.pp_input,
            self.weight,
            parallel_output=True,
            async_tensor_model_parallel_allreduce=self.async_tensor_model_parallel_allreduce,
            sequence_parallel=self.sequence_parallel,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            bias=self.bias,
        )
        # NOTE: the output is not the whole logits, but the logits for a part of tokens due to ColumnParallelLinear.
        # (However, data along the batch dim is all-gathered. No sequence parallel any more.)
        return x


class ParallelFlashMQATBase(nn.Module):

    def __init__(
        self,
        config: FlashMQATConfig,
        sequence_parallel: Optional[bool] = False,
        gradient_accumulation_fusion: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.device = device
        self.embedding_layer = ParallelVocabPositionEmbedding(
            config,
            sequence_parallel=sequence_parallel,
            dtype=dtype,
            device=device,
        )
        self.h = nn.ModuleList([
            ParallelFlashMQATBlock(
                config,
                layer_index=i,
                output_layernorm=(i == config.n_layers - 1),
                sequence_parallel=sequence_parallel,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
                dtype=dtype,
                device=device,
            ) for i in range(config.n_layers)
        ])

    def to_layers(self) -> List[nn.Module]:
        return [self.embedding_layer] + list(self.h)

    def forward(self, x: PipeTransferData, ys: List[PipeCacheData]) -> PipeTransferData:
        layers = self.to_layers()
        assert len(ys) == len(layers), (len(ys), len(layers))
        raw_pp_input = x.pp_input
        for i, (layer, y) in enumerate(zip(layers, ys)):
            x = layer(x, y)  # This will set pp_output.
            x.pp_input = x.pp_output
        # Finally, pp_input is the input of this pipeline stage,
        # pp_output is the output of this pipeline stage.
        # In the first stage, pp_input is None.
        x.pp_input = raw_pp_input
        return x


class ModelParallelModule(nn.Module):
    """TODO: change into lower level module substitution
    Currently only substitute FlashMQATModel into a ParallelFlashMQATBase
    """

    def __init__(
        self,
        module: Union[HuggingfaceLikeFlashMQATForCausalLM, DeepSpeedChatLikeFlashMQATCriticModel],
        config: FlashMQATConfig,
        sequence_parallel: Optional[bool] = False,
        gradient_accumulation_fusion: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()

        self.is_critic = module.is_critic
        self.sequence_parallel = sequence_parallel
        self.module = module
        self.module.transformer = ParallelFlashMQATBase(
            config,
            sequence_parallel=sequence_parallel,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
            dtype=dtype,
            device=device)  # overwrite
        if self.is_critic and sequence_parallel:
            # Critic, we only replace the head when using sequence_parallel,
            # as we need to gather input along the batch dim.
            self.module.head = SequenceParallelCriticHead(
                config.hidden_dim,
                1,
                bias=False,
                device=device,
                dtype=dtype,
            )
        elif not self.is_critic:
            # Actor. No matter sequence parallel or not, we replace the head
            # to output logits of partial tokens.
            self.module.head = SequenceParallelActorHead(
                config.hidden_dim,
                config.vocab_size,
                sequence_parallel=sequence_parallel,
                async_tensor_model_parallel_allreduce=not sequence_parallel,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
                bias=False,
                device=device,
                dtype=dtype,
            )

        self.num_checkpoint_shards = 1

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        packed_input_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        assert (packed_input_ids is not None
                and cu_seqlens is not None), "Model parallel module only accept packed inputs."
        pad_size = 0
        if self.sequence_parallel:
            packed_input_ids, cu_seqlens, max_seqlen, pad_size = pad_sequence_parallel_input(
                packed_input_ids, cu_seqlens, max_seqlen)
        output = self.module.forward(input_ids, attention_mask, packed_input_ids, cu_seqlens, max_seqlen)

        if self.sequence_parallel and pad_size > 0:
            if torch.is_tensor(output):
                output = output[:-pad_size]
            else:
                output.logits = output.logits[:-pad_size]
        return output

    def generate(self, *args, **kwargs):
        if self.sequence_parallel:
            raise NotImplementedError("Generation is not supported yet for pure model+sequence parallel.")
        return self.module.generate(*args, **kwargs)

    def gradient_checkpointing_enable(self, attn: bool = False, mlp: bool = False):
        self.module.gradient_checkpointing_enable(attn, mlp)

    def load(self, load_dir: str, init_critic_from_actor: bool = False):
        mp_rank = base.constants.model_parallel_rank()
        state_dict, n_shards = load_from_disk(load_dir,
                                              fn_pattern=r".*" + f"-pp-00-mp-{mp_rank:02d}-" + r"s-(\d{2}).*",
                                              return_n_shards=True)

        if init_critic_from_actor and f"{self.config.n_layers + 1}.weight" in state_dict:
            state_dict.pop(f"{self.config.n_layers + 1}.weight")
            self.module.load_state_dict(state_dict, strict=False)
        else:
            self.module.load_state_dict(state_dict, strict=True)

    def save(self, save_dir):
        dp_rank = base.constants.data_parallel_rank()
        mp_rank = base.constants.model_parallel_rank()
        if dp_rank > 0:  # only save on dp_rank = 0
            return

        save_to_disk(
            self.module.state_dict(),
            save_dir,
            output_fn=f"pytorch_model-pp-00-mp-{mp_rank:02d}-s-" + "{shard:02d}.bin",
            save_type="pt",
            n_shards=self.num_checkpoint_shards,
        )
