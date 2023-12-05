from typing import Callable, List, Optional, Tuple, Union
import copy
import dataclasses
import json
import math
import os
import queue

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers

from impl.model.nn.flash_mqat.flash_mqat_base import FlashMQATConfig
from impl.model.utils.data import PipeCacheData, PipeTransferData
from impl.model.utils.logits_warper import top_k_top_p_logits
from impl.model.utils.modules import LayerNormLinear, LayerNormMLP
import api.huggingface
import api.model
import base.logging as logging

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
except ModuleNotFoundError:
    pass
from impl.model.utils.model_parallel.modules import (ColumnParallelLinear, LayerNormColumnLinear,
                                                     LayerNormParallelMLP, ParallelEmbedding,
                                                     RowParallelLinear)
import base.logging as logging

logger = logging.getLogger("TensorParallelFlashMQAT")


class VocabPositionEmbedding(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        n_positions: int,
        hidden_dim: int,
        embed_pdrop: float,
        fixed_abs_position_ids: bool,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.n_positions = n_positions
        self.wte = ParallelEmbedding(vocab_size, hidden_dim, dtype=dtype, device=device)
        self.wpe = ParallelEmbedding(n_positions, hidden_dim, dtype=dtype, device=device)
        self.embed_drop = nn.Dropout(embed_pdrop)

        self.self_attention_mask = torch.tril(
            torch.ones((n_positions, n_positions), dtype=torch.bool, device=device))
        self.fixed_abs_position_ids = fixed_abs_position_ids

    def forward(self, x: PipeTransferData, y: PipeCacheData) -> PipeTransferData:
        is_gen = y.cache_seqlens is not None
        if is_gen and len(y.input_ids.shape) == 2:
            assert y.input_ids.shape[1] == 1
        elif is_gen and len(y.input_ids.shape) == 1:
            if x.cu_seqlens is None:
                y.input_ids = y.input_ids.unsqueeze(-1)
        packed = len(y.input_ids.shape) == 1
        if packed and ((x.cu_seqlens is None) or (x.max_seqlen is None)):
            raise ValueError("cu_seqlens and max_seqlen must be both provided for packed input.")

        # Set position ids.
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
        position_embeds = self.wpe(y.position_ids)
        x.pp_output = self.embed_drop(inputs_embeds + position_embeds)
        return x


class CausalSelfAttentionLayer(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        n_kv_heads: int,
        head_dim: int,
        resid_pdrop: float,
        attn_pdrop: float,
        layer_index: int,
        layer_norm_epsilon: float,
        scale_attn_by_inverse_layer_idx: bool,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        if dtype is None:
            dtype = torch.float16
        assert device != "cpu", "Tensor Parallel FlashMQAT does not support CPU."
        assert hidden_dim % head_dim == 0
        n_q_heads = hidden_dim // head_dim
        if n_kv_heads == n_q_heads:
            # multi-head attention, still use combined layer norm + linear
            self.c_attn = LayerNormColumnLinear(
                hidden_dim,
                head_dim * (n_q_heads + 2 * n_kv_heads),
                layer_norm_epsilon=layer_norm_epsilon,
                dtype=dtype,
                device=device,
            )
        elif n_kv_heads == 1:
            # multi-query attention, use separate layer norm + linear
            # becasue only query is splitted for tensor parallel
            self.ln = nn.LayerNorm(hidden_dim, eps=layer_norm_epsilon, dtype=dtype, device=device)
            self.q_attn = ColumnParallelLinear(
                hidden_dim,
                head_dim * n_q_heads,
                dtype=dtype,
                device=device,
            )
            self.kv_attn = nn.Linear(hidden_dim, head_dim * (n_kv_heads * 2), dtype=dtype, device=device)
        else:
            raise NotImplementedError("Currently tensor parallel FlashMQAT only supports MHA and MQA.")
        self.c_proj = RowParallelLinear(hidden_dim, hidden_dim, dtype=dtype, device=device)
        self.resid_dropout = nn.Dropout(resid_pdrop)

        self.attn_pdrop = attn_pdrop

        self.applied_attn_pdrop = attn_pdrop

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
        # input shape: [bs, seq, hidden_dim]
        # default upcast, scale
        if self.scale_attn_by_inverse_layer_idx:
            unscale = self.layer_index + 1
            scale_factor = unscale**-1
        else:
            unscale = 1.0
            scale_factor = 1
        scale_factor /= self.d**0.5

        if self.nkv == self.nq:
            # multi-head attention
            qkv, _ = self.c_attn(hidden_states)
            q, k, v = torch.split(qkv, (self.d * self.nq, self.d * self.nkv, self.d * self.nkv), dim=-1)
        elif self.nkv == 1:
            # multi-query attention
            hidden_states = self.ln(hidden_states)
            q, _ = self.q_attn(hidden_states)
            kv = self.kv_attn(hidden_states)
            k, v = torch.split(kv, (self.d * self.nkv, self.d * self.nkv), dim=-1)
        else:
            raise NotImplementedError("Currently tensor parallel FlashMQAT only supports MHA and MQA.")

        if k_cache is not None and len(qkv.shape) == 3:
            # k_cache/v_cache shape: [bs, max_seq, n_kv_heads, head_dim]
            if cache_seqlens is None:
                raise RuntimeError("cache_seqlens must be provided if kv_cache is not None.")
            assert q.shape[1] == 1, (qkv.shape, "Can only generate one token at a time.")
            # q, k, v = torch.split(qkv, (self.d * self.nq, self.d * self.nkv, self.d * self.nkv), dim=-1)
            q = q.view(*q.shape[:2], self.nq, self.d)
            v = v.view(*v.shape[:2], self.nkv, self.d)
            k = k.view(*k.shape[:2], self.nkv, self.d)
            # k_cache and v_cache will be modified in-place.
            hidden_states = flash_attn_with_kvcache(
                q,
                k_cache,
                v_cache,
                k=k,
                v=v,
                cache_seqlens=cache_seqlens,
                softmax_scale=scale_factor,
                causal=False,
                num_splits=1,
            )
        elif k_cache is not None and len(q.shape) == 2:
            # q, k, v = torch.split(qkv, (self.d * self.nq, self.d * self.nkv, self.d * self.nkv), dim=-1)
            q = q.view(q.shape[0], self.nq, self.d)
            v = v.view(v.shape[0], self.nkv, self.d)
            k = k.view(k.shape[0], self.nkv, self.d)
            # FIXME: The following code is crazily slow. We should implement them as a customized kernel?
            qlens = cu_seqlens[1:] - cu_seqlens[:-1]
            offset = 0
            new_k, new_v = [], []
            for i, (qlen, cache_len) in enumerate(zip(qlens, cache_seqlens)):
                new_k += [k_cache[i, :cache_len], k[offset:offset + qlen]]
                new_v += [v_cache[i, :cache_len], v[offset:offset + qlen]]
                with torch.no_grad():
                    k_cache[i, cache_len:cache_len + qlen] = k[offset:offset + qlen].detach()
                    v_cache[i, cache_len:cache_len + qlen] = v[offset:offset + qlen].detach()
                offset += qlen
            k, v = torch.cat(new_k), torch.cat(new_v)
            kv_seqlens = qlens + cache_seqlens
            max_kv_seqlen = int(kv_seqlens.max())
            kv_cu_seqlens = torch.cat([kv_seqlens.new_zeros(1), kv_seqlens.cumsum(0)])
            hidden_states = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens.int(),
                cu_seqlens_k=kv_cu_seqlens.int(),
                max_seqlen_q=int(max_seqlen),
                max_seqlen_k=int(max_kv_seqlen),
                dropout_p=0.0,
                softmax_scale=scale_factor,
                causal=True,
            )
        elif cu_seqlens is not None:
            assert max_seqlen is not None
            assert len(q.shape) == 2
            # q, k, v = torch.split(qkv, (self.d * self.nq, self.d * self.nkv, self.d * self.nkv), dim=-1)
            q = q.view(q.shape[0], self.nq, self.d)
            v = v.view(v.shape[0], self.nkv, self.d)
            k = k.view(k.shape[0], self.nkv, self.d)
            hidden_states = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens.int(),
                cu_seqlens.int(),
                int(max_seqlen),
                int(max_seqlen),
                dropout_p=self.applied_attn_pdrop,
                softmax_scale=scale_factor,
                causal=True,
            )
        else:
            # q, k, v = torch.split(qkv, (self.d * self.nq, self.d * self.nkv, self.d * self.nkv), dim=-1)
            k = k.view(*k.shape[:2], self.nkv, self.d)
            v = v.view(*v.shape[:2], self.nkv, self.d)
            q = q.view(*q.shape[:2], self.nq, self.d)
            hidden_states = flash_attn_func(q,
                                            k,
                                            v,
                                            dropout_p=self.applied_attn_pdrop,
                                            softmax_scale=scale_factor,
                                            causal=True)
        hidden_states, _ = self.c_proj(hidden_states.flatten(start_dim=-2))
        hidden_states = self.resid_dropout(hidden_states)
        return hidden_states, k, v


class FlashMQATBlock(nn.Module):

    def __init__(
        self,
        config: FlashMQATConfig,
        layer_index: int,
        output_layernorm: bool = False,
        ckpt_attn: bool = False,
        ckpt_mlp: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.attn = CausalSelfAttentionLayer(
            hidden_dim=config.hidden_dim,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            resid_pdrop=config.resid_pdrop,
            attn_pdrop=config.attn_pdrop,
            layer_index=layer_index,
            layer_norm_epsilon=config.layer_norm_epsilon,
            scale_attn_by_inverse_layer_idx=config.scale_attn_by_inverse_layer_idx,
            dtype=dtype,
            device=device,
        )
        self.mlp = LayerNormParallelMLP(
            hidden_dim=config.hidden_dim,
            intermediate_dim=config.intermediate_dim,
            resid_pdrop=config.resid_pdrop,
            activation_function=config.activation_function,
            layer_norm_epsilon=config.layer_norm_epsilon,
            dtype=dtype,
            device=device,
        )
        self.output_layernorm = output_layernorm
        if output_layernorm:
            self.ln_f = nn.LayerNorm(config.hidden_dim,
                                     eps=config.layer_norm_epsilon,
                                     dtype=dtype,
                                     device=device)

        self.ckpt_attn = ckpt_attn
        self.ckpt_mlp = ckpt_mlp

    def gradient_checkpointing_enable(self, attn: bool = True, mlp: bool = True):
        self.ckpt_attn = attn
        self.ckpt_mlp = mlp

    def forward(self, x: PipeTransferData, y: PipeCacheData) -> PipeTransferData:
        h = x.pp_input
        if self.ckpt_attn:
            attn_out, k, v = torch.utils.checkpoint.checkpoint(
                self.attn,
                h,
                x.cu_seqlens,
                y.k_cache,
                y.v_cache,
                y.cache_seqlens,
                x.attention_mask,
                x.max_seqlen,
                use_reentrant=True,
            )
        else:
            attn_out, k, v = self.attn(
                hidden_states=h,
                cu_seqlens=x.cu_seqlens,
                max_seqlen=x.max_seqlen,
                k_cache=y.k_cache,
                v_cache=y.v_cache,
                cache_seqlens=y.cache_seqlens,
                attention_mask=x.attention_mask,
            )
        h = h + attn_out
        if self.ckpt_mlp:
            h = torch.utils.checkpoint.checkpoint(self.mlp, h, use_reentrant=True) + h
        else:
            h = self.mlp(h) + h
        if self.output_layernorm:
            h = self.ln_f(h)
        x.pp_output = h
        # Set kv cache during the first forward pass of generation.
        # Do we need an option to disable this?
        # TODO: option to disable this to avoid redundant kvcache store
        if x.store_kv_cache:
            if y.k_cache is None:
                y.k_cache = k.detach()
            if y.v_cache is None:
                y.v_cache = v.detach()
            if y.cache_seqlens is None and x.cu_seqlens is not None:
                y.cache_seqlens = x.cu_seqlens[1:] - x.cu_seqlens[:-1]
        return x


class FlashMQATBase(nn.Module):

    def __init__(
        self,
        config: FlashMQATConfig,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.device = device
        self.embedding_layer = VocabPositionEmbedding(
            config.vocab_size,
            config.n_positions,
            config.hidden_dim,
            config.embd_pdrop,
            fixed_abs_position_ids=config.fixed_abs_position_ids,
            dtype=dtype,
            device=device,
        )
        self.h = nn.ModuleList([
            FlashMQATBlock(
                config,
                layer_index=i,
                output_layernorm=(i == config.n_layers - 1),
                ckpt_attn=(i > 0 and config.ckpt_attn),
                ckpt_mlp=(i > 0 and config.ckpt_mlp),
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
        for layer, y in zip(layers, ys):
            x = layer(x, y)  # This will set pp_output.
            x.pp_input = x.pp_output
        # Finally, pp_input is the input of this pipeline stage,
        # pp_output is the output of this pipeline stage.
        # In the first stage, pp_input is None.
        x.pp_input = raw_pp_input
        return x
