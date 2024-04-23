from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
import contextlib
import dataclasses
import functools
import json
import os

import torch
import torch.nn as nn
import torch.utils.checkpoint
import transformers

from reallm.api.quickstart.model import FlashMQATConfig
from reallm.impl.model.modules import CausalSelfAttentionLayer, LayerNormMLP, LlamaLayerNormMLP, LlamaRMSNorm
from reallm.impl.model.parallelism.model_parallel.modules import (ColumnParallelLinear, parallel_lm_logits,
                                                                  ParallelEmbedding, RowParallelLinear)
from reallm.impl.model.utils.data import PipeCacheData, PipeTransferData
from reallm.impl.model.utils.functional import compute_varlen_position_indices
from reallm.impl.model.utils.save_load import get_ckpt_spec, load_from_disk, save_to_disk
import reallm.base.constants as constants
import reallm.base.logging as logging
import reallm.impl.model.parallelism.model_parallel.mappings as tensor_parallel

logger = logging.getLogger("FlashMQATBase")


class FlashMQATBlock(nn.Module):

    def __init__(
        self,
        config: FlashMQATConfig,
        layer_index: int,
        output_layernorm: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        if dtype is None:
            dtype = torch.float16
        self.layer_index = layer_index
        self.attn = CausalSelfAttentionLayer(
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
            model_parallel=constants.model_parallel_world_size() > 1,
            sequence_parallel=config.sequence_parallel,
            gradient_accumulation_fusion=config.gradient_accumulation_fusion,
            dtype=dtype,
            device=device,
        )
        if config.mlp_type is None:
            self.mlp = LayerNormMLP(
                hidden_dim=config.hidden_dim,
                intermediate_dim=config.intermediate_dim,
                resid_pdrop=config.resid_pdrop,
                activation_function=config.activation_function,
                layer_norm_epsilon=config.layer_norm_epsilon,
                model_parallel=constants.model_parallel_world_size() > 1,
                sequence_parallel=config.sequence_parallel,
                gradient_accumulation_fusion=config.gradient_accumulation_fusion,
                dtype=dtype,
                device=device,
            )
        elif config.mlp_type == "llama":
            self.mlp = LlamaLayerNormMLP(
                hidden_dim=config.hidden_dim,
                intermediate_dim=config.intermediate_dim,
                activation_function=config.activation_function,
                layer_norm_epsilon=config.layer_norm_epsilon,
                model_parallel=constants.model_parallel_world_size() > 1,
                sequence_parallel=config.sequence_parallel,
                gradient_accumulation_fusion=config.gradient_accumulation_fusion,
                dtype=dtype,
                device=device,
            )
        self.output_layernorm = output_layernorm
        if output_layernorm:
            if config.layer_norm_type is None:
                layer_norm_fn = nn.LayerNorm
            elif config.layer_norm_type == "rms":
                layer_norm_fn = LlamaRMSNorm
            self.ln_f = layer_norm_fn(
                config.hidden_dim,
                eps=config.layer_norm_epsilon,
                dtype=dtype,
                device=device,
            )

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

    def gradient_checkpointing_disable(self):
        self.ckpt_attn = False
        self.ckpt_mlp = False
        self.ckpt_full = False

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
                use_reentrant=True,
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = pp_input
        if ckpt_attn:
            attn_out, k, v = torch.utils.checkpoint.checkpoint(
                self.attn,
                h,
                cu_seqlens,
                k_cache,
                v_cache,
                cache_seqlens,
                attention_mask,
                max_seqlen,
                use_reentrant=True,
            )
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
            h = torch.utils.checkpoint.checkpoint(self.mlp, h, use_reentrant=True) + h
        else:
            h = self.mlp(h) + h
        if self.output_layernorm:
            h = self.ln_f(h)
        return h, k, v


class VocabPositionEmbedding(nn.Module):

    def __init__(
        self,
        config: FlashMQATConfig,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.n_positions = config.n_positions
        self.sequence_parallel = config.sequence_parallel

        model_parallel = constants.model_parallel_world_size() > 1
        if model_parallel:
            embed_cls = ParallelEmbedding
        else:
            embed_cls = nn.Embedding

        self.wte = embed_cls(config.vocab_size, config.hidden_dim, dtype=dtype, device=device)

        self.apply_abs_pos_embed = not config.apply_rotary
        if self.apply_abs_pos_embed:
            self.wpe = embed_cls(config.n_positions, config.hidden_dim, dtype=dtype, device=device)

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
            y.position_ids = compute_varlen_position_indices(total_seqlen=y.input_ids.shape[0],
                                                             cu_seqlens=x.cu_seqlens,
                                                             seqlen_offsets=y.cache_seqlens)
            # lengths = x.cu_seqlens[1:] - x.cu_seqlens[:-1]
            # if y.cache_seqlens is None:
            #     y.position_ids = torch.cat(
            #         [torch.arange(int(l), dtype=torch.int32, device=y.input_ids.device) for l in lengths])
            #     assert (y.position_ids < x.max_seqlen).all() and y.position_ids.max() == x.max_seqlen - 1
            # else:
            #     y.position_ids = torch.cat([
            #         torch.arange(int(l), dtype=torch.int32, device=y.input_ids.device) + cache_len
            #         for l, cache_len in zip(lengths, y.cache_seqlens)
            #     ])
            if x.max_seqlen > self.n_positions:
                raise ValueError(f"max_seqlen ({x.max_seqlen}) must be <= n_positions ({self.n_positions}).")
            assert y.position_ids.shape == y.input_ids.shape, (
                y.position_ids.shape,
                y.input_ids.shape,
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

        x.pp_output = self._forward(y.input_ids, y.position_ids)
        return x

    def sequence_parallel_enable(self, mode: bool):
        self.sequence_parallel = mode

    def _forward(self, input_ids: torch.LongTensor, position_ids: torch.LongTensor) -> torch.Tensor:
        inputs_embeds = self.wte(input_ids)
        if self.apply_abs_pos_embed:
            inputs_embeds = inputs_embeds + self.wpe(position_ids)
        if self.sequence_parallel:
            inputs_embeds = tensor_parallel.scatter_to_sequence_parallel_region(inputs_embeds)
            # `scatter_to_sequence_parallel_region` returns a view, which prevents
            # the original tensor from being garbage collected. Clone to facilitate GC.
            # Has a small runtime cost (~0.5%).
            # if self.config.clone_scatter_output_in_embedding:
            #     embeddings = embeddings.clone()
            # with tensor_parallel.get_cuda_rng_tracker().fork():
            x = self.embed_drop(inputs_embeds)
        else:
            x = self.embed_drop(inputs_embeds)
        return x


class OutputHead(nn.Linear):

    def forward(self, x: PipeTransferData, y: PipeCacheData) -> PipeTransferData:
        x.pp_output = nn.functional.linear(x.pp_input, self.weight, self.bias)
        return x

    def _forward(self, x: torch.Tensor):
        return super().forward(x)


class SequenceParallelCriticHead(nn.Linear):

    def forward(self, x: PipeTransferData, y: PipeCacheData) -> PipeTransferData:
        all_gather_buffer = tensor_parallel.gather_from_sequence_parallel_region(x.pp_input)
        x.pp_output = nn.functional.linear(all_gather_buffer, self.weight, self.bias)
        return x

    def _forward(self, x: torch.Tensor):
        x = tensor_parallel.gather_from_sequence_parallel_region(x)
        return super().forward(x)


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

    def _forward(self, x: torch.Tensor):
        return parallel_lm_logits(
            x,
            self.weight,
            parallel_output=True,
            async_tensor_model_parallel_allreduce=self.async_tensor_model_parallel_allreduce,
            sequence_parallel=self.sequence_parallel,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            bias=self.bias,
        )


# Paramter count, used for partitioning pipeline stages
# Ignoring tensor model parallel since it will evenly partition parameters.
def flash_model_embed_param_count(config: FlashMQATConfig) -> int:
    count = config.vocab_size * config.hidden_dim
    if not config.apply_rotary:
        count += config.n_positions * config.hidden_dim
    return count


def flash_model_embedding_param_keys(config: FlashMQATConfig) -> int:
    keys = ["0.wte.weight"]
    if not config.apply_rotary:
        keys += ["0.wpe.weight"]
    return keys


def flash_model_tblock_param_count(config: FlashMQATConfig, idx: int) -> int:
    count = 0
    nq = config.hidden_dim // config.head_dim

    if config.layer_norm_type is None:
        # nn.LayerNorm
        ln_count = 2 * config.hidden_dim
    elif config.layer_norm_type == "rms":
        # Llama RMSNorm
        ln_count = config.hidden_dim
    else:
        raise NotImplementedError()

    # layernorm qkv linear
    count += ln_count + config.head_dim * (nq + config.n_kv_heads * 2) * config.hidden_dim
    if config.use_attention_bias:
        count += config.head_dim * (nq + config.n_kv_heads * 2)

    # attention projection
    count += config.hidden_dim * config.hidden_dim
    if config.use_attention_bias:
        config += config.hidden_dim
    # NOTE: we ignore the parameters of RotoaryEmbedding here

    # mlp
    count += ln_count
    if config.mlp_type is None:
        count += 2 * config.hidden_dim * config.intermediate_dim
    elif config.mlp_type == "llama":
        count += 3 * config.hidden_dim * config.intermediate_dim
    else:
        raise NotImplementedError()

    if idx == config.n_layers - 1:
        count += ln_count
    return count


def flash_model_tblock_param_keys(config: FlashMQATConfig, idx: int) -> List[str]:
    keys = [
        f"{idx + 1}.attn.c_attn.ln.weight",
        f"{idx + 1}.attn.c_attn.q_attn.weight",
        f"{idx + 1}.attn.c_attn.k_attn.weight",
        f"{idx + 1}.attn.c_attn.v_attn.weight",
    ]
    if config.use_attention_bias:
        keys += [
            f"{idx + 1}.attn.c_attn.q_attn.bias",
            f"{idx + 1}.attn.c_attn.k_attn.bias",
            f"{idx + 1}.attn.c_attn.v_attn.bias",
        ]
    keys += [f"{idx + 1}.attn.c_proj.weight"]
    if config.use_attention_bias:
        keys += [f"{idx + 1}.attn.c_proj.bias"]
    keys += [f"{idx + 1}.mlp.ln.weight"]
    if config.mlp_type is None:
        keys += [f"{idx + 1}.mlp.c_fc.weight", f"{idx + 1}.mlp.c_proj.weight"]
    elif config.mlp_type == "llama":
        keys += [
            f"{idx + 1}.mlp.gate_proj.weight", f"{idx + 1}.mlp.up_proj.weight",
            f"{idx + 1}.mlp.down_proj.weight"
        ]
    else:
        raise NotImplementedError()
    if idx == config.n_layers - 1:
        keys += [f"{idx + 1}.ln_f.weight"]
    return keys


def flash_model_head_param_count(config: FlashMQATConfig) -> int:
    # NOTE: To hold consistent partitions between actor and critic models,
    # we count the number of parameters of the critic head as config.hidden_dim * config.vocab_size.
    # This is the intended behavior rather than a bug.
    return config.hidden_dim * config.vocab_size


def flash_model_head_param_keys(config: FlashMQATConfig) -> List[str]:
    return [f"{config.n_layers + 1}.weight"]
