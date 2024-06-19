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

from realhf.api.core import model_api
from realhf.impl.model.modules import (
    CausalSelfAttentionLayer,
    LayerNormMLP,
    LlamaLayerNormMLP,
    LlamaRMSNorm,
)
from realhf.impl.model.parallelism.model_parallel.modules import (
    ColumnParallelLinear,
    parallel_lm_logits,
    ParallelEmbedding,
    RowParallelLinear,
)
from realhf.impl.model.utils.functional import compute_varlen_position_indices
import realhf.base.constants as constants
import realhf.base.logging as logging
import realhf.impl.model.parallelism.model_parallel.mappings as tensor_parallel

logger = logging.getLogger("ReaLModelBase")


@dataclasses.dataclass
class PipeTransferData:
    """Data structure for transferring data between stages.

    Each pipeline stage has exactly one PipeTransferData as the input and the output,
    no matter how many layers are in this stage.

    Attributes:
        pp_input: The input to the current stage. Usually hidden states
            with shape [bs, seq_len, hidden_dim].
        pp_output: The output of the current stage, also the input to the next stage.
            Usually hidden states with shape [bs, seq_len, hidden_dim].
        cu_seqlens: The cumulative sequence lengths of packed input_ids.
            Used by flash_attn_varlen_func. Will not be used during generation.
            It's configuration-like data that must be transfered from the first stage
            to the last. Shape [bs + 1].
        max_seqlen: The maximum sequence length of packed input_ids.
            Used by flash_attn_varlen_func. Will not be used during generation.
            It's configuration-like data that must be transfered from the first stage
            to the last.
        store_kv_cache: Whether to store the key and value cache for generation.
        attention_mask: The attention mask of the input, the same as huggingface transformers.
            Used by torch_attn_func to examine the outputs of PyTorch attention and flash
            attention are the same. Only for debugging. Shape [bs, seq_len].
    """

    pp_input: torch.Tensor = None
    pp_output: torch.Tensor = None

    # The followings are "configuration"-like data that should be passed across all stages.
    cu_seqlens: torch.Tensor = None
    max_seqlen: int = None
    store_kv_cache: bool = False


@dataclasses.dataclass
class PipeCacheData:
    """Data structure for caching data locally that will not be trasferred.

    Each layer has exactly one PipeCacheData as the input.
    If a pipeline stage has multiple layers, a list of PipeCacheData should be passed
    as the input. The cached tensors will be changed in-place.

    Attributes:
        input_ids: The input token ids. Used only at the first stage.
            Can be packed with shape [total_seq_len] or unpacked with shape [bs, seq].
        prompt_mask: Prompt mask used
        position_ids: Input position IDs. Can be resolved automatically in most cases.
            Used only at the first stage. The same shape as input_ids.
            If None, will be resolved automatically.
        k_cache: Key cache used for generation, shape [bs, max_seq, n_kv_heads, head_dim].
            Note that this is the cache for a specific layer, not for all layers.
        v_cache: Value cache used for generation, shape [bs, max_seq, n_kv_heads, head_dim].
            Note that this is the cache for a specific layer, not for all layers.
        cache_seqlens: The sequence lengths of the cached tokens. Used for generation. Shape [bs].
    """

    # Only cached in the first stage.
    packed_input_ids: torch.Tensor = None
    packed_position_ids: torch.Tensor = None
    # Cached in each transformer layer.
    k_cache: torch.Tensor = None
    v_cache: torch.Tensor = None
    cache_seqlens: torch.Tensor = None


class ReaLModelBlock(nn.Module):

    def __init__(
        self,
        config: model_api.ReaLModelConfig,
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

    def forward(
        self, x: PipeTransferData, y: PipeCacheData
    ) -> PipeTransferData:
        pp_input = x.pp_input
        cu_seqlens = x.cu_seqlens
        k_cache = y.k_cache
        v_cache = y.v_cache
        cache_seqlens = y.cache_seqlens
        max_seqlen = x.max_seqlen
        if constants.gradient_checkpointing():
            pp_output, k, v = torch.utils.checkpoint.checkpoint(
                self._forward,
                pp_input,
                cu_seqlens,
                k_cache,
                v_cache,
                cache_seqlens,
                max_seqlen,
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = pp_input
        attn_out, k, v = self.attn(
            hidden_states=h,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens,
        )
        h = h + attn_out
        h = self.mlp(h) + h
        if self.output_layernorm:
            h = self.ln_f(h)
        return h, k, v


class VocabPositionEmbedding(nn.Module):

    def __init__(
        self,
        config: model_api.ReaLModelConfig,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.n_positions = config.n_positions

        model_parallel = constants.model_parallel_world_size() > 1
        if model_parallel:
            embed_cls = ParallelEmbedding
        else:
            embed_cls = nn.Embedding

        self.wte = embed_cls(
            config.vocab_size, config.hidden_dim, dtype=dtype, device=device
        )

        self.apply_abs_pos_embed = not config.apply_rotary
        if self.apply_abs_pos_embed:
            self.wpe = embed_cls(
                config.n_positions,
                config.hidden_dim,
                dtype=dtype,
                device=device,
            )

        self.embed_drop = nn.Dropout(config.embd_pdrop)

        self.self_attention_mask = torch.tril(
            torch.ones(
                (config.n_positions, config.n_positions),
                dtype=torch.bool,
                device=device,
            )
        )

    def forward(
        self, x: PipeTransferData, y: PipeCacheData
    ) -> PipeTransferData:
        # Set position ids.
        # if y.packed_position_ids is not None:
        #     raise ValueError("In our use cases, position_ids must be None.")
        y.packed_position_ids = compute_varlen_position_indices(
            total_seqlen=y.packed_input_ids.shape[0],
            cu_seqlens=x.cu_seqlens,
            seqlen_offsets=y.cache_seqlens,
        )
        if x.max_seqlen > self.n_positions:
            raise ValueError(
                f"max_seqlen ({x.max_seqlen}) must be <= n_positions ({self.n_positions})."
            )
        assert y.packed_position_ids.shape == y.packed_input_ids.shape, (
            y.packed_position_ids.shape,
            y.packed_input_ids.shape,
            x.cu_seqlens,
        )

        x.pp_output = self._forward(y.packed_input_ids, y.packed_position_ids)
        return x

    def _forward(
        self, input_ids: torch.LongTensor, position_ids: torch.LongTensor
    ) -> torch.Tensor:
        inputs_embeds = self.wte(input_ids)
        if self.apply_abs_pos_embed:
            inputs_embeds = inputs_embeds + self.wpe(position_ids)
        if constants.sequence_parallel():
            inputs_embeds = tensor_parallel.scatter_to_sequence_parallel_region(
                inputs_embeds
            )
            # `scatter_to_sequence_parallel_region` returns a view, which prevents
            # the original tensor from being garbage collected. Clone to facilitate GC.
            # Has a small runtime cost (~0.5%).
            inputs_embeds = inputs_embeds.clone()
            # with tensor_parallel.get_cuda_rng_tracker().fork():
            x = self.embed_drop(inputs_embeds)
        else:
            x = self.embed_drop(inputs_embeds)
        return x


class OutputHead(nn.Linear):

    def forward(
        self, x: PipeTransferData, y: PipeCacheData
    ) -> PipeTransferData:
        x.pp_output = nn.functional.linear(x.pp_input, self.weight, self.bias)
        return x

    def _forward(self, x: torch.Tensor):
        return super().forward(x)


class SequenceParallelCriticHead(nn.Linear):

    def forward(
        self, x: PipeTransferData, y: PipeCacheData
    ) -> PipeTransferData:
        all_gather_buffer = (
            tensor_parallel.gather_from_sequence_parallel_region(x.pp_input)
        )
        x.pp_output = nn.functional.linear(
            all_gather_buffer, self.weight, self.bias
        )
        return x

    def _forward(self, x: torch.Tensor):
        x = tensor_parallel.gather_from_sequence_parallel_region(x)
        return super().forward(x)


class SequenceParallelActorHead(ColumnParallelLinear):

    def forward(
        self, x: PipeTransferData, y: PipeCacheData
    ) -> PipeTransferData:
        x.pp_output = parallel_lm_logits(
            x.pp_input,
            self.weight,
            parallel_output=True,
            async_tensor_model_parallel_allreduce=self.async_tensor_model_parallel_allreduce,
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
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            bias=self.bias,
        )


# Paramter count, used for partitioning pipeline stages
# Ignoring tensor model parallel since it will evenly partition parameters.
def real_model_embed_param_count(config: model_api.ReaLModelConfig) -> int:
    count = config.vocab_size * config.hidden_dim
    if not config.apply_rotary:
        count += config.n_positions * config.hidden_dim
    return count


def real_model_embedding_param_keys(config: model_api.ReaLModelConfig) -> int:
    keys = ["0.wte.weight"]
    if not config.apply_rotary:
        keys += ["0.wpe.weight"]
    return keys


def real_model_tblock_param_count(
    config: model_api.ReaLModelConfig, idx: int
) -> int:
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
    count += (
        ln_count
        + config.head_dim * (nq + config.n_kv_heads * 2) * config.hidden_dim
    )
    if config.use_attention_bias:
        count += config.head_dim * (nq + config.n_kv_heads * 2)

    # attention projection
    count += config.hidden_dim * config.hidden_dim
    if config.use_attention_bias:
        count += config.hidden_dim
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


def real_model_tblock_param_keys(
    config: model_api.ReaLModelConfig, idx: int
) -> List[str]:
    keys = [
        f"{idx + 1}.attn.c_attn.ln.weight",
        f"{idx + 1}.attn.c_attn.q_attn.weight",
        f"{idx + 1}.attn.c_attn.k_attn.weight",
        f"{idx + 1}.attn.c_attn.v_attn.weight",
    ]
    if config.layer_norm_type is None:
        keys += [f"{idx + 1}.attn.c_attn.ln.bias"]
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
    if config.layer_norm_type is None:
        keys += [f"{idx + 1}.mlp.ln.bias"]
    if config.mlp_type is None:
        keys += [
            f"{idx + 1}.mlp.c_fc.weight",
            f"{idx + 1}.mlp.c_proj.weight",
            f"{idx + 1}.mlp.c_fc.bias",
            f"{idx + 1}.mlp.c_proj.bias",
        ]
    elif config.mlp_type == "llama":
        keys += [
            f"{idx + 1}.mlp.gate_proj.weight",
            f"{idx + 1}.mlp.up_proj.weight",
            f"{idx + 1}.mlp.down_proj.weight",
        ]
    else:
        raise NotImplementedError()
    if idx == config.n_layers - 1:
        keys += [f"{idx + 1}.ln_f.weight"]
        if config.layer_norm_type is None:
            keys += [f"{idx + 1}.ln_f.bias"]
    return keys


def real_model_head_param_count(config: model_api.ReaLModelConfig) -> int:
    # NOTE: To hold consistent partitions between actor and critic models,
    # we count the number of parameters of the critic head as config.hidden_dim * config.vocab_size.
    # This is the intended behavior rather than a bug.
    return config.hidden_dim * config.vocab_size


def real_model_head_param_keys(config: model_api.ReaLModelConfig) -> List[str]:
    return [f"{config.n_layers + 1}.weight"]


def keys_from_layer_indices(
    config: model_api.ReaLModelConfig, layer_indices: List[int]
) -> List[str]:
    # assert _is_integer_list_contiguous(layer_indices)
    sd_keys = []
    for layer_idx in layer_indices:
        if layer_idx == 0:
            sd_keys += real_model_embedding_param_keys(config)
        elif layer_idx == config.n_layers + 1:
            sd_keys += real_model_head_param_keys(config)
        else:
            sd_keys += real_model_tblock_param_keys(config, layer_idx - 1)
    return sd_keys
