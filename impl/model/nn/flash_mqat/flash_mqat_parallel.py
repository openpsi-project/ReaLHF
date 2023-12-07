from typing import Callable, Dict, List, Optional, Tuple, Union
import functools
import os

import torch
import torch.nn as nn
import torch.utils.checkpoint
import transformers

from impl.model.nn.flash_mqat.flash_mqat_base import FlashMQATConfig, FlashMQATModel
from impl.model.utils.data import PipeCacheData, PipeTransferData
from impl.model.utils.model_parallel.modules import (ColumnParallelLinear, LayerNormColumnLinear,
                                                     LayerNormParallelMLP, LlamaLayerNormParallelMLP,
                                                     ParallelEmbedding, RowParallelLinear)
from impl.model.utils.modules import LlamaRMSNorm
import base.constants
import base.logging as logging

try:
    from flash_attn import (flash_attn_func, flash_attn_varlen_func, flash_attn_varlen_func_with_kvcache,
                            flash_attn_with_kvcache)
    from flash_attn.layers.rotary import RotaryEmbedding
except ModuleNotFoundError:
    pass

from impl.model.utils.save_load import load_from_disk, save_to_disk

logger = logging.getLogger("ParallelFlashMQAT")


class ParallelVocabPositionEmbedding(nn.Module):

    def __init__(
        self,
        config: FlashMQATConfig,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.n_positions = config.n_positions
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

        self.ln = nn.LayerNorm(hidden_dim, eps=layer_norm_epsilon, dtype=dtype, device=device)

        self.q_attn = ColumnParallelLinear(
            hidden_dim,
            head_dim * n_q_heads,
            dtype=dtype,
            device=device,
        )
        self.mp_world_size = base.constants.model_parallel_world_size()
        if n_kv_heads > 1 and \
            n_kv_heads % self.mp_world_size == 0:
            # split model parallel among heads if possible
            self.k_attn = ColumnParallelLinear(hidden_dim, head_dim * n_kv_heads, dtype=dtype, device=device)
            self.v_attn = ColumnParallelLinear(hidden_dim, head_dim * n_kv_heads, dtype=dtype, device=device)
        else:
            if n_kv_heads > 1:
                logger.warning(f"Cannot split {n_kv_heads} kv heads evenly among "
                               f"{self.mp_world_size} model parallel ranks, "
                               f"use unsplitted linear for kv heads instead")
            self.k_attn = nn.Linear(hidden_dim, head_dim * n_kv_heads, dtype=dtype, device=device)
            self.v_attn = nn.Linear(hidden_dim, head_dim * n_kv_heads, dtype=dtype, device=device)

        self.c_proj = RowParallelLinear(hidden_dim, hidden_dim, dtype=dtype, device=device)
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
        # default upcast, scale
        if self.scale_attn_by_inverse_layer_idx:
            unscale = self.layer_index + 1
            scale_factor = unscale**-1
        else:
            unscale = 1.0
            scale_factor = 1
        scale_factor /= self.d**0.5

        q: torch.Tensor = self.q_attn(hidden_states)
        k: torch.Tensor = self.k_attn(hidden_states)
        v: torch.Tensor = self.v_attn(hidden_states)
        kv = torch.cat([k, v], dim=-1)

        q = q.view(*q.shape[:-1], self.nq, self.d)
        kv = kv.view(*kv.shape[:-1], 2, self.nkv, self.d)

        if self.apply_rotary and k_cache is None:
            # otherwise, we input rotary cos/sin directly into flash_attn_with_kvcache
            q, kv = self.rotary_emb(
                q,
                kv,
                seqlen_offset=0,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        elif self.apply_rotary:
            self.rotary_emb._update_cos_sin_cache(k_cache.shape[1], device=q.device, dtype=q.dtype)
            # Rotary cos/sin will be automatically offset by cache_seqlens in flash_attn.
            rotary_cos, rotary_sin = self.rotary_emb._cos_cached, self.rotary_emb._sin_cached
        else:
            rotary_cos = rotary_sin = None

        k, v = kv.unbind(dim=-3)

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
                max_seqlen_q=int(max_seqlen),
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
                cu_seqlens.int(),
                cu_seqlens.int(),
                int(max_seqlen),
                int(max_seqlen),
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
        hidden_states, _ = self.c_proj(hidden_states.flatten(start_dim=-2))
        hidden_states = self.resid_dropout(hidden_states)
        return hidden_states, k, v


class ParallelFlashMQATBlock(nn.Module):

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
        self.attn = ParallelCausalSelfAttentionLayer(
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
        if config.mlp_type is None:
            self.mlp = LayerNormParallelMLP(
                hidden_dim=config.hidden_dim,
                intermediate_dim=config.intermediate_dim,
                resid_pdrop=config.resid_pdrop,
                activation_function=config.activation_function,
                layer_norm_epsilon=config.layer_norm_epsilon,
                dtype=dtype,
                device=device,
            )
        elif config.mlp_type == "llama":
            self.mlp = LlamaLayerNormParallelMLP(
                hidden_dim=config.hidden_dim,
                intermediate_dim=config.intermediate_dim,
                activation_function=config.activation_function,
                layer_norm_epsilon=config.layer_norm_epsilon,
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


class ParallelFlashMQATBase(nn.Module):

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
        self.embedding_layer = ParallelVocabPositionEmbedding(
            config.vocab_size,
            config.n_positions,
            config.hidden_dim,
            config.embd_pdrop,
            fixed_abs_position_ids=config.fixed_abs_position_ids,
            dtype=dtype,
            device=device,
        )
        self.h = nn.ModuleList([
            ParallelFlashMQATBlock(
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


class LanguageModelHead(nn.Linear):

    def forward(self, x: PipeTransferData, ys: List[PipeCacheData]) -> PipeTransferData:
        x.pp_output = nn.functional.linear(x.pp_input, self.weight, self.bias)
        return x


class ModelParallelModule(nn.Module):

    def __init__(
        self,
        module: FlashMQATModel,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.module = module
        self.module.transformer = ParallelFlashMQATBase(module.config,
                                                        dtype=module.dtype,
                                                        device=module.device)

    def load():
        pass


class ParallelFlashMQATModel(FlashMQATModel):
    """ TODO: change into lower level module substitution
    Currently only substitute FlashMQATModel into a ParallelFlashMQATBase
    """

    def __init__(
        self,
        config: FlashMQATConfig,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__(config, dtype, device)
        self.transformer = ParallelFlashMQATBase(config, dtype=dtype, device=device)  # overwrite
        self.num_checkpoint_shards = 1

    def load(self, load_dir: str, init_critic_from_actor: bool = False):
        mp_rank = base.constants.model_parallel_rank()
        state_dict, n_shards = load_from_disk(load_dir,
                                              fn_pattern=r".*" + f"-pp-00-mp-{mp_rank:02d}-" + r"s-(\d{2}).*",
                                              return_n_shards=True)

        if init_critic_from_actor and f'{self.config.n_layers + 1}.weight' in state_dict:
            state_dict.pop(f'{self.config.n_layers + 1}.weight')
            self.load_state_dict(state_dict, strict=False)
        else:
            self.load_state_dict(state_dict)

    def save(self, save_dir):
        dp_rank = base.constants.data_parallel_rank()
        mp_rank = base.constants.model_parallel_rank()
        if dp_rank > 0:  # only save on dp_rank = 0
            return

        save_to_disk(self.state_dict(),
                     save_dir,
                     output_fn=f"pytorch_model-pp-00-mp-{mp_rank:02d}-s-" + "{shard:02d}.bin",
                     save_type="pt",
                     n_shards=self.num_checkpoint_shards)
