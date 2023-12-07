from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
import dataclasses
import functools
import json
import os

import torch
import torch.nn as nn
import torch.utils.checkpoint
import transformers

from impl.model.utils.data import PipeCacheData, PipeTransferData
from impl.model.utils.functional import torch_attn_func
from impl.model.utils.modules import LayerNormLinear, LayerNormMLP, LlamaLayerNormMLP, LlamaRMSNorm
from impl.model.utils.save_load import load_from_disk, save_to_disk
import base.logging as logging

try:
    from flash_attn import (flash_attn_func, flash_attn_varlen_func, flash_attn_varlen_func_with_kvcache,
                            flash_attn_with_kvcache)
    from flash_attn.layers.rotary import RotaryEmbedding
except ModuleNotFoundError:
    pass
import base.logging as logging

logger = logging.getLogger("FlashMQATBase")


@dataclasses.dataclass
class FlashMQATConfig:
    n_layers: int
    n_kv_heads: int
    head_dim: int
    hidden_dim: int
    intermediate_dim: int  # for mlp, usually 4*h
    vocab_size: int
    n_positions: Optional[int] = None
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    activation_function: str = "gelu"
    ckpt_attn: bool = False
    ckpt_mlp: bool = False
    scale_attn_by_inverse_layer_idx: bool = True
    # llama does not use attention bias and uses special MLP/LayerNorm layers
    use_attention_bias: bool = True
    layer_norm_type: Optional[str] = None
    mlp_type: Optional[str] = None
    # rotary embedding
    apply_rotary: bool = False
    rotary_base: float = 10000.0
    rotary_interleaved: bool = False
    rotary_scaling: Optional[float] = None
    rotary_scaling_type: Optional[str] = None
    # only used for debugging, True for GPT2
    fixed_abs_position_ids: bool = False


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
        assert hidden_dim % head_dim == 0
        n_q_heads = hidden_dim // head_dim
        self.c_attn = LayerNormLinear(
            hidden_dim,
            head_dim * (n_q_heads + 2 * n_kv_heads),
            layer_norm_epsilon=layer_norm_epsilon,
            layer_norm_type=layer_norm_type,
            use_attention_bias=use_attention_bias,
            dtype=dtype,
            device=device,
        )
        self.c_proj = nn.Linear(hidden_dim, hidden_dim, bias=use_attention_bias, dtype=dtype, device=device)
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
            raise ValueError(f"n_kv_heads ({self.nkv}) must divide n_q_heads ({self.nq}).")
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

        qkv: torch.Tensor = self.c_attn(hidden_states)
        q, kv = torch.split(qkv, (self.d * self.nq, 2 * self.d * self.nkv), dim=-1)
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
            self.rotary_emb._update_cos_sin_cache(k_cache.shape[1], device=qkv.device, dtype=qkv.dtype)
            # Rotary cos/sin will be automatically offset by cache_seqlens in flash_attn.
            rotary_cos, rotary_sin = self.rotary_emb._cos_cached, self.rotary_emb._sin_cached
        else:
            rotary_cos = rotary_sin = None

        k, v = kv.unbind(dim=-3)

        if str(qkv.device) == "cpu":
            # Use vanilla pytorch attention, for debugging.
            hidden_states = torch_attn_func(
                q,
                k,
                v,
                causal=True,
                dropout_p=self.applied_attn_pdrop,
                softmax_scale=scale_factor,
                upcast_unscale=unscale,
                attention_mask=attention_mask,
            )
        elif k_cache is not None and len(qkv.shape) == 3:
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
                # num_splits=1,
            )
        elif k_cache is not None and len(qkv.shape) == 2:
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
            assert len(qkv.shape) == 2
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
        hidden_states = self.c_proj(hidden_states.flatten(start_dim=-2))
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
                dtype=dtype,
                device=device,
            )
        elif config.mlp_type == "llama":
            self.mlp = LlamaLayerNormMLP(
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
        if x.store_kv_cache:
            if y.k_cache is None:
                y.k_cache = k.detach()
            if y.v_cache is None:
                y.v_cache = v.detach()
            if y.cache_seqlens is None and x.cu_seqlens is not None:
                y.cache_seqlens = x.cu_seqlens[1:] - x.cu_seqlens[:-1]
        return x


class VocabPositionEmbedding(nn.Module):

    def __init__(
        self,
        config: FlashMQATConfig,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.n_positions = config.n_positions
        self.wte = nn.Embedding(config.vocab_size, config.hidden_dim, dtype=dtype, device=device)

        self.apply_abs_pos_embed = not config.apply_rotary
        if self.apply_abs_pos_embed:
            self.wpe = nn.Embedding(config.n_positions, config.hidden_dim, dtype=dtype, device=device)

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
            config,
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


class OutputHead(nn.Linear):
    # TODO: do we need to care about the initialization scale?

    def forward(self, x: PipeTransferData, ys: List[PipeCacheData]) -> PipeTransferData:
        x.pp_output = nn.functional.linear(x.pp_input, self.weight, self.bias)
        return x


class FlashMQATModel(nn.Module):

    def __init__(
        self,
        config: FlashMQATConfig,
        is_critic: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.config = config
        self.transformer = FlashMQATBase(config, dtype=dtype, device=device)
        self.head = OutputHead(
            config.hidden_dim,
            1 if is_critic else config.vocab_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self._is_critic = is_critic

    @property
    def is_critic(self):
        return self._is_critic

    def to_layers(self) -> List[nn.Module]:
        return self.transformer.to_layers() + [self.head]

    def gradient_checkpointing_enable(self):
        for l in self.transformer.h[1:]:
            # skip the first layer to enable lora together with grad checkpointing
            l: FlashMQATBlock
            l.gradient_checkpointing_enable()

    def forward(self, x: PipeTransferData, ys: List[PipeCacheData]) -> PipeTransferData:
        layers = self.to_layers()
        assert len(ys) == len(layers)
        raw_pp_input = x.pp_input
        for layer, y in zip(layers, ys):
            x = layer(x, y)  # This will set pp_output.
            x.pp_input = x.pp_output
        # Finally, pp_input is the input of this pipeline stage (maybe across several layers),
        # pp_output is the output of this pipeline stage.
        # In the first stage, pp_input is None.
        x.pp_input = raw_pp_input
        return x

    @staticmethod
    def map_to_pipe_state_dict(config: FlashMQATConfig, state_dict: Dict) -> Dict:
        """Map a FlashMQAT state dict to a state dict for the pipeline module.
        
        Note that pipeline module assumes a special state dict key format that
        every key starts with a f"{layer_idx}." prefix, which is different from
        the default keys of self.state_dict().
        """
        pipe_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("transformer.embedding_layer."):
                new_k = k.replace("transformer.embedding_layer.", "0.")
            elif k.startswith("transformer.h."):
                idx = int(k.split(".")[2])
                new_k = k.replace(f"transformer.h.{idx}.", f"{idx+1}.")
            elif k.startswith("head"):
                new_k = k.replace("head.", f"{config.n_layers+1}.")
            else:
                raise ValueError(f"Unexpected key: {k}")
            pipe_state_dict[new_k] = v
        return pipe_state_dict

    @staticmethod
    def from_pipe_state_dict(config: FlashMQATConfig, pipe_state_dict: Dict):
        """The reverse function of map_to_pipe_state_dict.
        """
        state_dict = {}
        for k, v in pipe_state_dict.items():
            if k.startswith("0."):
                new_k = k.replace("0.", "transformer.embedding_layer.")
            elif k.startswith(f"{config.n_layers+1}."):
                new_k = k.replace(f"{config.n_layers+1}.", "head.")
            else:
                idx = int(k.split(".")[0])
                new_k = k.replace(f"{idx}.", f"transformer.h.{idx-1}.")
            state_dict[new_k] = v
        return state_dict

    def pipe_state_dict(self):
        """Get a loadable state dict for the pipeline module.
        """
        state_dict = self.state_dict()
        return FlashMQATModel.map_to_pipe_state_dict(self.config, state_dict)

    # Template function used for converting HF model to FlashMQAT, similar to C++ template but is ugly in python.
    def _config_from_hf_template(
        config_converter: Callable[[transformers.PretrainedConfig], FlashMQATConfig],
        from_model: Optional[transformers.PreTrainedModel] = None,
        model_path: Optional[str] = None,
    ) -> FlashMQATConfig:
        if model_path is not None:
            hf_config = transformers.AutoConfig.from_pretrained(os.path.join(model_path, "config.json"))
        else:
            assert from_model is not None
            hf_config = from_model.config
        return config_converter(hf_config)

    # Template function used for converting HF model to FlashMQAT, similar to C++ template but is ugly in python.
    def _config_and_param_from_hf_template(
        config_converter: Callable[[transformers.PretrainedConfig], FlashMQATConfig],
        state_dict_converter: Optional[Callable[[Dict, FlashMQATConfig], Dict]] = None,
        from_model: Optional[transformers.PreTrainedModel] = None,
        model_path: Optional[str] = None,
        init_from_scratch: bool = False,
        force_load_from_hf_pretrained: bool = False,
    ) -> Tuple[FlashMQATConfig, Optional[Dict]]:
        if not init_from_scratch:
            assert state_dict_converter is not None
        config = FlashMQATModel._config_from_hf_template(config_converter, from_model, model_path)
        if model_path is not None:
            if init_from_scratch:
                state_dict = None
            elif force_load_from_hf_pretrained:
                logger.warning(f"Force to load from HuggingFace PreTrainedModel...")
                state_dict = transformers.AutoModelForCausalLM.from_pretrained(model_path).state_dict()
            else:
                try:
                    state_dict = load_from_disk(model_path)
                except Exception as e:
                    logger.critical(f"Failed to load state dict from {model_path}: {e}")
                    logger.critical("Degenerate to using huggingface model initialization. "
                                    "This will probably cause (CPU) OOM.")
                    state_dict = transformers.AutoModelForCausalLM.from_pretrained(model_path).state_dict()
        else:
            logger.warning(
                f"Note that HuggingFace PreTrainedModel may have different state dict keys from the saved one. "
                "Loading from HuggingFace `model_path` is ensured to be correct but the `from_model` argument may cause key mismatch."
            )
            assert from_model is not None
            state_dict = from_model.state_dict() if not init_from_scratch else None

        if not init_from_scratch:
            state_dict = state_dict_converter(state_dict, config)

        return config, state_dict

    # Template function used for converting HF model to FlashMQAT, similar to C++ template but is ugly in python.
    def _from_hf_template(
        cls,
        config_converter: Callable[[transformers.PretrainedConfig], FlashMQATConfig],
        state_dict_converter: Optional[Callable[[Dict, FlashMQATConfig], Dict]] = None,
        from_model: Optional[transformers.PreTrainedModel] = None,
        model_path: Optional[str] = None,
        init_from_scratch: bool = False,
        is_critic: bool = False,
        force_load_from_hf_pretrained: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        config, state_dict = FlashMQATModel._config_and_param_from_hf_template(
            config_converter=config_converter,
            state_dict_converter=state_dict_converter,
            from_model=from_model,
            model_path=model_path,
            init_from_scratch=init_from_scratch,
            force_load_from_hf_pretrained=force_load_from_hf_pretrained,
        )
        model = cls(config=config, is_critic=is_critic, dtype=dtype, device=device)
        if not init_from_scratch:
            if is_critic:
                state_dict['head.weight'] = model.state_dict()['head.weight']
            model.load_state_dict(state_dict)
        return model

    # Template function used for FlashMQAT to HF models, similar to C++ template but is ugly in python.
    def _to_hf_template(self, output_dir, state_dict_converter_to_hf):
        save_to_disk(state_dict_converter_to_hf(self.state_dict(), self.config), output_dir)

    @staticmethod
    def register_hf_model(
        model_name: str,
        config_converter: Callable[[transformers.PretrainedConfig], FlashMQATConfig],
        state_dict_converter: Callable[[Dict, FlashMQATConfig], Dict],
        state_dict_converter_to_hf: Optional[Callable[[Dict, FlashMQATConfig], Dict]] = None,
        force_load_from_hf_pretrained: bool = False,
    ):
        """Register a HuggingFace model with `model_name`, such that models can be converted back-and-forth.
        
        Example usage:
        
        ```
        # 1. Register a model called `starcoder` with helper functions.
        # Check `impl/model/nn/flash_mqat/flash_from_hf_impl.py` for details.
        FlashMQATModel.register_hf_model("starcoder",
                                         convert_config_starcoder, 
                                         state_dict_from_starcoder,
                                         state_dict_to_starcoder)
        
        # 2. Obtain the config
        config: FlashMQATConfig = FlashMQATModel.config_from_starcoder(model_path)

        # 3. Obtain config and state_dict (also support init_from_scratch=True)
        config, state_dict = FlashMQATModel.config_and_param_from_starcoder(model_path)

        # 4. Directly construct from HuggingFace model (also support init_from_scratch=True)
        model = FlashMQATModel.from_starcoder(model_path="/lustre/public/pretrained_model_weights/starcoder-16bit")

        # 5. Dump to HuggingFace model
        model.dump_to_starcoder(save_path)

        # 6. Use the dumped weights
        from impl.model.nn.utils.save_load import load_from_disk
        config = transformers.AutoConfig.from_pretrained(model_path)
        hf_model = transformers.AutoModelForCausalLM.from_config(config)
        hf_model.load_state_dict(load_from_disk(save_path))
        ```

        """
        if model_name == "pretrained":
            raise ValueError("model_name cannot be 'pretrained'.")
        setattr(
            FlashMQATModel,
            f"from_{model_name}",
            classmethod(
                functools.partial(
                    FlashMQATModel._from_hf_template,
                    config_converter=config_converter,
                    state_dict_converter=state_dict_converter,
                    force_load_from_hf_pretrained=force_load_from_hf_pretrained,
                )),
        )
        setattr(
            FlashMQATModel,
            f"config_from_{model_name}",
            staticmethod(
                functools.partial(
                    FlashMQATModel._config_from_hf_template,
                    config_converter=config_converter,
                )),
        )
        setattr(
            FlashMQATModel,
            f"config_and_param_from_{model_name}",
            staticmethod(
                functools.partial(
                    FlashMQATModel._config_and_param_from_hf_template,
                    config_converter=config_converter,
                    state_dict_converter=state_dict_converter,
                    force_load_from_hf_pretrained=force_load_from_hf_pretrained,
                )),
        )
        if state_dict_converter_to_hf:
            setattr(
                FlashMQATModel, f"dump_to_{model_name}",
                functools.partialmethod(FlashMQATModel._to_hf_template,
                                        state_dict_converter_to_hf=state_dict_converter_to_hf))

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        init_from_scratch: bool = False,
        is_critic: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Load from a pretrained FlashMQAT model (usually the SFT/RW model).
        """
        with open(os.path.join(model_path, "config.json"), "r") as f:
            config = FlashMQATConfig(**json.load(f))
        model = cls(config=config, is_critic=is_critic, dtype=dtype, device=device)
        if not init_from_scratch:
            state_dict = load_from_disk(model_path)
            model.load_state_dict(state_dict)
        return model

    @classmethod
    def from_pipeline_module(
        cls,
        model_path: str,
        is_critic: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Merge the state dict of a pipeline module to a single FlashMQAT.
        
        Used for loading weights for the reference model if SFT used pipeline parallel,
        """
        with open(os.path.join(model_path, "config.json"), "r") as f:
            config = FlashMQATConfig(**json.load(f))
        model = cls(config=config, is_critic=is_critic, dtype=dtype, device=device)
        state_dict = cls.from_pipe_state_dict(config, load_from_disk(model_path))
        model.load_state_dict(state_dict)
        return model
