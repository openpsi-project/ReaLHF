from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
import dataclasses
import functools
import json
import os

import torch
import torch.nn as nn
import torch.utils.checkpoint
import transformers

from impl.model.parallelism.model_parallel.modules import (ColumnParallelLinear, parallel_lm_logits,
                                                           ParallelEmbedding)
from impl.model.utils.data import PipeCacheData, PipeTransferData
from impl.model.utils.functional import compute_varlen_position_indices
from impl.model.modules import CausalSelfAttentionLayer, LayerNormMLP, LlamaLayerNormMLP, LlamaRMSNorm
from impl.model.utils.save_load import load_from_disk, save_to_disk
import base.constants
import base.logging as logging
import impl.model.parallelism.model_parallel.mappings as tensor_parallel

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
    # parallelism optimization
    sequence_parallel: bool = False
    gradient_accumulation_fusion: bool = False

    is_critic: bool = False

    # only used for debugging, True for GPT2
    fixed_abs_position_ids: bool = False


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
            model_parallel=base.constants.model_parallel_world_size() > 1,
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
                model_parallel=base.constants.model_parallel_world_size() > 1,
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
                model_parallel=base.constants.model_parallel_world_size() > 1,
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
            attn_out, k, v = torch.utils.checkpoint.checkpoint(
                self.attn,
                h,
                cu_seqlens,
                k_cache,
                v_cache,
                cache_seqlens,
                attention_mask,
                max_seqlen,
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
            h = torch.utils.checkpoint.checkpoint(self.mlp, h) + h
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

        model_parallel = base.constants.model_parallel_world_size() > 1
        if model_parallel:
            embed_cls = ParallelEmbedding
        else:
            if self.sequence_parallel:
                raise RuntimeError("sequence_parallel can only be used when model parallel size > 1.")
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


class FlashMQATBase(nn.Module):

    def __init__(
        self,
        config: FlashMQATConfig,
        no_param_instantiation: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.device = device
        if not no_param_instantiation:
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
                    dtype=dtype,
                    device=device,
                ) for i in range(config.n_layers)
            ])
        else:
            self.embedding_layer = self.h = None

    def to_layers(self) -> List[nn.Module]:
        return [self.embedding_layer] + list(self.h)

    def forward(self, x: PipeTransferData, ys: List[PipeCacheData]) -> PipeTransferData:
        ############## FIXME: we should ensure this outside the model ##############
        if x.max_seqlen is not None:
            x.max_seqlen = int(x.max_seqlen)
        if x.cu_seqlens is not None:
            x.cu_seqlens = x.cu_seqlens.int()
        ############## FIXME: we should ensure this outside the model ##############
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

# TODO: implement a customized output head class, merging the following 3 cases
class OutputHead(nn.Linear):
    # TODO: do we need to care about the initialization scale?

    def forward(self, x: PipeTransferData, y: PipeCacheData) -> PipeTransferData:
        x.pp_output = nn.functional.linear(x.pp_input, self.weight, self.bias)
        return x


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


class FlashMQATModel(nn.Module):

    def __init__(
        self,
        config: FlashMQATConfig,
        no_param_instantiation: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        if dtype is None:
            dtype = torch.float16
        self.config = config
        self.transformer = FlashMQATBase(config,
                                         no_param_instantiation=no_param_instantiation,
                                         dtype=dtype,
                                         device=device)
        if config.is_critic and config.sequence_parallel:
            self.head = SequenceParallelCriticHead(
                config.hidden_dim,
                1,
                bias=False,
                device=device,
                dtype=dtype,
            )
        elif not config.is_critic and base.constants.model_parallel_world_size() > 1:
            self.head = SequenceParallelActorHead(
                config.hidden_dim,
                config.vocab_size,
                bias=False,
                sequence_parallel=config.sequence_parallel,
                async_tensor_model_parallel_allreduce=not config.sequence_parallel,
                gradient_accumulation_fusion=config.gradient_accumulation_fusion,
                device=device,
                dtype=dtype,
            )
        else:
            self.head = OutputHead(
                config.hidden_dim,
                1 if config.is_critic else config.vocab_size,
                bias=False,
                device=device,
                dtype=dtype,
            )

    @property
    def is_critic(self):
        return self.config.is_critic

    def to_layers(self) -> List[nn.Module]:
        return self.transformer.to_layers() + [self.head]

    def gradient_checkpointing_enable(self, attn: Optional[bool] = False, mlp: Optional[bool] = False):
        for l in self.transformer.h[1:]:
            # skip the first layer to enable lora together with grad checkpointing
            l: FlashMQATBlock
            l.gradient_checkpointing_enable(attn, mlp)

    def forward(self, x: PipeTransferData, ys: List[PipeCacheData]) -> PipeTransferData:
        if self.config.sequence_parallel:
            from impl.model.utils.tensor import pad_sequence_parallel_input
            _packed_input_ids = ys[0].input_ids
            _cu_seqlens = x.cu_seqlens
            _max_seqlen = x.max_seqlen
            packed_input_ids, cu_seqlens, max_seqlen, pad_size = pad_sequence_parallel_input(
                ys[0].input_ids, x.cu_seqlens, x.max_seqlen)
            ys[0].input_ids = packed_input_ids
            x.cu_seqlens = cu_seqlens
            x.max_seqlen = max_seqlen
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
        if self.config.sequence_parallel and pad_size > 0:
            x.pp_output = x.pp_output[:-pad_size]
            ys[0].input_ids = _packed_input_ids
            x.cu_seqlens = _cu_seqlens
            x.max_seqlen = _max_seqlen
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
        """The reverse function of map_to_pipe_state_dict."""
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

    def state_dict(self):
        """Get a loadable state dict for the pipeline module."""
        state_dict = super().state_dict()
        return FlashMQATModel.map_to_pipe_state_dict(self.config, state_dict)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return super().load_state_dict(
            FlashMQATModel.from_pipe_state_dict(self.config, state_dict),
            strict=strict,
            assign=assign,
        )

    # Template function used for converting HF model to FlashMQAT, similar to C++ template but is ugly in python.
    def _config_from_hf_template(
        config_converter: Callable[[transformers.PretrainedConfig], FlashMQATConfig],
        from_model: Optional[transformers.PreTrainedModel] = None,
        model_path: Optional[str] = None,
        is_critic: bool = False,
        sequence_parallel: bool = False,
        gradient_accumulation_fusion: bool = False,
    ) -> FlashMQATConfig:
        if model_path is not None:
            hf_config = transformers.AutoConfig.from_pretrained(os.path.join(model_path, "config.json"))
        else:
            assert from_model is not None
            hf_config = from_model.config
        config = config_converter(hf_config)
        config.is_critic = is_critic
        config.sequence_parallel = sequence_parallel
        config.gradient_accumulation_fusion = gradient_accumulation_fusion
        return config

    # Template function used for converting HF model to FlashMQAT, similar to C++ template but is ugly in python.
    def _config_and_param_from_hf_template(
        config_converter: Callable[[transformers.PretrainedConfig], FlashMQATConfig],
        state_dict_converter: Optional[Callable[[Dict, FlashMQATConfig], Dict]] = None,
        from_model: Optional[transformers.PreTrainedModel] = None,
        model_path: Optional[str] = None,
        is_critic: bool = False,
        init_from_scratch: bool = False,
        no_param_instantiation: bool = False,
        sequence_parallel: bool = False,
        gradient_accumulation_fusion: bool = False,
        force_load_from_hf_pretrained: bool = False,
        load_model_parallel_as_list: bool = False,
    ) -> Tuple[FlashMQATConfig, Optional[Dict]]:
        if not init_from_scratch and not no_param_instantiation:
            assert state_dict_converter is not None
        config = FlashMQATModel._config_from_hf_template(
            config_converter,
            from_model,
            model_path,
            is_critic=is_critic,
            sequence_parallel=sequence_parallel,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
        )
        if model_path is not None:
            if init_from_scratch or no_param_instantiation:
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
            state_dict = (from_model.state_dict()
                          if not init_from_scratch and not no_param_instantiation else None)

        if not init_from_scratch and not no_param_instantiation:
            if load_model_parallel_as_list:
                # here state_dict_converter is for parallel mqat models
                state_dict = state_dict_converter(state_dict, config, load_model_parallel_as_list)
            else:
                state_dict = state_dict_converter(state_dict, config)

        return (
            config,
            FlashMQATModel.map_to_pipe_state_dict(config, state_dict) if state_dict is not None else None,
        )

    # Template function used for converting HF model to FlashMQAT, similar to C++ template but is ugly in python.
    def _from_hf_template(
        cls,
        config_converter: Callable[[transformers.PretrainedConfig], FlashMQATConfig],
        state_dict_converter: Optional[Callable[[Dict, FlashMQATConfig], Dict]] = None,
        from_model: Optional[transformers.PreTrainedModel] = None,
        model_path: Optional[str] = None,
        init_from_scratch: bool = False,
        no_param_instantiation: bool = False,
        is_critic: bool = False,
        force_load_from_hf_pretrained: bool = False,
        sequence_parallel: bool = False,
        gradient_accumulation_fusion: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if base.constants.pipe_parallel_world_size() > 1 and not no_param_instantiation:
            raise RuntimeError(
                "`from_$\{huggingface_model\}` can only be called without pipeline parallelism.")
        if base.constants.model_parallel_world_size() > 1:
            config = FlashMQATModel._config_from_hf_template(
                config_converter=config_converter,
                model_path=model_path,
                is_critic=is_critic,
                sequence_parallel=sequence_parallel,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
            )
        else:
            config, state_dict = FlashMQATModel._config_and_param_from_hf_template(
                config_converter=config_converter,
                state_dict_converter=state_dict_converter,
                from_model=from_model,
                model_path=model_path,
                is_critic=is_critic,
                init_from_scratch=init_from_scratch,
                force_load_from_hf_pretrained=force_load_from_hf_pretrained,
                no_param_instantiation=no_param_instantiation,
                sequence_parallel=sequence_parallel,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
            )
        model = cls(
            config=config,
            dtype=dtype,
            device=device,
            no_param_instantiation=no_param_instantiation,
        )
        if not init_from_scratch and not no_param_instantiation:
            if base.constants.model_parallel_world_size() > 1:
                model.load(model_path, init_critic_from_actor=is_critic)
            else:
                if is_critic:
                    state_dict["head.weight"] = model.state_dict()["head.weight"]
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
                FlashMQATModel,
                f"dump_to_{model_name}",
                functools.partialmethod(FlashMQATModel._to_hf_template,
                                        state_dict_converter_to_hf=state_dict_converter_to_hf),
            )

    def load(self, load_dir: str, init_critic_from_actor: bool = False, load_from_pipe: bool = False):
        if base.constants.pipe_parallel_world_size() > 1:
            raise RuntimeError("`load` should not be called when using pipeline parallelism. "
                               "Are your configuration correct?")
        mp_rank = base.constants.model_parallel_rank()

        if load_from_pipe:
            state_dict, n_shards = load_from_disk(load_dir,
                                                  fn_pattern=r".*" + f"mp-{mp_rank:02d}-" + r"s-(\d{2}).*",
                                                  return_n_shards=True)
        else:
            state_dict, n_shards = load_from_disk(
                load_dir,
                fn_pattern=r".*" + f"-pp-00-mp-{mp_rank:02d}-" + r"s-(\d{2}).*",
                return_n_shards=True,
            )
        if init_critic_from_actor and f"{self.config.n_layers + 1}.weight" in state_dict:
            state_dict.pop(f"{self.config.n_layers + 1}.weight")
            self.load_state_dict(state_dict, strict=False)
        else:
            self.load_state_dict(state_dict, strict=True)

    def save(self, save_dir: str):
        if base.constants.pipe_parallel_world_size() > 1:
            raise RuntimeError("`save` should not be called when using pipeline parallelism. "
                               "Are your configuration correct?")
        dp_rank = base.constants.data_parallel_rank()
        mp_rank = base.constants.model_parallel_rank()
        if dp_rank > 0:  # only save on dp_rank = 0
            return

        with open(os.path.join(save_dir, "flash_mqat_config.json"), "w") as f:
            json.dump(dataclasses.asdict(self.config), f)

        save_to_disk(
            self.state_dict(),
            save_dir,
            output_fn=f"pytorch_model-pp-00-mp-{mp_rank:02d}-s-" + "{shard:02d}.bin",
            save_type="pt",
            n_shards=int(os.getenv("FLASH_MQAT_N_SHARDS", "3")),
        )
