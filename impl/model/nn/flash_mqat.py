from typing import Callable, List, Optional, Tuple, Union
import copy
import dataclasses
import json
import logging
import math
import os
import queue

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers

from impl.model.utils.data import (
    build_packed_inputs,
    DuckGenerationOutput,
    DuckModelOutput,
    mask_eos_token,
    PipeCacheData,
    PipeTransferData,
    repeat_kv,
    TensorDataclassToTupleInterface,
    unpack_tensor,
    upcast_masked_softmax,
    upcast_softmax,
)
from impl.model.utils.logits_warper import top_k_top_p_logits
from impl.model.utils.modules import LayerNormLinear, LayerNormMLP
import api.huggingface
import api.model

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
except ModuleNotFoundError:
    pass

logger = logging.getLogger("FlashMQAT")


@dataclasses.dataclass
class FlashMQATConfig:
    n_layers: int
    n_kv_heads: int
    head_dim: int
    hidden_dim: int
    intermediate_dim: int  # for mlp, usually 4*h
    vocab_size: int
    n_positions: int
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    activation_function: str = "gelu"
    ckpt_attn: bool = False
    ckpt_mlp: bool = False
    scale_attn_by_inverse_layer_idx: bool = True
    # only used for debugging
    fixed_abs_position_ids: bool = False


@dataclasses.dataclass
class GenerationConfig:
    min_new_tokens: int = 1
    max_new_tokens: int = 10
    temperature: float = 1.0
    greedy: bool = True
    top_p: float = 1.0
    top_k: int = 0
    num_samples: int = 1


def torch_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    dropout_p: float,
    softmax_scale: float,
    upcast_unscale: float = 1.0,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """PyTorch implementation of the attention function with a flash-attn-like API.

    We use this function to compare the output of our model and huggingface models.
    Flash-attn/float16/CUDAkernels will all more or less suffer from float point errors.
    We call this function with float32 and CPU to get the "ground truth" output.

    Args:
        q (torch.Tensor): Shape [bs, seqlen, #q, head_dim].
        k (torch.Tensor): Shape [bs, seqlen, #kv, head_dim].
        v (torch.Tensor): Shape [bs, seqlen, #kv, head_dim].
        causal (bool): .
        dropout_p (float): .
        softmax_scale (float): .
        upcast_unscale (float, optional): Scale factor when upcastin attention scores.
            Defaults to 1.0.
        attention_mask (Optional[torch.Tensor], optional): Huggingface-like attention mask.
            Shape [*, seqlen, seqlen]. Will override the `causal` argument.
            Only used for debugging. Defaults to None.

    Returns:
        torch.Tensor: Attention score. Shape [bs, seqlen, #q, head_dim].
    """
    n_rep = q.shape[-2] // k.shape[-2]
    bsz, seqlen = q.shape[:2]
    # repeat k/v heads if n_kv_heads < n_heads
    k = repeat_kv(k, n_rep)  # (bs, seqlen, nq, head_dim)
    v = repeat_kv(v, n_rep)  # (bs, seqlen, nq, head_dim)

    q = q.transpose(1, 2)  # (bs, nq, seqlen, head_dim)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    scores = torch.matmul(q, k.transpose(2, 3)) * softmax_scale
    if attention_mask is not None:
        assert str(attention_mask.device) == "cpu"
        mask_softmax = True
        mask = attention_mask
    elif causal:
        mask_softmax = True
        mask = torch.tril(torch.ones(seqlen, seqlen, device=q.device, dtype=torch.bool))
    else:
        mask_softmax = False
    if mask_softmax:
        scores = upcast_masked_softmax(
            scores,
            mask,
            mask_value=torch.full(
                [], torch.finfo(torch.float32).min, device=scores.device, dtype=torch.float32
            ),
            scale=upcast_unscale,
            softmax_dtype=torch.float32,
        )
    else:
        scores = upcast_softmax(scores, scale=upcast_unscale, softmax_dtype=torch.float32)
    scores = nn.functional.dropout(scores, p=dropout_p)
    scores = scores.to(q.dtype)
    output = torch.matmul(scores, v)  # (bs, nq, seqlen, head_dim)
    output = output.transpose(1, 2).contiguous()
    return output


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
        assert hidden_dim % head_dim == 0
        n_q_heads = hidden_dim // head_dim
        self.c_attn = LayerNormLinear(
            hidden_dim,
            head_dim * (n_q_heads + 2 * n_kv_heads),
            layer_norm_epsilon=layer_norm_epsilon,
            dtype=dtype,
            device=device,
        )
        self.c_proj = nn.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device)
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

        qkv: torch.Tensor = self.c_attn(hidden_states)
        if str(qkv.device) == "cpu":
            # Use vanilla pytorch attention, for debugging.
            q, k, v = torch.split(qkv, (self.d * self.nq, self.d * self.nkv, self.d * self.nkv), dim=-1)
            k = k.view(*k.shape[:2], self.nkv, self.d)
            v = v.view(*v.shape[:2], self.nkv, self.d)
            q = q.view(*q.shape[:2], self.nq, self.d)
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
            assert qkv.shape[1] == 1, (qkv.shape, "Can only generate one token at a time.")
            q, k, v = torch.split(qkv, (self.d * self.nq, self.d * self.nkv, self.d * self.nkv), dim=-1)
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
        elif k_cache is not None and len(qkv.shape) == 2:
            q, k, v = torch.split(qkv, (self.d * self.nq, self.d * self.nkv, self.d * self.nkv), dim=-1)
            q = q.view(q.shape[0], self.nq, self.d)
            v = v.view(v.shape[0], self.nkv, self.d)
            k = k.view(k.shape[0], self.nkv, self.d)
            # FIXME: The following code is crazily slow. We should implement them as a customized kernel?
            qlens = cu_seqlens[1:] - cu_seqlens[:-1]
            offset = 0
            new_k, new_v = [], []
            for i, (qlen, cache_len) in enumerate(zip(qlens, cache_seqlens)):
                new_k += [k_cache[i, :cache_len], k[offset : offset + qlen]]
                new_v += [v_cache[i, :cache_len], v[offset : offset + qlen]]
                with torch.no_grad():
                    k_cache[i, cache_len : cache_len + qlen] = k[offset : offset + qlen].detach()
                    v_cache[i, cache_len : cache_len + qlen] = v[offset : offset + qlen].detach()
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
            assert len(qkv.shape) == 2
            q, k, v = torch.split(qkv, (self.d * self.nq, self.d * self.nkv, self.d * self.nkv), dim=-1)
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
            q, k, v = torch.split(qkv, (self.d * self.nq, self.d * self.nkv, self.d * self.nkv), dim=-1)
            k = k.view(*k.shape[:2], self.nkv, self.d)
            v = v.view(*v.shape[:2], self.nkv, self.d)
            q = q.view(*q.shape[:2], self.nq, self.d)
            hidden_states = flash_attn_func(
                q, k, v, dropout_p=self.applied_attn_pdrop, softmax_scale=scale_factor, causal=True
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
        self.mlp = LayerNormMLP(
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
            self.ln_f = nn.LayerNorm(
                config.hidden_dim, eps=config.layer_norm_epsilon, dtype=dtype, device=device
            )

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
        self.wte = nn.Embedding(vocab_size, hidden_dim, dtype=dtype, device=device)
        self.wpe = nn.Embedding(n_positions, hidden_dim, dtype=dtype, device=device)
        self.embed_drop = nn.Dropout(embed_pdrop)

        self.self_attention_mask = torch.tril(
            torch.ones((n_positions, n_positions), dtype=torch.bool, device=device)
        )
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
                    [torch.arange(int(l), dtype=torch.int32, device=y.input_ids.device) for l in lengths]
                )
                assert (y.position_ids < x.max_seqlen).all() and y.position_ids.max() == x.max_seqlen - 1
            else:
                y.position_ids = torch.cat(
                    [
                        torch.arange(int(l), dtype=torch.int32, device=y.input_ids.device) + cache_len
                        for l, cache_len in zip(lengths, y.cache_seqlens)
                    ]
                )
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
                y.position_ids = torch.arange(
                    y.input_ids.shape[-1], dtype=torch.long, device=y.input_ids.device
                ).unsqueeze(0)
            else:
                y.position_ids = attention_mask.long().cumsum(-1) - 1
                y.position_ids.masked_fill_(attention_mask == 0, 1)
            seqlen = y.input_ids.shape[-1]
            self_attention_mask = self.self_attention_mask[None, :seqlen, :seqlen]
            self_attention_mask = self_attention_mask * attention_mask.view(batch_size, 1, -1).to(
                dtype=torch.bool, device=self_attention_mask.device
            )
            x.attention_mask = self_attention_mask.unsqueeze(1)

        inputs_embeds = self.wte(y.input_ids)
        position_embeds = self.wpe(y.position_ids)
        x.pp_output = self.embed_drop(inputs_embeds + position_embeds)
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
        self.h = nn.ModuleList(
            [
                FlashMQATBlock(
                    config,
                    layer_index=i,
                    output_layernorm=(i == config.n_layers - 1),
                    ckpt_attn=(i > 0 and config.ckpt_attn),
                    ckpt_mlp=(i > 0 and config.ckpt_mlp),
                    dtype=dtype,
                    device=device,
                )
                for i in range(config.n_layers)
            ]
        )

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


class FlashMQATForCausalLM(nn.Module):
    def __init__(
        self,
        config: FlashMQATConfig,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.config = config
        self.transformer = FlashMQATBase(config, dtype=dtype, device=device)
        self.lm_head = LanguageModelHead(
            config.hidden_dim,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def to_layers(self) -> List[nn.Module]:
        return self.transformer.to_layers() + [self.lm_head]

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

    @classmethod
    def from_starcoder(
        cls,
        from_model: Optional[transformers.PreTrainedModel] = None,
        model_path: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if from_model is None:
            assert model_path is not None
            starcoder_config = transformers.AutoConfig.from_pretrained(
                os.path.join(model_path, "config.json")
            )
        else:
            starcoder_config = from_model.config
        config = FlashMQATConfig(
            n_layers=starcoder_config.n_layer,
            n_kv_heads=1,
            attn_pdrop=starcoder_config.attn_pdrop,
            embd_pdrop=starcoder_config.embd_pdrop,
            layer_norm_epsilon=starcoder_config.layer_norm_epsilon,
            hidden_dim=starcoder_config.n_embd,
            head_dim=starcoder_config.n_embd // starcoder_config.n_head,
            intermediate_dim=starcoder_config.n_inner,
            n_positions=starcoder_config.n_positions,
            resid_pdrop=starcoder_config.resid_pdrop,
            vocab_size=starcoder_config.vocab_size,
        )
        model = cls(config, dtype=dtype, device=device)

        if from_model is None:
            if os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
                state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
            elif os.path.exists(os.path.join(model_path, "pytorch_model.bin.index.json")):
                with open(os.path.join(model_path, "pytorch_model.bin.index.json"), "r") as f:
                    weight_map = json.load(f)["weight_map"]
                state_dict = {}
                for filename in list(set(list(weight_map.values()))):
                    assert os.path.exists(os.path.join(model_path, filename))
                    state_dict.update(torch.load(os.path.join(model_path, filename)))
            else:
                logger.warning(
                    "No pytorch_model.bin or pytorch_model.bin.index.json found, "
                    "using huggingface model initialization. "
                    "This will probably cause (CPU) OOM."
                )
                state_dict = transformers.AutoModelForCausalLM.from_pretrained(model_path).state_dict()
        else:
            state_dict = from_model.state_dict()

        new_state_dict = {}
        replace_from = [
            ".wte",
            ".wpe",
            ".ln_1.",
            ".ln_2.",
            ".c_attn.weight",
            ".c_attn.bias",
            "transformer.ln_f.",
        ]
        replace_to = [
            ".embedding_layer.wte",
            ".embedding_layer.wpe",
            ".attn.c_attn.ln.",
            ".mlp.ln.",
            ".c_attn.linear.weight",
            ".c_attn.linear.bias",
            f"transformer.h.{config.n_layers - 1}.ln_f.",
        ]
        for k, v in state_dict.items():
            for rf, rt in zip(replace_from, replace_to):
                if rf in k:
                    k = k.replace(rf, rt)
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        return model

    @classmethod
    def from_gpt2(
        cls,
        from_model: Optional[transformers.PreTrainedModel] = None,
        model_path: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if from_model is None:
            assert model_path is not None
            gpt2config: transformers.GPT2Config = transformers.AutoConfig.from_pretrained(
                os.path.join(model_path, "config.json")
            )
            # GPT2 is not that large, so this will not cause OOM
            from_model = transformers.AutoModelForCausalLM.from_pretrained(model_path)
        else:
            gpt2config = from_model.config
        config = FlashMQATConfig(
            n_layers=gpt2config.n_layer,
            n_kv_heads=gpt2config.n_head,
            attn_pdrop=gpt2config.attn_pdrop,
            embd_pdrop=gpt2config.embd_pdrop,
            layer_norm_epsilon=gpt2config.layer_norm_epsilon,
            hidden_dim=gpt2config.n_embd,
            head_dim=gpt2config.n_embd // gpt2config.n_head,
            intermediate_dim=gpt2config.n_inner if gpt2config.n_inner is not None else 4 * gpt2config.n_embd,
            n_positions=gpt2config.n_positions,
            resid_pdrop=gpt2config.resid_pdrop,
            vocab_size=gpt2config.vocab_size,
            activation_function=gpt2config.activation_function,
            scale_attn_by_inverse_layer_idx=False,
            fixed_abs_position_ids=True,
        )
        model = cls(config, dtype=dtype, device=device)

        state_dict = from_model.state_dict()

        new_state_dict = {}
        replace_from = [
            "wte.weight",
            "wpe.weight",
            ".ln_1.",
            ".ln_2.",
            ".c_attn.weight",
            ".c_attn.bias",
            "ln_f.weight",
            "ln_f.bias",
        ]
        replace_to = [
            "embedding_layer.wte.weight",
            "embedding_layer.wpe.weight",
            ".attn.c_attn.ln.",
            ".mlp.ln.",
            ".c_attn.linear.weight",
            ".c_attn.linear.bias",
            f"h.{config.n_layers - 1}.ln_f.weight",
            f"h.{config.n_layers - 1}.ln_f.bias",
        ]
        for k, v in state_dict.items():
            for rf, rt in zip(replace_from, replace_to):
                if rf in k:
                    k = k.replace(rf, rt)
            if k.endswith(".attn.bias"):
                continue
            if k.endswith(".linear.weight") or k.endswith("proj.weight") or k.endswith("fc.weight"):
                v = v.transpose(0, 1)
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        return model

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        with open(os.path.join(model_path, "config.json"), "r") as f:
            config = FlashMQATConfig(**json.load(f))
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        model = cls(config, dtype, device)
        model.load_state_dict(state_dict)
        return model


class HuggingfaceLikeFlashMQATForCausalLM(nn.Module):
    """__call__ on this model will return a huggingface-like output."""

    def __init__(self, net: FlashMQATForCausalLM):
        super().__init__()
        self.net = net

    @property
    def config(self):
        return self.net.config

    def gradient_checkpointing_enable(self):
        for l in self.net.transformer.h[1:]:
            # skip the first layer to enable lora together with grad checkpointing
            l: FlashMQATBlock
            l.gradient_checkpointing_enable()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        padding_side: Optional[str] = None,
        packed_input_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> DuckModelOutput:
        assert (packed_input_ids is None) == (cu_seqlens is None) == (max_seqlen is None)
        build_packed = False
        if packed_input_ids is None and attention_mask is not None:
            build_packed = True
            packed_input_ids, cu_seqlens, max_seqlen = build_packed_inputs(input_ids, attention_mask)
        if packed_input_ids is not None:
            x = PipeTransferData(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            ys = [PipeCacheData(input_ids=packed_input_ids)] + [
                PipeCacheData() for _ in range(self.config.n_layers + 1)
            ]
        else:
            x = PipeTransferData()
            ys = [PipeCacheData(input_ids=input_ids)] + [
                PipeCacheData() for _ in range(self.config.n_layers + 1)
            ]
        logits = self.net(x, ys).pp_output
        if build_packed:
            logits = unpack_tensor(logits, cu_seqlens, padding_side=padding_side)
        return DuckModelOutput(logits=logits)

    def generate(
        self,
        tokenizer: transformers.PreTrainedTokenizerFast,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        k_caches: Optional[List[torch.Tensor]] = None,
        v_caches: Optional[List[torch.Tensor]] = None,
        cache_seqlens: Optional[torch.Tensor] = None,
        gconfig: GenerationConfig = dataclasses.field(default_factory=GenerationConfig),
    ) -> DuckGenerationOutput:
        seq, scores, mask, _, _ = generate(
            self.net, tokenizer, input_ids, attention_mask, k_caches, v_caches, cache_seqlens, gconfig
        )
        return DuckGenerationOutput(seq, scores, mask)

    @classmethod
    def from_starcoder(
        cls,
        from_model: Optional[transformers.PreTrainedModel] = None,
        model_path: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        return cls(FlashMQATForCausalLM.from_starcoder(from_model, model_path, dtype, device))

    @classmethod
    def from_gpt2(
        cls,
        from_model: Optional[transformers.PreTrainedModel] = None,
        model_path: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        return cls(FlashMQATForCausalLM.from_gpt2(from_model, model_path, dtype, device))

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        with open(os.path.join(model_path, "config.json"), "r") as f:
            config = FlashMQATConfig(**json.load(f))
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        net = FlashMQATForCausalLM(config, dtype, device)
        model = cls(net)
        model.load_state_dict(state_dict)
        return model


def make_flash_mqat_clm_hf(
    name: str,
    device: torch.device,
    model_path: str,
    dtype: Optional[torch.dtype] = None,
    from_type: str = "starcoder",
    tokenizer_path: Optional[str] = None,
):
    if from_type == "starcoder":
        module = HuggingfaceLikeFlashMQATForCausalLM.from_starcoder(
            model_path=model_path, dtype=dtype, device=device
        )
        tokenizer = api.huggingface.load_hf_tokenizer(model_path)
    elif from_type == "self":
        module = HuggingfaceLikeFlashMQATForCausalLM.from_pretrained(
            model_path=model_path, dtype=dtype, device=device
        )
        if tokenizer_path is None:
            raise ValueError("tokenizer_path must be provided when from_type is 'self'.")
        tokenizer = api.huggingface.load_hf_tokenizer(tokenizer_path)
    elif from_type == "gpt2":
        module = HuggingfaceLikeFlashMQATForCausalLM.from_gpt2(
            model_path=model_path, dtype=dtype, device=device
        )
        tokenizer = api.huggingface.load_hf_tokenizer(model_path)
    else:
        raise NotImplementedError()
    return api.model.Model(name, module, tokenizer, device)


api.model.register_model("flash_mqat_clm_hf", make_flash_mqat_clm_hf)


class DeepSpeedChatLikeFlashMQATCriticModel(nn.Module):
    def __init__(self, net: FlashMQATBase, output_scaling: float = 1.0, output_bias: float = 0.0):
        super().__init__()
        self.net = net
        self.head = nn.Linear(
            net.config.hidden_dim, 1, bias=False, dtype=self.net.dtype, device=self.net.device
        )
        self.output_scaling = output_scaling
        self.output_bias = output_bias

    @property
    def config(self):
        return self.net.config

    def gradient_checkpointing_enable(self):
        for l in self.net.h[1:]:
            l: FlashMQATBlock
            l.gradient_checkpointing_enable()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        packed_input_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> DuckModelOutput:
        assert (packed_input_ids is None) == (cu_seqlens is None) == (max_seqlen is None)
        build_packed = False
        if packed_input_ids is None and attention_mask is not None:
            build_packed = True
            packed_input_ids, cu_seqlens, max_seqlen = build_packed_inputs(input_ids, attention_mask)
        if packed_input_ids is not None:
            x = PipeTransferData(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            ys = [PipeCacheData(input_ids=packed_input_ids)] + [
                PipeCacheData() for _ in range(self.config.n_layers)
            ]
        else:
            x = PipeTransferData()
            ys = [PipeCacheData(input_ids=input_ids)] + [PipeCacheData() for _ in range(self.config.n_layers)]
        hidden_states = self.net(x, ys).pp_output
        if build_packed:
            hidden_states = unpack_tensor(hidden_states, cu_seqlens, max_seqlen)
        return (self.head(hidden_states).squeeze() - self.output_bias) * self.output_scaling

    @classmethod
    def from_starcoder(
        cls,
        model_path: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
        v_head_path: Optional[str] = None,
        output_scaling: float = 1.0,
        output_bias: float = 0.0,
    ):
        from_model = HuggingfaceLikeFlashMQATForCausalLM.from_starcoder(
            model_path=model_path, dtype=dtype, device=device
        )
        model = cls(from_model.net.transformer, output_bias=output_bias, output_scaling=output_scaling)
        if v_head_path is not None:
            model.head.load_state_dict(torch.load(v_head_path))
        return model

    @classmethod
    def from_gpt2(
        cls,
        model_path: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
        v_head_path: Optional[str] = None,
        output_scaling: float = 1.0,
        output_bias: float = 0.0,
    ):
        from_model = HuggingfaceLikeFlashMQATForCausalLM.from_gpt2(
            model_path=model_path, dtype=dtype, device=device
        )
        model = cls(from_model.net.transformer, output_bias=output_bias, output_scaling=output_scaling)
        if v_head_path is not None:
            model.head.load_state_dict(torch.load(v_head_path))
        return model

    @classmethod
    def from_sft_model(
        cls,
        from_model: Optional[HuggingfaceLikeFlashMQATForCausalLM] = None,
        model_path: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
        output_scaling: float = 1.0,
        output_bias: float = 0.0,
    ):
        if from_model is None:
            from_model = HuggingfaceLikeFlashMQATForCausalLM.from_pretrained(model_path, dtype, device)
        model = cls(from_model.net.transformer, output_bias=output_bias, output_scaling=output_scaling)
        return model

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
        output_scaling: float = 1.0,
        output_bias: float = 0.0,
    ):
        with open(os.path.join(model_path, "config.json"), "r") as f:
            config = FlashMQATConfig(**json.load(f))
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        net = FlashMQATBase(config, dtype, device)
        model = cls(net, output_bias=output_bias, output_scaling=output_scaling)
        model.load_state_dict(state_dict)
        return model


def make_flash_mqat_critic(
    name: str,
    device: torch.device,
    model_path: str,
    dtype: Optional[torch.dtype] = None,
    from_type: str = "sft",
    tokenizer_path: Optional[str] = None,
    v_head_path: Optional[str] = None,
    output_scaling: float = 1.0,
    output_bias: float = 0.0,
):
    if tokenizer_path is None:
        tokenizer_path = model_path
    if from_type == "sft":
        module = DeepSpeedChatLikeFlashMQATCriticModel.from_sft_model(
            model_path=model_path,
            dtype=dtype,
            device=device,
            output_scaling=output_scaling,
            output_bias=output_bias,
        )
    elif from_type == "starcoder":
        module = DeepSpeedChatLikeFlashMQATCriticModel.from_starcoder(
            model_path=model_path,
            dtype=dtype,
            device=device,
            v_head_path=v_head_path,
            output_scaling=output_scaling,
            output_bias=output_bias,
        )
    elif from_type == "gpt2":
        module = DeepSpeedChatLikeFlashMQATCriticModel.from_gpt2(
            model_path=model_path,
            dtype=dtype,
            device=device,
            v_head_path=v_head_path,
            output_scaling=output_scaling,
            output_bias=output_bias,
        )
    elif from_type == "self":
        module = DeepSpeedChatLikeFlashMQATCriticModel.from_pretrained(
            model_path=model_path,
            dtype=dtype,
            device=device,
            output_scaling=output_scaling,
            output_bias=output_bias,
        )
    else:
        raise NotImplementedError()
    tokenizer = api.huggingface.load_hf_tokenizer(tokenizer_path)
    return api.model.Model(name, module, tokenizer, device)


api.model.register_model("flash_mqat_critic", make_flash_mqat_critic)


def genstep(
    next_token_logits: torch.Tensor,
    tokenizer: transformers.PreTrainedTokenizerFast,
    unfinished_sequences: torch.Tensor,
    generated_idx: Union[torch.IntTensor, int],
    gconfig: GenerationConfig,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], bool, torch.Tensor]:
    """Advance generation by one step given logits.

    Args:
        next_token_logits (torch.Tensor): Shape [bs, vocab_size].
        tokenizer (transformers.PreTrainedTokenizerFast): .
        unfinished_sequences (torch.Tensor): Bool tensor indicator of whether a sequence is finished.
            Shape [bs].
        generated_idx (int): The token index to be generated.
        gconfig (GenerationConfig): .

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, torch.Tensor]:
        A tuple of
            next_tokens: Shape [bs].
            logprob: The log probability of selected tokens. May be re-normalized
                according to the mask machanism. Shape [bs].
            logits_mask: The mask of logits. Shape [bs, vocab_size].
            terminate: Whether the generation should be terminated.
            unfinished_sequences: Bool tensor indicator of whether a sequence is finished.
                Shape [bs].
    """

    next_token_logits = next_token_logits.float()
    if isinstance(generated_idx, int):
        if generated_idx < gconfig.min_new_tokens:
            next_token_logits = mask_eos_token(next_token_logits, eos_token_id=tokenizer.eos_token_id)
    else:
        assert isinstance(generated_idx, torch.Tensor)
        if (generated_idx < gconfig.min_new_tokens).any():
            _batch_indices = (generated_idx < gconfig.min_new_tokens).unsqueeze(1)
            _vocab_indices = _batch_indices.new_zeros((1, next_token_logits.shape[1]))
            if tokenizer.eos_token_id is not None:
                _vocab_indices[:, tokenizer.eos_token_id] = 1
            next_token_logits.masked_fill_(
                _batch_indices * _vocab_indices, torch.finfo(next_token_logits.dtype).min
            )

    if not gconfig.greedy:
        next_token_logits /= gconfig.temperature
        next_token_logits = top_k_top_p_logits(
            next_token_logits,
            top_k=gconfig.top_k,
            top_p=gconfig.top_p,
            inplace=True,
            ordered=False,
        )

    distrb = torch.distributions.Categorical(logits=next_token_logits)
    next_tokens = distrb.mode if gconfig.greedy else distrb.sample()
    logprob = distrb.log_prob(next_tokens)

    if tokenizer.eos_token_id is not None:
        if tokenizer.pad_token_id is None:
            raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
        next_tokens = next_tokens * unfinished_sequences + tokenizer.pad_token_id * (1 - unfinished_sequences)
    unfinished_sequences = next_tokens.ne(tokenizer.eos_token_id).long() * unfinished_sequences

    # terminate check
    if isinstance(generated_idx, int):
        terminate = (generated_idx >= gconfig.max_new_tokens - 1) or (unfinished_sequences.max() == 0)
    else:
        unfinished_sequences.logical_and_(generated_idx < gconfig.max_new_tokens - 1)
        terminate = unfinished_sequences.max() == 0

    logits_mask = next_token_logits != torch.finfo(next_token_logits.dtype).min
    if logits_mask.all():
        logits_mask = None

    return next_tokens, logprob, logits_mask, terminate, unfinished_sequences


@torch.no_grad()
def generate(
    model: FlashMQATForCausalLM,
    tokenizer: transformers.PreTrainedTokenizerFast,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    k_caches: Optional[List[torch.Tensor]] = None,
    v_caches: Optional[List[torch.Tensor]] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    gconfig: GenerationConfig = dataclasses.field(default_factory=GenerationConfig),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[PipeCacheData], Optional[torch.Tensor]]:
    """Generete a sequence with a FlashMQAT.

    Args:
        model (FlashMQATForCausalLM): .
        tokenizer (transformers.PreTrainedTokenizerFast): .
        input_ids (torch.Tensor): Prompts, may be padded. Shape [bs, seqlen].
        attention_mask (Optional[torch.Tensor], optional): The same as huggingface.
            Shape [bs, seqlen]. If None, generate attention mask according to
            pad_token_id and eos_token_id. Defaults to None.
        k_caches (Optional[List[torch.Tensor]], optional): List of k_caches.
            Length equals to the number of transformer layers.
            Each tensor in the list has shape [bs, max_seqlen, #kv, head_dim].
            Used for resuming a previous generation state.
            If None, generate from scratch. Defaults to None.
        v_caches (Optional[List[torch.Tensor]], optional): List of v_caches.
            Length equals to the number of transformer layers.
            Each tensor in the list has shape [bs, max_seqlen, #kv, head_dim].
            Used for resuming a previous generation state.
            If None, generate from scratch. Defaults to None.
        cache_seqlens (Optional[torch.Tensor], optional): Shape [bs].
            Used for resuming a previous generation state. Defaults to None.
        gconfig (GenerationConfig, optional): .

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[PipeCacheData]]:
        The tuple of
            gen_tokens: Generated tokens. Shape [bs * num_samples, #new_tokens].
            log_probs: Log probabilities of generated tokens. Shape [bs * num_samples, #new_tokens].
            mask: The mask of logits. None if no mask otherwise a tensor of
                shape [bs * num_samples, #new_tokens, vocab_size].
                1 if the logits is valid else 0, e.g., should be used as
                `logits.masked_fill_(mask.logical_not(), -1e10)`.
            ys: List of PipeCacheData. Length equals to the number of transformer layers.
                Can be saved for continuing generation.
            prompt_logits: Output logits of prompts. None if k/v caches are passed in.
                Shape [#tot_prompt_tokens * num_samples].
    """
    if attention_mask is None:
        attention_mask = torch.logical_and(
            input_ids != tokenizer.pad_token_id, input_ids != tokenizer.eos_token_id
        )
    if (k_caches is None) != (v_caches is None) or (k_caches is None) != (cache_seqlens is None):
        raise ValueError("k_cache, v_cache, cache_seqlens must be all None or all not None")
    if gconfig.num_samples > 1 and k_caches is None:
        input_ids = input_ids.unsqueeze(1).repeat(1, gconfig.num_samples, 1).flatten(end_dim=1)
        attention_mask = attention_mask.unsqueeze(1).repeat(1, gconfig.num_samples, 1).flatten(end_dim=1)
    elif k_caches is not None:
        for k_cache, v_cache in zip(k_caches, v_caches):
            assert (
                k_cache.shape[0]
                == v_cache.shape[0]
                == input_ids.shape[0]
                == attention_mask.shape[0]
                == cache_seqlens.shape[0]
            )

    device = input_ids.device
    mconfig: FlashMQATConfig = model.config
    bs, prompt_padded_len = input_ids.shape[:2]

    terminate = False
    generated_idx = 0
    unfinished_sequences = torch.ones(bs, dtype=torch.long, device=device)

    gen_token_ph = []
    gen_logprob_ph = []
    gen_logits_mask_ph = []

    prompt_logits = None
    # Prepare inputs for generation iterations
    if k_caches is None:
        # Generate from scratch.
        # Input_ids may have different lengths, we should first pack them into a large batch
        # to use varlen flash attention, then record kv caches for the following inferences.
        packed_input_ids, cu_seqlens, max_seq_len = build_packed_inputs(input_ids, attention_mask)
        input_lens = cu_seqlens[1:] - cu_seqlens[:-1]

        x = PipeTransferData(cu_seqlens=cu_seqlens, max_seqlen=max_seq_len)
        # one embedding layer, n_layers transformer block, one output layer
        ys = [PipeCacheData(input_ids=packed_input_ids)] + [
            PipeCacheData() for _ in range(mconfig.n_layers + 1)
        ]
        # Model forward will set k/v cache in PipeCacheData.
        prompt_logits = model(x, ys).pp_output
        logits = prompt_logits[cu_seqlens[1:] - 1]
        for y in ys[1:-1]:
            assert y.k_cache is not None and y.v_cache is not None and y.cache_seqlens is not None
            kvcache_seqlen = max(
                max_seq_len + gconfig.max_new_tokens, mconfig.hidden_dim // mconfig.head_dim + 10
            )
            # fix of a flash attention bug
            k_cache = torch.zeros(
                (bs, kvcache_seqlen, *y.k_cache.shape[1:]), dtype=y.k_cache.dtype, device=device
            )
            v_cache = torch.zeros(
                (bs, kvcache_seqlen, *y.v_cache.shape[1:]), dtype=y.v_cache.dtype, device=device
            )
            for i in range(bs):
                k_cache[i, : input_lens[i]] = y.k_cache[cu_seqlens[i] : cu_seqlens[i + 1]]
                v_cache[i, : input_lens[i]] = y.v_cache[cu_seqlens[i] : cu_seqlens[i + 1]]
            y.k_cache = k_cache
            y.v_cache = v_cache
            y.cache_seqlens = input_lens.clone()
        x = PipeTransferData()
        ys[0].cache_seqlens = input_lens.clone()
        # Next, we will generate the next token after prompts.
        # cache_seqlens is exactly the lengths of prompts.
        next_tokens, logprob, logits_mask, terminate, unfinished_sequences = genstep(
            logits, tokenizer, unfinished_sequences, generated_idx, gconfig
        )
        gen_token_ph.append(next_tokens)
        gen_logprob_ph.append(logprob)
        gen_logits_mask_ph.append(logits_mask)
        generated_idx += 1
    else:
        # Resume from a previous generation state.
        if prompt_padded_len != 1:
            raise ValueError("prompt_padded_len must be 1 when resuming from a previous generation state.")
        max_seq_len = gconfig.max_new_tokens + int(max(cache_seqlens)) + 1
        for i in range(len(k_caches)):
            pad = (0, 0, 0, 0, 0, max_seq_len - k_caches[i].shape[1])
            if k_caches[i].shape[1] < max_seq_len:
                k_caches[i] = nn.functional.pad(k_caches[i], pad)
            if v_caches[i].shape[1] < max_seq_len:
                v_caches[i] = nn.functional.pad(v_caches[i], pad)
        x = PipeTransferData()
        ys = (
            [PipeCacheData(cache_seqlens=cache_seqlens.clone())]
            + [
                PipeCacheData(k_cache=k, v_cache=v, cache_seqlens=cache_seqlens.clone())
                for k, v in zip(k_caches, v_caches)
            ]
            + [PipeCacheData()]
        )
        next_tokens = input_ids[:, -1]

    # The main loop.
    while not terminate:
        # the next round of inference
        ys[0].input_ids = next_tokens.unsqueeze(-1)  # [bs, 1], seqlen=1
        ys[0].position_ids = None
        # K/v cache will be changed in-place with flash attention.
        logits = model(x, ys).pp_output.squeeze(dim=1)
        for yidx, y in enumerate(ys[:-1]):
            y.cache_seqlens += 1

        next_tokens, logprob, logits_mask, terminate, unfinished_sequences = genstep(
            logits, tokenizer, unfinished_sequences, generated_idx, gconfig
        )
        gen_token_ph.append(next_tokens)
        gen_logprob_ph.append(logprob)
        gen_logits_mask_ph.append(logits_mask)
        generated_idx += 1

    gen_tokens = torch.stack(gen_token_ph, -1)
    log_probs = torch.stack(gen_logprob_ph, -1)
    if all([m is None for m in gen_logits_mask_ph]):
        logits_mask = None
    else:
        mm = next(m for m in gen_logits_mask_ph if m is not None)
        gen_logits_mask_ph = [torch.ones_like(mm) if m is None else m for m in gen_logits_mask_ph]
        logits_mask = torch.stack(gen_logits_mask_ph, -2)

    return gen_tokens, log_probs, logits_mask, ys[1:-1], prompt_logits


@torch.no_grad()
def vanilla_packed_generate(
    model: FlashMQATForCausalLM,
    tokenizer: transformers.PreTrainedTokenizerFast,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    gconfig: GenerationConfig = dataclasses.field(default_factory=GenerationConfig),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Only used for debugging."""
    mconfig: FlashMQATConfig = model.config

    terminate = False
    generated_idx = 0
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    gen_token_ph = []
    gen_logprob_ph = []
    gen_logits_mask_ph = []

    # The main loop.
    while not terminate:
        packed_input_ids, cu_seqlens, max_seq_len = build_packed_inputs(input_ids, attention_mask)
        x = PipeTransferData(cu_seqlens=cu_seqlens, max_seqlen=max_seq_len)
        # one embedding layer, n_layers transformer block, one output layer
        ys = [PipeCacheData(input_ids=packed_input_ids)] + [
            PipeCacheData() for _ in range(mconfig.n_layers + 1)
        ]
        # Model forward will set k/v cache in PipeCacheData.
        logits = model(x, ys).pp_output
        logits = logits[cu_seqlens[1:] - 1]
        # Next, we will generate the next token after prompts.
        # cache_seqlens is exactly the lengths of prompts.
        next_tokens, logprob, logits_mask, terminate, unfinished_sequences = genstep(
            logits, tokenizer, unfinished_sequences, generated_idx, gconfig
        )
        gen_token_ph.append(next_tokens)
        gen_logprob_ph.append(logprob)
        gen_logits_mask_ph.append(logits_mask)
        generated_idx += 1

        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], 1)
        am = torch.logical_and(
            next_tokens.unsqueeze(-1).not_equal(tokenizer.eos_token_id),
            next_tokens.unsqueeze(-1).not_equal(tokenizer.pad_token_id),
        )
        attention_mask = torch.cat([attention_mask, am], 1)

    gen_tokens = torch.stack(gen_token_ph, -1)
    log_probs = torch.stack(gen_logprob_ph, -1)
    if all([m is None for m in gen_logits_mask_ph]):
        logits_mask = None
    else:
        mm = next(m for m in gen_logits_mask_ph if m is not None)
        gen_logits_mask_ph = [torch.ones_like(mm) if m is None else m for m in gen_logits_mask_ph]
        logits_mask = torch.stack(gen_logits_mask_ph, -2)

    return gen_tokens, log_probs, logits_mask


@torch.no_grad()
def vanilla_cpu_generate(
    model: FlashMQATForCausalLM,
    tokenizer: transformers.PreTrainedTokenizerFast,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    gconfig: GenerationConfig = dataclasses.field(default_factory=GenerationConfig),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Only used for debugging."""
    mconfig: FlashMQATConfig = model.config
    assert str(input_ids.device) == "cpu"

    terminate = False
    generated_idx = 0
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    gen_token_ph = []
    gen_logprob_ph = []
    gen_logits_mask_ph = []

    # The main loop.
    while not terminate:
        x = PipeTransferData(attention_mask=attention_mask)
        # one embedding layer, n_layers transformer block, one output layer
        ys = [PipeCacheData(input_ids=input_ids)] + [PipeCacheData() for _ in range(mconfig.n_layers + 1)]
        # Model forward will set k/v cache in PipeCacheData.
        logits = model(x, ys).pp_output[:, -1, :]
        # Next, we will generate the next token after prompts.
        # cache_seqlens is exactly the lengths of prompts.
        next_tokens, logprob, logits_mask, terminate, unfinished_sequences = genstep(
            logits, tokenizer, unfinished_sequences, generated_idx, gconfig
        )
        gen_token_ph.append(next_tokens)
        gen_logprob_ph.append(logprob)
        gen_logits_mask_ph.append(logits_mask)
        generated_idx += 1

        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], 1)
        am = torch.logical_and(
            next_tokens.unsqueeze(-1).not_equal(tokenizer.eos_token_id),
            next_tokens.unsqueeze(-1).not_equal(tokenizer.pad_token_id),
        )
        attention_mask = torch.cat([attention_mask, am], 1)

    gen_tokens = torch.stack(gen_token_ph, -1)
    log_probs = torch.stack(gen_logprob_ph, -1)
    if all([m is None for m in gen_logits_mask_ph]):
        logits_mask = None
    else:
        mm = next(m for m in gen_logits_mask_ph if m is not None)
        gen_logits_mask_ph = [torch.ones_like(mm) if m is None else m for m in gen_logits_mask_ph]
        logits_mask = torch.stack(gen_logits_mask_ph, -2)

    return gen_tokens, log_probs, logits_mask


class InflightBatchingGenerator:
    def __init__(
        self,
        inqueue: queue.Queue,
        outqueue: queue.Queue,
        model: FlashMQATForCausalLM,
        tokenizer: transformers.PreTrainedTokenizerFast,
        gconfig: GenerationConfig,
        batch_size: int,
        max_prompt_len: int,
    ):
        self.inqueue = inqueue
        self.outqueue = outqueue

        self.model = model
        self.mconfig = mconfig = model.config
        self.tokenizer = tokenizer

        self.gconfig = gconfig
        self.batch_size = batch_size

        kvcache_seqlen = max(
            max_prompt_len + gconfig.max_new_tokens, mconfig.hidden_dim // mconfig.head_dim + 10
        )
        _p = next(self.model.parameters())
        dtype, device = _p.dtype, _p.device

        # internel state/input buffers
        self.k_caches = torch.zeros(
            (self.mconfig.n_layers, batch_size, kvcache_seqlen, mconfig.n_kv_heads, mconfig.head_dim),
            dtype=dtype,
            device=device,
        )
        self.v_caches = torch.zeros_like(self.k_caches)
        self.cache_seqlens = torch.zeros((batch_size,), dtype=torch.int32, device=device)
        # generate_idx and cache_seqlens differ at a prompt_len
        self.generate_idx = torch.zeros((batch_size,), dtype=torch.int32, device=device)
        self.input_buf = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        self.prompt_tokens = [None for _ in range(batch_size)]
        self.unfinished_sequences = torch.zeros((batch_size,), dtype=torch.float32, device=device)

        # output buffers
        self.output_tokens_buf = [[] for _ in range(batch_size)]
        self.output_logprob_buf = [[] for _ in range(batch_size)]
        self.output_logits_mask = [[] for _ in range(batch_size)]

    def _get_non_eos_logits(self) -> torch.FloatTensor:
        x = PipeTransferData()
        ys = (
            [
                PipeCacheData(
                    cache_seqlens=self.cache_seqlens.clone(),
                    input_ids=self.input_buf.clone(),
                )
            ]
            + [
                PipeCacheData(k_cache=k, v_cache=v, cache_seqlens=self.cache_seqlens.clone())
                for k, v in zip(self.k_caches, self.v_caches)
            ]
            + [PipeCacheData()]
        )
        logits = self.model(x, ys).pp_output.squeeze(dim=1)

        self.cache_seqlens += 1
        return logits.float()

    def _get_inflight_logits(self) -> torch.FloatTensor:
        finished_sequences = self.unfinished_sequences.logical_not()
        assert finished_sequences.any()

        finish_indices = finished_sequences.nonzero().squeeze(-1).tolist()

        # pop out finished sequences and clear corresponding buffers
        for i in finish_indices:
            prompt_tokens = self.prompt_tokens[i]

            # Used to skip the first call.
            if prompt_tokens is not None:
                gen_tokens = torch.stack(self.output_tokens_buf[i])
                gen_logp = torch.stack(self.output_logprob_buf[i])
                if all([m is None for m in self.output_logits_mask[i]]):
                    gen_logits_mask = None
                else:
                    mm = next(m for m in self.output_logits_mask[i] if m is not None)
                    gen_logits_mask = [
                        torch.ones_like(mm) if m is None else m for m in self.output_logits_mask[i]
                    ]
                    gen_logits_mask = torch.stack(gen_logits_mask, -2)

                res = dict(prompt=prompt_tokens, gen=gen_tokens, logp=gen_logp, logits_mask=gen_logits_mask)
                try:
                    self.outqueue.put_nowait(res)
                except queue.Full as e:
                    raise RuntimeError("Output queue is full. Please set a larger queue size.") from e

            self.k_caches[:, i] = 0
            self.v_caches[:, i] = 0
            self.input_buf[i] = self.tokenizer.pad_token_id
            self.prompt_tokens[i] = None
            self.cache_seqlens[i] = 0
            self.generate_idx[i] = 0
            self.unfinished_sequences[i] = 1

            self.output_logits_mask[i] = []
            self.output_tokens_buf[i] = []
            self.output_logprob_buf[i] = []

        # build packed input ids with variable lengths for the next-step inference
        packed_input_ids = []
        for i in range(self.batch_size):
            if i in finish_indices:
                try:
                    prompt = self.inqueue.get_nowait()
                    self.prompt_tokens[i] = prompt
                    packed_input_ids.append(prompt)
                except queue.Empty as e:
                    raise RuntimeError("Input queue is empty. This should not happen.") from e
            else:
                packed_input_ids.append(self.input_buf[i])
        seqlens = [x.shape[0] for x in packed_input_ids]
        packed_input_ids = torch.cat(packed_input_ids)
        max_seqlen = int(max(seqlens))
        input_lens = torch.tensor(seqlens, device=packed_input_ids.device)
        cu_seqlens = torch.cat([input_lens.new_zeros(1), input_lens.cumsum(0)]).to(
            device=packed_input_ids.device, dtype=torch.int32
        )

        x = PipeTransferData(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        ys = (
            [
                PipeCacheData(
                    cache_seqlens=self.cache_seqlens.clone(),
                    input_ids=packed_input_ids,
                )
            ]
            + [
                PipeCacheData(k_cache=k, v_cache=v, cache_seqlens=self.cache_seqlens.clone())
                for k, v in zip(self.k_caches, self.v_caches)
            ]
            + [PipeCacheData()]
        )
        logits = self.model(x, ys).pp_output
        logits = logits[cu_seqlens[1:] - 1]

        self.cache_seqlens += input_lens

        return logits.float()

    def advance_one_genstep(self):
        if self.unfinished_sequences.logical_not().any():
            logits = self._get_inflight_logits()
        else:
            logits = self._get_non_eos_logits()

        next_tokens, logprob, logits_mask, _, self.unfinished_sequences = genstep(
            logits, self.tokenizer, self.unfinished_sequences, self.generate_idx, self.gconfig
        )

        for i in range(self.batch_size):
            self.output_tokens_buf[i].append(next_tokens[i].long())
            self.output_logprob_buf[i].append(logprob[i].float())
            if logits_mask is not None:
                self.output_logits_mask[i].append(logits_mask[i].bool())
            else:
                self.output_logits_mask[i].append(None)

        self.generate_idx += 1
        self.input_buf[:, 0] = next_tokens

    def step_for(self, n: int):
        for _ in range(n):
            self.advance_one_genstep()
