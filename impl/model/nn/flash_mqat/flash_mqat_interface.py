from typing import List, Optional, Union
import dataclasses
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers

from impl.model.nn.flash_mqat.flash_generate import generate, GenerationConfig
from impl.model.nn.flash_mqat.flash_mqat_base import (FlashMQATBase, FlashMQATBlock, FlashMQATConfig,
                                                      FlashMQATForCausalLM)
from impl.model.utils.data import DuckGenerationOutput, DuckModelOutput, PipeCacheData, PipeTransferData
from impl.model.utils.save_load import load_from_disk
import api.huggingface
import api.model
import base.logging as logging

try:
    from flash_attn.bert_padding import pad_input, unpad_input
except ModuleNotFoundError:
    pass
import base.logging as logging

logger = logging.getLogger("FlashMQAT Interface")


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
        packed_input_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> DuckModelOutput:
        assert (packed_input_ids is None) == (cu_seqlens is None) == (max_seqlen is None)
        build_packed = False
        if packed_input_ids is None and attention_mask is not None:
            batch_size, seqlen = input_ids.shape
            packed_input_ids, indices, cu_seqlens, max_seqlen = unpad_input(input_ids, attention_mask)
            build_packed = True
        if packed_input_ids is not None:
            x = PipeTransferData(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            ys = [PipeCacheData(input_ids=packed_input_ids)
                  ] + [PipeCacheData() for _ in range(self.config.n_layers + 1)]
        else:
            x = PipeTransferData()
            ys = [PipeCacheData(input_ids=input_ids)
                  ] + [PipeCacheData() for _ in range(self.config.n_layers + 1)]
        logits = self.net(x, ys).pp_output
        if build_packed:
            logits = pad_input(logits, indices, batch_size, seqlen)
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
            self.net,
            tokenizer,
            input_ids,
            attention_mask,
            k_caches,
            v_caches,
            cache_seqlens,
            gconfig,
        )
        return DuckGenerationOutput(seq, scores, mask)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        with open(os.path.join(model_path, "config.json"), "r") as f:
            config = FlashMQATConfig(**json.load(f))
        state_dict = load_from_disk(model_path)
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
) -> api.model.Model:
    if from_type == "self":
        module = HuggingfaceLikeFlashMQATForCausalLM.from_pretrained(model_path=model_path,
                                                                     dtype=dtype,
                                                                     device=device)
        if tokenizer_path is None:
            raise ValueError("tokenizer_path must be provided when from_type is 'self'.")
        tokenizer = api.huggingface.load_hf_tokenizer(tokenizer_path)
    else:
        module = HuggingfaceLikeFlashMQATForCausalLM(
            getattr(FlashMQATForCausalLM, f"from_{from_type}")(model_path=model_path,
                                                               dtype=dtype,
                                                               device=device))
        tokenizer = api.huggingface.load_hf_tokenizer(model_path)
    return api.model.Model(name, module, tokenizer, device)


api.model.register_model("flash_mqat_clm_hf", make_flash_mqat_clm_hf)


class DeepSpeedChatLikeFlashMQATCriticModel(nn.Module):

    def __init__(self, net: FlashMQATBase, output_scaling: float = 1.0, output_bias: float = 0.0):
        super().__init__()
        self.net = net
        self.head = nn.Linear(net.config.hidden_dim,
                              1,
                              bias=False,
                              dtype=self.net.dtype,
                              device=self.net.device)
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
            packed_input_ids, indices, cu_seqlens, max_seqlen = unpad_input(input_ids, attention_mask)
            batch_size, seqlen = input_ids
        if packed_input_ids is not None:
            x = PipeTransferData(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            ys = [PipeCacheData(input_ids=packed_input_ids)
                  ] + [PipeCacheData() for _ in range(self.config.n_layers)]
        else:
            x = PipeTransferData()
            ys = [PipeCacheData(input_ids=input_ids)] + [PipeCacheData() for _ in range(self.config.n_layers)]
        hidden_states = self.net(x, ys).pp_output
        if build_packed:
            hidden_states = pad_input(hidden_states, indices, batch_size, seqlen)
        return (self.head(hidden_states).squeeze() - self.output_bias) * self.output_scaling

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
        state_dict = load_from_disk(model_path)
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
    elif from_type == "self":
        module = DeepSpeedChatLikeFlashMQATCriticModel.from_pretrained(
            model_path=model_path,
            dtype=dtype,
            device=device,
            output_scaling=output_scaling,
            output_bias=output_bias,
        )
    else:
        from_model = make_flash_mqat_clm_hf(
            name=name,
            device=device,
            model_path=model_path,
            dtype=dtype,
            from_type=from_type,
            tokenizer_path=tokenizer_path,
        )
        model = DeepSpeedChatLikeFlashMQATCriticModel(from_model.module.net.transformer,
                                                      output_bias=output_bias,
                                                      output_scaling=output_scaling)
        if v_head_path is not None:
            model.head.load_state_dict(torch.load(v_head_path))
    tokenizer = api.huggingface.load_hf_tokenizer(tokenizer_path)
    return api.model.Model(name, module, tokenizer, device)


api.model.register_model("flash_mqat_critic", make_flash_mqat_critic)
