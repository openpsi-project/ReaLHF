from typing import List, Optional, Union
import dataclasses
import functools
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers

from impl.model.nn.flash_mqat.flash_generate import generate, GenerationConfig
from impl.model.nn.flash_mqat.flash_mqat_base import (FlashMQATBase, FlashMQATBlock, FlashMQATConfig,
                                                      FlashMQATModel)
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


class HuggingfaceLikeFlashMQATForCausalLM(FlashMQATModel):
    """__call__ on this model will return a huggingface-like output."""

    def __init__(self,
                 config: FlashMQATConfig,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 **kwargs):
        super().__init__(config, is_critic=False, dtype=dtype, device=device)

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
        logits = FlashMQATModel.forward(self, x, ys).pp_output
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
        current_forward = self.forward
        self.forward = functools.partial(FlashMQATModel.forward, self)
        seq, scores, mask, _, _ = generate(
            self,
            tokenizer,
            input_ids,
            attention_mask,
            k_caches,
            v_caches,
            cache_seqlens,
            gconfig,
        )
        self.forward = current_forward
        return DuckGenerationOutput(seq, scores, mask)


class DeepSpeedChatLikeFlashMQATCriticModel(FlashMQATModel):

    def __init__(self,
                 config: FlashMQATConfig,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 output_scaling: float = 1.0,
                 output_bias: float = 0.0,
                 **kwargs):
        super().__init__(config, is_critic=True, dtype=dtype, device=device)
        self.output_scaling = output_scaling
        self.output_bias = output_bias

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
                  ] + [PipeCacheData() for _ in range(self.config.n_layers + 1)]
        else:
            x = PipeTransferData()
            ys = [PipeCacheData(input_ids=input_ids)
                  ] + [PipeCacheData() for _ in range(self.config.n_layers + 1)]
        scores = FlashMQATModel.forward(self, x, ys).pp_output
        if build_packed:
            scores = pad_input(scores, indices, batch_size, seqlen)
        return (scores.squeeze(-1) - self.output_bias) * self.output_scaling


class DummyNet(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config


def make_flash_model(module_cls: FlashMQATModel,
                     name: str,
                     device: torch.device,
                     model_path: str,
                     dtype: Optional[torch.dtype] = None,
                     from_type: str = "starcoder",
                     tokenizer_path: Optional[str] = None,
                     init_from_scratch: bool = False,
                     v_head_path: Optional[str] = None,
                     output_scaling: Optional[float] = None,
                     output_bias: Optional[float] = None,
                     config_only: Optional[bool] = False) -> api.model.Model:
    is_critic = module_cls == DeepSpeedChatLikeFlashMQATCriticModel
    if config_only:
        config = getattr(FlashMQATModel, f"config_from_{from_type}")(model_path=model_path)
        net = DummyNet(config)
    elif from_type == "self":
        # Initialize from a self-trained model, e.g., PPO actor loading SFT model.
        net = FlashMQATModel.from_pretrained(
            model_path=model_path,
            init_from_scratch=init_from_scratch,
            is_critic=is_critic,
            dtype=dtype,
            device=device,
        )
        if tokenizer_path is None:
            raise ValueError("tokenizer_path must be provided when from_type is 'self'.")
        tokenizer = api.huggingface.load_hf_tokenizer(tokenizer_path)
    elif from_type == "pipe":
        # Merge weights of the pipeline model into a single one, probably used for inference.
        if init_from_scratch:
            raise ValueError("init_from_scratch must be False when from_type is 'pipe'.")
        net = FlashMQATModel.from_pipeline_module(model_path=model_path,
                                                  is_critic=is_critic,
                                                  dtype=dtype,
                                                  device=device)
        if tokenizer_path is None:
            raise ValueError("tokenizer_path must be provided when from_type is 'self'.")
        tokenizer = api.huggingface.load_hf_tokenizer(tokenizer_path)
    elif from_type == "sft":
        # Special load for reward model loading from SFT. The value head will be re-initialized.
        if not is_critic:
            raise RuntimeError("from_type 'sft' is only supported for critic model.")
        with open(os.path.join(model_path, "config.json"), "r") as f:
            config = FlashMQATConfig(**json.load(f))
        net = module_cls(
            config,
            dtype=dtype,
            device=device,
            output_scaling=output_scaling,
            output_bias=output_bias,
        )
        if not init_from_scratch:
            state_dict = load_from_disk(model_path)
            state_dict["head.weight"] = net.state_dict()["head.weight"]
            net.load_state_dict(state_dict)
        tokenizer = api.huggingface.load_hf_tokenizer(tokenizer_path)
    else:
        # Convert a HuggingFace model into FlashMQAT.
        net = getattr(FlashMQATModel, f"from_{from_type}")(model_path=model_path,
                                                           dtype=dtype,
                                                           device=device,
                                                           is_critic=is_critic,
                                                           init_from_scratch=init_from_scratch)
        tokenizer = api.huggingface.load_hf_tokenizer(model_path)
    if not isinstance(net, module_cls):
        net.forward = functools.partial(module_cls.forward, net)
        if hasattr(module_cls, "generate"):
            net.generate = functools.partial(module_cls.generate, net)
        if is_critic:
            net.output_scaling = output_scaling
            net.output_bias = output_bias
    if v_head_path is not None and not init_from_scratch:
        net.head.load_state_dict(torch.load(v_head_path, map_location="cpu"))
    return api.model.Model(name, net, tokenizer, device)


api.model.register_model("flash_mqat_actor",
                         functools.partial(make_flash_model, HuggingfaceLikeFlashMQATForCausalLM))
api.model.register_model("flash_mqat_critic",
                         functools.partial(make_flash_model, DeepSpeedChatLikeFlashMQATCriticModel))
