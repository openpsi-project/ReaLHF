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
from impl.model.nn.flash_mqat.flash_mqat_base import FlashMQATConfig, FlashMQATModel
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


# a helper function to make flash_mqat look like huggingface model
def generate_helper(
        self: FlashMQATModel,
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


# a helper function to make flash_mqat look like huggingface model
def forward_helper(
    self: FlashMQATModel,
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
        batch_size, seqlen = input_ids.shape[:2]
    if packed_input_ids is not None:
        x = PipeTransferData(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        ys = [PipeCacheData(input_ids=packed_input_ids)
              ] + [PipeCacheData() for _ in range(self.config.n_layers + 1)]
    else:
        x = PipeTransferData()
        ys = [PipeCacheData(input_ids=input_ids)] + [PipeCacheData() for _ in range(self.config.n_layers + 1)]
    scores = FlashMQATModel.forward(self, x, ys).pp_output
    if build_packed:
        scores = pad_input(scores, indices, batch_size, seqlen)
    return scores


def add_helper_functions(m: FlashMQATModel):
    m.forward = functools.partial(forward_helper, m)
    m.generate = functools.partial(generate_helper, m)
    return m


def make_flash_model(
    name: str,
    device: torch.device,
    model_path: str,
    from_type: str,
    dtype: Optional[str] = None,
    hf_model_type: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    sequence_parallel: bool = False,
    gradient_accumulation_fusion: bool = False,
) -> api.model.Model:
    """Make a FlashMQATModel.

    There are 5 primitive model types that we can construct FlashMQAT from:
        + HF models
        + actors w. PP
        + actors w./o. PP
        + critics w. PP
        + critics w.o. PP
    We can construct either actor or critic from them (but constructing actor
    from critic is not allowed).

    And there are 2 special types of FlashMQAT models:
        random (aka init_from_scratch) and empty (e.g. used for pipeline module).
    """
    if dtype == "fp16" or dtype == None:
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp32":
        dtype == torch.float32
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")

    tokenizer = None
    if from_type == "hf_as_critic":
        # Convert a HuggingFace model into FlashMQAT.
        m = getattr(FlashMQATModel, f"from_{hf_model_type}")(
            model_path=model_path,
            dtype=dtype,
            device=device,
            is_critic=True,
            no_param_instantiation=False,
            init_from_scratch=False,
            sequence_parallel=sequence_parallel,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
        )
        tokenizer = api.huggingface.load_hf_tokenizer(model_path)
    elif from_type == "hf_as_actor":
        # Convert a HuggingFace model into FlashMQAT.
        m = getattr(FlashMQATModel, f"from_{hf_model_type}")(
            model_path=model_path,
            dtype=dtype,
            device=device,
            is_critic=False,
            no_param_instantiation=False,
            init_from_scratch=False,
            sequence_parallel=sequence_parallel,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
        )
        tokenizer = api.huggingface.load_hf_tokenizer(model_path)
    elif from_type == "actor_as_critic":
        # initialize a critic from actor
        with open(os.path.join(model_path, "flash_mqat_config.json"), "r") as f:
            config = FlashMQATConfig(**json.load(f))
        config.is_critic = True
        config.sequence_parallel = sequence_parallel
        config.gradient_accumulation_fusion = gradient_accumulation_fusion
        m = FlashMQATModel(config=config, no_param_instantiation=False, dtype=dtype, device=device)
        m.load(model_path, init_critic_from_actor=True, load_from_pipe=False)
    elif from_type == "pp_actor_as_critic":
        # initialize a critic from a pipeline-parallel actor
        with open(os.path.join(model_path, "flash_mqat_config.json"), "r") as f:
            config = FlashMQATConfig(**json.load(f))
        config.is_critic = True
        config.sequence_parallel = sequence_parallel
        config.gradient_accumulation_fusion = gradient_accumulation_fusion
        m = FlashMQATModel(config=config, no_param_instantiation=False, dtype=dtype, device=device)
        m.load(model_path, init_critic_from_actor=True, load_from_pipe=True)
    elif from_type == "pp_self":
        # load a pipeline-parallel actor/critic
        with open(os.path.join(model_path, "flash_mqat_config.json"), "r") as f:
            config = FlashMQATConfig(**json.load(f))
        config.sequence_parallel = sequence_parallel
        config.gradient_accumulation_fusion = gradient_accumulation_fusion
        m = FlashMQATModel(config=config, no_param_instantiation=False, dtype=dtype, device=device)
        m.load(model_path, init_critic_from_actor=False, load_from_pipe=True)
    elif from_type == "random_actor":
        # randomly initialize a actor
        m = getattr(FlashMQATModel, f"from_{hf_model_type}")(
            model_path=tokenizer_path,
            dtype=dtype,
            device=device,
            is_critic=False,
            no_param_instantiation=False,
            init_from_scratch=True,
            sequence_parallel=sequence_parallel,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
        )
    elif from_type == "random_critic":
        # randomly initialize a critic
        m = getattr(FlashMQATModel, f"from_{hf_model_type}")(
            model_path=tokenizer_path,
            dtype=dtype,
            device=device,
            is_critic=True,
            no_param_instantiation=False,
            init_from_scratch=True,
            sequence_parallel=sequence_parallel,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
        )
    elif from_type == "empty_actor":
        # initialize an empty actor, probably used for pipeline module
        m = getattr(FlashMQATModel, f"from_{hf_model_type}")(
            model_path=tokenizer_path,
            dtype=dtype,
            device=device,
            is_critic=False,
            no_param_instantiation=True,
            init_from_scratch=True,
            sequence_parallel=sequence_parallel,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
        )
    elif from_type == "empty_critic":
        # initialize a empty critic, probably used for pipeline module
        m = getattr(FlashMQATModel, f"from_{hf_model_type}")(
            model_path=tokenizer_path,
            dtype=dtype,
            device=device,
            is_critic=True,
            no_param_instantiation=True,
            init_from_scratch=True,
            sequence_parallel=sequence_parallel,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
        )
    else:
        # load a non-pipeline actor/critic
        assert from_type == "self"
        with open(os.path.join(model_path, "flash_mqat_config.json"), "r") as f:
            config = FlashMQATConfig(**json.load(f))
        config.sequence_parallel = sequence_parallel
        config.gradient_accumulation_fusion = gradient_accumulation_fusion
        m = FlashMQATModel(config=config, no_param_instantiation=False, dtype=dtype, device=device)
        m.load(model_path, init_critic_from_actor=False, load_from_pipe=False)

    if tokenizer is None:
        tokenizer = api.huggingface.load_hf_tokenizer(tokenizer_path)

    m = add_helper_functions(m)
    return api.model.Model(name, m, tokenizer, device, dtype=dtype)


api.model.register_model("flash_mqat", make_flash_model)
