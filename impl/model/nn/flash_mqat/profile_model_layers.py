from typing import Optional
import json
import os

import torch

from impl.model.nn.flash_mqat.flash_mqat_base import (FlashMQATBlock, FlashMQATConfig, OutputHead,
                                                      SequenceParallelActorHead, SequenceParallelCriticHead,
                                                      VocabPositionEmbedding)
import api.huggingface
import api.model


def make_flash_mqat_block_layer(
    name: str,
    device: torch.device,
    model_path: str,
    dtype: Optional[str] = None,
):
    if dtype == "fp16" or dtype == None:
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp32":
        dtype == torch.float32
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")

    with open(os.path.join(model_path, "flash_mqat_config.json"), "r") as f:
        config = FlashMQATConfig(**json.load(f))

    m = FlashMQATBlock(config=config, layer_index=0, dtype=dtype, device=device)
    tokenizer = api.huggingface.load_hf_tokenizer(model_path)
    return api.model.Model(name=name, module=m, tokenizer=tokenizer, device=device, dtype=dtype)


def make_vocab_pos_embedding_layer(
    name: str,
    device: torch.device,
    model_path: str,
    dtype: Optional[str] = None,
):
    if dtype == "fp16" or dtype == None:
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp32":
        dtype == torch.float32
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")

    with open(os.path.join(model_path, "flash_mqat_config.json"), "r") as f:
        config = FlashMQATConfig(**json.load(f))

    m = VocabPositionEmbedding(config=config, dtype=dtype, device=device)
    tokenizer = api.huggingface.load_hf_tokenizer(model_path)
    return api.model.Model(name=name, module=m, tokenizer=tokenizer, device=device, dtype=dtype)


api.model.register_model("flash_mqat_block_layer", make_flash_mqat_block_layer)
api.model.register_model("vocab_pos_embedding_layer", make_vocab_pos_embedding_layer)
