from typing import Dict, List, Optional, Tuple, Union
import functools

import torch

from base.monitor import process_memory_mb
from base.topology import PipeDataParallelTopology
from impl.model.nn.flash_mqat.flash_mqat_base import (FlashMQATBlock, FlashMQATConfig, FlashMQATForCausalLM,
                                                      LanguageModelHead, VocabPositionEmbedding)
from impl.model.utils.pipeline_module import LayerSpec, PipelineModule
import api.huggingface
import api.model
import base.logging as logging

logger = logging.getLogger("pipe_nn")


def make_causal_flash_mqat_pipe_module(
    config: FlashMQATConfig,
    topology: PipeDataParallelTopology,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
):
    layer_specs = []
    # vocab pos embedding
    embedding_layer = LayerSpec(VocabPositionEmbedding, config, dtype=dtype, device=device)

    layer_specs.append(embedding_layer)

    for i in range(config.n_layers):
        flash_mqat_block = LayerSpec(
            FlashMQATBlock,
            config,
            layer_index=i,
            output_layernorm=(i == config.n_layers - 1),
            ckpt_attn=(i > 0 and config.ckpt_attn),
            ckpt_mlp=(i > 0 and config.ckpt_mlp),
            dtype=dtype,
            device=device,
        )
        layer_specs.append(flash_mqat_block)

    lm_head = LayerSpec(
        LanguageModelHead,
        config.hidden_dim,
        config.vocab_size,
        bias=False,
        device=device,
        dtype=dtype,
    )
    layer_specs.append(lm_head)

    def compute_loss(output, label):
        return output.loss

    return PipelineModule(
        layers=layer_specs,
        loss_fn=compute_loss,
        topology=topology,
        config=config,
    )


def make_flash_mqat_pipe_model(
    name: str,
    device: torch.device,
    model_path: str,
    num_pp: int,
    num_dp: int,
    from_type: str,
    dtype: torch.dtype = torch.float16,
    tokenizer_path: Optional[str] = None,
):
    if tokenizer_path is None:
        tokenizer_path = model_path
    tokenizer = api.huggingface.load_hf_tokenizer(model_path)
    topology = PipeDataParallelTopology(num_pp=num_pp, num_dp=num_dp)
    config = getattr(FlashMQATForCausalLM, f"config_from_{from_type}")(model_path=model_path)
    module = make_causal_flash_mqat_pipe_module(config, topology, dtype, device)
    process_memory_mb("before_load")
    module.load(model_path)
    process_memory_mb("after_load")
    return api.model.Model(name, module, tokenizer, device)


api.model.register_model("flash_mqat_pipe", make_flash_mqat_pipe_model)
