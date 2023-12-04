from typing import Dict, List, Optional, Tuple, Union
import functools

import torch

from base.monitor import process_memory_mb
from base.topology import PipeDataParallelTopology
from impl.model.nn.flash_mqat.flash_mqat_base import (FlashMQATBlock, FlashMQATConfig, FlashMQATForCausalLM,
                                                      LanguageModelHead, VocabPositionEmbedding)
from impl.model.nn.flash_mqat.flash_mqat_interface import (DeepSpeedChatLikeFlashMQATCriticModel,
                                                           HuggingfaceLikeFlashMQATForCausalLM)
from impl.model.utils.pipeline_module import LayerSpec, PipelineModule
import api.huggingface
import api.model
import base.logging as logging

logger = logging.getLogger("pipe_nn")


def make_causal_flash_mqat_pipe_module(
    config: FlashMQATConfig,
    topology: PipeDataParallelTopology,
    is_critic: bool = False,
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

    if not is_critic:
        lm_head = LayerSpec(
            LanguageModelHead,
            config.hidden_dim,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
    else:
        lm_head = LayerSpec(
            LanguageModelHead,
            config.hidden_dim,
            1,
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


def pipe_wrap_fn(model_path: str, num_pp: int, num_dp: int, is_critic: bool, init_from_scratch: bool = False):

    def pipe_wrap_fn_(model: api.model.Model) -> api.model.Model:
        topology = PipeDataParallelTopology(num_pp=num_pp, num_dp=num_dp)
        if not isinstance(
                model.module,
            (DeepSpeedChatLikeFlashMQATCriticModel, HuggingfaceLikeFlashMQATForCausalLM),
        ):
            raise RuntimeError(f"Only FlashMQAT models can be wrapped as "
                               f"pipeline module, provided type {type(model.module)}")
        config = model.module.config
        module = make_causal_flash_mqat_pipe_module(config, topology, is_critic, device=model.device)
        if not init_from_scratch:
            process_memory_mb("before_load")
            module.load(model_path)
            process_memory_mb("after_load")
        model.module = module
        return model

    return pipe_wrap_fn_


api.model.register_wrapper("pipe", pipe_wrap_fn)
