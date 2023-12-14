from typing import Optional, Union

import torch

from base.monitor import process_memory_mb
from base.topology import PipeModelDataParallelTopology
from impl.model.nn.flash_mqat.flash_mqat_base import FlashMQATConfig, FlashMQATModel, OutputHead
from impl.model.nn.flash_mqat.flash_mqat_parallel import (ModelParallelModule, ParallelFlashMQATBlock,
                                                          ParallelVocabPositionEmbedding)
from impl.model.utils.pipeline_module import LayerSpec, PipelineModule
import api.huggingface
import api.model
import base.logging as logging

logger = logging.getLogger("model_parallel_nn")


def make_causal_flash_mqat_parallel_pipe_module(
    config: FlashMQATConfig,
    topology: PipeModelDataParallelTopology,
    is_critic: bool = False,
    partition_method: str = "parameters",
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
):
    layer_specs = []
    # vocab pos embedding
    embedding_layer = LayerSpec(ParallelVocabPositionEmbedding, config, dtype=dtype, device=device)

    layer_specs.append(embedding_layer)

    for i in range(config.n_layers):
        flash_mqat_block = LayerSpec(
            ParallelFlashMQATBlock,
            config,
            layer_index=i,
            output_layernorm=(i == config.n_layers - 1),
            ckpt_attn=(i > 0 and config.ckpt_attn),
            ckpt_mlp=(i > 0 and config.ckpt_mlp),
            dtype=dtype,
            device=device,
        )
        layer_specs.append(flash_mqat_block)

    head = LayerSpec(
        OutputHead,
        config.hidden_dim,
        # 1 if is_critic else config.vocab_size,
        # here preserve the original head for critic and swap it in pipeline modules
        # to preserve same pipe stage division.
        config.vocab_size,
        bias=False,
        device=device,
        dtype=dtype,
    )

    layer_specs.append(head)

    def compute_loss(output, label):
        return output.loss

    return PipelineModule(
        layers=layer_specs,
        loss_fn=compute_loss,
        topology=topology,
        is_critic=is_critic,
        partition_method=partition_method,
        dtype=dtype,
        device=device,
        config=config,
    )


def model_pipe_wrap_fn(
    model_path: str,
    num_pp: int,
    num_mp: int,
    num_dp: int,
    is_critic: bool,
    partition_method: str = "parameters",
    init_critic_from_actor: bool = False,
    init_from_scratch: bool = False,
):

    def model_pipe_wrap_fn_(model: api.model.Model) -> api.model.Model:

        if not isinstance(model.module, FlashMQATModel):
            raise RuntimeError(f"Only FlashMQAT models can be wrapped as "
                               f"pipeline module, provided type {type(model.module)}")
        config = model.module.config
        topology = PipeModelDataParallelTopology(num_pp=num_pp, num_mp=num_mp, num_dp=num_dp)
        module = make_causal_flash_mqat_parallel_pipe_module(config,
                                                             topology,
                                                             is_critic,
                                                             partition_method=partition_method,
                                                             device=model.device)
        if not init_from_scratch:
            process_memory_mb("before_load")
            module.load(model_path, init_critic_from_actor=init_critic_from_actor)
            process_memory_mb("after_load")
        model.module = module
        return model

    return model_pipe_wrap_fn_


api.model.register_wrapper("model_pipe_parallel", model_pipe_wrap_fn)


def model_parallel_wrap_fn(
    model_path: str,
    is_critic: bool,
    init_critic_from_actor: bool = False,
    init_from_scratch: bool = False,
):

    def model_parallel_wrap_fn_(model: api.model.Model) -> api.model.Model:
        if not isinstance(model.module, FlashMQATModel):
            raise RuntimeError(f"Only FlashMQAT models can be wrapped as "
                               f"pipeline module, provided type {type(model.module)}")
        config = model.module.config
        module = ModelParallelModule(model.module, config, device=model.device)
        if not init_from_scratch:
            process_memory_mb("before_load")
            module.load(model_path, init_critic_from_actor=init_critic_from_actor)
            process_memory_mb("after_load")
        model.module = module
        return model

    return model_parallel_wrap_fn_


api.model.register_wrapper("model_parallel", model_parallel_wrap_fn)
