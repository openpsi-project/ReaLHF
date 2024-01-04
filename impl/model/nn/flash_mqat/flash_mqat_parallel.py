from typing import *

import torch

from .flash_mqat_base import FlashMQATConfig
from base.monitor import process_memory_mb
from impl.model.nn.flash_mqat.flash_mqat_base import (FlashMQATBlock, FlashMQATConfig, FlashMQATModel,
                                                      OutputHead, SequenceParallelActorHead,
                                                      SequenceParallelCriticHead, VocabPositionEmbedding)
from impl.model.parallelism.pipeline_parallel.pipeline_module import LayerSpec, PipelineModule
import api.huggingface
import api.model
import base.constants
import base.logging as logging

logger = logging.getLogger("flash mqat parallel")


# model parallel partition util functions
def mp_partition(tensor: torch.Tensor, mp_rank: Optional[int], mp_world_size: int, dim: int) -> torch.Tensor:
    assert tensor.shape[dim] % mp_world_size == 0
    splits = torch.split(tensor, tensor.shape[dim] // mp_world_size, dim=dim)
    if mp_rank is None:
        return [s.contiguous() for s in splits]
    else:
        return splits[mp_rank].contiguous()
    # return tensor.narrow(dim, mp_rank * tensor.shape[dim] // mp_world_size,
    #                      tensor.shape[dim] // mp_world_size)


def mp_partition_flash_mqat_state_dict(
    state_dict: Dict[str, torch.Tensor],
    config: FlashMQATConfig,
    mp_size: int,
) -> List[Dict]:
    # the qkv linear in non-paralleled model is merged. We should split it first.
    for i in range(1, config.n_layers + 1):
        print(list(state_dict.keys()))
        w = state_dict[f"{i}.attn.c_attn.linear.weight"]
        nq = config.hidden_dim // config.head_dim
        q_proj_w = w[:nq * config.head_dim]
        k_proj_w = w[nq * config.head_dim:(nq + config.n_kv_heads) * config.head_dim]
        v_proj_w = w[(nq + config.n_kv_heads) * config.head_dim:]
        state_dict[f"{i}.attn.c_attn.q_attn.weight"] = q_proj_w
        state_dict[f"{i}.attn.c_attn.k_attn.weight"] = k_proj_w
        state_dict[f"{i}.attn.c_attn.v_attn.weight"] = v_proj_w
        state_dict.pop(f"{i}.attn.c_attn.linear.weight")
        if f"{i}.attn.c_attn.linear.bias" in state_dict:
            b = state_dict[f"{i}.attn.c_attn.linear.bias"]
            state_dict[f"{i}.attn.c_attn.q_attn.weight"] = b[:nq * config.head_dim]
            state_dict[f"{i}.attn.c_attn.k_attn.bias"] = b[nq * config.head_dim:(nq + config.n_kv_heads) *
                                                           config.head_dim]
            state_dict[f"{i}.attn.c_attn.v_attn.bias"] = b[(nq + config.n_kv_heads) * config.head_dim:]
            state_dict.pop(f"{i}.attn.c_attn.linear.bias")

    # converted to normal llama, now partition into model parallel ckpt for current rank
    # keys used to identify modules
    embedding_keys = [".wte", ".wpe"]  # dim=0 no bias
    column_linear_keys = [
        ".attn.c_attn.q_attn",
        ".attn.c_attn.k_attn",
        ".attn.c_attn.v_attn",
        ".mlp.c_fc",
        ".mlp.gate_proj",
        ".mlp.up_proj",
        f"{config.n_layers + 1}.weight",
    ]  # dim=0 + partition bias
    row_linear_keys = [".attn.c_proj", ".mlp.down_proj"]  # dim=-1 + no partition bias

    for k, v in state_dict.items():
        # print(f"key {k}:: ")
        # print(f"before partition shape {state_dict[k].shape}")
        if any([ek in k for ek in embedding_keys]):
            if "weight" in k:
                state_dict[k] = mp_partition(v, None, mp_size, dim=0)
        elif any([ck in k for ck in column_linear_keys]):
            if "weight" in k:
                state_dict[k] = mp_partition(v, None, mp_size, dim=0)
            if "bias" in k:
                state_dict[k] = mp_partition(v, None, mp_size, dim=0)
        elif any([rk in k for rk in row_linear_keys]):
            if "weight" in k:
                state_dict[k] = mp_partition(v, None, mp_size, dim=-1)
        else:
            # replicate weights across all models
            state_dict[k] = [state_dict[k] for _ in range(mp_size)]
        # print(f"after partition shape {state_dict[k].shape}")

    return [{k: v[mp_rank] for k, v in state_dict.items()} for mp_rank in range(mp_size)]


def make_causal_flash_mqat_pipe_module(
    config: FlashMQATConfig,
    partition_method: str = "parameters",
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    output_layer_specs_only: bool = False,
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
            dtype=dtype,
            device=device,
        )
        layer_specs.append(flash_mqat_block)

    if config.is_critic and config.sequence_parallel:
        head = LayerSpec(
            SequenceParallelCriticHead,
            config.hidden_dim,
            # here preserve the original head for critic and swap it in pipeline modules
            # to preserve same pipe stage division.
            config.vocab_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
    elif not config.is_critic and base.constants.model_parallel_world_size() > 1:
        head = LayerSpec(
            SequenceParallelActorHead,
            config.hidden_dim,
            config.vocab_size,
            sequence_parallel=config.sequence_parallel,
            gradient_accumulation_fusion=config.gradient_accumulation_fusion,
            async_tensor_model_parallel_allreduce=not config.sequence_parallel,
            bias=False,
            device=device,
            dtype=dtype,
        )
    else:
        head = LayerSpec(
            OutputHead,
            config.hidden_dim,
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

    if output_layer_specs_only:
        return layer_specs

    return PipelineModule(
        layers=layer_specs,
        loss_fn=compute_loss,
        is_critic=config.is_critic,
        partition_method=partition_method,
        topology=base.constants.grid()._topo,
        config=config,
    )


def pipe_wrap_fn(
    model_path: str,
    partition_method: str = "parameters",
    init_critic_from_actor: bool = False,
    init_from_scratch: bool = False,
):

    def pipe_wrap_fn_(model: api.model.Model) -> api.model.Model:
        if not isinstance(model.module, FlashMQATModel):
            raise RuntimeError(f"Only FlashMQAT models can be wrapped as "
                               f"pipeline module, provided type {type(model.module)}")
        config = model.module.config
        module = make_causal_flash_mqat_pipe_module(
            config,
            partition_method=partition_method,
            dtype=model.dtype,
            device=model.device,
        )
        if not init_from_scratch:
            process_memory_mb("before_load")
            module.load(model_path, init_critic_from_actor=init_critic_from_actor)
            process_memory_mb("after_load")
        model.module = module
        return model

    return pipe_wrap_fn_


api.model.register_wrapper("pipe_flash_mqat", pipe_wrap_fn)
