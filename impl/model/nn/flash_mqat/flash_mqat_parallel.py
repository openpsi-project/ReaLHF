from typing import *

import torch

from .flash_mqat_base import FlashMQATConfig
from base.monitor import process_memory_mb
from impl.model.nn.flash_mqat.flash_mqat_base import (
    FlashMQATBlock,
    FlashMQATConfig,
    FlashMQATModel,
    OutputHead,
    SequenceParallelActorHead,
    SequenceParallelCriticHead,
    VocabPositionEmbedding,
)
from impl.model.parallelism.pipeline_parallel.pipeline_module import LayerSpec, PipelineModule
import api.huggingface
import api.model
import base.constants
import base.logging as logging

logger = logging.getLogger("flash mqat parallel")

# keys used to identify modules
_embedding_keys = lambda config: [".wte", ".wpe"]  # dim=0 no bias
_column_linear_keys = lambda config: [
    ".attn.c_attn.q_attn",
    ".attn.c_attn.k_attn",
    ".attn.c_attn.v_attn",
    ".mlp.c_fc",
    ".mlp.gate_proj",
    ".mlp.up_proj",
    f"{config.n_layers + 1}.weight",
]  # dim=0 + partition bias
_row_linear_keys = lambda config: [".attn.c_proj", ".mlp.down_proj"]  # dim=-1 + no partition bias


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
        for key in ["weight", "bias"]:
            if f"{i}.attn.c_attn.linear.{key}" not in state_dict:
                continue
            w = state_dict[f"{i}.attn.c_attn.linear.{key}"]
            nq = config.hidden_dim // config.head_dim
            q_proj_w = w[: nq * config.head_dim]
            k_proj_w = w[nq * config.head_dim : (nq + config.n_kv_heads) * config.head_dim]
            v_proj_w = w[(nq + config.n_kv_heads) * config.head_dim :]
            state_dict[f"{i}.attn.c_attn.q_attn.{key}"] = q_proj_w
            state_dict[f"{i}.attn.c_attn.k_attn.{key}"] = k_proj_w
            state_dict[f"{i}.attn.c_attn.v_attn.{key}"] = v_proj_w
            state_dict.pop(f"{i}.attn.c_attn.linear.{key}")

    embedding_keys = _embedding_keys(config)
    column_linear_keys = _column_linear_keys(config)
    row_linear_keys = _row_linear_keys(config)

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


def mp_merge_flash_mqat_state_dict(
    state_dicts: List[Dict[str, torch.Tensor]],
    config: FlashMQATConfig,
) -> Dict:
    mp_size = len(state_dicts)
    if mp_size == 1:
        return state_dicts[0]

    embedding_keys = _embedding_keys(config)
    column_linear_keys = _column_linear_keys(config)
    row_linear_keys = _row_linear_keys(config)

    state_dict = dict()
    for k in state_dicts[0].keys():
        i = int(k.split(".")[0])
        if any([ek in k for ek in embedding_keys]) and "weight" in k:
            state_dict[k] = torch.cat([sd[k] for sd in state_dicts], dim=0)
        elif (
            any([ck in k for ck in column_linear_keys]) and state_dicts[0][k].shape[0] > 1
        ):  # exclude critic head
            state_dict[k] = torch.cat([sd[k] for sd in state_dicts], dim=0)
        elif any([rk in k for rk in row_linear_keys]) and "weight" in k:
            state_dict[k] = torch.cat([sd[k] for sd in state_dicts], dim=1)
        else:
            state_dict[k] = state_dicts[0][k]

    # the qkv linear in non-paralleled model is merged.
    for i in range(1, config.n_layers + 1):
        for key in ["weight", "bias"]:
            if f"{i}.attn.c_attn.q_attn.{key}" not in state_dict:
                continue
            qw = state_dict[f"{i}.attn.c_attn.q_attn.{key}"]
            kw = state_dict[f"{i}.attn.c_attn.k_attn.{key}"]
            vw = state_dict[f"{i}.attn.c_attn.v_attn.{key}"]
            state_dict[f"{i}.attn.c_attn.linear.{key}"] = torch.cat([qw, kw, vw], dim=0)
            state_dict.pop(f"{i}.attn.c_attn.q_attn.{key}")
            state_dict.pop(f"{i}.attn.c_attn.k_attn.{key}")
            state_dict.pop(f"{i}.attn.c_attn.v_attn.{key}")

    return state_dict


def make_causal_flash_mqat_pipe_module(
    config: FlashMQATConfig,
    partition_method: str = "parameters_balanced",
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
    partition_method: str = "parameters_balanced",
    init_critic_from_actor: bool = False,
    init_from_scratch: bool = False,
):
    def pipe_wrap_fn_(model: api.model.Model) -> api.model.Model:
        if not isinstance(model.module, FlashMQATModel):
            raise RuntimeError(
                f"Only FlashMQAT models can be wrapped as "
                f"pipeline module, provided type {type(model.module)}"
            )
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
