from typing import *
import os

import numpy as np
import torch

from api.config.config_flash_model import FlashMQATConfig
import base.logging as logging

try:
    import transformer_engine.pytorch as te

    TE_ENABLED = True
except ImportError:
    TE_ENABLED = False
USE_TE_BACKEND = TE_ENABLED and os.getenv("FLASH_MQAT_USE_TE") == "1"

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

if USE_TE_BACKEND:
    _column_linear_keys = lambda config: [
        ".attn.c_attn.q_attn",
        ".attn.c_attn.k_attn",
        ".attn.c_attn.v_attn",
        ".mlp.c_fc",
        ".mlp.fc1_weight",
        f"{config.n_layers + 1}.weight",
    ]  # dim=0 + partition bias
    _row_linear_keys = lambda config: [".attn.c_proj", ".mlp.fc2_weight"]


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
    mp_rank: Optional[int] = None,
) -> List[Dict]:
    # the qkv linear in non-paralleled model is merged. We should split it first.
    if mp_size == 1:
        if mp_rank is None:
            return [state_dict]
        else:
            return state_dict

    embedding_keys = _embedding_keys(config)
    column_linear_keys = _column_linear_keys(config)
    row_linear_keys = _row_linear_keys(config)

    for k, v in state_dict.items():
        # print(f"key {k}:: ")
        # print(f"before partition shape {state_dict[k].shape}")
        if any([ek in k for ek in embedding_keys]):
            if "weight" in k:
                state_dict[k] = mp_partition(v, mp_rank, mp_size, dim=0)
        elif any([ck in k for ck in column_linear_keys]):
            if ("k_attn" or "v_attn" in k) and config.n_kv_heads % mp_size != 0:
                logger.warning(f"Cannot split {config.n_kv_heads} kv heads evenly among "
                               f"{mp_size} model parallel ranks, "
                               f"use unsplitted linear for kv heads instead")
                if mp_rank is None:
                    state_dict[k] = [state_dict[k] for _ in range(mp_size)]
                continue
            if k == f"{config.n_layers + 1}.weight" and v.shape[0] == 1:
                if mp_rank is None:
                    state_dict[k] = [state_dict[k] for _ in range(mp_size)]
                continue
            if "weight" in k:
                state_dict[k] = mp_partition(v, mp_rank, mp_size, dim=0)
            if "bias" in k:
                state_dict[k] = mp_partition(v, mp_rank, mp_size, dim=0)
        elif any([rk in k for rk in row_linear_keys]):
            if "weight" in k:
                state_dict[k] = mp_partition(v, mp_rank, mp_size, dim=-1)
        else:
            # replicate weights across all models
            if mp_rank is None:
                state_dict[k] = [state_dict[k] for _ in range(mp_size)]
        # print(f"after partition shape {state_dict[k].shape}")

    if mp_rank is None:
        return [{k: v[mp_rank] for k, v in state_dict.items()} for mp_rank in range(mp_size)]
    else:
        return state_dict

def get_flash_model_param_shape(k: str, config: FlashMQATConfig, mp_size: int):
    column_linear_keys = _column_linear_keys(config)
    row_linear_keys = _row_linear_keys(config)

    if "wte.weight" in k:
        assert config.vocab_size % mp_size == 0
        return (config.vocab_size // mp_size, config.hidden_dim )
    elif "wpe.weight" in k:
        assert config.n_positions % mp_size == 0
        return (config.n_positions // mp_size, config.hidden_dim)
    elif any([ck in k for ck in column_linear_keys]):
        if "k_attn" in k or "v_attn" in k:
            if "weight" in k:
                if config.n_kv_heads % mp_size == 0:
                    return (config.head_dim * config.n_kv_heads // mp_size, config.hidden_dim)
                else:
                    return (config.head_dim * config.n_kv_heads, config.hidden_dim)
        if k == f"{config.n_layers + 1}.weight":
            return (config.hidden_dim, )
        if "weight" in k:
            assert config.hidden_dim // config.head_dim % mp_size == 0
            return (config.hidden_dim // mp_size, config.hidden_dim)
    elif any([rk in k for rk in row_linear_keys]):
        if "mlp" in k and "weight" in k:
            return (config.hidden_dim, config.intermediate_dim // mp_size)
        elif "attn" in k and "weight" in k:
            return (config.hidden_dim, config.hidden_dim // mp_size)
        elif "bias" in k:
            return (config.hidden_dim // mp_size, )
    else:
        raise NotImplementedError(f"unkown shape of key {k}.")

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
        elif (any([ck in k for ck in column_linear_keys])
              and state_dicts[0][k].shape[0] > 1):  # exclude critic head
            state_dict[k] = torch.cat([sd[k] for sd in state_dicts], dim=0)
        elif any([rk in k for rk in row_linear_keys]) and "weight" in k:
            state_dict[k] = torch.cat([sd[k] for sd in state_dicts], dim=1)
        else:
            state_dict[k] = state_dicts[0][k]

    return state_dict


def partition_pipeline_layers(
    config: FlashMQATConfig,
    num_stages: int,
    embed_param_counter: Callable[[FlashMQATConfig], int],
    transformer_block_param_counter: Callable[[FlashMQATConfig, int], int],
    head_param_counter: Callable[[FlashMQATConfig], int],
    method: str = "parameters_balanced",
) -> Dict[int, Tuple[int, int]]:
    from deepspeed.runtime import utils as ds_utils

    from base.datapack import partition_balanced as true_partition_balanced

    # Each stage gets a simple uniform number of layers.
    param_counts = ([embed_param_counter(config)] +
                    [transformer_block_param_counter(config, i)
                     for i in range(config.n_layers)] + [head_param_counter(config)])
    parts = None
    if method == "uniform":
        parts = ds_utils.partition_uniform(num_items=config.n_layers + 2, num_parts=num_stages)
    elif method == "parameters":
        parts = ds_utils.partition_balanced(weights=param_counts, num_parts=num_stages)
    elif method == "parameters_balanced":
        param_counts = np.array(param_counts)
        parts = true_partition_balanced(nums=param_counts, k=num_stages)
    else:
        raise NotImplementedError(f"Partitioning method {method} not implemented.")

    stage_to_layer_idx = {}
    for stage in range(num_stages):
        start = parts[stage]
        stop = parts[stage + 1]
        stage_to_layer_idx[stage] = (start, stop)
    return stage_to_layer_idx


def pipeline_repartition_strategy(
    layer_mapping1: Dict[int, List[int]],
    layer_mapping2: Dict[int, List[int]],
):
    assert set(sum(layer_mapping1.values(), [])) == set(sum(layer_mapping2.values(), []))

    layer_map: Dict[Tuple[int, int], List[int]] = {}
    for pp_rank2, layer_indices2 in layer_mapping2.items():
        for pp_rank1, layer_indices1 in layer_mapping1.items():
            layer_map[(pp_rank1,
                       pp_rank2)] = sorted(list(set(layer_indices1).intersection(set(layer_indices2))))

    return layer_map
