from typing import *

import numpy as np
import torch

from realhf.api.core import model_api
from realhf.base import constants, logging

logger = logging.getLogger("ReaL parallel")

# FIXME: gradients of layer norm layers (replicated for tensor parallel) may be incorrect
# keys used to identify modules
EMBEDDING_KEYS = [".wte", ".wpe"]  # dim=0 no bias
COLUMN_LINEAR_KEYS = [
    ".attn.c_attn.q_attn",
    ".attn.c_attn.k_attn",
    ".attn.c_attn.v_attn",
    ".mlp.c_fc",
    ".mlp.gate_proj",
    ".mlp.up_proj",
]  # dim=0 + partition bias
ROW_LINEAR_KEYS = [
    ".attn.c_proj",
    ".mlp.down_proj",
    ".mlp.c_proj",
]  # dim=1 + no partition bias

if constants.use_te_impl():
    COLUMN_LINEAR_KEYS = [
        ".attn.c_attn.q_attn",
        ".attn.c_attn.k_attn",
        ".attn.c_attn.v_attn",
        ".mlp.c_fc",
        ".mlp.fc1_weight",
    ]  # dim=0 + partition bias
    ROW_LINEAR_KEYS = [".attn.c_proj", ".mlp.fc2_weight"]


def tensor_slice_partition_fn(
    tensor: torch.Tensor,
    mp_rank: Optional[int],
    mp_world_size: int,
    dim: Optional[int],
) -> Union[List[torch.Tensor], torch.Tensor]:
    """Partition a pytorch tensor for model parallelism."""
    if dim is None:
        splits = [tensor for _ in range(mp_world_size)]
    else:
        assert tensor.shape[dim] % mp_world_size == 0
        splits = torch.split(
            tensor, tensor.shape[dim] // mp_world_size, dim=dim
        )
    if mp_rank is None:
        return [s.contiguous() for s in splits]
    else:
        return splits[mp_rank].contiguous()


def intervals_partition_fn(
    shape: torch.Size,
    mp_rank: Optional[int],
    mp_world_size: int,
    dim: Optional[int],
) -> Union[List[torch.Tensor], torch.Tensor]:
    """Get the intervals of a MP-partitioned tensor in the flatten view.

    For example, if a tensor of shape (2, 4) is partitioned along the second dimension
    into 2 parts, then the intervals are [(0, 2), (2, 4)].
    """
    assert mp_rank is not None
    param_size = int(np.prod(shape))
    if dim is None:
        return np.array([(0, param_size)], dtype=np.int64)

    if dim < 0:
        dim = len(shape) + dim
    assert shape[dim] % mp_world_size == 0

    if len(shape) == 1:
        assert dim == 0
        partition_size = shape[0] // mp_world_size
        return np.array(
            [(partition_size * mp_rank, partition_size * (mp_rank + 1))],
            dtype=np.int64,
        )
    else:
        assert len(shape) == 2, shape
        if dim == 0:
            row_start = mp_rank * shape[0] // mp_world_size
            row_end = (mp_rank + 1) * shape[0] // mp_world_size
            return np.array(
                [(row_start * shape[1], row_end * shape[1])], dtype=np.int64
            )
        else:
            assert dim == 1
            col_start = mp_rank * shape[1] // mp_world_size
            col_end = (mp_rank + 1) * shape[1] // mp_world_size
            return np.arange(shape[0], dtype=np.int64)[:, None] * shape[
                1
            ] + np.array([(col_start, col_end)], dtype=np.int64)


def shape_partition_fn(
    shape: torch.Size,
    mp_rank: Optional[int],
    mp_world_size: int,
    dim: Optional[int],
):
    """Get the partitioned shape of a tensor for model parallelism."""
    if dim is None:
        splits = [shape for _ in range(mp_world_size)]
    else:
        if dim < 0:
            dim = len(shape) + dim
        assert shape[dim] % mp_world_size == 0
        splits = [
            (*shape[:dim], shape[dim] // mp_world_size, *shape[dim + 1 :])
            for _ in range(mp_world_size)
        ]
    if mp_rank is None:
        return [s for s in splits]
    else:
        return splits[mp_rank]


def mp_partition_key(
    key: str,
    tensor_or_shape: torch.Tensor | torch.Size,
    mp_rank: Optional[int],
    mp_size: Optional[int],
    config: model_api.ReaLModelConfig,
    partition_fn: Callable[
        [torch.Tensor, Optional[int], int, Optional[int]],
        Union[List[torch.Tensor], torch.Tensor],
    ] = tensor_slice_partition_fn,
) -> torch.Tensor:
    """Partition a parameter of ReaLModel with name `key` by the method `partition_fn`."""
    if any([ek in key for ek in EMBEDDING_KEYS]):
        assert "weight" in key
        return partition_fn(tensor_or_shape, mp_rank, mp_size, dim=0)
    elif key == f"{config.n_layers + 1}.weight":  # output head
        if (
            isinstance(tensor_or_shape, torch.Tensor)
            and tensor_or_shape.shape[0] == 1
        ) or (
            not isinstance(tensor_or_shape, torch.Tensor)
            and tensor_or_shape[0] == 1
        ):
            assert config.is_critic
            return partition_fn(tensor_or_shape, mp_rank, mp_size, dim=None)
        else:
            return partition_fn(tensor_or_shape, mp_rank, mp_size, dim=0)
    elif any([ck in key for ck in COLUMN_LINEAR_KEYS]):
        if (
            ("k_attn" in key) or ("v_attn" in key)
        ) and config.n_kv_heads % mp_size != 0:
            # logger.warning(f"Cannot split {config.n_kv_heads} kv heads evenly among "
            #                f"{mp_size} model parallel ranks, "
            #                f"use unsplitted linear for kv heads instead")
            return partition_fn(tensor_or_shape, mp_rank, mp_size, dim=None)
        # partition both weight and bias
        return partition_fn(tensor_or_shape, mp_rank, mp_size, dim=0)
    elif any([rk in key for rk in ROW_LINEAR_KEYS]):
        # only partition weight
        if "weight" in key:
            return partition_fn(tensor_or_shape, mp_rank, mp_size, dim=1)
        else:
            assert "bias" in key, key
            return partition_fn(tensor_or_shape, mp_rank, mp_size, dim=None)
    else:
        return partition_fn(tensor_or_shape, mp_rank, mp_size, dim=None)


def mp_partition_real_model_state_dict(
    state_dict: Dict[str, torch.Tensor],
    config: model_api.ReaLModelConfig,
    mp_size: int,
    mp_rank: Optional[int] = None,
) -> Union[Dict, List[Dict]]:
    if mp_size == 1:
        if mp_rank is None:
            return [state_dict]
        else:
            return state_dict

    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k] = mp_partition_key(k, v, mp_rank, mp_size, config)

    if mp_rank is None:
        return [
            {k: v[mp_rank] for k, v in new_state_dict.items()}
            for mp_rank in range(mp_size)
        ]
    else:
        return new_state_dict


def get_real_model_param_shape(
    k: str,
    config: model_api.ReaLModelConfig,
    mp_size: int,
    sequence_parallel: bool,
) -> Tuple:
    if "wte.weight" in k:
        assert config.vocab_size % mp_size == 0
        return (config.vocab_size // mp_size, config.hidden_dim)
    elif "wpe.weight" in k:
        assert config.n_positions % mp_size == 0
        return (config.n_positions // mp_size, config.hidden_dim)
    elif ".ln." in k or ".ln_f." in k:
        return (config.hidden_dim,)
    elif k == f"{config.n_layers + 1}.weight":  # output head
        if config.is_critic and sequence_parallel:
            return (1, config.hidden_dim)
        elif not config.is_critic and mp_size > 1:
            assert config.vocab_size % mp_size == 0
            return (config.vocab_size // mp_size, config.hidden_dim)
        else:
            return (
                config.vocab_size if not config.is_critic else 1,
                config.hidden_dim,
            )
    elif any([ck in k for ck in COLUMN_LINEAR_KEYS]):
        if "k_attn" in k or "v_attn" in k:
            if "weight" in k:
                if config.n_kv_heads % mp_size == 0:
                    return (
                        config.head_dim * config.n_kv_heads // mp_size,
                        config.hidden_dim,
                    )
                else:
                    return (
                        config.head_dim * config.n_kv_heads,
                        config.hidden_dim,
                    )
            else:
                assert "bias" in k
                if config.n_kv_heads % mp_size == 0:
                    return (config.head_dim * config.n_kv_heads // mp_size,)
                else:
                    return (config.head_dim * config.n_kv_heads,)
        if "mlp" in k:
            if "weight" in k:
                return (config.intermediate_dim // mp_size, config.hidden_dim)
            else:
                assert "bias" in k
                return (config.intermediate_dim // mp_size,)
        if "weight" in k:
            assert config.hidden_dim // config.head_dim % mp_size == 0
            return (config.hidden_dim // mp_size, config.hidden_dim)
        else:
            assert "bias" in k
            return (config.hidden_dim // mp_size,)
    elif any([rk in k for rk in ROW_LINEAR_KEYS]):
        if "mlp" in k and "weight" in k:
            return (config.hidden_dim, config.intermediate_dim // mp_size)
        elif "attn" in k and "weight" in k:
            return (config.hidden_dim, config.hidden_dim // mp_size)
        elif "bias" in k:
            return (config.hidden_dim,)
        else:
            raise NotImplementedError(f"unkown shape of key {k}.")
    else:
        raise NotImplementedError(f"unkown shape of key {k}.")


def mp_merge_key(
    k: str,
    tensors: List[torch.Tensor],
    config: model_api.ReaLModelConfig,
) -> torch.Tensor:
    if any([ek in k for ek in EMBEDDING_KEYS]) and "weight" in k:
        return torch.cat(tensors, dim=0)
    elif k == f"{config.n_layers + 1}.weight" and not config.is_critic:
        return torch.cat(tensors, dim=0)
    elif any([ck in k for ck in COLUMN_LINEAR_KEYS]):
        return torch.cat(tensors, dim=0)
    elif any([rk in k for rk in ROW_LINEAR_KEYS]) and "weight" in k:
        return torch.cat(tensors, dim=1)
    else:
        return tensors[0]


def mp_merge_real_model_state_dict(
    state_dicts: List[Dict[str, torch.Tensor]],
    config: model_api.ReaLModelConfig,
) -> Dict:
    mp_size = len(state_dicts)
    if mp_size == 1:
        return state_dicts[0]

    new_state_dict = {}
    for k in state_dicts[0].keys():
        new_state_dict[k] = mp_merge_key(
            k, [sd[k] for sd in state_dicts], config
        )

    return new_state_dict


def partition_pipeline_layers(
    config: model_api.ReaLModelConfig,
    num_stages: int,
    embed_param_counter: Callable[[model_api.ReaLModelConfig], int],
    transformer_block_param_counter: Callable[
        [model_api.ReaLModelConfig, int], int
    ],
    head_param_counter: Callable[[model_api.ReaLModelConfig], int],
    method: str = "parameters_balanced",
) -> Dict[int, Tuple[int, int]]:
    from deepspeed.runtime import utils as ds_utils

    from realhf.base.datapack import (
        partition_balanced as true_partition_balanced,
    )

    # Each stage gets a simple uniform number of layers.
    param_counts = (
        [embed_param_counter(config)]
        + [
            transformer_block_param_counter(config, i)
            for i in range(config.n_layers)
        ]
        + [head_param_counter(config)]
    )
    parts = None
    if method == "uniform":
        parts = ds_utils.partition_uniform(
            num_items=config.n_layers + 2, num_parts=num_stages
        )
    elif method == "parameters":
        parts = ds_utils.partition_balanced(
            weights=param_counts, num_parts=num_stages
        )
    elif method == "parameters_balanced":
        param_counts = np.array(param_counts)
        parts = true_partition_balanced(nums=param_counts, k=num_stages)
    else:
        raise NotImplementedError(
            f"Partitioning method {method} not implemented."
        )

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
    assert set(sum(layer_mapping1.values(), [])) == set(
        sum(layer_mapping2.values(), [])
    )
    assert all(isinstance(i, int) for i in layer_mapping1)
    assert all(isinstance(i, int) for i in layer_mapping2)

    layer_mapping1 = dict(sorted(layer_mapping1.items()))
    layer_mapping2 = dict(sorted(layer_mapping2.items()))

    layer_map: Dict[Tuple[int, int], List[int]] = {}
    for pp_rank2, layer_indices2 in layer_mapping2.items():
        for pp_rank1, layer_indices1 in layer_mapping1.items():
            layer_map[(pp_rank1, pp_rank2)] = sorted(
                list(set(layer_indices1).intersection(set(layer_indices2)))
            )

    return layer_map
