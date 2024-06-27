import dataclasses
import math
from typing import *

import numpy as np
import torch

from realhf.api.core import model_api
from realhf.api.core.config import ModelName
from realhf.base import constants, logging

try:
    import realhf._C.interval_op_cuda as interval_op_cuda
except ImportError:
    print(
        "interval_op_cuda not found. "
        "This should only appear on workers without CUDA supports."
    )

from .real_llm_base import (
    real_model_embed_param_count,
    real_model_embedding_param_keys,
    real_model_head_param_count,
    real_model_head_param_keys,
    real_model_tblock_param_count,
    real_model_tblock_param_keys,
)
from .real_llm_parallel import (
    get_real_model_param_shape,
    intervals_partition_fn,
    mp_partition_key,
    partition_pipeline_layers,
    shape_partition_fn,
)

logger = logging.getLogger("FlattenParam")

MAX_PYTORCH_N_INTERVALS = 1024
CUDA_INTERVAL_OP_CHUNK_SIZE = 2048
_FLAT_PARAM_INDICES_CACHE = {}


@dataclasses.dataclass
class ContiguousParamSpec:
    start_idx: int
    end_idx: int
    shape: torch.Size


def _is_integer_list_contiguous(l: List[int]) -> bool:
    return np.all(np.array(l) == np.arange(len(l)) + l[0])


def _are_intervals_contiguous(l: List[Tuple[int, int]]) -> bool:
    l = sorted(l, key=lambda x: x[0])
    res = True
    for i in range(len(l) - 1):
        res &= l[i][1] == l[i + 1][0]
    return res


def recursive_getattr(obj, attr_string):
    attrs = attr_string.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def slice_intervals(
    tensor: torch.Tensor,
    intervals: torch.IntTensor,
    intervals_cpu: List[Tuple[int, int]],
    max_interval_size: int,
    output_size: int,
) -> torch.Tensor:
    assert len(tensor.shape) == 1
    # NOTE: these following assertions will take ten years to finish.
    # assert len(intervals) == len(intervals_cpu)
    # assert len(set([x[0] for x in intervals_cpu])) == len(intervals_cpu)
    if len(intervals_cpu) == 1:
        return tensor[intervals_cpu[0][0] : intervals_cpu[0][1]]
    elif len(intervals_cpu) <= MAX_PYTORCH_N_INTERVALS:
        return torch.cat([tensor[start:end] for start, end in intervals_cpu])

    interval_sizes = intervals[:, 1] - intervals[:, 0]
    offsets = torch.nn.functional.pad(interval_sizes.cumsum(0)[:-1], (1, 0), value=0)
    assert tensor.dtype == torch.half
    return interval_op_cuda.slice_intervals_cuda_half(
        tensor,
        intervals,
        interval_sizes,
        offsets,
        max_interval_size,
        output_size,
    )


def set_intervals(
    src: torch.Tensor,
    dst: torch.Tensor,
    intervals: torch.IntTensor,
    intervals_cpu: List[Tuple[int, int]],
    max_interval_size: int,
):
    assert len(dst.shape) == len(src.shape) == 1
    if len(intervals_cpu) <= MAX_PYTORCH_N_INTERVALS:
        offset = 0
        for i, j in intervals_cpu:
            dst[i:j] = src[offset : offset + j - i]
            offset += j - i
        assert offset == src.shape[0]
        return
    interval_sizes = intervals[:, 1] - intervals[:, 0]
    offsets = torch.nn.functional.pad(interval_sizes.cumsum(0)[:-1], (1, 0), value=0)
    interval_op_cuda.set_intervals_cuda_half(
        src,
        dst,
        intervals,
        interval_sizes,
        offsets,
        max_interval_size,
    )
    return


def build_param_spec(
    layer_indices: List[int],
    config: model_api.ReaLModelConfig,
    dp_size: int,
    mp_size: int,
    pp_size: int,
    sequence_parallel: bool,
    head_param_point_to_embedding: bool,
    bucket_size: int = 40000000,
) -> Tuple[Dict[str, ContiguousParamSpec], int]:
    # TODO: omit parameters that do not require gradient?
    if len(layer_indices) == 0:
        return {}, 0

    disable_bucketing = 0 not in layer_indices

    sd_keys = []
    for layer_idx in layer_indices:
        if layer_idx == 0:
            sd_keys += real_model_embedding_param_keys(config)
        elif layer_idx == config.n_layers + 1:
            sd_keys += real_model_head_param_keys(config)
        else:
            sd_keys += real_model_tblock_param_keys(config, layer_idx - 1)

    # In the reverse order as backpropagation, consistent with Megatron.
    sd_keys = reversed(sd_keys)

    data_start_index = 0
    bucket_data_start_index = data_start_index
    bucket_params = set()

    def _requires_new_allreduce_bucket(k):
        if pp_size == 1:
            return False
        if config.is_critic:
            return False
        if not config.share_embeddings_and_output_weights:
            return False
        return k == f"{config.n_layers + 1}.weight" or k == "0.wte.weight"

    def _pad_to_multiple(x, m):
        return int(math.ceil(x / m)) * m

    def _create_fake_bucket(data_end_index) -> int:
        nonlocal bucket_data_start_index, bucket_params
        data_end_index = _pad_to_multiple(data_end_index, dp_size)
        # Update bucket metadata.
        bucket_data_start_index = data_end_index
        # Re-set bucket_params and increment bucket_id for next bucket.
        bucket_params = set()
        # Return the potentially padded data_end_index.
        return data_end_index

    param_spec = {}
    for k in sd_keys:
        if head_param_point_to_embedding and k == f"{config.n_layers + 1}.weight":
            continue

        shape = get_real_model_param_shape(k, config, mp_size, sequence_parallel)
        numel = int(np.prod(shape))
        data_end_index = data_start_index + numel

        if _requires_new_allreduce_bucket(k) and len(bucket_params) > 0:
            _create_fake_bucket(data_start_index)

        param_spec[k] = ContiguousParamSpec(
            data_start_index,
            data_end_index,
            shape,
        )
        bucket_params.add(k)
        if (
            not disable_bucketing
            and (data_end_index - bucket_data_start_index) >= bucket_size
        ) or _requires_new_allreduce_bucket(k):
            data_end_index = _create_fake_bucket(data_end_index)

        data_start_index = data_end_index

    if len(bucket_params) > 0:
        data_end_index = _create_fake_bucket(data_end_index)

    if head_param_point_to_embedding:
        param_spec[f"{config.n_layers + 1}.weight"] = param_spec["0.wte.weight"]
    return param_spec, data_end_index


def param_intervals_from_keys(
    model_name: ModelName,
    config: model_api.ReaLModelConfig,
    head_param_point_to_embedding: bool,
    param_spec: Dict[str, ContiguousParamSpec],
    mp_size: int,
    sequence_parallel: bool,
    sd_keys: List[str],
    portion_size: int,
    portion_rank: int,
) -> List[int]:
    # if portion_size == 1:
    #     if not head_param_point_to_embedding or f"{config.n_layers + 1}.weight" not in sd_keys:
    #         # In this case, all parameters have unique unoverlapped memory allocations.
    #         # If the portion size is also one, we can merge these contiguous intervals.
    #         start, end = None, None
    #         for k in sd_keys:
    #             if start is None or param_spec[k].start_idx < start:
    #                 start = param_spec[k].start_idx
    #             if end is None or param_spec[k].end_idx > end:
    #                 end = param_spec[k].end_idx
    #         return [(start, end)]
    #     else:
    #         # The embeddings and output weights are shared.
    #         intervals = [(param_spec[f"{config.n_layers + 1}.weight"].start_idx, param_spec[f"{config.n_layers + 1}.weight"].end_idx)]
    #         start, end = None, None
    #         for k in sd_keys:
    #             if k == f"{config.n_layers + 1}.weight":
    #                 continue
    #             if start is None or param_spec[k].start_idx < start:
    #                 start = param_spec[k].start_idx
    #             if end is None or param_spec[k].end_idx > end:
    #                 end = param_spec[k].end_idx
    #         intervals += [(start, end)]
    #         intervals = sorted(intervals, key=lambda x: x[0])
    #         # Merge these two intervals if possible.
    #         if intervals[0][1] == intervals[1][0]:
    #             return [(intervals[0][0], intervals[1][1])]
    #         else:
    #             return intervals

    intervals = []
    for k in sd_keys:
        if head_param_point_to_embedding and k == f"{config.n_layers + 1}.weight":
            continue
        if (
            model_name,
            k.split(".", 1)[1],
            mp_size,
            portion_rank,
            portion_size,
        ) not in _FLAT_PARAM_INDICES_CACHE:
            zero_start_intervals = mp_partition_key(
                k,
                get_real_model_param_shape(k, config, mp_size, sequence_parallel),
                portion_rank,
                portion_size,
                config,
                partition_fn=intervals_partition_fn,
            )
            _FLAT_PARAM_INDICES_CACHE[
                (
                    model_name,
                    k.split(".", 1)[1],
                    mp_size,
                    portion_rank,
                    portion_size,
                )
            ] = zero_start_intervals
        else:
            zero_start_intervals = _FLAT_PARAM_INDICES_CACHE[
                (
                    model_name,
                    k.split(".", 1)[1],
                    mp_size,
                    portion_rank,
                    portion_size,
                )
            ]
        intervals += (zero_start_intervals + param_spec[k].start_idx).tolist()
    # assert len(set([x[0] for x in intervals])) == len(intervals)
    intervals = sorted(intervals, key=lambda x: x[0])
    return intervals


def map_param_to_contigous_memory(
    layers: torch.nn.ModuleList,
    config: model_api.ReaLModelConfig,
    head_param_point_to_embedding: bool,
    param_spec: Dict[str, ContiguousParamSpec],
    contiguous_param: torch.Tensor,
    layer_idx_offset: int,
    allocate_only: bool,
):
    for local_layer_idx, l in enumerate(layers):
        layer_idx = local_layer_idx + layer_idx_offset
        for k, v in l.named_parameters():
            spec = param_spec[f"{layer_idx}.{k}"]
            old_param_data = v.data
            target = contiguous_param[spec.start_idx : spec.end_idx].view(spec.shape)
            if not allocate_only:
                target.copy_(old_param_data)
            else:
                if not (
                    head_param_point_to_embedding and layer_idx == config.n_layers + 1
                ):
                    assert old_param_data.shape == torch.Size([0]), (
                        old_param_data.shape,
                        spec.shape,
                        f"{layer_idx}.{k}",
                    )
            recursive_getattr(l, k).data = target
