import contextlib
import dataclasses
import math
import os
import subprocess
from typing import *

import numpy as np
import torch
import torch.utils.cpp_extension as torch_cpp_ext
from packaging.version import Version, parse

import realhf
from realhf.api.core import model_api
from realhf.api.core.config import ModelName
from realhf.base import logging

from .real_llm_base import (
    real_model_embedding_param_keys,
    real_model_head_param_keys,
    real_model_tblock_param_keys,
)
from .real_llm_parallel import (
    get_real_model_param_shape,
    intervals_partition_fn,
    mp_partition_key,
    shape_partition_fn,
)

try:
    from realhf._C.interval_op import merge_intervals
except ImportError:
    merge_intervals = None
logger = logging.getLogger("FlattenParam")

_FLAT_PARAM_INDICES_CACHE = {}
_SET_INTERVAL_FN = {}
_GET_INTERVAL_FN = {}


def get_nvcc_cuda_version(cuda_dir: str) -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    nvcc_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def _torch_dtype_to_cpp_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float"
    if dtype == torch.float64:
        return "double"
    if dtype == torch.float16:
        return "at::Half"
    if dtype == torch.bfloat16:
        return "at::BFloat16"
    raise ValueError(f"Unsupported dtype: {dtype}")


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


def _slice_interval_dispatch(N: int) -> Callable:
    global _GET_INTERVAL_FN
    if N in _GET_INTERVAL_FN:
        return _GET_INTERVAL_FN[N]
    src_path = os.path.join(
        realhf.__path__[0], os.pardir, "csrc", "interval_op", "slice_interval.cpp"
    )
    with open(src_path, "r") as f:
        src = f.read()
    # NOTE: We embed the C++ binding code in python docstring because we want to dynamically compile
    # for a specific template value N. This ensures the optimal performance.
    src += f"""
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
  m.def("slice_intervals", &slice_intervals<{N}>, "Slice intervals of a 1D tensor");
}}
"""
    ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
    logger.info(f">>> Building slice interval extension with N={N}. <<<")
    slice_fn = torch_cpp_ext.load_inline(
        f"slice_intervals_{N}",
        cpp_sources=[src],
        extra_cflags=[
            "-O3",
            "-std=c++17",
            f"-D_GLIBCXX_USE_CXX11_ABI={ABI}",
            "-fopenmp",
        ],
        extra_ldflags=["-fopenmp"],
        verbose=False,
    ).slice_intervals
    _GET_INTERVAL_FN[N] = slice_fn
    return slice_fn


def _set_interval_dispatch(N: int, torch_dtype: torch.dtype) -> Callable:
    global _SET_INTERVAL_FN
    dtype = _torch_dtype_to_cpp_dtype(torch_dtype)
    if (N, dtype) in _SET_INTERVAL_FN:
        return _SET_INTERVAL_FN[(N, dtype)]

    src_path = os.path.join(
        realhf.__path__[0], os.pardir, "csrc", "interval_op", "set_interval.cu"
    )
    with open(src_path, "r") as f:
        src = f.read()
    # NOTE: We embed the C++ binding code in python docstring because we want to dynamically compile
    # for a specific template value N. This ensures the optimal performance.
    src += f"""
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
  m.def("set_intervals", &set_intervals<{N}, {dtype}>, "Set intervals of a 1D tensor");
}}
"""
    ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
    NVCC_FLAGS = ["-O3", "-std=c++17", f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
    nvcc_cuda_version = get_nvcc_cuda_version(torch_cpp_ext.CUDA_HOME)
    # Use NVCC threads to parallelize the build.
    if nvcc_cuda_version >= Version("11.2"):
        nvcc_threads = int(os.getenv("NVCC_THREADS", 8))
        num_threads = min(os.cpu_count(), nvcc_threads)
        NVCC_FLAGS += ["--threads", str(num_threads)]

    if nvcc_cuda_version >= Version("11.8"):
        NVCC_FLAGS += ["-DENABLE_FP8_E5M2"]
    NVCC_FLAGS += torch_cpp_ext.COMMON_NVCC_FLAGS
    REMOVE_NVCC_FLAGS = [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    for flag in REMOVE_NVCC_FLAGS:
        with contextlib.suppress(ValueError):
            torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
    logger.info(
        f">>> Building set interval extension with N={N} and dtype={dtype}. <<<"
    )
    set_fn = torch_cpp_ext.load_inline(
        f"set_intervals_{N}_{str(torch_dtype).split('.')[1]}",
        cpp_sources=[],
        cuda_sources=[src],
        extra_cuda_cflags=NVCC_FLAGS,
        with_cuda=True,
        verbose=False,
    ).set_intervals
    _SET_INTERVAL_FN[(N, dtype)] = set_fn
    return set_fn


def _slice_intervals_py(src: torch.Tensor, intervals: List[Tuple[int, int]]):
    # Drop-in replacement for the C++ implementation.
    assert len(src.shape) == 1
    assert all([x[0] >= 0 for x in intervals])
    assert all([x[1] <= src.shape[0] for x in intervals])
    N = len(intervals)
    slices = []
    for i, j in intervals:
        slices.append(src[i:j])
    return torch.cat(slices, dim=0)


def _set_intervals_py(
    src: torch.Tensor,
    dst: torch.Tensor,
    intervals: List[Tuple[int, int]],
):
    # Drop-in replacement for the C++ implementation.
    assert len(dst.shape) == len(src.shape) == 1
    offset = 0
    for i, j in intervals:
        assert i >= 0
        assert j <= dst.shape[0], (j, dst.shape[0])
        dst[i:j] = src[offset : offset + j - i]
        offset += j - i
    assert offset == src.shape[0]


def slice_intervals(
    src: torch.Tensor,
    intervals: List[Tuple[int, int]],
):
    N = len(intervals)
    if N < 1000:
        # NOTE: The C++ implementation has no negative effect, but for small Ns
        # using the python implementation can omit compliation.
        return _slice_intervals_py(src, intervals)
    slice_fn = _slice_interval_dispatch(N)
    return slice_fn(src, intervals)


def set_intervals(
    src: torch.Tensor,
    dst: torch.Tensor,
    intervals: List[Tuple[int, int]],
):
    if len(intervals) < 2048 or not (src.is_cuda and dst.is_cuda):
        # NOTE: The CUDA implementation will launch a thread for each interval,
        # which has a negative effect when the number of intervals is small.
        return _set_intervals_py(src, dst, intervals)
    set_fn = _set_interval_dispatch(len(intervals), src.dtype)
    return set_fn(src, dst, intervals)


def param_size_from_keys(
    config: model_api.ReaLModelConfig,
    src_mp_size: int,
    sequence_parallel: bool,
    sd_keys: List[str],
    src2dst_tp_size: int,
    src2dst_tp_rank: int,
    head_param_point_to_embedding: bool,
) -> Tuple[List[int], int]:
    param_size = 0
    for k in sd_keys:
        if (
            head_param_point_to_embedding
            and k == f"{config.n_layers + 1}.weight"
            and "0.wte.weight" in sd_keys
        ):
            continue
        new_shape = mp_partition_key(
            k,
            get_real_model_param_shape(k, config, src_mp_size, sequence_parallel),
            src2dst_tp_rank,
            src2dst_tp_size,
            config,
            partition_fn=shape_partition_fn,
        )
        param_size += int(np.prod(new_shape))
    return param_size


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
    for layer_idx in sorted(layer_indices):
        if layer_idx == 0:
            sd_keys += real_model_embedding_param_keys(config)
        elif layer_idx == config.n_layers + 1:
            sd_keys += real_model_head_param_keys(config)
        else:
            sd_keys += real_model_tblock_param_keys(config, layer_idx - 1)

    # In the reverse order as backpropagation, consistent with Megatron.
    sd_keys = list(reversed(sd_keys))

    data_start_index = 0
    bucket_data_start_index = data_start_index
    bucket_params = set()

    def _requires_new_allreduce_bucket(k):
        if pp_size == 1:
            return False
        if config.is_critic:
            return False
        if not config.tied_embedding:
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

    if head_param_point_to_embedding and f"{config.n_layers + 1}.weight" in sd_keys:
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
    param_size = param_size_from_keys(
        config=config,
        src_mp_size=mp_size,
        sequence_parallel=sequence_parallel,
        sd_keys=sd_keys,
        src2dst_tp_size=portion_size,
        src2dst_tp_rank=portion_rank,
        head_param_point_to_embedding=head_param_point_to_embedding,
    )

    interval_size = 0
    intervals = []
    for k in sd_keys:
        if (
            head_param_point_to_embedding
            and k == f"{config.n_layers + 1}.weight"
            and "0.wte.weight" in sd_keys
        ):
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
        interval_size += sum(zero_start_intervals[:, 1] - zero_start_intervals[:, 0])
    # assert len(set([x[0] for x in intervals])) == len(intervals)
    assert interval_size == param_size, (interval_size, param_size)
    intervals = sorted(intervals, key=lambda x: x[0])
    if merge_intervals is not None:
        intervals = merge_intervals(intervals)
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
