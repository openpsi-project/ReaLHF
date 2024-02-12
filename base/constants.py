# log format constants
from typing import Any, Dict
import copy
import getpass

from base.cluster import spec as cluster_spec

# constants in experiment instance scope
MODEL_SAVE_ROOT = f"{cluster_spec.fileroot}/checkpoints/{getpass.getuser()}"
LOG_ROOT = f"{cluster_spec.fileroot}/logs/{getpass.getuser()}"

SLURM_LOCK_FILE_NAME = f"{cluster_spec.fileroot}/logs/slurm_scheduler.lock"

PYTORCH_KERNEL_CACHE_PATH = f"{cluster_spec.fileroot}/.cache/{getpass.getuser()}/torch/kernels"
TRITON_CACHE_PATH = f"{cluster_spec.fileroot}/.cache/{getpass.getuser()}/triton"
DATASET_CACHE_PATH = f"{cluster_spec.fileroot}/.cache/{getpass.getuser()}/datasets"
TORCH_EXTENSIONS_DIR = f"{cluster_spec.fileroot}/.cache/{getpass.getuser()}/torch/extensions"

QUICKSTART_EXPR_CACHE_PATH = f"{cluster_spec.fileroot}/.cache/{getpass.getuser()}/quickstart.pkl"

_experiment_name = None
_trial_name = None

# constants in worker/process scope

_model_name: str = None
_grids: Dict[str, Any] = {}  # PipelineParallelGrid, not type hint here to avoid circular import
_pgroups: Dict[str, Any] = {}  # torch.distributed.ProcessGroup, not type hint here to avoid importing torch

_global_memory_buffer = None  # type GlobalMemoryBuffer, not type hint here to avoid circular import

# used only in scripts and tests
_fake_mp_world_size = None
_fake_mp_rank = None

# TODO: As in Megatron, we can set NCCL group options. Is it necessary?


def set_model_name(model_name: str):
    global _model_name
    assert _model_name is None, "Cannot set model_name twice."
    from impl.model.parallelism.model_parallel.utils import GlobalMemoryBuffer
    set_global_memory_buffer(GlobalMemoryBuffer())
    _model_name = copy.deepcopy(model_name)


def set_grid(model_name: str, grid):
    global _grids
    _grids[model_name] = grid


def grid():
    if _model_name is None:
        raise RuntimeError("Global constant `model_name` is accessed before set.")
    if _grids.get(_model_name, None) is None:
        raise RuntimeError(f"Grid for model {_model_name} is not set.")
    return _grids[_model_name]

def grid_of_model(model_name: str):
    if _grids.get(model_name, None) is None:
        raise RuntimeError(f"Grid for model {model_name} is not set.")
    return _grids[model_name]

def set_parallelism_group(model_name: str, pgroup):
    global _pgroups
    _pgroups[model_name] = pgroup


def parallelism_group():
    """Returns the 3D parallelism group of a specific model."""
    if _model_name is None:
        raise RuntimeError("Global constant `model_name` is accessed before set.")
    if _pgroups.get(_model_name, None) is None:
        raise RuntimeError(f"Parallelism group for model {_model_name} is not set.")
    return _pgroups[_model_name]


def parallelism_group_size() -> int:
    """The 3D parallelism group size of a specific model, normally dp_size * pp_size * mp_size."""
    import torch.distributed as dist

    return dist.get_world_size(group=parallelism_group())


def parallelism_rank() -> int:
    """Return the rank of a specific model in its 3D parallelism group."""
    import torch.distributed as dist

    return dist.get_rank(group=parallelism_group())


def process_group_offset() -> int:
    """Return the offset of the model's parallelism group w.r.t. the global process group (all models + master)."""
    import torch.distributed as dist

    return dist.get_global_rank(group=parallelism_group(), group_rank=0)


def pipe_parallel_rank() -> int:
    return grid().get_pipe_parallel_rank()


def pipe_parallel_world_size() -> int:
    return grid().get_pipe_parallel_world_size()


def pipe_parallel_group():
    return grid().get_pipe_parallel_group()


def model_parallel_rank() -> int:
    try:
        return grid().get_tensor_model_parallel_rank()
    except RuntimeError as e:  # used only in scripts and tests
        if _fake_mp_rank is not None:
            return _fake_mp_rank
        else:
            raise e


def model_parallel_world_size() -> int:
    try:
        return grid().get_tensor_model_parallel_world_size()
    except RuntimeError as e:  # used only in scripts and tests
        if _fake_mp_world_size is not None:
            return _fake_mp_world_size
        else:
            raise e


def model_parallel_group():
    return grid().get_tensor_model_parallel_group()


def data_parallel_rank() -> int:
    return grid().get_data_parallel_rank()


def data_parallel_world_size() -> int:
    return grid().get_data_parallel_world_size()


def data_parallel_group():
    return grid().get_data_parallel_group()


def set_fake_mp_world_size(world_size):
    # used only in scripts and tests
    global _fake_mp_world_size
    _fake_mp_world_size = world_size


def set_fake_mp_rank(rank):
    # used only in scripts and tests
    global _fake_mp_rank
    _fake_mp_rank = rank


def set_global_memory_buffer(buffer):
    global _global_memory_buffer
    assert _global_memory_buffer is None, "cannot set global memory buffer twice"
    _global_memory_buffer = buffer


def get_global_memory_buffer():
    global _global_memory_buffer
    assert _global_memory_buffer is not None, "global memory buffer is not set"
    return _global_memory_buffer
