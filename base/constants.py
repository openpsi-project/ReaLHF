# log format constants
import getpass

from base.cluster import spec as cluster_spec
from base.topology import PipelineParallelGrid

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

_grid: PipelineParallelGrid = None

# used only in scripts and tests
_fake_mp_world_size = None
_fake_mp_rank = None


def set_grid(grid):
    global _grid
    _grid = grid


def grid() -> PipelineParallelGrid:
    if _grid == None:
        raise RuntimeError("Global constant `grid` is accessed before set.")
    return _grid


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
