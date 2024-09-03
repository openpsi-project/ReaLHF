# log format constants
import contextlib
import copy
import datetime
import getpass
import os
import pathlib
from collections import defaultdict
from typing import *

import numpy as np

import realhf.base.logging as logging
from realhf.base.cluster import spec as cluster_spec

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from realhf.api.core.config import ModelName
    from realhf.api.core.system_api import ModelShardID
    from realhf.base.topology import ParallelGrid, PipeModelDataParallelTopology


class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations.

    Caller should ensure that buffers of the same name are not used
    concurrently.
    """

    def __init__(self):
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name, force_zero: bool = False):
        import torch

        required_len = int(np.prod(tensor_shape))
        if self.buffer.get((name, dtype), None) is None:
            self.buffer[(name, dtype)] = torch.empty(
                required_len,
                dtype=dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
        elif self.buffer[(name, dtype)].numel() < required_len:
            self.buffer[(name, dtype)] = torch.nn.functional.pad(
                self.buffer[(name, dtype)],
                (0, required_len - self.buffer[(name, dtype)].numel()),
                value=0,
            )
        res = self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)
        if force_zero:
            res.zero_()
        return res


# 30 minutes. Transferring super-large batches via NCCL bcast
# for the first time may consumer over 600 secs, which is the
# pytorch's default. Increase this value to 30 minutes.
NCCL_DEFAULT_TIMEOUT = datetime.timedelta(seconds=1800)

# constants in experiment instance scope
MODEL_SAVE_ROOT = f"{cluster_spec.fileroot}/checkpoints/{getpass.getuser()}"
LOG_ROOT = f"{cluster_spec.fileroot}/logs/{getpass.getuser()}"
RECOVER_ROOT = f"{cluster_spec.fileroot}/recover/{getpass.getuser()}"
SLURM_LOCK_FILE_NAME = f"{cluster_spec.fileroot}/logs/slurm_scheduler.lock"
PYTORCH_KERNEL_CACHE_PATH = (
    f"{cluster_spec.fileroot}/.cache/{getpass.getuser()}/torch/kernels"
)
TRITON_CACHE_PATH = f"{cluster_spec.fileroot}/.cache/{getpass.getuser()}/triton"
DATASET_CACHE_PATH = f"{cluster_spec.fileroot}/.cache/{getpass.getuser()}/datasets"
PROFILER_CACHE_PATH = f"{cluster_spec.fileroot}/.cache/{getpass.getuser()}/profiler"
TORCH_EXTENSIONS_DIR = (
    f"{cluster_spec.fileroot}/.cache/{getpass.getuser()}/torch/extensions"
)
QUICKSTART_EXPR_CACHE_PATH = f"{cluster_spec.fileroot}/.cache/{getpass.getuser()}/"
BASE_ENVIRONS = {
    "PYTHONPATH": "/realhf",
    "REAL_IS_REMOTE": "1",
    # "NCCL_P2P_DISABLE": "1",
    # "NCCL_IB_DISABLE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "PYTORCH_KERNEL_CACHE_PATH": PYTORCH_KERNEL_CACHE_PATH,
    "TRITON_CACHE_DIR": TRITON_CACHE_PATH,
    "TOKENIZERS_PARALLELISM": "true",
    "TORCH_EXTENSIONS_DIR": TORCH_EXTENSIONS_DIR,
    # "NCCL_DEBUG": "INFO",
    # "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
    # "NCCL_SOCKET_IFNAME": "ibp71s0",
    # "GLOO_SOCKET_IFNAME": "ibp71s0",
    # "TORCH_USE_CUDA_DSA": "1",
    # "NCCL_IGNORE_DISABLED_P2P": "1",
    # "CUDA_LAUNCH_BLOCKING": "1",  # NOTE: CUDAGraph Capturing will not work if CUDA_LAUNCH_BLOCKING is set to 1.
    # "NCCL_COMM_BLOCKING": "1",  # NOTE: CUDAGraph Capturing will not work if NCCL_COMM_BLOCKING is set to 1.
    # "NCCL_BLOCKING_WAIT": "1",  # NOTE: CUDAGraph Capturing will not work if NCCL_BLOCKING_WAIT is set to 1.
    # "TORCH_SHOW_CPP_STACKTRACES": "1",
    "RAY_DEDUP_LOGS": "0",  # disable ray log deduplication
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "PYTHONUSERBASE": "/nonsense",  # a random PYTHONUSERBASE to avoid local user site-packages interference
    "OMP_NUM_THREADS": str(min(os.cpu_count(), 32)),
    # torch.distributed.all_reduce does not free the input tensor until
    # the synchronization point. This causes the memory usage to grow
    # as the number of all_reduce calls increases. This env var disables
    # this behavior.
    # Related issue:
    # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
    # Whether to enable time mark to plot timelines.
    "REAL_CUDA_TMARK": os.getenv("REAL_CUDA_TMARK", "0"),
    "REAL_DUMP_TRACE": os.getenv("REAL_DUMP_TRACE", "0"),
    "REAL_DUMP_MEMORY": os.getenv("REAL_DUMP_MEMORY", "0"),
}


# make directories if does not exist
os.makedirs(MODEL_SAVE_ROOT, exist_ok=True)
os.makedirs(LOG_ROOT, exist_ok=True)
os.makedirs(RECOVER_ROOT, exist_ok=True)
os.makedirs(PYTORCH_KERNEL_CACHE_PATH, exist_ok=True)
os.makedirs(TRITON_CACHE_PATH, exist_ok=True)
os.makedirs(DATASET_CACHE_PATH, exist_ok=True)
os.makedirs(PROFILER_CACHE_PATH, exist_ok=True)
os.makedirs(TORCH_EXTENSIONS_DIR, exist_ok=True)
os.makedirs(QUICKSTART_EXPR_CACHE_PATH, exist_ok=True)

# _model_name will be changed in the model_scope context manager
_model_name: "ModelName" = None

# constants in worker/process scope
_experiment_name = None
_trial_name = None

_grids: Dict["ModelName", "ParallelGrid"] = {}
_pgroups: Dict["ModelName", Any] = (
    {}
)  # torch.distributed.ProcessGroup, not type hint here to avoid importing torch
_pgroup_ranks: Dict["ModelName", List[int]] = {}
_self_group = None
_rank_mapping: Dict["ModelName", Dict["ModelShardID", int]] = {}
_global_memory_buffer: GlobalMemoryBuffer = GlobalMemoryBuffer()

# used only in scripts and tests
_fake_mp_world_size = None
_fake_mp_rank = None

# GLOBAL_STATS_TRACKER is used to track and log training stats that cannot be gracefully obtained via model outputs
# in interface implementations, e.g. load balancing loss in each MoE layer.
GLOBAL_STATS_TRACKER = defaultdict(dict)
GLOBAL_STATS_TRACKER_LOG_HOOKS = defaultdict(dict)

# TODO: As in Megatron, we can set NCCL group options. Is it necessary?


def reset_run():
    global _model_name, _grids, _pgroups, _pgroup_ranks, _self_group, _rank_mapping, _global_memory_buffer, _fake_mp_world_size, _fake_mp_rank, GLOBAL_STATS_TRACKER, GLOBAL_STATS_TRACKER_LOG_HOOKS
    _model_name = None
    _grids = {}
    _pgroups = {}
    _pgroup_ranks = {}
    _self_group = None
    _rank_mapping = {}
    _global_memory_buffer = GlobalMemoryBuffer()
    _fake_mp_world_size = None
    _fake_mp_rank = None
    GLOBAL_STATS_TRACKER = defaultdict(dict)
    GLOBAL_STATS_TRACKER_LOG_HOOKS = defaultdict(dict)


@contextlib.contextmanager
def model_scope(model_name: "ModelName"):
    global _model_name
    assert _model_name is None
    _model_name = model_name
    yield
    assert _model_name == model_name
    _model_name = None


@contextlib.contextmanager
def model_scope_disabled():
    global _model_name
    assert _model_name is not None
    t, _model_name = _model_name, None
    yield
    _model_name = t


################# setter functions #################
def set_experiment_trial_names(expr_name: str, trial_name: str):
    global _experiment_name, _trial_name
    if _experiment_name is not None or _trial_name is not None:
        raise RuntimeError("Experiment and trial names are already set.")
    _experiment_name = expr_name
    _trial_name = trial_name


def set_grid(model_name: "ModelName", grid: "ParallelGrid"):
    global _grids
    if model_name in _grids:
        raise RuntimeError(f"Grid for model {model_name} is already set.")
    _grids[model_name] = grid


def set_parallelism_group(model_name: "ModelName", pgroup, ranks):
    global _pgroups
    if model_name in _pgroups:
        raise RuntimeError(f"Parallelism group for model {model_name} is already set.")
    _pgroups[model_name] = pgroup
    _pgroup_ranks[model_name] = ranks


def set_self_group(pgroup):
    global _self_group
    if _self_group is not None:
        raise RuntimeError("Self group is already set.")
    _self_group = pgroup


def set_rank_mapping(
    model_name: "ModelName",
    topo: "PipeModelDataParallelTopology",
    msid2mwid: Optional[Dict["ModelShardID", int]] = None,
):
    global _rank_mapping
    if model_name in _rank_mapping:
        raise RuntimeError(f"Rank mapping for model {model_name} is already set.")
    if msid2mwid is None:
        _rank_mapping[model_name] = {i: i for i in range(topo.world_size())}
    else:
        msid2mwid = {k: v for k, v in msid2mwid.items() if k.model_name == model_name}
        _rank_mapping[model_name] = {
            topo.get_rank(data=s.dp_rank, model=s.mp_rank, pipe=s.pp_rank): mw_id
            for s, mw_id in msid2mwid.items()
        }


################# attribute functions #################
def use_te_impl() -> bool:
    try:
        import transformer_engine.pytorch as te

        TE_ENABLED = True
    except ImportError:
        TE_ENABLED = False
    return TE_ENABLED and os.getenv("REAL_LLM_USE_TE") == "1"


def sequence_parallel() -> bool:
    return grid().topology().sequence_parallel


def gradient_accumulation_fusion() -> bool:
    _grad_accum_fusion_available = True
    try:
        import fused_weight_gradient_mlp_cuda
    except ImportError:
        _grad_accum_fusion_available = False
    return (
        _grad_accum_fusion_available and grid().topology().gradient_accumulation_fusion
    )


def max_prompt_len() -> int:
    return grid().topology().max_prompt_len


def gradient_checkpointing() -> bool:
    return grid().topology().gradient_checkpointing


def has_model_name(name: str) -> bool:
    return name in _grids and _grids[name].global_rank != -1


def self_group():
    global _self_group
    assert _self_group is not None
    return _self_group


def model_name():
    if _model_name == None:
        raise RuntimeError(
            "Global constant `model_name` should be accessed in the `model_scope` context."
        )
    return _model_name


def experiment_name():
    if _experiment_name == None:
        raise RuntimeError("Global constant `experiment_name` is accessed before set.")
    return _experiment_name


def trial_name():
    if _trial_name == None:
        raise RuntimeError("Global constant `trial_name` is accessed before set.")
    return _trial_name


def grid() -> "ParallelGrid":
    if _model_name is None:
        raise RuntimeError("Global constant `model_name` is accessed before set.")
    if _grids.get(_model_name, None) is None:
        raise RuntimeError(f"Grid for model {_model_name} is not set.")
    return _grids[_model_name]


def grid_of_model(model_name: str) -> "ParallelGrid":
    if _grids.get(model_name, None) is None:
        raise RuntimeError(f"Grid for model {model_name} is not set.")
    return _grids[model_name]


def parallelism_group():
    """Returns the 3D parallelism group of a specific model."""
    if _model_name is None:
        raise RuntimeError("Global constant `model_name` is accessed before set.")
    if _pgroups.get(_model_name, None) is None:
        raise RuntimeError(f"Parallelism group for model {_model_name} is not set.")
    return _pgroups[_model_name]


def parallelism_group_ranks():
    if _model_name is None:
        raise RuntimeError("Global constant `model_name` is accessed before set.")
    if _pgroup_ranks.get(_model_name, None) is None:
        raise RuntimeError(
            f"Parallelism group ranks for model {_model_name} is not set."
        )
    return _pgroup_ranks[_model_name]


def parallelism_group_size() -> int:
    """The 3D parallelism group size of a specific model, normally dp_size *
    pp_size * mp_size."""
    import torch.distributed as dist

    return dist.get_world_size(group=parallelism_group())


def parallelism_rank() -> int:
    """Return the rank of a specific model in its 3D parallelism group."""
    import torch.distributed as dist

    return dist.get_rank(group=parallelism_group())


def to_global_pg_rank(local_rank: int) -> int:
    global _rank_mapping
    if _rank_mapping is None or model_name() not in _rank_mapping:
        raise RuntimeError("Rank mapping is not set.")
    return _rank_mapping[model_name()][local_rank]


def rank_mapping_of_model(model_name: str) -> Dict["ModelShardID", int]:
    global _rank_mapping
    if _rank_mapping is None or _rank_mapping.get(model_name, None) is None:
        raise RuntimeError(f"Rank mapping for model {model_name} is not set.")
    return _rank_mapping[model_name]


def pipe_parallel_rank() -> int:
    return grid().get_pipe_parallel_rank()


def pipe_parallel_world_size() -> int:
    return grid().get_pipe_parallel_world_size()


def pipe_parallel_group():
    return grid().get_pipe_parallel_group()


def is_last_pipe_stage():
    return pipe_parallel_rank() == pipe_parallel_world_size() - 1


def is_first_pipe_stage():
    return pipe_parallel_rank() == 0


def next_pipe_stage():
    return (pipe_parallel_rank() + 1) % pipe_parallel_world_size()


def prev_pipe_stage():
    return (
        pipe_parallel_world_size() + pipe_parallel_rank() - 1
    ) % pipe_parallel_world_size()


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


def model_parallel_cpu_group():
    return grid().get_tensor_model_parallel_cpu_group()


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


def set_fake_grid(model_name, rank, topo):
    # used only in scripts and tests
    from realhf.base.topology import FakeGrid

    global _grids
    _grids[model_name] = FakeGrid(rank=rank, topo=topo)


def get_global_memory_buffer():
    global _global_memory_buffer
    assert _global_memory_buffer is not None, "global memory buffer is not set"
    return _global_memory_buffer


def clear_global_memory_buffer():
    global _global_memory_buffer
    _global_memory_buffer = GlobalMemoryBuffer()


def get_repo_path() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent.parent


def get_env_vars(**kwargs):
    return {
        **kwargs,
        "REAL_PACKAGE_PATH": str(get_repo_path()),
        **BASE_ENVIRONS,
    }


################# logging related #################


def save_to_global_stats_tracker(
    key: str, value: Any, hook: Optional[Callable] = None, **hook_kwargs
):
    """Save kv-pair to global stats tracker for current model.

    :param key: Key
    :type key: str
    :param value: Value
    :type value: Any
    :param hook: Hook function to be called before logging the stats in `log_global_stats_tracker`.
        For example, this hook can be used to gather and average stats across parallel ranks.
    :type hook: Optional[Callable]
    :param hook_kwargs: Keyword arguments to be passed to the hook function.
    """
    if _model_name is None:
        raise RuntimeError("Global constant `model_name` is accessed before set.")
    GLOBAL_STATS_TRACKER[_model_name][key] = value
    if hook is not None:
        GLOBAL_STATS_TRACKER_LOG_HOOKS[_model_name][key] = (hook, hook_kwargs)


def get_from_global_stats_tracker(key: str):
    if _model_name is None:
        raise RuntimeError("Global constant `model_name` is accessed before set.")
    return GLOBAL_STATS_TRACKER[_model_name].get(key, None)


def clear_global_stats_tracker():
    if _model_name is None:
        raise RuntimeError("Global constant `model_name` is accessed before set.")
    global GLOBAL_STATS_TRACKER
    GLOBAL_STATS_TRACKER[_model_name] = dict()


def log_global_stats_tracker(
    return_dict: bool = True, clear_stats_after_logging: bool = True
):
    """Log the global stats tracker and optionally return the stats as a
    dictionary. This method is expected to be called in interface
    implementations.

    :param return_dict: Whether to return the stats as a dictionary.
    :type return_dict: bool
    :param clear_stats_after_logging: Whether to clear the stats after
        logging.
    :type clear_stats_after_logging: bool
    """
    if _model_name is None:
        raise RuntimeError("Global constant `model_name` is accessed before set.")
    stats = GLOBAL_STATS_TRACKER[_model_name]
    hooks = GLOBAL_STATS_TRACKER_LOG_HOOKS[_model_name]
    for key in stats.keys():
        hook, hook_kwargs = hooks.get(key, None)
        if hook is not None:
            hook(**hook_kwargs)

    res = {}
    if not return_dict:
        logger.info(f"Logging global stats tracker:")
    for key, value in stats.items():
        res[key] = value
        if not return_dict:
            logger.info(f"{key}: {value}")

    if clear_stats_after_logging:
        clear_global_stats_tracker()

    if return_dict:
        return res
