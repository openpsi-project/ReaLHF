from collections import defaultdict
from statistics import mean
from typing import Callable, List, Optional, TYPE_CHECKING, Union
import contextlib
import dataclasses
import enum
import os
import pickle
import time

import numpy as np
import psutil
import pynvml

import realhf.base.constants as constants
import realhf.base.logging as logging

if TYPE_CHECKING:
    from realhf.api.core.config import ModelName

logger = logging.getLogger("benchmark")

IF_MARK = False


def process_memory_mb(name):
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / 1024**2
    pid = process.pid
    logger.debug(f"Process PID {pid} memory usage @{name}: {memory_usage_mb}.")


def gpu_memory_mb(name):
    from deepspeed.accelerator import get_accelerator
    import torch.distributed as dist

    logger.debug(
        f"{name} GPU rank {dist.get_rank()}: memory usage: {round(get_accelerator().memory_allocated() / 1024**2, 2)}MB, "
        f"max memory usage: {round(get_accelerator().max_memory_allocated() / 1024**2, 2)}MB"
    )


def mock_time_mark(name, identifier, t, step):
    if IF_MARK:
        logger.debug(f"*{name}* #{identifier}#  ${t}$ ns step &{step}&")


def time_mark(name, identifier, step=0):
    if IF_MARK:
        logger.debug(
            f"*{name}* #{identifier}#  ${int(time.time_ns())}$ ns step &{step}&"
        )


def parse_time_mark_in_line(line, name, step_range=None):
    if f"*{name}*" in line:
        identifer, t, step = (
            line.split("#")[1],
            int(line.split("$")[1]),
            int(line.split("&")[1]),
        )
        if step_range:
            if step >= step_range[1] or step < step_range[0]:
                return None
        return identifer, t

    else:
        return None


def parse_time_mark_in_file(file, name, step_range=None):
    time_points = defaultdict(list)
    with open(file, "r") as f:
        count = 0
        res_count = 0
        for line in f.readlines():
            count += 1
            res = parse_time_mark_in_line(line, name, step_range=step_range)
            if res is not None:
                res_count += 1
                identifier, time_point = res
                time_points[identifier].append(time_point)
        # print(f"file {file} name {name} line count {count} res count {res_count}")
    return time_points


def parse_time_mark_in_dir(dir, name, step_range=None):
    time_points = {}
    for file in os.listdir(dir):
        file_path = os.path.join(dir, file)
        tpsf = parse_time_mark_in_file(file_path, name, step_range=step_range)
        for k, v in tpsf.items():
            if k not in time_points:
                time_points[k] = v
            else:
                time_points[k].extend(v)
    return time_points


MATPLOTLIB_COLORS = [
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "black",
    "brown",
    "gray",
    "cyan",
    "magenta",
    "lime",
    "olive",
    "navy",
]


def summary_time_points(
    start_keys,
    end_keys,
    identifiers,
    dir_name=None,
    file_name=None,
    start_time=None,
    figsize=(12, 4),
    end_time=None,
    step_range=None,
    save_fig_path="time_points.png",
    draw_boundary=False,
):
    """Plot and summary time marks in logs"""
    import matplotlib.pyplot as plt

    assert file_name or dir_name, "dir or file name must be specified"
    all_time_points = {}
    if file_name is None:
        for k in start_keys:
            all_time_points[k] = parse_time_mark_in_dir(
                dir_name, k, step_range=step_range
            )
        for k in end_keys:
            all_time_points[k] = parse_time_mark_in_dir(
                dir_name, k, step_range=step_range
            )
    else:
        for k in start_keys:
            all_time_points[k] = parse_time_mark_in_file(
                file_name, k, step_range=step_range
            )
        for k in end_keys:
            all_time_points[k] = parse_time_mark_in_file(
                file_name, k, step_range=step_range
            )

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_ylim(-1, len(identifiers))
    ax.set_yticks(list(range(len(identifiers))))
    ax.set_yticklabels(identifiers)

    label_set = {sk: False for sk in start_keys}
    infos = {}
    min_time = None
    max_time = None
    for id_index, identifier in enumerate(identifiers):
        time_sum = {}
        time_list = {}
        for start_key_idx, (start_key, end_key) in enumerate(
            zip(start_keys, end_keys)
        ):
            # print(start_key, identifier, all_time_points[start_key])
            time_sum[start_key] = 0
            time_list[start_key] = []
            try:
                start_time_points = np.array(
                    all_time_points[start_key][identifier]
                )
                end_time_points = np.array(all_time_points[end_key][identifier])
            except KeyError:
                continue
            assert len(start_time_points) == len(end_time_points)

            if start_time is not None:
                valid_indices_st = np.where(start_time_points > start_time)
                valid_indices_et = np.where(start_time_points < end_time)
                valid_indices = np.intersect1d(
                    valid_indices_st, valid_indices_et
                )
                start_time_points = start_time_points[valid_indices]
                end_time_points = end_time_points[valid_indices]

            # print(id_index, identifier, start_key_idx, start_key, end_key, start_time_points, end_time_points)

            # plot time point pairs
            for stp, etp in zip(list(start_time_points), list(end_time_points)):
                min_time = stp if min_time is None else min(min_time, stp)
                max_time = etp if max_time is None else max(max_time, etp)
                time_sum[start_key] += etp - stp
                time_list[start_key].append(etp - stp)

                if label_set[start_key] is False:
                    label = start_key
                    label_set[start_key] = True
                else:
                    label = None

                # print(f"id={identifier} start_key={start_key} left={stp%1000} width={etp-stp}")
                # print((etp-stp)//1e6)
                ax.barh(
                    y=id_index,
                    width=etp - stp,
                    left=stp,
                    height=0.8,
                    color=MATPLOTLIB_COLORS[start_key_idx],
                    label=label,
                )

                if draw_boundary:
                    ax.plot(
                        [stp, stp],
                        [id_index - 0.4, id_index + 0.4],
                        color="black",
                        linestyle="-",
                        linewidth=0.5,
                    )
                    ax.plot(
                        [etp, etp],
                        [id_index - 0.4, id_index + 0.4],
                        color="black",
                        linestyle="-",
                        linewidth=0.5,
                    )

        infos[identifier] = (time_sum, time_list)

    ax.set_xlim(min_time, max_time)
    total_width = max_time - min_time
    xticks = np.arange(
        min_time - total_width // 12, max_time - total_width // 12, 10 * 1e9
    )
    xtick_labels = [f"{int((i//1e9)%1000)}" for i in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)

    # summary time cost percent
    for id_index, identifier in enumerate(identifiers):
        print("=" * 30)
        print(f"Identifier {identifier} time cost percent:")
        bubble_time = 100
        time_sum, time_list = infos[identifier]
        for k in time_sum:
            time_perc = round(time_sum[k] / (max_time - min_time) * 100, 2)
            # print time cost percent
            avg_val = (
                round(mean(time_list[k]) / 10e6, 2)
                if len(time_list[k]) > 0
                else "-"
            )
            max_val = (
                round(max(time_list[k]) / 10e6, 2)
                if len(time_list[k]) > 0
                else "-"
            )
            min_val = (
                round(min(time_list[k]) / 10e6, 2)
                if len(time_list[k]) > 0
                else "-"
            )

            bubble_time -= time_perc
            print(
                f"{k} -- {time_perc} %, "
                f"avg, min, max = {avg_val}, {min_val}, {max_val} ms, "
                f"sum, n = {round(time_sum[k]/10e6, 2)} ms, {len(time_list[k])}"
            )
        print(f"bubble time -- {round(bubble_time, 2)}%")

    plt.legend(loc=(1.01, 0.0))
    plt.tight_layout()

    plt.savefig(save_fig_path)


def gpu_utilization_monitor(worker_idx: int, interval: float, ttl: float):
    pynvml.nvmlInit()
    gpu_idx = worker_idx % 8
    tik = time.time()
    while time.time() - tik < ttl:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_memory = memory_info.total / (
            1024**2
        )  # Convert bytes to megabytes
        used_memory = memory_info.used / (1024**2)
        memory_usage_percentage = (used_memory / total_memory) * 100
        logger.debug(
            f"Worker Index {worker_idx}, GPU {gpu_idx}: "
            f"Compute Utilization - {utilization.gpu}%, "
            f"Total Memory - {total_memory:.2f}MB, Used Memory - {used_memory:.2f}MB, "
            f"Memory Usage - {memory_usage_percentage:.2f}%"
        )
        time.sleep(interval)
    pynvml.nvmlShutdown()


# Helper function to calculate FLOPs using the Megatron-LM paper's formula
def calculate_llama_train_flops(
    checkpoint_activations_factor: int,
    batch_size: int,
    seqlens: List[int],
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    vocab_size: int,
):
    return checkpoint_activations_factor * caculuate_llama_forward_flops(
        batch_size,
        seqlens,
        num_layers,
        hidden_size,
        intermediate_size,
        vocab_size,
    )


def caculuate_llama_forward_flops(
    batch_size: int,
    seqlens: List[int],
    num_layers: int,
    hidden_size: int,
    intermediate_size: int,
    vocab_size: int,
):
    assert len(seqlens) == batch_size
    attn_flops = sum(x**2 for x in seqlens) * hidden_size
    return (
        2
        * num_layers
        * (
            4 * sum(seqlens) * hidden_size**2
            + 2 * attn_flops
            + 3 * sum(seqlens) * hidden_size * intermediate_size
        )
        + 4 * sum(seqlens) * vocab_size * hidden_size
    )


def calculate_llama_gen_flops(
    batch_size,
    prompt_lens,
    gen_len,
    num_layers,
    hidden_size,
    intermediate_size,
    vocab_size,
):
    flops = caculuate_llama_forward_flops(
        batch_size,
        prompt_lens,
        num_layers,
        hidden_size,
        intermediate_size,
        vocab_size,
    )
    for i in range(gen_len):
        prefix_lens = [x + i for x in prompt_lens]
        flops += (
            2
            * num_layers
            * (
                4 * batch_size * hidden_size**2
                + 2 * (sum(prefix_lens) + batch_size) * hidden_size
                + 3 * batch_size * hidden_size * intermediate_size
            )
            + 4 * batch_size * vocab_size * hidden_size
        )
    return flops


class CUDATimeMarkType(enum.Enum):
    forward = "forward"
    backward = "backward"
    optim_step = "optim_step"
    comm = "comm"
    misc = "misc"
    mem_layout = "memory_layout"


@dataclasses.dataclass
class TimeMarkEntry:
    name: str
    model_name: "ModelName"
    type_: CUDATimeMarkType
    start_time: int
    end_time: int


TIME_MARK_DB = []


def cuda_tmark(name: str, type_: CUDATimeMarkType):
    if os.getenv("REAL_CUDA_TMARK", None) == "1":

        def wrapper(f: Callable):

            def _wrapped_f(*args, **kwargs):
                import torch

                from realhf.base.constants import _model_name

                torch.cuda.synchronize()
                tik = time.time_ns()
                res = f(*args, **kwargs)
                torch.cuda.synchronize()
                tok = time.time_ns()
                global TIME_MARK_DB
                TIME_MARK_DB.append(
                    TimeMarkEntry(name, _model_name, type_, tik, tok)
                )
                return res

            return _wrapped_f

    else:

        def wrapper(f):
            return f

    return wrapper


@contextlib.contextmanager
def cuda_tmarked(name: str, type_: CUDATimeMarkType):
    if os.getenv("REAL_CUDA_TMARK", None) == "1":
        import torch

        from realhf.base.constants import _model_name

        torch.cuda.synchronize()
        tik = time.time_ns()
    yield
    if os.getenv("REAL_CUDA_TMARK", None) == "1":
        torch.cuda.synchronize()
        tok = time.time_ns()
        global TIME_MARK_DB
        TIME_MARK_DB.append(TimeMarkEntry(name, _model_name, type_, tik, tok))


def fetch_latest_tmark():
    global TIME_MARK_DB
    return TIME_MARK_DB[-1]


def dump_tmark_db(worker_idx):
    if os.getenv("REAL_CUDA_TMARK", None) != "1":
        return
    fn = os.path.join(
        constants.LOG_ROOT,
        constants.experiment_name(),
        constants.trial_name(),
        f"time_marks{worker_idx}.pkl",
    )
    global TIME_MARK_DB
    with open(fn, "wb") as f:
        pickle.dump(TIME_MARK_DB, f)
    TIME_MARK_DB.clear()


COMPUTE_KERNEL_KEYS = [
    "elementwise_kernel",
    "gemm_",
    "aten::",
    "at::native::",
    "flash",
    "backward_kernel",
    "reduce_kernel",
    "multi_tensor_apply",
]

COMM_KERNEL_KEYS = [
    "c10d::",
    "nccl::",
    "ncclDevKernel",
]

MEM_KERNEL_KEYS = [
    "Memcpy",
    "Memset",
]

MISC_KERNEL_KEYS = [
    "at_cuda_detail",
    "CudaCodeGen",
]


@dataclasses.dataclass
class CUDAKernelTime:  # in us
    compute: int
    comm: int
    mem: int
    misc: int

    @classmethod
    def from_profiler(cls, p):
        import torch

        compute_time = comm_time = mem_time = misc_time = 0
        unknown_keys = []
        for x in p.key_averages():
            if x.device_type != torch.autograd.DeviceType.CUDA:
                continue
            if x.self_cuda_time_total <= 0:
                continue
            if "waitevent" in x.key.lower():
                print(x.key)
            if any(k in x.key for k in COMPUTE_KERNEL_KEYS):
                compute_time += x.self_cuda_time_total
            elif any(k in x.key for k in COMM_KERNEL_KEYS):
                comm_time += x.self_cuda_time_total
            elif any(k in x.key for k in MEM_KERNEL_KEYS):
                mem_time += x.self_cuda_time_total
            elif any(k in x.key for k in MISC_KERNEL_KEYS):
                misc_time += x.self_cuda_time_total
            else:
                unknown_keys.append(x)
        if unknown_keys:
            raise NotImplementedError(
                f"Unknown keys: {[(x.key, x.self_cuda_time_total) for x in unknown_keys]}"
            )
        return cls(
            compute=compute_time, comm=comm_time, mem=mem_time, misc=misc_time
        )

    def __add__(self, other):
        return CUDAKernelTime(
            compute=self.compute + other.compute,
            comm=self.comm + other.comm,
            mem=self.mem + other.mem,
            misc=self.misc + other.misc,
        )

    @property
    def total_secs(self):
        return (self.compute + self.comm + self.mem + self.misc) / 1e6

    def __truediv__(self, x):
        return CUDAKernelTime(
            compute=self.compute / x,
            comm=self.comm / x,
            mem=self.mem / x,
            misc=self.misc / x,
        )

    def __repr__(self):
        return f"CUDAKernelTime(compute={self.compute}us, comm={self.comm}us, mem={self.mem}us, misc={self.misc}us)"
