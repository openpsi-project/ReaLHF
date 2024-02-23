from collections import defaultdict
from statistics import mean
from typing import List, Optional, Union
import os
import time

import numpy as np
import psutil
import viztracer
import pynvml

import base.logging as logging

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
        f"max memory usage: {round(get_accelerator().max_memory_allocated() / 1024**2, 2)}MB")


def mock_time_mark(name, identifier, t, step):
    if IF_MARK:
        logger.debug(f"*{name}* #{identifier}#  ${t}$ ns step &{step}&")


def time_mark(name, identifier, step=0):
    if IF_MARK:
        logger.debug(f"*{name}* #{identifier}#  ${int(time.time_ns())}$ ns step &{step}&")


def parse_time_mark_in_line(line, name, step_range=None):
    if f"*{name}*" in line:
        identifer, t, step = line.split("#")[1], int(line.split("$")[1]), int(line.split("&")[1])
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


def summary_time_points(start_keys,
                        end_keys,
                        identifiers,
                        dir_name=None,
                        file_name=None,
                        start_time=None,
                        figsize=(12, 4),
                        end_time=None,
                        step_range=None,
                        save_fig_path="time_points.png",
                        draw_boundary=False):
    """ Plot and summary time marks in logs
    """
    import matplotlib.pyplot as plt

    assert file_name or dir_name, "dir or file name must be specified"
    all_time_points = {}
    if file_name is None:
        for k in start_keys:
            all_time_points[k] = parse_time_mark_in_dir(dir_name, k, step_range=step_range)
        for k in end_keys:
            all_time_points[k] = parse_time_mark_in_dir(dir_name, k, step_range=step_range)
    else:
        for k in start_keys:
            all_time_points[k] = parse_time_mark_in_file(file_name, k, step_range=step_range)
        for k in end_keys:
            all_time_points[k] = parse_time_mark_in_file(file_name, k, step_range=step_range)

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
        for start_key_idx, (start_key, end_key) in enumerate(zip(start_keys, end_keys)):
            # print(start_key, identifier, all_time_points[start_key])
            time_sum[start_key] = 0
            time_list[start_key] = []
            try:
                start_time_points = np.array(all_time_points[start_key][identifier])
                end_time_points = np.array(all_time_points[end_key][identifier])
            except KeyError:
                continue
            assert len(start_time_points) == len(end_time_points)

            if start_time is not None:
                valid_indices_st = np.where(start_time_points > start_time)
                valid_indices_et = np.where(start_time_points < end_time)
                valid_indices = np.intersect1d(valid_indices_st, valid_indices_et)
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
                ax.barh(y=id_index,
                        width=etp - stp,
                        left=stp,
                        height=0.8,
                        color=MATPLOTLIB_COLORS[start_key_idx],
                        label=label)

                if draw_boundary:
                    ax.plot([stp, stp], [id_index - 0.4, id_index + 0.4],
                            color='black',
                            linestyle='-',
                            linewidth=0.5)
                    ax.plot([etp, etp], [id_index - 0.4, id_index + 0.4],
                            color='black',
                            linestyle='-',
                            linewidth=0.5)

        infos[identifier] = (time_sum, time_list)

    ax.set_xlim(min_time, max_time)
    total_width = max_time - min_time
    xticks = np.arange(min_time - total_width // 12, max_time - total_width // 12, 10 * 1e9)
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
            avg_val = round(mean(time_list[k]) / 10e6, 2) if len(time_list[k]) > 0 else "-"
            max_val = round(max(time_list[k]) / 10e6, 2) if len(time_list[k]) > 0 else "-"
            min_val = round(min(time_list[k]) / 10e6, 2) if len(time_list[k]) > 0 else "-"

            bubble_time -= time_perc
            print(f"{k} -- {time_perc} %, "
                  f"avg, min, max = {avg_val}, {min_val}, {max_val} ms, "
                  f"sum, n = {round(time_sum[k]/10e6, 2)} ms, {len(time_list[k])}")
        print(f"bubble time -- {round(bubble_time, 2)}%")

    plt.legend(loc=(1.01, 0.0))
    plt.tight_layout()

    plt.savefig(save_fig_path)


class NoopTracer:
    """Dumb alternative for VizTracer."""

    def __init__(self, **kwargs):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def save(self, *args, **kwargs):
        pass


def get_tracer(
    tracer_entries: int = 1000000,
    verbose: int = 1,
    max_stack_depth: int = -1,
    include_files: Optional[List[str]] = None,
    exclude_files: Optional[List[str]] = None,
    ignore_c_function: bool = False,
    ignore_frozen: bool = False,
    log_func_retval: bool = False,
    log_func_args: bool = False,
    log_print: bool = False,
    log_gc: bool = False,
    log_async: bool = False,
    pid_suffix: bool = False,
    register_global: bool = True,
    min_duration: int = 0,
    output_file: str = "result.json",
) -> viztracer.VizTracer:
    if os.environ.get("DLLM_TRACE") == "1":
        return viztracer.VizTracer(
            tracer_entries=tracer_entries,
            verbose=verbose,
            max_stack_depth=max_stack_depth,
            include_files=include_files,
            exclude_files=exclude_files,
            ignore_c_function=ignore_c_function,
            ignore_frozen=ignore_frozen,
            log_func_retval=log_func_retval,
            log_func_args=log_func_args,
            log_print=log_print,
            log_gc=log_gc,
            log_async=log_async,
            pid_suffix=pid_suffix,
            register_global=register_global,
            min_duration=min_duration,
            output_file=output_file,
        )
    else:
        return NoopTracer()

def gpu_utilization_monitor(gpu_idx:int, ttl:float):
    pynvml.nvmlInit()
    tik = time.time()
    while time.time() - tik < ttl:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_memory = memory_info.total / (1024 ** 2)  # Convert bytes to megabytes
        used_memory = memory_info.used / (1024 ** 2)
        memory_usage_percentage = (used_memory / total_memory) * 100
        logger.debug(f"GPU {gpu_idx}: Compute Utilization - {utilization.gpu}%, Total Memory - {total_memory:.2f}MB, Used Memory - {used_memory:.2f}MB, Memory Usage - {memory_usage_percentage:.2f}%")
        time.sleep(10)
    pynvml.nvmlShutdown()
