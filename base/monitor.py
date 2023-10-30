from collections import defaultdict
import logging
import os
import time

from deepspeed.accelerator import get_accelerator
import numpy as np
import psutil

logger = logging.getLogger("benchmarkutils")


def process_memory_mb(name):
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / 1024**2
    pid = process.pid
    logger.info(f"Process PID {pid} memory usage @{name}: {memory_usage_mb}.")


def gpu_memory_mb(name):
    import torch.distributed as dist
    logger.info(
        f"{name} GPU rank {dist.get_rank()}: memory usage: {round(get_accelerator().memory_allocated() / 1024**2, 2)}MB, "
        f"max memory usage: {round(get_accelerator().max_memory_allocated() / 1024**2, 2)}MB")


def time_mark(name, identifier, step=0):
    logger.info(f"*{name}* #{identifier}#  ${int(time.time_ns())}$ ns step &{step}&")


def time_mark_ms(name, identifier, step=0):
    logger.info(f"*{name}* #{identifier}#  ${int(time.time_ns()/10e6)}$ ms step &{step}&")


def time_mark_s(name, identifier, step=0):
    logger.info(f"*{name}* #{identifier}#  ${int(time.time_ns()/10e9)}$ s step &{step}&")


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
        for line in f.readlines():
            res = parse_time_mark_in_line(line, name, step_range=step_range)
            if res is not None:
                identifier, time_point = res
                time_points[identifier].append(time_point)
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


MATPLOTLIB_COLORS = ["b", "g", "r", "c", "m", "y", "k", "w"]


def plot_time_points(start_keys,
                     end_keys,
                     identifiers,
                     dir_name=None,
                     file_name=None,
                     start_time=None,
                     figsize=(12, 4),
                     end_time=None,
                     step_range=None,
                     save_fig_path="time_points.png"):
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
    if start_time is not None:
        ax.set_xlim(start_time, end_time)
        xticks = np.arange(start_time, end_time, 100)
        xtick_labels = [f"{i%1000}" for i in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)
    label_set = {sk: False for sk in start_keys}
    # f = True
    # print(all_time_points)

    for id_index, identifier in enumerate(identifiers):
        for start_key_idx, (start_key, end_key) in enumerate(zip(start_keys, end_keys)):
            # print(start_key, identifier, all_time_points[start_key])
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

            print(id_index, identifier, start_key_idx, start_key, end_key, start_time_points, end_time_points)

            # plot time point pairs
            for stp, etp in zip(list(start_time_points), list(end_time_points)):
                if label_set[start_key] is False:
                    label = start_key
                    label_set[start_key] = True
                else:
                    label = None
                print(f"id={identifier} start_key={start_key} left={stp%1000} width={etp-stp}")
                ax.barh(y=id_index,
                        width=etp - stp,
                        left=stp,
                        color=MATPLOTLIB_COLORS[start_key_idx],
                        label=label)

                # print(int(identifier))
                # print(start_key, MATPLOTLIB_COLORS[start_key_idx])
                # f=False

    plt.legend()

    plt.savefig(save_fig_path)
