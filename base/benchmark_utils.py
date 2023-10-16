from collections import defaultdict
import logging
import os
import time

import psutil

logger = logging.getLogger("benchmarkutils")

LINE_START_INDENTIFIER = "__TIME_LOGPOINT__"


def set_time_point(name: str, use_gpu_global_rank=False):
    log_entries = [f"{LINE_START_INDENTIFIER}"]
    log_entries.append(name)
    log_entries.append(str(time.time_ns()))
    if not use_gpu_global_rank:
        pid = psutil.Process().pid
        log_entries.append(str(pid))
    else:
        import torch.distributed as dist
        log_entries.append(str(dist.get_rank()))
    log_string = " " + " ".join(log_entries) + " "
    logging.info(log_string)


def parse_time_point_line(line: str):
    assert LINE_START_INDENTIFIER in line
    words = line.split(" ")
    start_index = words.index(LINE_START_INDENTIFIER)
    return words[start_index + 1], int(words[start_index + 2]), int(words[start_index + 3])


def parse_time_point_file(log_file_name):
    time_points = {}
    with open(log_file_name, "r") as f:
        lines = f.readlines()
        for line in lines:
            if LINE_START_INDENTIFIER in line:
                name, t, pid = parse_time_point_line(line)
                if pid in time_points:
                    time_points[pid][name].append(t)
                else:
                    time_points[pid] = defaultdict(list)
    return time_points


def parse_time_point_dir(dir_name):
    tps = {}
    for fn in os.listdir(dir_name):
        abs_fn = os.path.join(dir_name, fn)
        if os.path.isdir(abs_fn):
            continue
        tp = parse_time_point_file(abs_fn)
        for pid, d in tp:
            if pid in tps:
                tps[pid].extend(tp[pid])
            else:
                tps[pid] = tp[pid]
    return tps


def plot_time_info(tps, name_pairs, time_range):
    """ 
    Arguments:
        tps: Time points returned by parse_time_point_dir. Keys are process ids or GPU global ranks.
             
    """
    import matplotlib.pyplot as plt
