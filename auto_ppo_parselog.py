import datetime
import enum
import os
import re
import subprocess

import numpy as np

bs_seqlen = [(128, 896), (256, 384), (512, 128)]
model_sizes = [7, 13, 34, 70]


def _parse_log(rootdir: str, model_size, seqlen, bs, mode):
    # throughput is (4*gen_bs*n_actor_gpus / e2e_time)
    record_times = []
    oom = False
    # print(">>>", os.path.join(rootdir, f"model_worker-0"))
    with open(os.path.join(rootdir, f"model_worker-0"), "r") as f:
        begin_line_no = 0
        lines = f.readlines()
        for line_no, line in enumerate(lines):
            if "SBATCH SCRIPT BEGIN" in line:
                begin_line_no = line_no
        for line_no, line in enumerate(lines):
            if line_no < begin_line_no:
                continue
            if "out of memory" in line:
                oom = True
                break

    with open(os.path.join(rootdir, f"master_worker-0"), "r") as f:
        begin_line_no = 0
        lines = f.readlines()
        for line_no, line in enumerate(lines):
            if "SBATCH SCRIPT BEGIN" in line:
                begin_line_no = line_no

        for line_no, line in enumerate(lines):
            if line_no < begin_line_no:
                continue
            if oom:
                break
            if "End to end" not in line:
                continue
            x = float(line.split("#End to end# execution time: *")[1].split("*s.")[0])
            record_times.append(x)
    if oom:
        print(f"{model_size}B mode {mode} seqlen={seqlen} bs={bs} OOM")
    else:
        # assert len(record_times) >= 12, len(record_times)
        record_times = record_times[2:17]
        print(
            f"{model_size}B mode {mode} seqlen={seqlen} bs={bs} mean step time {np.mean(record_times):.2f}s")


def parse_log():

    modes = ["s", "m"]
    # modes = ["s", "m"]
    for r in range(0, 1):
        for model_size in model_sizes:
            for bs, seqlen in bs_seqlen:
                for mode in modes:
                    trial_name = f"20240407-{r}"
                    exp_name = f"sosp-a{model_size}s{seqlen}g{bs}t{bs}-{mode}"
                    root_dir = f"/lustre/aigc/llm/logs/meizy/{exp_name}/{trial_name}"
                    print(exp_name)
                    try:
                        _parse_log(root_dir, model_size, seqlen, bs, mode)
                    except FileNotFoundError:
                        pass


if __name__ == "__main__":
    parse_log()
