import enum
import os
import re
import subprocess

import numpy as np


class ModelSize(enum.Enum):
    SMALL = 7
    MEDIUM = 13
    LARGE = 34
    XLARGE = 70


def _parse_log(rootdir: str, model_size, dp_size, pp_size, exp_identifier=""):
    # throughput is (4*gen_bs*n_actor_gpus / e2e_time)
    record_times = []
    oom = False
    # print(">>>", os.path.join(rootdir, f"model_worker-0"))
    with open(os.path.join(rootdir, f"model_worker-0"), "r") as f:
        for line in f.readlines():
            if "CUDA out of memory" in line:
                oom = True
                break
    with open(os.path.join(rootdir, f"master_worker-0"), "r") as f:
        for line in f.readlines():
            if oom:
                break
            if "End to end" not in line:
                continue
            x = float(line.split("#End to end# execution time: *")[1].split("*s.")[0])
            record_times.append(x)
    if oom:
        print(f"{model_size}B {exp_identifier} dp_size={dp_size} pp_size={pp_size} OOM")
    else:
        # assert len(record_times) >= 12, len(record_times)
        record_times = record_times[2:17]
        print(
            f"{model_size}B {exp_identifier} dp_size={dp_size} pp_size={pp_size} mean step time {np.mean(record_times):.2f}s"
        )


def parse_log(model_size: ModelSize):
    print("-" * 40)
    n_gpus = (64 if model_size == ModelSize.XLARGE else 32 if model_size == ModelSize.LARGE else 16)
    pp_sizes = [1, 2, 4, 8]
    if model_size == ModelSize.XLARGE:
        pp_sizes.append(16)
        pp_sizes.remove(1)
    for pp_size in pp_sizes:
        dp_size = n_gpus // pp_size
        exp_name = f"sosp-baseline-a{model_size.value}-{dp_size}x{pp_size}-c7r7"
        rootdir = f"/lustre/aigc/llm/logs/fw/{exp_name}/benchmark"
        _parse_log(rootdir, model_size.value, dp_size, pp_size)
        if pp_size > 1:
            rootdir = f"/lustre/aigc/llm/logs/fw/{exp_name}-mb1/benchmark"
            _parse_log(rootdir, model_size.value, dp_size, pp_size, "mb1")
            rootdir = f"/lustre/aigc/llm/logs/fw/{exp_name}-mb1gen/benchmark"
            _parse_log(rootdir, model_size.value, dp_size, pp_size, "mb1gen")
    print("-" * 40)


def main():
    parse_log(ModelSize.SMALL)
    parse_log(ModelSize.MEDIUM)
    parse_log(ModelSize.LARGE)
    parse_log(ModelSize.XLARGE)


if __name__ == "__main__":
    main()
