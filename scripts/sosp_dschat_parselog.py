from collections import defaultdict
import argparse
import itertools
import json
import os
import subprocess

import numpy as np
import pandas as pd

benchmark_db = defaultdict(list)


def _parselog(
    model_size: int,
    actor_zero_stage: int,
    critic_zero_stage: int,
    seqlen: int,
    gen_bs: int,
    offload: bool,
):
    exp_name = f"rerun-dschat-a{model_size}-z{actor_zero_stage}-c7r7-cz{critic_zero_stage}-seqlen{seqlen}-g{gen_bs}"
    if offload:
        exp_name += "-offload"
    logpath = f"/lustre/aigc/llm/logs/fw/{exp_name}/benchmark/rlhf-0"
    oom = False
    time_records = []
    tflops_records = []
    thpt_records = []
    max_mem = 0.0
    try:
        with open(logpath, "r", errors="ignore") as f:
            lines = f.readlines()
            for line in lines:
                if "CUDA out of memory" in line or "not enough memory" in line:
                    oom = True
                    break
                elif ("torch.distributed.DistBackendError: NCCL error in: /tmp/pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1331, internal error - please report this issue to the NCCL developers, NCCL version 2.18.5"
                      in line):
                    oom = True
                    break
                if "End-to-End" in line:
                    step_time = float(line.split("End-to-End => Latency: ")[1].split("s,")[0])
                    time_records.append(step_time)
                    tflops = float(line.split("TFLOPs: ")[1].split(",")[0])
                    tflops_records.append(tflops)
                    thpt = float(line.split(", Samples/sec: ")[1].split(",")[0])
                    thpt_records.append(thpt)
                if "Compute Utilization - " in line:
                    mem = float(line.split("Used Memory - ")[1].split("MB,")[0])
                    max_mem = max(max_mem, mem)
    except FileNotFoundError:
        return False
    if not oom:
        if (len(time_records) == 0 or len(tflops_records) == 0 or len(thpt_records) == 0 or max_mem == 0.0):
            return False
        avg_time = np.mean(time_records[1:])
        avg_tflops = np.mean(tflops_records[1:])
        thpt = np.mean(thpt_records[1:])
    else:
        avg_time = float("inf")
        avg_tflops = -float("inf")
        thpt = -float("inf")
        max_mem = 0.0
    d = dict(
        model_size=model_size,
        actor_zero_stage=actor_zero_stage,
        critic_zero_stage=critic_zero_stage,
        seqlen=seqlen,
        gen_bs=gen_bs,
        offload=offload,
        OOM=oom,
        Throughput=thpt,
        MaxGPUMemory=max_mem,
        avg_tflops=avg_tflops,
    )
    for k, v in d.items():
        benchmark_db[k].append(v)
    return True


def parselog(model_size: int):
    # takeaways
    # 1. when actor zero=3, critic zero=2 or 3 makes no difference. So by default we set critic zero=3;
    # 2. when actor zero=2, inference_tp_size must be 1;
    # 3. max GPU memory used is usually determined by gen_bs;
    critic_zero_stages = [3]
    actor_zero_stages = [3, 2]
    gen_batch_sizes = range(1, 100)
    seqlens = [256, 512, 1024]
    offloads = [True, False]
    for critic_zero_stage, actor_zero_stage in itertools.product(critic_zero_stages, actor_zero_stages):
        for max_answer_len, gen_bs, offload in itertools.product(seqlens, gen_batch_sizes, offloads):
            _parselog(
                model_size,
                actor_zero_stage,
                critic_zero_stage,
                max_answer_len,
                gen_bs,
                offload,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", "-x", type=int, default=7, choices=[7, 13, 34, 70], nargs="+")
    parser.add_argument("--max", action="store_true")
    args = parser.parse_args()
    for model_size in args.model_size:
        parselog(model_size)
    df = pd.DataFrame(benchmark_db)
    if not args.max:
        print(df.to_string(index=False))
    else:
        max_throughput_df = df.loc[df.groupby(["model_size", "seqlen"])["Throughput"].idxmax()]
        print(max_throughput_df.to_string(index=False))
