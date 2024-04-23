from typing import *
import argparse
import collections
import dataclasses
import datetime
import functools
import gc
import itertools
import json
import multiprocessing as mp
import os
import queue
import time

import pynvml
import torch
import torch.distributed
import torch.profiler

from base.topology import PipeModelDataParallelTopology
from tests.utils import (clear_gpu_cache, clear_name_resolve, get_llama7b_flash_config, get_memory,
                         get_pytorch_profiler, MODEL_NAME, pytorch_memory_burnin)
import reallm.base.constants
import reallm.base.gpu_utils
import reallm.base.topology


def get_model(mconfig):
    from impl.model.backend.pipe_inf import InferencePipelineEngine
    from impl.model.nn.flash_mqat.flash_mqat_api import add_helper_functions, FlashMQATModel

    m = FlashMQATModel(mconfig, device=torch.device("cuda:0"), dtype=torch.float16)
    m.instantiate()
    if base.constants.pipe_parallel_world_size() == 1:
        add_helper_functions(m)
        engine = m
    else:
        engine = InferencePipelineEngine(m)
    torch.cuda.synchronize()
    return m, engine


@torch.no_grad()
def test_impl(rank, world_size, topo, profile, check, n_iterations, record_cost_to_file):
    mconfig = get_llama7b_flash_config()
    with base.constants.model_scope(MODEL_NAME):
        m, engine = get_model(mconfig)
        if check:
            original_state_dict = m.state_dict()

        m2, engine2 = get_model(mconfig)

        for it in range(n_iterations):
            if it == n_iterations - 1 and profile:
                profiler = get_pytorch_profiler(f"offload{rank}.json")
                profiler.start()

            packed_input_ids = torch.randint(
                0,
                mconfig.vocab_size,
                (2**17 // base.constants.data_parallel_world_size(),),
                dtype=torch.long,
                device="cuda",
            )
            bs = 2**17 // base.constants.data_parallel_world_size() // 256
            if bs == 0:
                return
            cu_seqlens = torch.linspace(0, 256 * bs, bs + 1, dtype=torch.int32, device="cuda")
            seqlens_cpu = [256 for _ in range(bs)]
            assert cu_seqlens[-1] == packed_input_ids.shape[0]

            # normal forward, no offload
            torch.distributed.barrier()
            torch.cuda.synchronize()
            tik = time.perf_counter_ns()
            if base.constants.pipe_parallel_world_size() == 1:
                engine.forward(packed_input_ids=packed_input_ids, cu_seqlens=cu_seqlens, max_seqlen=256)
            else:
                engine.forward(
                    seqlens_cpu=seqlens_cpu,
                    packed_input_ids=packed_input_ids,
                    cu_seqlens=cu_seqlens,
                    num_micro_batches=base.constants.pipe_parallel_world_size(),
                )
            torch.cuda.synchronize()
            normal_fwd_t = time.perf_counter_ns() - tik

            m.async_offload()
            m.wait_for_offload()

            # overlapped load-and-forward
            torch.distributed.barrier()
            torch.cuda.synchronize()
            tik = time.perf_counter_ns()
            if base.constants.pipe_parallel_world_size() == 1:
                engine.forward(packed_input_ids=packed_input_ids, cu_seqlens=cu_seqlens, max_seqlen=256)
            else:
                engine.forward(
                    seqlens_cpu=seqlens_cpu,
                    packed_input_ids=packed_input_ids,
                    cu_seqlens=cu_seqlens,
                    num_micro_batches=base.constants.pipe_parallel_world_size(),
                )
            torch.cuda.synchronize()
            load_fwd_t = time.perf_counter_ns() - tik

            if check:
                new_state_dict = m.state_dict()
                for k in original_state_dict:
                    # print(k, original_state_dict[k])
                    assert torch.allclose(original_state_dict[k], new_state_dict[k]), (
                        k,
                        original_state_dict[k],
                        new_state_dict[k],
                    )

            clear_gpu_cache()
            init_mem = get_memory(0)
            print("After model creation", get_memory(0))

            # overlapped offload and other model forward
            torch.distributed.barrier()
            torch.cuda.synchronize()
            tik = time.perf_counter_ns()
            m.async_offload()
            if base.constants.pipe_parallel_world_size() == 1:
                engine2.forward(packed_input_ids=packed_input_ids, cu_seqlens=cu_seqlens, max_seqlen=256)
            else:
                engine2.forward(
                    seqlens_cpu=seqlens_cpu,
                    packed_input_ids=packed_input_ids,
                    cu_seqlens=cu_seqlens,
                    num_micro_batches=base.constants.pipe_parallel_world_size(),
                )
            torch.cuda.synchronize()
            offload_fwd_t = time.perf_counter_ns() - tik

            clear_gpu_cache()
            mem_diff = get_memory(0) - init_mem
            print("After model offload", get_memory(0))

            if it == n_iterations - 1 and profile:
                profiler.__exit__(None, None, None)

            if record_cost_to_file:
                normal_fwd_t = torch.tensor(normal_fwd_t, device="cuda", dtype=torch.long) / world_size
                torch.distributed.all_reduce(normal_fwd_t)

                load_fwd_t = torch.tensor(load_fwd_t, device="cuda", dtype=torch.long) / world_size
                torch.distributed.all_reduce(load_fwd_t)

                offload_fwd_t = torch.tensor(offload_fwd_t, device="cuda", dtype=torch.long) / world_size
                torch.distributed.all_reduce(offload_fwd_t)

                mem_diff = torch.tensor(mem_diff, device="cuda", dtype=torch.long)
                torch.distributed.all_reduce(mem_diff, op=torch.distributed.ReduceOp.MAX)

                if rank == 0 and it == n_iterations - 1:
                    with open("offload_cost.jsonl", "a") as f:
                        d = dict(
                            world_size=world_size,
                            pp_mp_dp=(topo.get_dim("pipe"), topo.get_dim("model"), topo.get_dim("data")),
                            normal_fwd_t=normal_fwd_t.item(),
                            load_fwd_t=load_fwd_t.item(),
                            offload_fwd_t=offload_fwd_t.item(),
                            mem_diff_per_device=mem_diff.item(),
                        )
                        f.write(json.dumps(d, ensure_ascii=False) + "\n")


def setup_gpu(rank, world_size):
    os.environ["DLLM_MODE"] = "LOCAL"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % 8)
    os.environ["GPU_DEVICES_ISOLATED"] = str(1)
    info = base.gpu_utils.setup_ddp(
        "offload-test",
        f"w{world_size}",
        rank,
        world_size=world_size,
        global_rank=rank,
    )
    world_size = info.world_size
    # print(f"rank {rank} setup ddp")
    import deepspeed

    deepspeed.init_distributed()
    # print(f"rank {rank} setup deepspeed")
    pynvml.nvmlInit()
    pytorch_memory_burnin(rank)
    return info


def test(
    idx,
    world_size,
    topo,
    err_queue,
    profile: bool,
    check: bool,
    n_iterations: int,
    record_cost_to_file: bool,
):
    # PyTorch CUDA setup
    assert world_size == topo.get_dim("pipe") * topo.get_dim("model") * topo.get_dim("data")
    setup_gpu(idx, world_size)

    with base.constants.model_scope(MODEL_NAME):
        base.constants.set_rank_mapping(MODEL_NAME, topo)
        wg = base.topology.new_or_get_group(
            ranks=[base.constants.to_global_pg_rank(i) for i in range(world_size)])

        base.constants.set_parallelism_group(model_name=MODEL_NAME, pgroup=wg)
        grid = base.topology.ParallelGrid(process_group=wg, topology=topo)
        base.constants.set_grid(model_name=MODEL_NAME, grid=grid)

    print("After setup memory", get_memory(0))

    try:

        test_impl(idx, world_size, topo, profile, check, n_iterations, record_cost_to_file)

        torch.distributed.barrier()
        if idx == 0:
            clear_name_resolve("offload-test", f"w{world_size}")
        torch.distributed.barrier()
    except Exception as e:
        err_queue.put(e)
        raise e


def decompose_to_three_factors(n: int):
    factors = []
    for i in range(1, int(n**(1 / 2)) + 1):
        if n % i == 0:
            for j in range(i, int((n // i)**(1 / 2)) + 1):
                if (n // i) % j == 0:
                    k = (n // i) // j
                    factors += list(set(itertools.permutations([i, j, k])))
    return factors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node_idx", "-i", type=int, default=0)
    parser.add_argument("--num_nodes", "-n", type=int, default=1)
    args = parser.parse_args()

    err_queue = mp.Queue(100)
    for world_size in [8]:
        three_factors = decompose_to_three_factors(world_size)
        if os.path.exists("offload_cost.jsonl"):
            with open("offload_cost.jsonl", "r") as f:
                cost_data = [json.loads(line) for line in f]
        all_configs = list(filter(lambda x: x[0] <= 16 and x[1] <= 8 and x[2] <= 8, three_factors))
        print(f">>>>>>>>> running {len(all_configs)} configurations >>>>>>>")
        for config_id, x in enumerate(all_configs):
            if args.node_idx == args.num_nodes - 1 and config_id == 0:
                clear_name_resolve("offload-test", f"w{world_size}")
            if os.path.exists("offload_cost.jsonl"):
                with open("offload_cost.jsonl", "r") as f:
                    cost_data = [json.loads(line) for line in f]
                if any(
                    [d["world_size"] == world_size and tuple(d["pp_mp_dp"]) == tuple(x) for d in cost_data]):
                    continue
            print(f"Testing offload with topo {x}")

            topo = PipeModelDataParallelTopology(num_pp=x[0], num_mp=x[1], num_dp=x[2])
            procs = []
            for i in range(8):
                p = mp.Process(
                    target=test,
                    args=(
                        i + args.node_idx * 8,
                        args.num_nodes * 8,
                        topo,
                        err_queue,
                    ),
                    kwargs=dict(
                        profile=False,
                        n_iterations=3,
                        check=False,
                        record_cost_to_file=True,
                    ),
                )
                p.start()
                procs.append(p)
            for proc in procs:
                proc.join()
            for i in range(8):
                try:
                    err_code = err_queue.get_nowait()
                    print("error!!!!!!!!!!!!")
                    exit(0)
                except queue.Empty:
                    pass
