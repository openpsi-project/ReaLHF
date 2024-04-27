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

from reallm.base.topology import PipeModelDataParallelTopology
from tests.utils import (clear_gpu_cache, clear_name_resolve, get_llama7b_real_config, get_memory,
                         get_pytorch_profiler, MODEL_NAME, pytorch_memory_burnin)
import reallm.base.constants as constants
import reallm.base.gpu_utils
import reallm.base.topology


def get_model(mconfig):
    from reallm.impl.model.backend.pipe_inf import InferencePipelineEngine
    from reallm.impl.model.nn.real_llm_api import add_helper_functions, ReaLModel

    m = ReaLModel(mconfig, device=torch.device("cuda:0"), dtype=torch.float16)
    m.instantiate()
    m.forward = m._forward
    # add_helper_functions(m)
    assert constants.pipe_parallel_world_size() == 1
    torch.cuda.synchronize()
    return m


@torch.no_grad()
def test_impl(rank, world_size, topo, profile, check, n_iterations, record_cost_to_file):
    from reallm.impl.model.backend.deepspeed import deepspeed_initialize, get_eval_ds_config

    mconfig = get_llama7b_real_config()
    with constants.model_scope(MODEL_NAME):
        m = get_model(mconfig)

        ds_config = get_eval_ds_config(offload=True, stage=3, enable_fp16=True)
        engine, *_ = deepspeed_initialize(
            model=m,
            config=ds_config,
            engine_type="deepspeed",
            sequence_parallel=False,
            enable_async_p2p_communication=False,
        )

        for it in range(n_iterations):
            if it == n_iterations - 1 and profile:
                profiler = get_pytorch_profiler(f"zero3offload{rank}.json")
                profiler.start()

            clear_gpu_cache()
            init_mem = get_memory(0)
            print("After model creation", get_memory(0))

            packed_input_ids = torch.randint(
                0,
                mconfig.vocab_size,
                (2**17 // constants.data_parallel_world_size(),),
                dtype=torch.long,
                device="cuda",
            )
            bs = 2**17 // constants.data_parallel_world_size() // 256
            position_ids = torch.arange(0, 256, dtype=torch.long, device="cuda").repeat(bs, 1).flatten()
            if bs == 0:
                return
            cu_seqlens = torch.linspace(0, 256 * bs, bs + 1, dtype=torch.int32, device="cuda")
            assert cu_seqlens[-1] == packed_input_ids.shape[0]

            # normal forward, no offload
            torch.distributed.barrier()
            torch.cuda.synchronize()
            tik = time.perf_counter_ns()
            engine.forward(
                input_ids=packed_input_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=256,
                k_caches=None,
                v_caches=None,
                cache_seqlens=None,
                position_ids=position_ids,
                hidden_states=None,
            )
            torch.cuda.synchronize()
            fwd_t = time.perf_counter_ns() - tik

            clear_gpu_cache()
            fwd_mem = get_memory(0)
            print("After model forward", get_memory(0))

            if it == n_iterations - 1 and profile:
                profiler.__exit__(None, None, None)

            if record_cost_to_file:
                fwd_t = torch.tensor(fwd_t, device="cuda", dtype=torch.long) / world_size
                torch.distributed.all_reduce(fwd_t)

                init_mem = torch.tensor(init_mem, device="cuda", dtype=torch.long)
                torch.distributed.all_reduce(init_mem, op=torch.distributed.ReduceOp.MAX)
                fwd_mem = torch.tensor(fwd_mem, device="cuda", dtype=torch.long)
                torch.distributed.all_reduce(fwd_mem, op=torch.distributed.ReduceOp.MAX)

                if rank == 0 and it == n_iterations - 1:
                    with open("zero3offload_cost.jsonl", "a") as f:
                        d = dict(
                            world_size=world_size,
                            pp_mp_dp=(topo.get_dim("pipe"), topo.get_dim("model"), topo.get_dim("data")),
                            fwd_t=fwd_t.item(),
                            init_mem=init_mem.item(),
                            fwd_mem=fwd_mem.item(),
                        )
                        f.write(json.dumps(d, ensure_ascii=False) + "\n")


def setup_gpu(rank, world_size):
    os.environ["DLLM_MODE"] = "LOCAL"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % 8)
    os.environ["GPU_DEVICES_ISOLATED"] = str(1)
    info = reallm.base.gpu_utils.setup_ddp(
        "zero3offload-test",
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

    with constants.model_scope(MODEL_NAME):
        constants.set_rank_mapping(MODEL_NAME, topo)
        wg = reallm.base.topology.new_or_get_group(
            ranks=[constants.to_global_pg_rank(i) for i in range(world_size)])

        constants.set_parallelism_group(model_name=MODEL_NAME, pgroup=wg)
        grid = reallm.base.topology.ParallelGrid(process_group=wg, topology=topo)
        constants.set_grid(model_name=MODEL_NAME, grid=grid)

    print("After setup memory", get_memory(0))

    try:

        test_impl(idx, world_size, topo, profile, check, n_iterations, record_cost_to_file)

        torch.distributed.barrier()
        if idx == 0:
            clear_name_resolve("zero3offload-test", f"w{world_size}")
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
        if os.path.exists("zero3offload_cost.jsonl"):
            with open("zero3offload_cost.jsonl", "r") as f:
                cost_data = [json.loads(line) for line in f]
        all_configs = list(filter(lambda x: x[0] == 1 and x[1] <= 8 and x[2] <= 8, three_factors))
        print(f">>>>>>>>> running {len(all_configs)} configurations >>>>>>>")
        # for config_id, x in enumerate(all_configs):
        for config_id, x in enumerate([(1, 1, 8)]):
            if args.node_idx == args.num_nodes - 1 and config_id == 0:
                clear_name_resolve("zero3offload-test", f"w{world_size}")
            if os.path.exists("zero3offload_cost.jsonl"):
                with open("zero3offload_cost.jsonl", "r") as f:
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
                        profile=True,
                        n_iterations=3,
                        check=False,
                        record_cost_to_file=False,
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
