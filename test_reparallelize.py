from typing import *
import argparse
import dataclasses
import itertools
import multiprocessing as mp
import os
import queue
import random
import time

import torch
import torch.distributed

from api.config.config_base import ModelName, ModelShardID
from api.config.config_system import ModelName, ModelShardID
from base.monitor import cuda_tmark, cuda_tmarked, CUDATimeMarkType, fetch_latest_tmark
from base.topology import PipeModelDataParallelTopology
from tests.utils import *
import base.constants
import base.gpu_utils


def test_impl(
    rank,
    world_size,
    from_model_name,
    to_model_name,
    from_topo,
    to_topo,
    pg_info,
    profile=False,
    check=True,
    n_iterations=1,
    profile_compile=False,
    record_cost_to_file=False,
):
    assert not (profile and profile_compile)
    from impl.model.backend.pipe_inf import InferencePipelineEngine
    from impl.model.nn.flash_mqat.flash_mqat_api import add_helper_functions, FlashMQATModel

    mconfig = get_llama7b_flash_config()
    os.environ["DLLM_CUDA_TMARK"] = "1"

    if rank < from_topo.world_size():
        with base.constants.model_scope(from_model_name):
            m1 = FlashMQATModel(mconfig, dtype=torch.float16, device="cuda")
            m1.instantiate()
            if check:
                m1.load_from_saved_flash_model("/lustre/aigc/llm/checkpoints/reparallelize_test/")
            if base.constants.pipe_parallel_world_size() > 1:
                engine1 = InferencePipelineEngine(m1)
            else:
                add_helper_functions(m1)
                engine1 = m1

        if profile_compile:
            profiler = get_pytorch_profiler(f"repara{rank}_m1_compile.json")
            profiler.start()
        print("building m1 reparallelization plan... 1...")
        tik = time.perf_counter()
        m1.build_reparallelization_plan(to_model_name, from_model_name, to_topo, from_topo, mconfig, pg_info)
        print(f"building m1 reparallelization plan... 1... time {time.perf_counter() - tik:.4f}")

        print("building m1 reparallelization plan... 2...")
        tik = time.perf_counter()
        m1.build_reparallelization_plan(from_model_name, to_model_name, from_topo, to_topo, mconfig, pg_info)
        print(f"building m1 reparallelization plan... 2... time {time.perf_counter() - tik:.4f}")
        if profile_compile:
            profiler.__exit__(None, None, None)
    else:
        m1 = None

    if rank >= world_size - to_topo.world_size():
        with base.constants.model_scope(to_model_name):
            m2 = FlashMQATModel(mconfig, dtype=torch.float16, device="cuda")
            if base.constants.pipe_parallel_world_size() > 1:
                engine2 = InferencePipelineEngine(m2)
            else:
                add_helper_functions(m2)
                engine2 = m2
        print("building m2 reparallelization plan... 1...")
        tik = time.perf_counter()
        m2.build_reparallelization_plan(from_model_name, to_model_name, from_topo, to_topo, mconfig, pg_info)
        print(f"building m2 reparallelization plan... 1... time {time.perf_counter() - tik:.4f}")

        tik = time.perf_counter()
        print("building m2 reparallelization plan... 2...")
        m2.build_reparallelization_plan(to_model_name, from_model_name, to_topo, from_topo, mconfig, pg_info)
        print(f"building m2 reparallelization plan... 2... time {time.perf_counter() - tik:.4f}")
    else:
        m2 = None

    for it in range(n_iterations):
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.synchronize()
        if it > 0:
            print("//////////// warmup finish, real runs start //////////")

        if it == n_iterations - 1 and profile:
            profiler = get_pytorch_profiler(f"repara{rank}.json")
            profiler.start()
        # from m1 to m2
        if m1 is not None:
            tik = time.perf_counter()
            res = m1.build_reparallelized_layers_async(from_model_name, to_model_name, from_topo, to_topo,
                                                       mconfig, pg_info)
            cpu_time = time.perf_counter() - tik
            torch.cuda.synchronize()
            print(f"param sync request time: {time.perf_counter() - tik:.4f}s, cpu time: {cpu_time:.4f}s")
        else:
            tik = time.perf_counter()
            res = m2.build_reparallelized_layers_async(from_model_name, to_model_name, from_topo, to_topo,
                                                       mconfig, pg_info)
            cpu_time = time.perf_counter() - tik
            torch.cuda.synchronize()
            print(f"param sync request time: {time.perf_counter() - tik:.4f}s, cpu time: {cpu_time:.4f}s")

        if m1 is not None:
            for k, p in m1.named_parameters():
                assert p.data.shape == torch.Size([0]), (k, p, it)
        if m2 is not None:
            for k, p in m2.named_parameters():
                assert p.data.shape == torch.Size([0]), (k, p, it)

        if m2 is not None:
            m2.patch_reparallelization(res[:2])

            with base.constants.model_scope(to_model_name):
                comm_volume = res[-1]
                dist.all_reduce(comm_volume, group=base.constants.parallelism_group())
                entry = fetch_latest_tmark()
                assert entry.type_ == CUDATimeMarkType.mem_layout
                mem_shift_time_ns = (
                    torch.tensor(entry.end_time - entry.start_time, dtype=torch.long, device="cuda") /
                    to_topo.world_size())
                dist.all_reduce(mem_shift_time_ns, group=base.constants.parallelism_group())

                if check:
                    torch.cuda.synchronize()
                    m3 = FlashMQATModel(mconfig, dtype=torch.float16, device="cuda")
                    m3.instantiate()
                    m3.load_from_saved_flash_model("/lustre/aigc/llm/checkpoints/reparallelize_test/")
                    assert len(m2.state_dict()) == len(m3.state_dict()) > 0
                    for k in m2.state_dict().keys():
                        v1 = m2.state_dict()[k]
                        v2 = m3.state_dict()[k]
                        assert torch.allclose(v1, v2), (k, v1, v2)

                with torch.no_grad():
                    packed_input_ids = torch.randint(
                        0,
                        mconfig.vocab_size,
                        (2**17 // to_topo.get_dim("data"),),
                        dtype=torch.long,
                        device="cuda",
                    )
                    bs = 2**17 // to_topo.get_dim("data") // 256 + 1
                    cu_seqlens = torch.linspace(0,
                                                256,
                                                2**17 // to_topo.get_dim("data") // 256 + 1,
                                                dtype=torch.int32,
                                                device="cuda")
                    seqlens_cpu = [256 for _ in range(bs)]
                    max_seqlen = 256

                    dist.barrier(group=base.constants.parallelism_group())
                    torch.cuda.synchronize()
                    tik = time.time_ns()
                    if isinstance(engine2, InferencePipelineEngine):
                        engine2.forward(seqlens_cpu=seqlens_cpu,
                                        packed_input_ids=packed_input_ids,
                                        cu_seqlens=cu_seqlens)
                    else:
                        engine2(packed_input_ids=packed_input_ids,
                                cu_seqlens=cu_seqlens,
                                max_seqlen=max_seqlen)
                    dist.barrier(group=base.constants.parallelism_group())
                    torch.cuda.synchronize()
                fwd_time_ns = (torch.tensor(time.time_ns() - tik, dtype=torch.long, device="cuda") /
                               to_topo.world_size())
                dist.all_reduce(fwd_time_ns, group=base.constants.parallelism_group())
                with open("memshift_cost.jsonl", "a") as f:
                    import json

                    d = dict(
                        from_mp_size=from_topo.get_dim("model"),
                        from_pp_size=from_topo.get_dim("pipe"),
                        from_dp_size=from_topo.get_dim("data"),
                        to_mp_size=to_topo.get_dim("model"),
                        to_pp_size=to_topo.get_dim("pipe"),
                        to_dp_size=to_topo.get_dim("data"),
                        mem_shift_time_ns=mem_shift_time_ns.item(),
                        fwd_time_ns=fwd_time_ns.item(),
                        comm_volume=comm_volume.item(),
                        world_size=world_size,
                    )
                    if (it == n_iterations - 1 and record_cost_to_file and dist.get_rank()
                            == dist.get_process_group_ranks(base.constants.parallelism_group())[0]):
                        f.write(json.dumps(d, ensure_ascii=False) + "\n")

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.synchronize()
        # convert m2 back to m1
        if m2 is not None:
            tik = time.perf_counter()
            res = m2.build_reparallelized_layers_async(to_model_name, from_model_name, to_topo, from_topo,
                                                       mconfig, pg_info)
            cpu_time = time.perf_counter() - tik
            torch.cuda.synchronize()
            print(f"param sync request time: {time.perf_counter() - tik:.4f}s, cpu time: {cpu_time:.4f}s")
        else:
            tik = time.perf_counter()
            res = m1.build_reparallelized_layers_async(to_model_name, from_model_name, to_topo, from_topo,
                                                       mconfig, pg_info)
            cpu_time = time.perf_counter() - tik
            torch.cuda.synchronize()
            print(f"param sync request time: {time.perf_counter() - tik:.4f}s, cpu time: {cpu_time:.4f}s")

        if m1 is not None:
            for k, p in m1.named_parameters():
                assert p.data.shape == torch.Size([0]), (k, p, it)
        if m2 is not None:
            for k, p in m2.named_parameters():
                assert p.data.shape == torch.Size([0]), (k, p, it)

        if m1 is not None:
            m1.patch_reparallelization(res[:2])
            with base.constants.model_scope(from_model_name):
                comm_volume = res[-1]
                dist.all_reduce(comm_volume, group=base.constants.parallelism_group())
                entry = fetch_latest_tmark()
                assert entry.type_ == CUDATimeMarkType.mem_layout
                mem_shift_time_ns = (
                    torch.tensor(entry.end_time - entry.start_time, dtype=torch.long, device="cuda") /
                    from_topo.world_size())
                dist.all_reduce(mem_shift_time_ns, group=base.constants.parallelism_group())

                if check:
                    torch.cuda.synchronize()
                    m4 = FlashMQATModel(mconfig, dtype=torch.float16, device="cuda")
                    m4.instantiate()
                    m4.load_from_saved_flash_model("/lustre/aigc/llm/checkpoints/reparallelize_test/")
                    assert len(m1.state_dict()) == len(m4.state_dict()) > 0
                    for k in m1.state_dict().keys():
                        v1 = m1.state_dict()[k]
                        v2 = m4.state_dict()[k]
                        assert torch.allclose(v1, v2), (k, v1, v2)

                with torch.no_grad():
                    packed_input_ids = torch.randint(
                        0,
                        mconfig.vocab_size,
                        (2**17 // to_topo.get_dim("data"),),
                        dtype=torch.long,
                        device="cuda",
                    )
                    bs = 2**17 // to_topo.get_dim("data") // 256 + 1
                    cu_seqlens = torch.linspace(0,
                                                256,
                                                2**17 // to_topo.get_dim("data") // 256 + 1,
                                                dtype=torch.int32,
                                                device="cuda")
                    seqlens_cpu = [256 for _ in range(bs)]
                    max_seqlen = 256

                    dist.barrier(group=base.constants.parallelism_group())
                    torch.cuda.synchronize()
                    tik = time.time_ns()
                    if isinstance(engine1, InferencePipelineEngine):
                        engine1.forward(seqlens_cpu=seqlens_cpu,
                                        packed_input_ids=packed_input_ids,
                                        cu_seqlens=cu_seqlens)
                    else:
                        engine1(packed_input_ids=packed_input_ids,
                                cu_seqlens=cu_seqlens,
                                max_seqlen=max_seqlen)
                    dist.barrier(group=base.constants.parallelism_group())
                    torch.cuda.synchronize()
                fwd_time_ns = (torch.tensor(time.time_ns() - tik, dtype=torch.long, device="cuda") /
                               from_topo.world_size())
                dist.all_reduce(fwd_time_ns, group=base.constants.parallelism_group())
                with open("memshift_cost.jsonl", "a") as f:
                    import json

                    d = dict(
                        from_mp_size=to_topo.get_dim("model"),
                        from_pp_size=to_topo.get_dim("pipe"),
                        from_dp_size=to_topo.get_dim("data"),
                        to_mp_size=from_topo.get_dim("model"),
                        to_pp_size=from_topo.get_dim("pipe"),
                        to_dp_size=from_topo.get_dim("data"),
                        mem_shift_time_ns=mem_shift_time_ns.item(),
                        fwd_time_ns=fwd_time_ns.item(),
                        comm_volume=comm_volume.item(),
                        world_size=world_size,
                    )
                    if (it == n_iterations - 1 and record_cost_to_file and dist.get_rank()
                            == dist.get_process_group_ranks(base.constants.parallelism_group())[0]):
                        f.write(json.dumps(d, ensure_ascii=False) + "\n")
        if it == n_iterations - 1 and profile:
            profiler.__exit__(None, None, None)


def setup_gpu(rank, world_size, barrier, model_topos, msid2mwid, param_sync_pairs):
    os.environ["DLLM_MODE"] = "LOCAL"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % 8)
    os.environ["GPU_DEVICES_ISOLATED"] = str(1)
    info = base.gpu_utils.setup_ddp(
        EXPR_NAME,
        TRIAL_NAME,
        rank,
        model_topos=model_topos,
        msid2mwid=msid2mwid,
        param_sync_pairs=param_sync_pairs,
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
    rank,
    world_size,
    barrier,
    from_topo,
    to_topo,
    err_queue,
    profile=False,
    check=False,
    n_iterations=2,
    profile_compile=False,
    record_cost_to_file=False,
):
    from_model_name = ModelName("actor", 0)
    to_model_name = ModelName("actor", 1)
    param_sync_pairs = [(from_model_name, to_model_name), (to_model_name, from_model_name)]
    model_topos = {from_model_name: from_topo, to_model_name: to_topo}
    msid2mwid = {}
    for i in range(from_topo.world_size()):
        msid2mwid[ModelShardID.from_parallelism_rank(from_model_name, from_topo, i)] = i
    for i in range(to_topo.world_size()):
        msid2mwid[ModelShardID.from_parallelism_rank(to_model_name, to_topo,
                                                     i)] = (i + world_size - to_topo.world_size())
    pg_info = setup_gpu(rank, world_size, barrier, model_topos, msid2mwid, param_sync_pairs)
    if rank < from_topo.world_size():
        init_global_constants(topo=from_topo, model_name=from_model_name, msid2mwid=msid2mwid)
    if rank >= world_size - to_topo.world_size():
        init_global_constants(topo=to_topo, model_name=to_model_name, msid2mwid=msid2mwid)

    from impl.model.nn.flash_mqat.flash_mqat_api import FlashMQATModel

    try:
        mconfig = get_llama7b_flash_config()
        torch.distributed.barrier()
        # if check and base.constants.has_model_name(from_model_name):
        #     with base.constants.model_scope(from_model_name):
        #         global_m = FlashMQATModel(mconfig, dtype=torch.float16, device="cuda")
        #         global_m.instantiate()
        #         # if os.path.exists("/lustre/aigc/llm/checkpoints/reparallelize_test/"):
        #         #     global_m.load_from_saved_flash_model("/lustre/aigc/llm/checkpoints/reparallelize_test/")
        #         # else:
        #         # torch.distributed.barrier(group=base.constants.parallelism_group())
        #         # os.system("rm -rf /lustre/aigc/llm/checkpoints/reparallelize_test/")
        #         torch.distributed.barrier(group=base.constants.parallelism_group())
        #         global_m.save("/lustre/aigc/llm/checkpoints/reparallelize_test/")

        torch.distributed.barrier()
        test_impl(
            rank,
            world_size,
            from_model_name,
            to_model_name,
            from_topo,
            to_topo,
            pg_info=pg_info,
            profile=profile,
            check=check,
            n_iterations=n_iterations,
            profile_compile=profile_compile,
            record_cost_to_file=record_cost_to_file,
        )
    except Exception as e:
        err_queue.put(1)
        raise e

    # print("====================")
    # test_impl(
    #     rank, world_size, from_model_name, to_model_name, from_topo, to_topo, pg_info=pg_info, profile=True
    # )

    # assert len(state_dict) == 0, (idx, state_dict.keys(), new_state_dict.keys())

    # layer_indices_before = {}
    # param_portion_before = {}
    # for i in range(8):
    #     if i < from_topo.world_size():
    #         coord = from_topo.get_coord(i)
    #         layer_indices_before[i] = list(
    #             range(
    #                 coord.pipe * n_layers // from_topo.get_dim("pipe"),
    #                 (coord.pipe + 1) * n_layers // from_topo.get_dim("pipe"),
    #             )
    #         )
    #         param_portion_before[i] = (
    #             coord.model * param_size // from_topo.get_dim("model"),
    #             (coord.model + 1) * param_size // from_topo.get_dim("model"),
    #         )
    #     else:
    #         layer_indices_before[i] = []
    #         param_portion_before[i] = (0, 0)

    # layer_indices_after = {}
    # param_portion_after = {}
    # for i in range(8):
    #     if i >= 8 - to_topo.world_size():
    #         coord = to_topo.get_coord(i - (8 - to_topo.world_size()))
    #         layer_indices_after[i] = list(
    #             range(
    #                 coord.pipe * n_layers // to_topo.get_dim("pipe"),
    #                 (coord.pipe + 1) * n_layers // to_topo.get_dim("pipe"),
    #             )
    #         )
    #         param_portion_after[i] = (
    #             coord.model * param_size // to_topo.get_dim("model"),
    #             (coord.model + 1) * param_size // to_topo.get_dim("model"),
    #         )
    #     else:
    #         layer_indices_after[i] = []
    #         param_portion_after[i] = (0, 0)

    # est_comm_volume = 0
    # for i in range(8):
    #     for layer_idx in layer_indices_after[i]:
    #         if layer_idx in layer_indices_before[i]:
    #             est_comm_volume += len(
    #                 set(range(param_portion_after[i][0], param_portion_after[i][1])).difference(
    #                     range(param_portion_before[i][0], param_portion_before[i][1])
    #                 )
    #             )
    #         else:
    #             est_comm_volume += param_portion_after[i][1] - param_portion_after[i][0]

    # comm_volume = torch.tensor(comm_volume, dtype=torch.int32, device="cuda")
    # torch.distributed.all_reduce(comm_volume)
    # assert comm_volume == est_comm_volume, (comm_volume, est_comm_volume, from_topo, to_topo)
    torch.distributed.barrier()
    if rank % 8 == 0:
        print("success!")


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

    for a, b in [(8, 8)]:
        if a == b:
            three_factors = decompose_to_three_factors(a)
            all_configs = []
            for i in range(len(three_factors)):
                for j in range(i):
                    all_configs.append((three_factors[i], three_factors[j]))
        else:
            all_configs = list(itertools.product(decompose_to_three_factors(a),
                                                 decompose_to_three_factors(b)))
        all_configs = list(filter(lambda x: x[0][1] % x[1][1] == 0 or x[1][1] % x[0][1] == 0, all_configs))
        random.shuffle(all_configs)
        print(f">>>>>>>>> running {len(all_configs)} configurations >>>>>>>")
        # for x1, x2 in all_configs:
        for x1, x2 in itertools.product([(4, 2, 1)], [(1, 1, 8)]):
            barrier = mp.Barrier(8)
            if args.node_idx == args.num_nodes - 1:
                clear_name_resolve()
            print(f"testing from {x1} to {x2}")

            from_topo = PipeModelDataParallelTopology(num_pp=x1[0], num_mp=x1[1], num_dp=x1[2])
            to_topo = PipeModelDataParallelTopology(num_pp=x2[0], num_mp=x2[1], num_dp=x2[2])
            procs = []
            for i in range(8):
                proc = mp.Process(
                    target=test,
                    args=(
                        i + args.node_idx * 8,
                        args.num_nodes * 8,
                        barrier,
                        from_topo,
                        to_topo,
                        err_queue,
                    ),
                    kwargs=dict(profile=True,
                                check=False,
                                n_iterations=3,
                                profile_compile=False,
                                record_cost_to_file=False),
                )
                procs.append(proc)
                proc.start()
            for proc in procs:
                proc.join()
            for i in range(8):
                try:
                    err_code = err_queue.get_nowait()
                    print("error!!!!!!!!!!!!")
                    exit(0)
                except queue.Empty:
                    pass
