from typing import *
import torch.distributed as dist
import torch
import random
import multiprocessing as mp
import dataclasses
import os
import time

import torch.distributed
from api.config.config_system import ModelShardID, ModelName
from base.topology import PipeModelDataParallelTopology
import itertools
import queue
from tests.utils import *

import base.constants
from api.config.config_base import ModelName, ModelShardID
import base.gpu_utils


def test_impl(rank, world_size, from_model_name, to_model_name, from_topo, to_topo, pg_info, profile=False):
    from impl.model.nn.flash_mqat.flash_mqat_api import FlashMQATModel, add_helper_functions
    from impl.model.backend.pipe_inf import InferencePipelineEngine

    mconfig = get_llama7b_flash_config()
    packed_input_ids = torch.randint(0, mconfig.vocab_size, (32 * 256,), dtype=torch.long, device="cuda")
    cu_seqlens = torch.linspace(0, 32 * 256, 33, dtype=torch.int32, device="cuda")
    seqlens_cpu = [256 for _ in range(32)]
    max_seqlen = 256

    if rank < from_topo.world_size():
        with base.constants.model_scope(from_model_name):
            m1 = FlashMQATModel(mconfig, dtype=torch.float16, device="cuda")
            m1.instantiate()
            m1.load_from_saved_flash_model("/tmp/reparallelize_test/")
            if base.constants.pipe_parallel_world_size() > 1:
                engine1 = InferencePipelineEngine(m1)
            else:
                add_helper_functions(m1)
                engine1 = m1
        m1.build_reparallelization_plan(to_model_name, from_model_name, to_topo, from_topo, mconfig, pg_info)
        m1.build_reparallelization_plan(from_model_name, to_model_name, from_topo, to_topo, mconfig, pg_info)
    else:
        m1 = None

    if rank >= 8 - to_topo.world_size():
        with base.constants.model_scope(to_model_name):
            m2 = FlashMQATModel(mconfig, dtype=torch.float16, device="cuda")
            if base.constants.pipe_parallel_world_size() > 1:
                engine2 = InferencePipelineEngine(m2)
            else:
                add_helper_functions(m2)
                engine2 = m2
        m2.build_reparallelization_plan(from_model_name, to_model_name, from_topo, to_topo, mconfig, pg_info)
        m2.build_reparallelization_plan(to_model_name, from_model_name, to_topo, from_topo, mconfig, pg_info)
    else:
        m2 = None

    if profile:
        profiler = get_pytorch_profiler(f"repara{rank}.json")
        profiler.start()

    for it in range(3):
        # from m1 to m2
        if m1 is not None:
            tik = time.perf_counter()
            res = m1.build_reparallelized_layers_async(
                from_model_name, to_model_name, from_topo, to_topo, mconfig, pg_info
            )
            print(f"param sync request cpu time: {time.perf_counter() - tik:.2f}s")
        else:
            tik = time.perf_counter()
            res = m2.build_reparallelized_layers_async(
                from_model_name, to_model_name, from_topo, to_topo, mconfig, pg_info
            )
            print(f"param sync request cpu time: {time.perf_counter() - tik:.2f}s")

        if m1 is not None:
            for k, p in m1.named_parameters():
                assert p.data.shape == torch.Size([0]), (k, p, it)
        if m2 is not None:
            for k, p in m2.named_parameters():
                assert p.data.shape == torch.Size([0]), (k, p, it)

        if m2 is not None:
            m2.patch_reparallelization(res)

            with base.constants.model_scope(to_model_name):

                torch.cuda.synchronize()
                m3 = FlashMQATModel(mconfig, dtype=torch.float16, device="cuda")
                m3.instantiate()
                m3.load_from_saved_flash_model("/tmp/reparallelize_test/")
                assert len(m2.state_dict()) == len(m3.state_dict()) > 0
                for k in m2.state_dict().keys():
                    v1 = m2.state_dict()[k]
                    v2 = m3.state_dict()[k]
                    assert torch.allclose(v1, v2), (k, v1, v2)

                if isinstance(engine2, InferencePipelineEngine):
                    engine2.forward(
                        seqlens_cpu=seqlens_cpu, packed_input_ids=packed_input_ids, cu_seqlens=cu_seqlens
                    )
                else:
                    engine2(packed_input_ids=packed_input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        # convert m2 back to m1
        if m2 is not None:
            tik = time.perf_counter()
            res = m2.build_reparallelized_layers_async(
                to_model_name, from_model_name, to_topo, from_topo, mconfig, pg_info
            )
            print(f"param sync request cpu time: {time.perf_counter() - tik:.2f}s")
        else:
            tik = time.perf_counter()
            res = m1.build_reparallelized_layers_async(
                to_model_name, from_model_name, to_topo, from_topo, mconfig, pg_info
            )
            print(f"param sync request cpu time: {time.perf_counter() - tik:.2f}s")

        if m1 is not None:
            for k, p in m1.named_parameters():
                assert p.data.shape == torch.Size([0]), (k, p, it)
        if m2 is not None:
            for k, p in m2.named_parameters():
                assert p.data.shape == torch.Size([0]), (k, p, it)

        if m1 is not None:
            m1.patch_reparallelization(res)
            with base.constants.model_scope(from_model_name):
                torch.cuda.synchronize()
                m4 = FlashMQATModel(mconfig, dtype=torch.float16, device="cuda")
                m4.instantiate()
                m4.load_from_saved_flash_model("/tmp/reparallelize_test/")
                assert len(m1.state_dict()) == len(m4.state_dict()) > 0
                for k in m1.state_dict().keys():
                    v1 = m1.state_dict()[k]
                    v2 = m4.state_dict()[k]
                    assert torch.allclose(v1, v2), (k, v1, v2)

                if isinstance(engine1, InferencePipelineEngine):
                    engine1.forward(
                        seqlens_cpu=seqlens_cpu, packed_input_ids=packed_input_ids, cu_seqlens=cu_seqlens
                    )
                else:
                    engine1(packed_input_ids=packed_input_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
    if profile:
        profiler.__exit__(None, None, None)


def setup_gpu(rank, world_size, barrier, model_topos, msid2mwid, param_sync_pairs):
    os.environ["DLLM_MODE"] = "LOCAL"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    barrier.wait()
    base.gpu_utils.isolate_cuda_device(WORKER_TYPE, rank, world_size, EXPR_NAME, TRIAL_NAME)
    barrier.wait()
    # print(f"rank {rank} isolated cuda device")
    base.gpu_utils.reveal_ddp_identity(EXPR_NAME, TRIAL_NAME, rank)
    # print(f"rank {rank} revealed ddp identity")
    barrier.wait()
    info = base.gpu_utils.setup_ddp(
        EXPR_NAME,
        TRIAL_NAME,
        rank,
        model_topos=model_topos,
        msid2mwid=msid2mwid,
        param_sync_pairs=param_sync_pairs,
    )
    world_size = info.world_size
    # print(f"rank {rank} setup ddp")
    import deepspeed

    deepspeed.init_distributed()
    # print(f"rank {rank} setup deepspeed")
    pynvml.nvmlInit()
    pytorch_memory_burnin(rank)
    return info


def test(rank, world_size, barrier, from_topo, to_topo, err_queue):
    from_model_name = ModelName("actor", 0)
    to_model_name = ModelName("actor", 1)
    param_sync_pairs = [(from_model_name, to_model_name), (to_model_name, from_model_name)]
    model_topos = {from_model_name: from_topo, to_model_name: to_topo}
    msid2mwid = {}
    for i in range(from_topo.world_size()):
        msid2mwid[ModelShardID.from_parallelism_rank(from_model_name, from_topo, i)] = i
    for i in range(to_topo.world_size()):
        msid2mwid[ModelShardID.from_parallelism_rank(to_model_name, to_topo, i)] = (
            i + 8 - to_topo.world_size()
        )
    pg_info = setup_gpu(rank, world_size, barrier, model_topos, msid2mwid, param_sync_pairs)
    init_global_constants(topo=from_topo, model_name=from_model_name, msid2mwid=msid2mwid)
    init_global_constants(topo=to_topo, model_name=to_model_name, msid2mwid=msid2mwid)

    from impl.model.nn.flash_mqat.flash_mqat_api import FlashMQATModel

    mconfig = get_llama7b_flash_config()
    with base.constants.model_scope(from_model_name):
        global_m = FlashMQATModel(mconfig, dtype=torch.float16, device="cuda")
        global_m.instantiate()
        if os.path.exists("/tmp/reparallelize_test/"):
            global_m.load_from_saved_flash_model("/tmp/reparallelize_test/")
        else:
            global_m.save("/tmp/reparallelize_test/")

    try:
        test_impl(rank, world_size, from_model_name, to_model_name, from_topo, to_topo, pg_info=pg_info)
    except Exception:
        err_queue.put(1)
        raise

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
    if rank == 0:
        print("success!")


def decompose_to_three_factors(n: int):
    factors = []
    for i in range(1, int(n ** (1 / 2)) + 1):
        if n % i == 0:
            for j in range(i, int((n // i) ** (1 / 2)) + 1):
                if (n // i) % j == 0:
                    k = (n // i) // j
                    factors += list(set(itertools.permutations([i, j, k])))
    return factors


if __name__ == "__main__":
    err_queue = mp.Queue(8)
    for a, b in [(8, 8)]:
        for x1, x2 in itertools.product(decompose_to_three_factors(a), decompose_to_three_factors(b)):
            # for x1, x2 in itertools.product([(1, 4, 2)], [(2, 2, 2)]):
            if not (x1[1] % x2[1] == 0 or x2[1] % x1[1] == 0):
                continue
            barrier = mp.Barrier(8)
            clear_name_resolve()
            print(f"testing from {x1} to {x2}")
            from_topo = PipeModelDataParallelTopology(num_pp=x1[0], num_mp=x1[1], num_dp=x1[2])
            to_topo = PipeModelDataParallelTopology(num_pp=x2[0], num_mp=x2[1], num_dp=x2[2])
            procs = []
            for i in range(8):
                proc = mp.Process(target=test, args=(i, 8, barrier, from_topo, to_topo, err_queue))
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
