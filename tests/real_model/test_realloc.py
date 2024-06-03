from typing import *
import argparse
import dataclasses
import itertools
import json
import multiprocessing as mp
import os
import pickle
import queue
import random
import subprocess
import time

import numpy as np
import pandas as pd
import pynvml
import torch
import torch.distributed as dist
import tqdm
import transformers

from reallm.api.core.model_api import MODEL_FAMILY_TO_PATH, ModelFamily, ModelName, ModelShardID
from reallm.api.core.system_api import ModelName, ModelShardID
from reallm.api.quickstart.model import REAL_MODEL_CONFIG_CONVERTER
from reallm.base.monitor import cuda_tmark, cuda_tmarked, CUDATimeMarkType, fetch_latest_tmark
from reallm.base.testing import (clear_name_resolve, get_pytorch_profiler, init_global_constants,
                                 pytorch_memory_burnin)
from reallm.base.topology import PipeModelDataParallelTopology
from reallm.scheduler.client import make as make_scheduer
import reallm.base.constants as constants
import reallm.base.gpu_utils as gpu_utils

EXPR_NAME = "test_reparallelize"
TRIAL_NAME = "test"

NODELIST = {
    0: "QH-com47",
    1: "QH-com47",
    2: "QH-com[47-48]",
    3: "QH-com[26,47-48]",
    4: "QH-com[25-26,47-48]",
    5: "QH-com[43-47]",
    6: "QH-com[43-48]",
    7: "QH-com[28,43-48]",
    8: "QH-com[27-28,43-48]",
    16: "QH-com[20-22,25-26,41-48]",
}


def get_n_gpus(main_model_size: int, case):
    default_ngpus = (8 if main_model_size == 7 else
                     16 if main_model_size == 13 else 32 if main_model_size == 34 else 64)
    return default_ngpus if case <= 1 else 2 * default_ngpus


def get_mem_shift_settings():
    with open("/lustre/meizy/res_df.pkl", "rb") as f:
        data = pickle.load(f)

    settings = []
    data: pd.DataFrame = data[data["mode"] == "s"]
    for actor_size, critic_size in itertools.product([7, 13, 34, 70], [7, 13, 34, 70]):
        seqlen = 896
        main_model_size = max(actor_size, critic_size)
        # HACK:
        if actor_size == 70 and critic_size == 70:
            continue
        if actor_size != critic_size and actor_size > 7 and critic_size > 7:
            continue
        if critic_size == 7:
            case = 0
        elif actor_size == 7:
            case = 1
        else:
            case = 2
        if case == 0:
            ref_size = actor_size
            rew_size = 7
        elif case == 1:
            ref_size = 7
            rew_size = critic_size
        else:
            assert actor_size == critic_size > 7
            ref_size = rew_size = actor_size
        n_gpus = get_n_gpus(main_model_size, case)
        bs = 2**17 // (seqlen + 128)
        df = data[(data["actor_model_size"] == actor_size)
                  & (data["critic_model_size"] == critic_size)
                  & (data["seqlen"] == seqlen)
                  & (data["n_nodes"] == n_gpus // 8)]
        assert len(df) == 1, len(df)
        logpath = df["log_path"].tolist()[0]
        with open(os.path.join(logpath, "device_mapping.pkl"), "rb") as f:
            device_mapping = pickle.load(f)
        actor_topos = []
        critic_topos = []
        for k, v in device_mapping.items():
            role = k.split("ModelName(role='")[1].split("'")[0]
            handle_name = k.split("@")[1]
            if role not in ["actor", "critic"]:
                continue
            topo = (
                v.train_eval_config.parallel.pipeline_parallel_size,
                v.train_eval_config.parallel.model_parallel_size,
                v.train_eval_config.parallel.data_parallel_size,
            )
            if role == "actor":
                actor_topos.append((topo, v.mapping))
            else:
                critic_topos.append((topo, v.mapping))
        assert len(actor_topos) == 2
        assert len(critic_topos) == 2
        if actor_topos[0][0] != actor_topos[1][0]:
            from_topo, from_mapping = actor_topos[0]
            to_topo, to_mapping = actor_topos[1]
            world_size = np.logical_or(from_mapping, to_mapping).sum()
            settings.append((actor_size, world_size, from_topo, to_topo))
        if critic_topos[0][0] != critic_topos[1][0]:
            from_topo, from_mapping = critic_topos[0]
            to_topo, to_mapping = critic_topos[1]
            world_size = np.logical_or(from_mapping, to_mapping).sum()
            settings.append((critic_size, world_size, from_topo, to_topo))

    new_settings = set()
    for s in settings:
        size, world_size, ft, tt = s
        if s in new_settings or (size, world_size, tt, ft) in new_settings:
            continue
        new_settings.add(s)
    return list(sorted(new_settings, key=lambda x: x[1]))


def test_impl(
    rank,
    world_size,
    model_size,
    from_model_name,
    to_model_name,
    from_topo,
    to_topo,
    pg_info,
    profile=False,
    check=True,
    n_iterations=1,
    profile_compile=False,
    dump_to_file=None,
):
    from_pp_mp_dp = (from_topo.get_dim("pipe"), from_topo.get_dim("model"), from_topo.get_dim("data"))
    to_pp_mp_dp = (to_topo.get_dim("pipe"), to_topo.get_dim("model"), to_topo.get_dim("data"))
    dump_key = str((model_size, world_size, from_pp_mp_dp, to_pp_mp_dp))

    assert not (profile and profile_compile)
    from reallm.impl.model.backend.pipe_inf import InferencePipelineEngine
    from reallm.impl.model.nn.real_llm_api import add_helper_functions, ReaLModel

    hf_model_type = "llama" if model_size != 34 else "codellama"
    hf_config = transformers.AutoConfig.from_pretrained(MODEL_FAMILY_TO_PATH[ModelFamily(
        hf_model_type, model_size, False)])
    mconfig = REAL_MODEL_CONFIG_CONVERTER[hf_model_type](hf_config)
    os.environ["REAL_CUDA_TMARK"] = "1"

    if rank < from_topo.world_size():
        with constants.model_scope(from_model_name):
            m1 = ReaLModel(mconfig, dtype=torch.float16, device="cuda")
            m1.instantiate()
            if check:
                m1.load_from_saved_real_model("/lustre/aigc/llm/checkpoints/reparallelize_test/")
            if constants.pipe_parallel_world_size() > 1:
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
        with constants.model_scope(to_model_name):
            m2 = ReaLModel(mconfig, dtype=torch.float16, device="cuda")
            if constants.pipe_parallel_world_size() > 1:
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

            with constants.model_scope(to_model_name):
                comm_volume = res[-1]
                dist.all_reduce(comm_volume, group=constants.parallelism_group())
                entry = fetch_latest_tmark()
                assert entry.type_ == CUDATimeMarkType.mem_layout
                mem_shift_time_ns = (
                    torch.tensor(entry.end_time - entry.start_time, dtype=torch.long, device="cuda") /
                    to_topo.world_size())
                dist.all_reduce(mem_shift_time_ns, group=constants.parallelism_group())

                if check:
                    torch.cuda.synchronize()
                    m3 = ReaLModel(mconfig, dtype=torch.float16, device="cuda")
                    m3.instantiate()
                    m3.load_from_saved_real_model("/lustre/aigc/llm/checkpoints/reparallelize_test/")
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

                    dist.barrier(group=constants.parallelism_group())
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
                    dist.barrier(group=constants.parallelism_group())
                    torch.cuda.synchronize()
                fwd_time_ns = (torch.tensor(time.time_ns() - tik, dtype=torch.long, device="cuda") /
                               to_topo.world_size())
                dist.all_reduce(fwd_time_ns, group=constants.parallelism_group())

                d = dict(
                    mem_shift_time_ns=mem_shift_time_ns.item(),
                    fwd_time_ns=fwd_time_ns.item(),
                    comm_volume=comm_volume.item(),
                )
                if (it == n_iterations - 1 and dump_to_file is not None and dist.get_rank()
                        == dist.get_process_group_ranks(constants.parallelism_group())[0]):
                    assert dump_to_file.endswith("json"), dump_to_file
                    with open(dump_to_file, "r") as f:
                        _dump = json.load(f)
                    assert dump_key not in _dump, _dump.keys()
                    _dump[dump_key] = d
                    with open(dump_to_file, "w") as f:
                        json.dump(_dump, f, ensure_ascii=False)

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
            with constants.model_scope(from_model_name):
                comm_volume = res[-1]
                dist.all_reduce(comm_volume, group=constants.parallelism_group())
                entry = fetch_latest_tmark()
                assert entry.type_ == CUDATimeMarkType.mem_layout
                mem_shift_time_ns = (
                    torch.tensor(entry.end_time - entry.start_time, dtype=torch.long, device="cuda") /
                    from_topo.world_size())
                dist.all_reduce(mem_shift_time_ns, group=constants.parallelism_group())

                if check:
                    torch.cuda.synchronize()
                    m4 = ReaLModel(mconfig, dtype=torch.float16, device="cuda")
                    m4.instantiate()
                    m4.load_from_saved_real_model("/lustre/aigc/llm/checkpoints/reparallelize_test/")
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

                    dist.barrier(group=constants.parallelism_group())
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
                    dist.barrier(group=constants.parallelism_group())
                    torch.cuda.synchronize()
                fwd_time_ns = (torch.tensor(time.time_ns() - tik, dtype=torch.long, device="cuda") /
                               from_topo.world_size())
                dist.all_reduce(fwd_time_ns, group=constants.parallelism_group())
                d = dict(
                    mem_shift_time_ns=mem_shift_time_ns.item(),
                    fwd_time_ns=fwd_time_ns.item(),
                    comm_volume=comm_volume.item(),
                )
                if (it == n_iterations - 1 and dump_to_file is not None and dist.get_rank()
                        == dist.get_process_group_ranks(constants.parallelism_group())[0]):
                    assert dump_to_file.endswith("json"), dump_to_file
                    with open(dump_to_file, "r") as f:
                        _dump = json.load(f)
                    assert dump_key in _dump, _dump.keys()
                    _dump[dump_key] = {k: v1 + v2 for (k, v1), v2 in zip(d.items(), _dump[dump_key].values())}
                    with open(dump_to_file, "w") as f:
                        json.dump(_dump, f, ensure_ascii=False)
        if it == n_iterations - 1 and profile:
            profiler.__exit__(None, None, None)


def setup_gpu(rank, world_size, model_topos, msid2mwid, param_realloc_pairs):
    os.environ["REAL_MODE"] = "LOCAL"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % 8)
    os.environ["GPU_DEVICES_ISOLATED"] = str(1)
    info = reallm.base.gpu_utils.setup_ddp(
        EXPR_NAME,
        TRIAL_NAME,
        rank,
        model_topos=model_topos,
        msid2mwid=msid2mwid,
        param_realloc_pairs=param_realloc_pairs,
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


def test(args):
    rank = args.rank
    world_size = args.world_size
    from_topo = PipeModelDataParallelTopology(num_pp=args.from_pp, num_mp=args.from_mp, num_dp=args.from_dp)
    to_topo = PipeModelDataParallelTopology(num_pp=args.to_pp, num_mp=args.to_mp, num_dp=args.to_dp)

    from_model_name = ModelName("actor", 0)
    to_model_name = ModelName("actor", 1)
    param_realloc_pairs = [(from_model_name, to_model_name), (to_model_name, from_model_name)]
    model_topos = {from_model_name: from_topo, to_model_name: to_topo}
    msid2mwid = {}
    for i in range(from_topo.world_size()):
        msid2mwid[ModelShardID.from_parallelism_rank(from_model_name, from_topo, i)] = i
    for i in range(to_topo.world_size()):
        msid2mwid[ModelShardID.from_parallelism_rank(to_model_name, to_topo,
                                                     i)] = (i + world_size - to_topo.world_size())
    pg_info = setup_gpu(rank, world_size, model_topos, msid2mwid, param_realloc_pairs)
    if rank < from_topo.world_size():
        init_global_constants(topo=from_topo, model_name=from_model_name, msid2mwid=msid2mwid)
    if rank >= world_size - to_topo.world_size():
        init_global_constants(topo=to_topo, model_name=to_model_name, msid2mwid=msid2mwid)

    hf_model_type = "llama" if args.model_size != 34 else "codellama"
    hf_config = transformers.AutoConfig.from_pretrained(MODEL_FAMILY_TO_PATH[ModelFamily(
        hf_model_type, args.model_size, False)])
    mconfig = REAL_MODEL_CONFIG_CONVERTER[hf_model_type](hf_config)
    torch.distributed.barrier()
    # if check and constants.has_model_name(from_model_name):
    #     with constants.model_scope(from_model_name):
    #         global_m = ReaLModel(mconfig, dtype=torch.float16, device="cuda")
    #         global_m.instantiate()
    #         # if os.path.exists("/lustre/aigc/llm/checkpoints/reparallelize_test/"):
    #         #     global_m.load_from_saved_real_model("/lustre/aigc/llm/checkpoints/reparallelize_test/")
    #         # else:
    #         # torch.distributed.barrier(group=constants.parallelism_group())
    #         # os.system("rm -rf /lustre/aigc/llm/checkpoints/reparallelize_test/")
    #         torch.distributed.barrier(group=constants.parallelism_group())
    #         global_m.save("/lustre/aigc/llm/checkpoints/reparallelize_test/")

    torch.distributed.barrier()
    test_impl(
        rank,
        world_size,
        args.model_size,
        from_model_name,
        to_model_name,
        from_topo,
        to_topo,
        pg_info=pg_info,
        profile=False,
        check=False,
        n_iterations=2,
        profile_compile=False,
        dump_to_file=args.dump_to_file,
    )

    torch.distributed.barrier()
    if rank % 8 == 0:
        print("success!")
    if rank == 0:
        clear_name_resolve()
    torch.distributed.barrier()


def decompose_to_three_factors(n: int):
    factors = []
    for i in range(1, int(n**(1 / 2)) + 1):
        if n % i == 0:
            for j in range(i, int((n // i)**(1 / 2)) + 1):
                if (n // i) % j == 0:
                    k = (n // i) // j
                    factors += list(set(itertools.permutations([i, j, k])))
    return factors


def launch(args):
    settings = get_mem_shift_settings()
    for s in settings:
        assert isinstance(s, tuple) and len(s) == 4, settings
        assert isinstance(s[-1], tuple) and isinstance(s[-2], tuple), settings
        assert s[0] in [7, 13, 34, 70], settings
        assert s[1] >= np.prod(s[-1]) and s[1] >= np.prod(s[-2]), settings
    print(settings)

    dump_file = "/lustre/fw/sosp24/reallocation-cost-exp.json"
    if os.path.exists(dump_file):
        with open(dump_file, "r") as f:
            _dump = json.load(f)
        settings = list(filter(lambda x: str(x) not in _dump, settings))

    for config_id, (model_size, world_size, from_pp_mp_dp, to_pp_mp_dp) in enumerate(tqdm.tqdm(settings)):
        # for config_id, (x1, x2) in enumerate(itertools.product([(4, 2, 1)], [(1, 1, 8)])):

        clear_name_resolve(EXPR_NAME, TRIAL_NAME)
        print(f"testing from {from_pp_mp_dp} to {to_pp_mp_dp}...")

        cmd = [
            "python3",
            "test_reparallelize.py",
            "start_testing",
            "--rank",
            "%t",
            "--world_size",
            world_size,
            "--model_size",
            model_size,
            "--from_mp",
            from_pp_mp_dp[1],
            "--from_pp",
            from_pp_mp_dp[0],
            "--from_dp",
            from_pp_mp_dp[2],
            "--to_dp",
            to_pp_mp_dp[2],
            "--to_mp",
            to_pp_mp_dp[1],
            "--to_pp",
            to_pp_mp_dp[0],
            "--dump_to_file",
            dump_file,
        ]
        cmd = " ".join([str(x) for x in cmd])
        os.makedirs(os.path.join(constants.LOG_ROOT, EXPR_NAME, TRIAL_NAME), exist_ok=True)
        multiprog_path = os.path.join(constants.LOG_ROOT, EXPR_NAME, TRIAL_NAME, "test.multiprog")
        with open(multiprog_path, "w") as f:
            f.write(f"0-{world_size - 1} {cmd}\n")
        srun_flags = [
            f"--ntasks={world_size}",
            f"--nodes={max(1, world_size // 8)}",
            f"--cpus-per-task=1",
            f"--gpus-per-task=tesla:1",
            f"--mem-per-cpu={100000}",
            "--job-name=test-reparallelize",
            f"--container-image=llm/llm-gpu",
            f"--nodelist={NODELIST[world_size // 8]}",
            f"--container-mounts=/lustre:/lustre,/dev/infiniband:/dev/infiniband,/sys/class/infiniband_verbs:/sys/class/infiniband_verbs",
            f"--container-mount-home",
            f"--container-workdir={os.getcwd()}",
        ]
        srun_cmd = f"srun -l --multi-prog {' '.join(srun_flags)} {multiprog_path}"
        print(f"Running the following cmd:\n{srun_cmd}\n")
        try:
            subprocess.run(srun_cmd, shell=True)
        except KeyboardInterrupt:
            os.system("pkill -9 srun")
            print("KeyboardInterrupt, exiting...")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", help="sub-command help")
    subparsers.required = True

    subparser = subparsers.add_parser("launch")
    subparser.set_defaults(func=launch)

    subparser = subparsers.add_parser("start_testing")
    subparser.add_argument("--rank", "-r", type=int, required=True)
    subparser.add_argument("--world_size", "-w", type=int, required=True)
    subparser.add_argument("--model_size", type=int, required=True)
    subparser.add_argument("--from_mp", type=int, required=True)
    subparser.add_argument("--from_pp", type=int, required=True)
    subparser.add_argument("--from_dp", type=int, required=True)
    subparser.add_argument("--to_dp", type=int, required=True)
    subparser.add_argument("--to_mp", type=int, required=True)
    subparser.add_argument("--to_pp", type=int, required=True)
    subparser.add_argument("--dump_to_file", type=str, default=None)
    subparser.set_defaults(func=test)

    args = parser.parse_args()

    args.func(args)
