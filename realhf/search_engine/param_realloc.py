import dataclasses
import itertools
import json
import time
from collections import defaultdict
from typing import *

import torch
import torch.distributed

import realhf.api.core.system_api as system_api
import realhf.base.constants as constants
import realhf.base.topology as topology
from realhf.api.core.config import ModelFamily, ModelName
from realhf.api.core.model_api import ReaLModelConfig
from realhf.base.topology import decompose_to_three_factors


def bcast_cost(
    param_size: float, bw: float, src: int, dsts: List[int], n_nodes_per_gpu=8
):
    src_node = src // n_nodes_per_gpu
    dst_nodes = [dst // n_nodes_per_gpu for dst in dsts]
    if src_node == dst_nodes[0] and all(
        dst_node == dst_nodes[0] for dst_node in dst_nodes
    ):
        return param_size * 2 * 8 / (1800 * 1024**3)
        # return 0.0
    else:
        # param size is in float16, bw is in Gbps
        return param_size * 2 * 8 / (bw * 1024**3) * len(set(dst_nodes))


def compute_cost(
    world_size: int,
    from_model_name: ModelName,
    to_model_name: ModelName,
    from_topo: topology.PipeModelDataParallelTopology,
    to_topo: topology.PipeModelDataParallelTopology,
    model_config: ReaLModelConfig,
    bw: float,  # Gbps
    set_interval_cost: float,
) -> int:
    from realhf.impl.model.comm.param_realloc import (
        ParamReallocInfo,
        ReparallelizeReceiverStep,
        ReparallelizeSenderStep,
        _create_param_realloc_groups,
        _derive_reparallelize_comm_plan,
    )

    param_sync_groups = {}
    param_sync_src_ranks = {}
    param_sync_dst_ranks = {}
    msid2mwid = {}
    for i in range(from_topo.world_size()):
        msid2mwid[
            system_api.ModelShardID.from_parallelism_rank(from_model_name, from_topo, i)
        ] = i
    for i in range(to_topo.world_size()):
        msid2mwid[
            system_api.ModelShardID.from_parallelism_rank(to_model_name, to_topo, i)
        ] = (i + world_size - to_topo.world_size())
    _create_param_realloc_groups(
        from_topo,
        to_topo,
        from_model_name,
        to_model_name,
        msid2mwid,
        param_sync_groups,
        param_sync_src_ranks,
        param_sync_dst_ranks,
    )
    pg_info = ParamReallocInfo(
        param_sync_groups,
        param_sync_src_ranks,
        param_sync_dst_ranks,
    )
    comm_plan = _derive_reparallelize_comm_plan(
        from_model_name,
        to_model_name,
        from_topo,
        to_topo,
        model_config,
        model_config,
        pg_info,
    )

    # Run boradcast!
    max_cost = max_comm_volume = max_bcast_cnt = 0
    for _rank in range(world_size):
        cost = comm_volume = bcast_cnt = 0
        for step in comm_plan:
            if isinstance(step, ReparallelizeReceiverStep) and step.rank == _rank:
                if step.rank != step.src:
                    cost += bcast_cost(step.param_size, bw, step.src, step.dst_ranks)
                    comm_volume += step.param_size
                    bcast_cnt += 1
                cost += set_interval_cost
            if isinstance(step, ReparallelizeSenderStep) and step.rank == _rank:
                if step.group is not None:
                    cost += bcast_cost(step.param_size, bw, step.rank, step.dst_ranks)
                    bcast_cnt += 1
        max_cost = max(max_cost, cost)
        max_comm_volume = max(max_comm_volume, comm_volume)
        max_bcast_cnt = max(max_bcast_cnt, bcast_cnt)

    return max_cost


def dump_table(
    n_nodes: int,
    model_family: ModelFamily,
    model_path: str,
    rank: int = 0,
    parallel: int = 1,
):
    from_model_name = ModelName("actor", 0)
    to_model_name = ModelName("actor", 1)

    def hash_tuple_into_str(t) -> str:
        return ",".join([str(i) for i in t])

    import tqdm

    res = {}
    device_mesh_sizes = [4] + [8 * i for i in range(1, n_nodes + 1)]
    space = list(itertools.product(device_mesh_sizes, device_mesh_sizes))
    sub_space = space[rank::parallel]
    # for a, b in set(small_spaces + large_spaces):
    for a, b in sub_space:
        mtik = time.perf_counter()
        all_configs = list(
            itertools.product(
                decompose_to_three_factors(a), decompose_to_three_factors(b)
            )
        )
        all_configs = list(filter(lambda x: x[0][1] <= 8 and x[1][1] <= 8, all_configs))
        all_configs = list(filter(lambda x: x[0][2] <= 8 and x[1][2] <= 8, all_configs))
        all_configs = list(
            filter(
                lambda x: x[0][1] in [1, 2, 4, 8] and x[1][1] in [1, 2, 4, 8],
                all_configs,
            )
        )
        all_configs = list(
            filter(lambda x: x[0][0] <= 16 and x[1][0] <= 16, all_configs)
        )
        all_configs = list(
            filter(
                lambda x: x[0][1] % x[1][1] == 0 or x[1][1] % x[0][1] == 0,
                all_configs,
            )
        )
        for config_id, (from_pp_mp_dp, to_pp_mp_dp) in tqdm.tqdm(
            enumerate(all_configs)
        ):
            world_size = max(a, b)

            from_topo = topology.PipeModelDataParallelTopology(
                *from_pp_mp_dp, False, False
            )
            to_topo = topology.PipeModelDataParallelTopology(*to_pp_mp_dp, False, False)
            assert world_size >= from_topo.world_size()
            assert world_size >= to_topo.world_size()

            from realhf.search_engine.utils import load_model_config

            mconfig = load_model_config(model_family._class, model_path)

            cost = compute_cost(
                world_size,
                from_model_name,
                to_model_name,
                from_topo,
                to_topo,
                mconfig,
                bw=200.0,
                set_interval_cost=0.03,
            )
            res[
                hash_tuple_into_str((model_family.size, *from_pp_mp_dp, *to_pp_mp_dp))
            ] = int(cost * 1000 * 1000)
        print(
            f"Time for model size {model_family.size} {a} -> {b} {rank}/{parallel}: "
            f"{time.perf_counter() - mtik:.4f}, num res entries {len(res)}"
        )

    print(f"Rank {rank} of model {model_family} finished, res size {len(res)}.")

    import os
    import pickle

    dump_path = os.path.join(constants.PROFILER_CACHE_PATH, "param_realloc")
    fn = f"prtc_{model_family}_n{n_nodes}_{rank}_{parallel}.pkl"
    if not os.path.exists(dump_path):
        os.makedirs(dump_path, exist_ok=True)
    with open(os.path.join(dump_path, fn), "wb") as f:
        pickle.dump(res, f)
    print(f"dumped table with {len(res)} entries to {model_family}-{rank}-{parallel}.")


def dump_table_parallel(
    n_nodes: int,
    model_family_to_path: Dict[ModelFamily, str],
    parallel: int = 4,
):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    rq = mp.Queue()
    ps = []
    for model_family, model_path in model_family_to_path.items():
        for rank in range(parallel):
            ps.append(
                mp.Process(
                    target=dump_table,
                    args=(n_nodes, model_family, model_path, rank, parallel),
                )
            )

    for p in ps:
        p.start()

    for p in ps:
        p.join()


def merge_tables(
    n_nodes: int,
    model_family_to_path: Dict[ModelFamily, str],
    parallel: int = 4,
):
    import os
    import pickle

    res_path = os.path.join(constants.PROFILER_CACHE_PATH, "param_realloc")

    for model_family in model_family_to_path.keys():
        prefix = f"prtc_{model_family}_n{n_nodes}"

        r = {}
        counter = 0
        for fn in os.listdir(res_path):
            if fn.endswith(".pkl") and fn.startswith(prefix):
                counter += 1
                path = os.path.join(res_path, fn)
                with open(path, "rb") as f:
                    r.update(pickle.load(f))
                os.remove(path)
        if counter < parallel:
            raise RuntimeError(
                "missing sub-tables, probably some sub-processes failed "
                "during param realloc time cost estimation."
            )
        with open(os.path.join(res_path, f"{prefix}.pkl"), "wb") as f:
            pickle.dump(r, f)
        print(f"merged parallel tables into {prefix}.pkl, total entries {len(r)}")


def estimate_param_realloc_time_cost(
    n_nodes: int,
    model_family_to_path: Dict[ModelFamily, str],
    parallel: int = 4,
):
    dump_table_parallel(n_nodes, model_family_to_path, parallel)
    merge_tables(n_nodes, model_family_to_path, parallel)
