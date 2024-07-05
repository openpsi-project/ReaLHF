import argparse
import functools
import json
import os
import pickle
import pprint
import re
from typing import Any, Dict, List, Literal, Optional

import numpy as np

try:
    import realhf._C.mdm_search as mdm_search
except ModuleNotFoundError:
    mdm_search = None

import realhf.base.constants as constants
from realhf.api.core.config import ModelInterfaceType
from realhf.api.core.dfg import MFCDef
from realhf.api.quickstart.device_mesh import DeviceMesh, RPCAllocation
from realhf.api.quickstart.model import ModelTrainEvalConfig, ParallelismConfig
from realhf.api.quickstart.search import RPCExecution


def search_rpc_allocations(
    device_mesh: DeviceMesh,
    rpcs: List[MFCDef],
    num_gen_tokens: int = 256,
    n_ppo_minibatches: int = 1,
    seq_len: int = 256,
    gradient_checkpointing: bool = True,
    use_cache: bool = False,
) -> List[RPCAllocation]:
    from realhf.search_engine.enumerate import build_graph
    from realhf.search_engine.estimate import get_param_realloc_stats

    from_file = os.environ.get("REAL_IS_REMOTE", "0") == "1"
    dump_dir = os.path.join(
        constants.LOG_ROOT,
        constants.experiment_name(),
        constants.trial_name(),
        "device_mapping.pkl",
    )
    log_dir = os.path.join(
        constants.LOG_ROOT,
        constants.experiment_name(),
        constants.trial_name(),
        "device_mapping",
    )
    rs_dir = os.path.join(
        constants.LOG_ROOT,
        constants.experiment_name(),
        constants.trial_name(),
        "raw_search_result",
    )
    rpc_exe_dir = os.path.join(
        constants.LOG_ROOT,
        constants.experiment_name(),
        constants.trial_name(),
        "rpc_exe_info",
    )

    if from_file or (use_cache and os.path.exists(dump_dir)):
        with open(dump_dir, "r") as f:
            s = json.load(f)
            rpc_allocs = [RPCAllocation.from_dict(d) for d in s]
        return rpc_allocs
    else:
        os.makedirs(os.path.dirname(dump_dir), exist_ok=True)

    n_nodes = device_mesh.n_nodes
    table = {}
    for rpc in rpcs:
        print(f"Getting param realloc stats for {rpc.model_type} at {rpc.model_path}")
        t = get_param_realloc_stats(rpc.model_type, rpc.model_path, n_nodes, True)
        table.update(t)

    rpc_exe_list = make_rpc_exe_list(
        rpcs,
        device_mesh,
        num_gen_tokens=num_gen_tokens,
        n_ppo_minibatches=n_ppo_minibatches,
        seq_len=seq_len,
        gradient_checkpointing=gradient_checkpointing,
        log_dir=rpc_exe_dir,
        if_print=False,
    )
    graph = build_graph(rpcs, 5, 1, if_print=False)
    model_size_dict = make_model_size_dict(rpcs, if_print=False)

    n_nodes = device_mesh.n_nodes
    search_time = 120

    rs: List[Dict[str, List]] = mdm_search.multi_mcmc_search(
        rpcs,
        rpc_exe_list,
        graph,
        table,
        model_size_dict,
        0.001,  # beta min
        0.002,  # beta max
        0.001,  # beta step
        search_time,  # time limit for each search
        1,  # repeat
    )
    if not from_file:
        with open(rs_dir, "w") as f:

            pprint.pprint(rs, stream=f)

    r: Dict[str, Dict[str, Any]] = rs[-1]
    pprint.pprint(r)

    rpc_name_to_rpcs = {rpc.name: rpc for rpc in rpcs}
    rpc_allocs = []
    for rpc_name, alloc_info in r.items():
        if rpc_name in ["end_time", "mem_cost"]:
            continue
        # rpc = rpc_dict[rpc_name]
        rpc = rpc_name_to_rpcs[rpc_name]
        parallel = ParallelismConfig(
            pipeline_parallel_size=alloc_info["num_pp"],
            data_parallel_size=alloc_info["num_dp"],
            model_parallel_size=alloc_info["num_mp"],
            use_sequence_parallel=(
                alloc_info["num_mp"] > 1
                and rpc.interface_type == ModelInterfaceType.TRAIN_STEP
            ),
        )
        sub_device_mesh = DeviceMesh(
            n_nodes=device_mesh.n_nodes,
            n_gpus_per_node=device_mesh.n_gpus_per_node,
            mapping=alloc_info["device_mesh_mapping"],
            name=alloc_info["device_mesh_name"],
            global_mesh_name=device_mesh.global_mesh_name,
        )
        rpc_alloc = RPCAllocation(
            rpc=rpc,
            device_mesh=sub_device_mesh,
            parallel=parallel,
        )
        rpc_allocs.append(rpc_alloc)

    if not from_file:
        with open(dump_dir, "w") as f:
            json.dump([rpc_alloc.to_dict() for rpc_alloc in rpc_allocs], f, indent=4)
        with open(log_dir, "w") as f:

            pprint.pprint(rpc_allocs, stream=f)

    return rpc_allocs


def make_rpc_exe_list(
    rpcs: List[MFCDef],
    device_mesh: DeviceMesh,
    num_gen_tokens: int,
    n_ppo_minibatches: int,
    seq_len: int,
    gradient_checkpointing: bool,
    if_print: bool = False,
    log_dir: Optional[str] = None,
) -> List[RPCExecution]:
    from realhf.search_engine.enumerate import enumerate_rpc_executions

    rpc_exe_list = []
    log_flag = False
    for rpc in rpcs:
        # real_model_config = load_model_config(rpc)
        feasible = enumerate_rpc_executions(
            rpc,
            device_mesh,
            seq_len=seq_len,
            num_gen_tokens=num_gen_tokens,
            n_ppo_minibatches=n_ppo_minibatches,
            gradient_checkpointing=gradient_checkpointing,
        )
        rpc_exe_list.extend(feasible)

        if log_dir is not None:
            mode = "w" if not log_flag else "a"
            with open(log_dir, mode) as f:
                f.write(f"{rpc.name} feasible: {len(feasible)}\n")
                feasible.sort(key=lambda x: x.time_cost)
                # feasible = feasible[:30]
                for i, rpc_exe in enumerate(feasible):
                    f.write(
                        f"{i}: time_cost: {rpc_exe.time_cost} ms, {rpc_exe.time_cost} "
                        f"sub_device_mesh: {rpc_exe.device_mesh}, "
                        f"parallel_strategy: {rpc_exe.parallel_strategy}, "
                        f"mem_cost: {rpc_exe.mem/(1024*1024*1024):02f} GB, "
                        f"static_mem_cost: {rpc_exe.static_mem/(1024*1024*1024):02f} GB\n"
                    )
                f.write("\n")
                log_flag = True

        if if_print:
            print(f"{rpc.name} feasible: {len(feasible)}")
            feasible.sort(key=lambda x: x.time_cost)
            # feasible = feasible[:10]
            for i, rpc_exe in enumerate(feasible):
                print(
                    f"{i}: time_cost: {rpc_exe.time_cost} ms, "
                    f"sub_device_mesh: {rpc_exe.device_mesh}, "
                    f"parallel_strategy: {rpc_exe.parallel_strategy}, "
                    f"mem_cost: {rpc_exe.mem/(1024*1024*1024):02f} GB, "
                    f"static_mem_cost: {rpc_exe.static_mem/(1024*1024*1024):02f} GB"
                )

    return rpc_exe_list


def make_model_size_dict(rpcs: List[MFCDef], if_print: bool = False) -> Dict[str, int]:
    model_size_dict = {}

    for rpc in rpcs:
        if rpc.model_name.role in model_size_dict:
            continue
        # model_configs = load_model_config(rpc)
        # model_size_dict[rpc.model_name.role] = estimate_model_size(real_model_config)
        model_size_dict[rpc.model_name.role] = rpc.model_type.size

        if if_print:
            print(
                f"model_name: {rpc.model_name.role}, "
                f"model_size: {rpc.model_type.size}"
            )
    return model_size_dict
