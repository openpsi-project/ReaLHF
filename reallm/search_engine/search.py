from typing import Any, Dict, List, Literal, Optional
import argparse
import functools
import json
import os
import pickle
import re

import numpy as np

try:
    import reallm._C.mdm_search as mdm_search
except ModuleNotFoundError:
    mdm_search = None

from reallm.api.core.dfg import ModelInterfaceType, ModelRPC
from reallm.api.core.model_api import ModelFamily
from reallm.api.quickstart.device_mesh import DeviceMesh, RPCAllocation
from reallm.api.quickstart.model import ModelTrainEvalConfig, ParallelismConfig
from reallm.api.quickstart.search import RPCExecution
import reallm.base.constants as constants


def search_rpc_allocations(
    device_mesh: DeviceMesh,
    rpcs: List[ModelRPC],
    num_gen_tokens: int = 256,
    n_ppo_minibatches: int = 1,
    seq_len: int = 256,
    gradient_checkpointing: bool = True,
    use_cache: bool = False,
) -> List[RPCAllocation]:
    from reallm.search_engine.enumerate import build_graph
    from reallm.search_engine.estimate import get_param_realloc_stats

    from_file = os.environ.get("IS_REMOTE", "0") == "1"
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
            import pprint

            pprint.pprint(rs, stream=f)

    import pprint

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
            use_sequence_parallel=(alloc_info["num_mp"] > 1
                                   and rpc.interface_type == ModelInterfaceType.TRAIN_STEP),
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
            import pprint

            pprint.pprint(rpc_allocs, stream=f)

    return rpc_allocs


def make_rpc_exe_list(
    rpcs: List[ModelRPC],
    device_mesh: DeviceMesh,
    num_gen_tokens: int,
    n_ppo_minibatches: int,
    seq_len: int,
    gradient_checkpointing: bool,
    if_print: bool = False,
    log_dir: Optional[str] = None,
) -> List[RPCExecution]:
    from reallm.search_engine.enumerate import enumerate_rpc_executions

    rpc_exe_list = []
    log_flag = False
    for rpc in rpcs:
        # flash_mqat_config = load_model_config(rpc)
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
                    f.write(f"{i}: time_cost: {rpc_exe.time_cost} ms, {rpc_exe.time_cost} "
                            f"sub_device_mesh: {rpc_exe.device_mesh}, "
                            f"parallel_strategy: {rpc_exe.parallel_strategy}, "
                            f"mem_cost: {rpc_exe.mem/(1024*1024*1024):02f} GB, "
                            f"static_mem_cost: {rpc_exe.static_mem/(1024*1024*1024):02f} GB\n")
                f.write("\n")
                log_flag = True

        if if_print:
            print(f"{rpc.name} feasible: {len(feasible)}")
            feasible.sort(key=lambda x: x.time_cost)
            # feasible = feasible[:10]
            for i, rpc_exe in enumerate(feasible):
                print(f"{i}: time_cost: {rpc_exe.time_cost} ms, "
                      f"sub_device_mesh: {rpc_exe.device_mesh}, "
                      f"parallel_strategy: {rpc_exe.parallel_strategy}, "
                      f"mem_cost: {rpc_exe.mem/(1024*1024*1024):02f} GB, "
                      f"static_mem_cost: {rpc_exe.static_mem/(1024*1024*1024):02f} GB")

    return rpc_exe_list


def make_model_size_dict(rpcs: List[ModelRPC], if_print: bool = False) -> Dict[str, int]:
    model_size_dict = {}

    for rpc in rpcs:
        if rpc.model_name.role in model_size_dict:
            continue
        # model_configs = load_model_config(rpc)
        # model_size_dict[rpc.model_name.role] = estimate_model_size(flash_mqat_config)
        model_size_dict[rpc.model_name.role] = rpc.model_type.size

        if if_print:
            print(f"model_name: {rpc.model_name.role}, "
                  f"model_size: {rpc.model_type.size}")
    return model_size_dict


def dump_search_settings(
    device_mesh: DeviceMesh,
    rpcs: List[ModelRPC],
    num_gen_tokens: int = 256,
    n_ppo_minibatches: int = 1,
    gradient_checkpointing: bool = True,
    seq_len: int = 256,
):

    from reallm.search_engine.enumerate import build_graph

    dump_dir = "/home/meizy/model_device_mapping_search/test_case/"
    rpc_exe_list = make_rpc_exe_list(
        rpcs,
        device_mesh,
        num_gen_tokens=num_gen_tokens,
        n_ppo_minibatches=n_ppo_minibatches,
        seq_len=seq_len,
        gradient_checkpointing=gradient_checkpointing,
        # log_dir=rpc_exe_dir,
        if_print=True,
    )
    graph = build_graph(rpcs, 5, 1, if_print=True)
    model_size_dict = make_model_size_dict(rpcs, if_print=True)

    with open(dump_dir + "rpc_list.pkl", "wb") as f:
        pickle.dump(rpcs, f)
    with open(dump_dir + "rpc_exe_list.pkl", "wb") as f:
        pickle.dump(rpc_exe_list, f)
    with open(dump_dir + "graph.pkl", "wb") as f:
        pickle.dump(graph, f)
    with open(dump_dir + "model_size_dict.pkl", "wb") as f:
        pickle.dump(model_size_dict, f)


def print_model_device_mapping_by_index(rpcs: List[ModelRPC], device_mesh: DeviceMesh, index):
    # print_list = [0, 30, 36, 38, 38, 27, ]
    rpc_exe_table = {}
    avg_time_cost = []

    from reallm.search_engine.enumerate import enumerate_rpc_executions
    from reallm.search_engine.estimate import load_model_config

    for i, rpc in enumerate(rpcs):
        flash_mqat_config = load_model_config(rpc)
        feasible = enumerate_rpc_executions(rpc, device_mesh)
        print(f"{rpc.name} feasible: {len(feasible)}")
        feasible.sort(key=lambda x: x.time_cost)
        # feasible = feasible[:10]
        rpc_exe_table[rpc.name] = feasible
        avg_time_cost.append((sum([x.time_cost for x in feasible][:10]) + i / 10, rpc.name))

    avg_time_cost.sort(key=lambda x: x[0], reverse=True)
    sorted_model_rpc_names = [x[1] for x in avg_time_cost]

    for pindex, n in zip(index, sorted_model_rpc_names):
        feasible = rpc_exe_table[n]
        print(feasible[pindex])


if __name__ == "__main__":
    # profile_search(rank=args.rank)
    from reallm.experiments.common.ppo_exp import PPOConfig, PPOHyperparameters

    constants.set_experiment_trial_names("test", "test")
    actor = ModelTrainEvalConfig(
        type=ModelFamily("llama", 7, False),
        path="/lustre/public/pretrained_model_weights/Llama-2-7b-hf",
        gradient_checkpointing=True,
    )
    critic = ModelTrainEvalConfig(
        type=ModelFamily("llama", 7, True),
        path="/lustre/public/pretrained_model_weights/Llama-2-7b-hf",
        gradient_checkpointing=True,
    )
    ref = ModelTrainEvalConfig(
        type=ModelFamily("llama", 7, False),
        path="/lustre/public/pretrained_model_weights/Llama-2-7b-hf",
    )
    rew = ModelTrainEvalConfig(
        type=ModelFamily("llama", 7, True),
        path="/lustre/public/pretrained_model_weights/Llama-2-7b-hf",
    )
    ppo_cfg = PPOHyperparameters(ppo_n_minibatches=4, max_new_tokens=256, min_new_tokens=256)
    exp = PPOConfig(
        experiment_name="test",
        trial_name="test",
        allocation_mode="search",
        global_train_bs=128,
        global_gen_bs=128,
        n_nodes=1,
        n_gpus_per_node=8,
        actor=actor,
        critic=critic,
        ref=ref,
        rew=rew,
        ppo=ppo_cfg,
    )
    device_mesh = DeviceMesh(1, 8, np.ones((1, 8), dtype=np.int32), None, None)
    print(device_mesh.global_mesh_name)
    exp.initial_setup()
