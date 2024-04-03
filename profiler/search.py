from typing import Dict, List, Optional
import pickle

from profiler.device_mesh import DeviceMesh, make_device_mesh_from_name, ModelParallelStrategy
from profiler.enumerate import build_graph, enumerate_rpc_executions
from profiler.estimate import (comm_stats, estimate_function_call_memory, estimate_model_size,
                               estimate_rpc_cost, load_model_config)
from profiler.experiments import ppo_rpcs_example, ProfileExperiment
from profiler.rpc import RPC, RPCExecution
import profiler.cppsearch.mdm_search as mdm_search

from api.config.dfg import ModelRPC
from experiments.autoexp.auto_ppo import ClusterDeviceMesh
from experiments.autoexp.device_mapping import RPCAllocation
from impl.model.nn.flash_mqat.flash_mqat_base import FlashMQATConfig


def optimal_device_mapping(
    device_mesh: ClusterDeviceMesh,
    model_rpcs: List[ModelRPC],
    model_configs: Dict[str, FlashMQATConfig],
    nodelist: Optional[str] = None,
) -> Dict[str, RPCAllocation]:
    device_mesh = make_device_mesh_from_name(nodelist)

    rpc_exe_list = make_rpc_exe_list(model_rpcs, device_mesh, if_print=True)
    rpc_list = make_rpc_list(model_rpcs, if_print=True)
    graph = build_graph(model_rpcs, 5, 2, if_print=True)
    comm_stats_ = comm_stats(None, if_print=True)
    model_size_dict = make_model_size_dict(model_rpcs, if_print=True)

    r = mdm_search.multi_mcmc_search(rpc_list, rpc_exe_list, graph, comm_stats_, model_size_dict, 0.01, 0.03,
                                     0.01, 10)
    print(r)


# def handpick_model_device_mapping(rpcs: List[ModelRPC], device_mesh_name: str):
#     device_mesh = make_device_mesh_from_name(device_mesh_name)
#     total_time_cost = 0
#     total_model_cost = 0
#     dynamic_model_cost = 0
#     models = []
#     for rpc in exp.model_rpcs:
#         model_type = exp.model_names_to_types[rpc.model_name]
#         flash_mqat_config = load_model_config(exp.model_paths[exp.model_types.index(model_type)])
#         parallel_strategy = ModelParallelStrategy(num_pp=exp.n_nodes, num_dp=8, num_mp=1)
#         mem_cost, static_mem = estimate_function_call_memory(rpc.interface_type, rpc.min_n_seqs,
#                                                              rpc.max_n_tokens // rpc.min_n_seqs,
#                                                              flash_mqat_config, parallel_strategy)
#         mem_cost = int(mem_cost)
#         static_mem = int(static_mem)
#         time_cost = estimate_rpc_cost(exp, rpc, flash_mqat_config, parallel_strategy)
#         total_time_cost += time_cost
#         if mem_cost - static_mem > dynamic_model_cost:
#             dynamic_model_cost = mem_cost - static_mem
#         if rpc.model_name not in models:
#             models.append(rpc.model_name)
#             total_model_cost += static_mem

#     print(f"total_time_cost: {5*total_time_cost/(1e3)} ms")
#     print(f"total_model_cost: {total_model_cost/(1024*1024*1024):02f} GB")
#     print(f"dynamic_model_cost: {dynamic_model_cost/(1024*1024*1024):02f} GB")


def make_rpc_exe_list(rpcs: List[ModelRPC], device_mesh: DeviceMesh, if_print: bool = False):
    rpc_exe_list = []
    for rpc in rpcs:
        # flash_mqat_config = load_model_config(rpc)
        feasible = enumerate_rpc_executions(rpc, device_mesh)
        rpc_exe_list.extend(feasible)

        if if_print:
            print(f"{rpc.name} feasible: {len(feasible)}")
            feasible.sort(key=lambda x: x.time_cost)
            feasible = feasible[:30]
            for rpc_exe in feasible:
                print(f"time_cost: {rpc_exe.time_cost/(1e3)} ms, {rpc_exe.time_cost} "
                      f"sub_device_mesh: {rpc_exe.device_mesh}, "
                      f"parallel_strategy: {rpc_exe.parallel_strategy}, "
                      f"mem_cost: {rpc_exe.mem/(1024*1024*1024):02f} GB, "
                      f"static_mem_cost: {rpc_exe.static_mem/(1024*1024*1024):02f} GB")

    return rpc_exe_list


def make_model_size_dict(rpcs: List[ModelRPC], if_print: bool = False):
    model_size_dict = {}

    for rpc in rpcs:
        if rpc.model_name.role in model_size_dict:
            continue
        flash_mqat_config = load_model_config(rpc)
        model_size_dict[rpc.model_name.role] = estimate_model_size(flash_mqat_config)

        if if_print:
            print(f"model_name: {rpc.model_name.role}, "
                  f"model_size: {estimate_model_size(flash_mqat_config)} Bytes")
    return model_size_dict


def make_rpc_list(rpcs: List[ModelRPC], if_print: bool = False):
    rpc_list = []
    for rpc in rpcs:
        rpc_list.append(RPC(rpc))
        if if_print:
            print(f"rpc_name: {rpc.name}, model_name: {rpc.model_name}, interface_type: {rpc.interface_type}")
    return rpc_list


def search_model_device_mappings(rpcs: List[ModelRPC], device_mesh: DeviceMesh, method="mcmc", beta=0.75):
    rpc_exe_list = make_rpc_exe_list(rpcs, device_mesh)
    rpc_list = make_rpc_list(rpcs)
    graph = build_graph(rpcs, 5, 2)
    comm_stats_ = comm_stats(None)
    model_size_dict = make_model_size_dict(rpcs)

    if method == "mcmc":
        mdm_search.mcmc_search(rpc_list, rpc_exe_list, graph, comm_stats_, model_size_dict, beta)
    elif method == "brute_force":
        mdm_search.brute_force_search(rpc_list, rpc_exe_list, graph, comm_stats_, model_size_dict)
    else:
        raise NotImplementedError(f"method {method} not supported")


def dump_search_settings(rpcs: List[ModelRPC], device_mesh: DeviceMesh):
    rpc_exe_list = make_rpc_exe_list(rpcs, device_mesh, if_print=True)
    rpc_list = make_rpc_list(rpcs, if_print=True)
    graph = build_graph(rpcs, 5, 2, if_print=True)
    comm_stats_ = comm_stats(None, if_print=True)
    model_size_dict = make_model_size_dict(rpcs, if_print=True)

    dump_dir = "/home/meizy/model_device_mapping_search/test_case/"
    with open(dump_dir + "rpc_list.pkl", "wb") as f:
        pickle.dump(rpc_list, f)
    with open(dump_dir + "rpc_exe_list.pkl", "wb") as f:
        pickle.dump(rpc_exe_list, f)
    with open(dump_dir + "graph.pkl", "wb") as f:
        pickle.dump(graph, f)
    with open(dump_dir + "model_size_dict.pkl", "wb") as f:
        pickle.dump(model_size_dict, f)
    with open(dump_dir + "comm_stats.pkl", "wb") as f:
        pickle.dump(comm_stats_, f)


def print_model_device_mapping_by_index(rpcs: List[ModelRPC], device_mesh: DeviceMesh, index):
    # print_list = [0, 30, 36, 38, 38, 27, ]
    rpc_exe_table = {}
    avg_time_cost = []

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
    # exp = ProfileExperiment()
    size = 34
    n_nodes = 4
    node_start = 40
    node_end = node_start + n_nodes - 1
    nodelist = f"QH-com[{node_start:02d}-{node_end:02d}]"
    rpcs = ppo_rpcs_example(size)
    device_mesh = make_device_mesh_from_name(nodelist)
    # handpick_model_device_mapping(exp)
    # search_model_device_mappings(exp)ModelInterfaceType
    # dump_search_settings(rpcs, device_mesh)
    optimal_device_mapping(None, rpcs, None, nodelist)

    # 7b 13b 2x8:
    # [0, 2, 2, 1, 11, 3, ]
    # [0, 2, 0, 8, 33, 1, ] = 75s

    # 7b 34b 4x8
    # [0, 0, 9, 1, 4, 3, ] 87441235
    # [0, 0, 9, 1, 1, 3, ] 86986420
    # print_model_device_mapping_by_index(rpcs, device_mesh, [0, 0, 9, 1, 1, 3, ])
