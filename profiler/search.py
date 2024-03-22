import pickle

from profiler.device_mesh import DeviceMesh, make_device_mesh_from_name, ModelParallelStrategy
from profiler.enumerate import build_graph, enumerate_rpc_executions
from profiler.estimate import (comm_stats, estimate_function_call_memory, estimate_model_size,
                               estimate_rpc_cost, load_model_config)
from profiler.experiments import ProfileExperiment
from profiler.rpc import RPC, RPCExecution
import profiler.cppsearch.mdm_search as mdm_search


def handpick_model_device_mapping(exp: ProfileExperiment):
    device_mesh = make_device_mesh_from_name(exp.device_mesh_name)
    total_time_cost = 0
    total_model_cost = 0
    dynamic_model_cost = 0
    models = []
    for rpc in exp.model_rpcs:
        model_type = exp.model_names_to_types[rpc.model_name]
        flash_mqat_config = load_model_config(exp.model_paths[exp.model_types.index(model_type)])
        parallel_strategy = ModelParallelStrategy(num_pp=exp.n_nodes, num_dp=8, num_mp=1)
        mem_cost, static_mem = estimate_function_call_memory(rpc.interface_type, rpc.min_n_seqs,
                                                             rpc.max_n_tokens // rpc.min_n_seqs,
                                                             flash_mqat_config, parallel_strategy)
        mem_cost = int(mem_cost)
        static_mem = int(static_mem)
        time_cost = estimate_rpc_cost(exp, rpc, flash_mqat_config, parallel_strategy)
        total_time_cost += time_cost
        if mem_cost - static_mem > dynamic_model_cost:
            dynamic_model_cost = mem_cost - static_mem
        if rpc.model_name not in models:
            models.append(rpc.model_name)
            total_model_cost += static_mem

    print(f"total_time_cost: {5*total_time_cost/(1e3)} ms")
    print(f"total_model_cost: {total_model_cost/(1024*1024*1024):02f} GB")
    print(f"dynamic_model_cost: {dynamic_model_cost/(1024*1024*1024):02f} GB")


def make_rpc_exe_list(exp: ProfileExperiment, if_print: bool = False):
    device_mesh = make_device_mesh_from_name(exp.device_mesh_name)
    rpc_exe_list = []
    for rpc in exp.model_rpcs:
        model_type = exp.model_names_to_types[rpc.model_name]
        flash_mqat_config = load_model_config(exp.model_paths[exp.model_types.index(model_type)])
        feasible = enumerate_rpc_executions(exp, rpc, device_mesh, flash_mqat_config)
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


def make_model_size_dict(exp: ProfileExperiment, if_print: bool = False):
    model_size_dict = {}
    device_mesh = make_device_mesh_from_name(exp.device_mesh_name)

    for model_name in exp.model_names:
        model_type = exp.model_names_to_types[model_name]
        flash_mqat_config = load_model_config(exp.model_paths[exp.model_types.index(model_type)])
        model_size_dict[model_name] = estimate_model_size(flash_mqat_config)

        if if_print:
            print(f"model_name: {model_name}, "
                  f"model_size: {estimate_model_size(flash_mqat_config)} Bytes")
    return model_size_dict


def make_rpc_list(exp: ProfileExperiment, if_print: bool = False):
    rpc_list = []
    for rpc in exp.model_rpcs:
        rpc_list.append(RPC(rpc))
        if if_print:
            print(f"rpc_name: {rpc.name}, model_name: {rpc.model_name}, interface_type: {rpc.interface_type}")
    return rpc_list


def search_model_device_mappings(exp: ProfileExperiment, method="mcmc"):
    rpc_exe_list = make_rpc_exe_list(exp)
    rpc_list = make_rpc_list(exp)
    graph = build_graph(exp, 5, 2)
    comm_stats_ = comm_stats(exp)
    model_size_dict = make_model_size_dict(exp)
    if method == "mcmc":
        mdm_search.mcmc_search(rpc_list, rpc_exe_list, graph, model_size_dict)
    elif method == "brute_force":
        mdm_search.brute_force_search(rpc_list, rpc_exe_list, graph, model_size_dict)
    else:
        raise NotImplementedError(f"method {method} not supported")


def dump_search_settings(exp: ProfileExperiment):
    rpc_exe_list = make_rpc_exe_list(exp, if_print=True)
    rpc_list = make_rpc_list(exp, if_print=True)
    graph = build_graph(exp, 5, 2, if_print=True)
    comm_stats_ = comm_stats(exp, if_print=True)
    model_size_dict = make_model_size_dict(exp, if_print=True)

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


def print_model_device_mapping_by_index(exp: ProfileExperiment, index):
    device_mesh = make_device_mesh_from_name(exp.device_mesh_name)
    # print_list = [0, 30, 36, 38, 38, 27, ]
    rpc_exe_table = {}
    avg_time_cost = []

    for i, rpc in enumerate(exp.model_rpcs):
        model_type = exp.model_names_to_types[rpc.model_name]
        flash_mqat_config = load_model_config(exp.model_paths[exp.model_types.index(model_type)])
        feasible = enumerate_rpc_executions(exp, rpc, device_mesh, flash_mqat_config)
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
    exp = ProfileExperiment()
    handpick_model_device_mapping(exp)
    # search_model_device_mappings(exp)
    # dump_search_settings(exp)

    # 7b 13b 2x8:
    # [0, 2, 2, 1, 11, 3, ]
    # [0, 2, 0, 8, 33, 1, ] = 75s
    # print_model_device_mapping_by_index(exp, [2, 4, 8, 0, 10, 0, ])
