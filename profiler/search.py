import pickle

from profiler.device_mesh import DeviceMesh, make_device_mesh_from_name, ModelParallelStrategy
from profiler.enumerate import enumerate_rpc_executions
from profiler.estimate import load_model_config
from profiler.experiments import ProfileExperiment
from profiler.rpc import model_rpc_name, RPC, RPCExecution
import profiler.cppsearch.mdm_search as mdm_search


def search_model_device_mappings(exp: ProfileExperiment):
    device_mesh = make_device_mesh_from_name(exp.device_mesh_name)
    rpc_exe_list = []
    rpc_list = []

    for rpc in exp.model_rpcs:
        model_type = exp.model_names_to_types[rpc.model_name]
        flash_mqat_config = load_model_config(exp.model_paths[exp.model_types.index(model_type)])
        feasible = enumerate_rpc_executions(exp, rpc, device_mesh, flash_mqat_config)
        print(f"{model_rpc_name(rpc)} feasible: {len(feasible)}")
        feasible.sort(key=lambda x: x.time_cost)
        # feasible = feasible[:10]

        for rpc_exe in feasible[:10]:
            rpc_exe: RPCExecution
            print(f"time_cost: {rpc_exe.time_cost/(1e3)} ms, "
                  f"sub_device_mesh: {rpc_exe.device_mesh}, "
                  f"parallel_strategy: {rpc_exe.parallel_strategy}, "
                  f"mem_cost: {rpc_exe.mem/(1024*1024*1024):02f} GB, "
                  f"static_mem_cost: {rpc_exe.static_mem/(1024*1024*1024):02f} GB")

        rpc_list.append(RPC(rpc))
        rpc_exe_list.extend(feasible)

    # dump_dir = "/home/meizy/model_device_mapping_search/test_case/"
    # with open(dump_dir + "rpc_list.pkl", "wb") as f:
    #     pickle.dump(rpc_list, f)
    # with open(dump_dir + "rpc_exe_list.pkl", "wb") as f:
    #     pickle.dump(rpc_exe_list, f)
    mdm_search.brute_force_search(rpc_list, rpc_exe_list)


if __name__ == "__main__":
    exp = ProfileExperiment()
    search_model_device_mappings(exp)
