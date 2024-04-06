from typing import Dict, List, Optional
import argparse
import os
import pickle

from profiler.device_mesh import DeviceMesh, make_device_mesh_from_name, ModelParallelStrategy
from profiler.enumerate import build_graph, enumerate_rpc_executions
from profiler.estimate import (comm_stats, estimate_model_size, estimate_rpc_memory, estimate_rpc_time,
                               load_model_config)
from profiler.experiments import ppo_rpcs_example
from profiler.profile_layers import profile_rpcs
from profiler.rpc import RPC, RPCExecution
import profiler.cppsearch.mdm_search as mdm_search

from api.config.config_base import MODEL_TYPE_TO_PATH
from api.config.config_device_mesh import ClusterDeviceMesh, RPCAllocation
from api.config.config_flash_model import ModelTrainEvalConfig, OptimizerConfig
from api.config.dfg import ModelInterfaceType, ModelName, ModelRPC, OffloadHook, SyncParamHook
from impl.model.nn.flash_mqat.flash_mqat_base import FlashMQATConfig
import api.config.config_system as config_package
import base.constants


def optimal_device_mapping(
    device_mesh: ClusterDeviceMesh,
    model_rpcs: List[ModelRPC],
    model_configs: Dict[str, FlashMQATConfig],
    nodelist: Optional[str] = None,
    profile_layers: bool = False,
    num_gen_tokens: int = 256,
    n_ppo_minibatches: int = 1,
    top_k: int = 10,
) -> Dict[str, RPCAllocation]:
    from_file = os.environ["IS_REMOTE"] == "1"
    dump_dir = os.path.join(base.constants.LOG_ROOT, base.constants.experiment_name(),
                            base.constants.trial_name(), "device_mapping.pkl")
    log_dir = os.path.join(base.constants.LOG_ROOT, base.constants.experiment_name(),
                           base.constants.trial_name(), "device_mapping")

    if from_file:
        with open(dump_dir, "rb") as f:
            return pickle.load(f)

    # profile layers for each model type
    if profile_layers:
        profile_rpcs(model_rpcs)

    # hack, only suitable for configs in experiments/autoexp/auto_ppo.py
    # rollout, rew_inf, ref_inf, critic_inf, actor_train, critic_train = model_rpcs
    # actor_train.pre_hooks.append(SyncParamHook(source=ModelName("actor", 0)))
    # actor_train.model_name = ModelName("actor", 1)
    # actor_train.post_hooks.append(SyncParamHook(target=ModelName("actor", 0)))
    # critic_train.pre_hooks.append(SyncParamHook(source=ModelName("critic", 0)))
    # critic_train.model_name = ModelName("critic", 1)
    # critic_train.post_hooks.append(SyncParamHook(target=ModelName("critic", 0)))
    # rew_inf.post_hooks.append(OffloadHook())
    # ref_inf.post_hooks.append(OffloadHook())

    search_device_mesh = make_device_mesh_from_name(nodelist)

    rpc_exe_list = make_rpc_exe_list(model_rpcs,
                                     search_device_mesh,
                                     num_gen_tokens=num_gen_tokens,
                                     n_ppo_minibatches=n_ppo_minibatches,
                                     if_print=True)
    rpc_list = make_rpc_list(model_rpcs, if_print=True)
    graph = build_graph(model_rpcs, 5, 2, if_print=True)
    comm_stats_ = comm_stats(if_print=True)
    model_size_dict = make_model_size_dict(model_rpcs, if_print=True)

    # rpc_dict = {rpc.name: rpc for rpc in model_rpcs}

    n_nodes = device_mesh.n_nodes
    search_time = 30 * n_nodes

    rs: List[Dict[str, List]] = mdm_search.multi_mcmc_search(
        rpc_list,
        rpc_exe_list,
        graph,
        comm_stats_,
        model_size_dict,
        0.01,  # beta min
        0.05,  # beta max
        0.01,  # beta step
        search_time,  # time limit for each search
        1,  # repeat
    )
    r = rs[-1]
    # print(r)

    # hack, only suitable for configs in experiments/autoexp/auto_ppo.py
    rollout, rew_inf, ref_inf, critic_inf, actor_train, critic_train = model_rpcs
    rollout_topo = tuple(r[rollout.name].values())
    actor_train_topo = tuple(r[actor_train.name].values())
    old_name = actor_train.name
    optim_model_name = []
    if rollout_topo != actor_train_topo:
        actor_train.pre_hooks.append(SyncParamHook(source=ModelName("actor", 0)))
        actor_train.model_name = ModelName("actor", 1)
        actor_train.post_hooks.append(SyncParamHook(target=ModelName("actor", 0)))
        r[actor_train.name] = r.pop(old_name)

    critic_inf_topo = tuple(r[critic_inf.name].values())
    critic_train_topo = tuple(r[critic_train.name].values())
    old_name = critic_train.name
    if critic_inf_topo != critic_train_topo:
        critic_train.pre_hooks.append(SyncParamHook(source=ModelName("critic", 0)))
        critic_train.model_name = ModelName("critic", 1)
        critic_train.post_hooks.append(SyncParamHook(target=ModelName("critic", 0)))
        r[critic_train.name] = r.pop(old_name)

    rew_inf.post_hooks.append(OffloadHook())
    ref_inf.post_hooks.append(OffloadHook())

    rpc_dict = {rpc.name: rpc for rpc in model_rpcs}
    # import pprint
    # pprint.pprint(rpc_dict)

    rpc_alloc_dict = {}
    for rpc_name, alloc_info in r.items():
        rpc = rpc_dict[rpc_name]
        ps = ModelParallelStrategy(
            num_pp=alloc_info["num_pp"],
            num_dp=alloc_info["num_dp"],
            num_mp=alloc_info["num_mp"],
        )
        use_sequence_parallel = False\
            if rpc.model_name.role in ["reward", "ref"] else (ps.num_mp > 1)
        parallel_config = ps.to_config(use_sequence_parallel=use_sequence_parallel,)
        gradient_checkpointing = True
        if rpc.model_name in [actor_train.model_name, critic_train.model_name]:
            optim_config = OptimizerConfig(type="adam", offload=False)
        else:
            optim_config = OptimizerConfig()
        # optim_config = OptimizerConfig(type="adam", offload=False)\
        #     if rpc.interface_type == ModelInterfaceType.TRAIN_STEP else OptimizerConfig()
        # print(alloc_info["device_mesh"])
        sub_device_mesh = make_device_mesh_from_name(alloc_info["device_mesh"])
        train_eval_config = ModelTrainEvalConfig(
            type=rpc.model_type._class,
            path=MODEL_TYPE_TO_PATH[rpc.model_type],
            base_model_path=MODEL_TYPE_TO_PATH[rpc.model_type],
            gradient_checkpointing=gradient_checkpointing,
            parallel=parallel_config,
            optimizer=optim_config,
        )
        rpc_alloc = sub_device_mesh.to_config(
            device_mesh=search_device_mesh,
            train_eval_config=train_eval_config,
            rpc=rpc_dict[rpc_name],
        )
        # print(f"{rpc_name}: mapping: {rpc_alloc.mapping}, "
        #       f" parallel: {parallel_config}")
        rpc_alloc_dict[rpc_name] = rpc_alloc

    if not from_file:
        os.makedirs(os.path.dirname(dump_dir), exist_ok=True)
        with open(dump_dir, "wb") as f:
            pickle.dump(rpc_alloc_dict, f)
        with open(log_dir, "w") as f:
            import pprint
            pprint.pprint(rpc_alloc_dict, stream=f)

    return rpc_alloc_dict


def make_rpc_exe_list(rpcs: List[ModelRPC],
                      device_mesh: DeviceMesh,
                      num_gen_tokens: int = 256,
                      n_ppo_minibatches: int = 1,
                      if_print: bool = False) -> List[RPCExecution]:
    rpc_exe_list = []
    for rpc in rpcs:
        # flash_mqat_config = load_model_config(rpc)
        feasible = enumerate_rpc_executions(rpc,
                                            device_mesh,
                                            num_gen_tokens=num_gen_tokens,
                                            n_ppo_minibatches=n_ppo_minibatches)
        rpc_exe_list.extend(feasible)

        if if_print:
            print(f"{rpc.name} feasible: {len(feasible)}")
            feasible.sort(key=lambda x: x.time_cost)
            # feasible = feasible[:30]
            for rpc_exe in feasible:
                print(f"time_cost: {rpc_exe.time_cost/(1e3)} ms, {rpc_exe.time_cost} "
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
        flash_mqat_config = load_model_config(rpc)
        model_size_dict[rpc.model_name.role] = estimate_model_size(flash_mqat_config)

        if if_print:
            print(f"model_name: {rpc.model_name.role}, "
                  f"model_size: {estimate_model_size(flash_mqat_config)} Bytes")
    return model_size_dict


def make_rpc_list(rpcs: List[ModelRPC], if_print: bool = False) -> List[RPC]:
    rpc_list = []
    for rpc in rpcs:
        rpc_list.append(RPC.from_config(rpc))
        if if_print:
            print(f"rpc_name: {rpc.name}, model_name: {rpc.model_name}, "
                  f"interface_type: {rpc.interface_type}")
    return rpc_list


def search_model_device_mappings(rpcs: List[ModelRPC], device_mesh: DeviceMesh, method="mcmc", beta=0.75):
    rpc_exe_list = make_rpc_exe_list(rpcs, device_mesh)
    rpc_list = make_rpc_list(rpcs)
    graph = build_graph(rpcs, 5, 2)
    comm_stats_ = comm_stats()
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
    comm_stats_ = comm_stats(if_print=True)
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
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--expr_name", type=str)
    # args = parser.parse_args()

    # experiment = config_package.make_experiment(args.expr_name)

    os.environ.setdefault("IS_REMOTE", "0")
    size = 13
    n_nodes = 2
    node_start = 42
    node_end = node_start + n_nodes - 1
    nodelist = f"QH-com[{node_start:02d}-{node_end:02d}]"
    cluster_device_mesh = ClusterDeviceMesh(n_nodes=n_nodes, n_gpus_per_node=8, mem=80)
    rpcs = ppo_rpcs_example(size)
    device_mesh = make_device_mesh_from_name(nodelist)
    dump_search_settings(rpcs, device_mesh)
    # optimal_device_mapping(cluster_device_mesh, rpcs, None, nodelist)

    # 7b 13b 2x8:
    # [0, 2, 2, 1, 11, 3, ]
    # [0, 2, 0, 8, 33, 1, ] = 75s

    # 7b 34b 4x8
    # [0, 0, 9, 1, 4, 3, ] 87441235
    # [0, 0, 9, 1, 1, 3, ] 86986420
    # print_model_device_mapping_by_index(rpcs, device_mesh, [0, 0, 9, 1, 1, 3, ])
