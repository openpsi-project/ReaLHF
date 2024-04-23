from typing import Dict, List, Literal, Optional
import argparse
import functools
import os
import pickle

import numpy as np

from reallm.api.core.config import MODEL_TYPE_TO_PATH
from reallm.api.core.dfg import ModelInterfaceType, ModelName, ModelRPC, OffloadHook, SyncParamHook
from reallm.api.quickstart.device_mesh import ClusterDeviceMesh, RPCAllocation
from reallm.api.quickstart.model import (FlashMQATConfig, ModelTrainEvalConfig, OptimizerConfig,
                                         ParallelismConfig)
import reallm._C.mdm_search as mdm_search
import reallm.api.core.system as config_package
import reallm.base.constants

from .device_mesh import DeviceMesh, make_device_mesh_from_name, ModelParallelStrategy
from .enumerate import build_graph, enumerate_rpc_executions
from .estimate import (comm_stats, estimate_model_size, estimate_rpc_memory, estimate_rpc_time,
                       load_model_config)
from .experiments import ppo_rpcs_example
from .profile_layers import profile_rpcs
from .rpc import RPC, RPCExecution


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
    dump_dir = os.path.join(base.constants.LOG_ROOT, reallm.base.constants.experiment_name(),
                            reallm.base.constants.trial_name(), "device_mapping.pkl")
    log_dir = os.path.join(base.constants.LOG_ROOT, reallm.base.constants.experiment_name(),
                           reallm.base.constants.trial_name(), "device_mapping")
    rs_dir = os.path.join(base.constants.LOG_ROOT, reallm.base.constants.experiment_name(),
                          reallm.base.constants.trial_name(), "raw_search_result")
    rpc_exe_dir = os.path.join(base.constants.LOG_ROOT, reallm.base.constants.experiment_name(),
                               reallm.base.constants.trial_name(), "rpc_exe_info")

    if from_file:
        with open(dump_dir, "rb") as f:
            return pickle.load(f)
    else:
        os.makedirs(os.path.dirname(dump_dir), exist_ok=True)

    # profile layers for each model type
    if profile_layers:
        profile_rpcs(model_rpcs)

    # hack, only suitable for configs in reallm.experiments/autoexp/auto_ppo.py
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
                                     log_dir=rpc_exe_dir)
    rpc_list = make_rpc_list(model_rpcs, if_print=False)
    graph = build_graph(model_rpcs, 5, 1, if_print=False)
    cost_table = pickle.load(open("profile_result/param_sync_cost_table_parallel.pkl", "rb"))
    model_size_dict = make_model_size_dict(model_rpcs, if_print=False)

    # rpc_dict = {rpc.name: rpc for rpc in model_rpcs}

    n_nodes = device_mesh.n_nodes
    search_time = 120  # 60 * n_nodes

    rs: List[Dict[str, List]] = mdm_search.multi_mcmc_search(
        rpc_list,
        rpc_exe_list,
        graph,
        cost_table,
        model_size_dict,
        0.001,  # beta min
        0.004,  # beta max
        0.001,  # beta step
        search_time,  # time limit for each search
        1,  # repeat
    )
    if not from_file:
        with open(rs_dir, "w") as f:
            import pprint
            pprint.pprint(rs, stream=f)
    r = rs[-1]
    # print(r)

    # hack, only suitable for configs in reallm.experiments/autoexp/auto_ppo.py
    rollout, rew_inf, ref_inf, critic_inf, actor_train, critic_train = model_rpcs
    rollout_topo = (r[rollout.name]["device_mesh"], r[rollout.name]["num_pp"], r[rollout.name]["num_dp"],
                    r[rollout.name]["num_mp"])
    actor_train_topo = (r[actor_train.name]["device_mesh"], r[actor_train.name]["num_pp"],
                        r[actor_train.name]["num_dp"], r[actor_train.name]["num_mp"])
    old_name = actor_train.name
    optim_model_name = []
    if rollout_topo != actor_train_topo:
        actor_train.pre_hooks.append(SyncParamHook(source=ModelName("actor", 0)))
        actor_train.model_name = ModelName("actor", 1)
        actor_train.post_hooks.append(SyncParamHook(target=ModelName("actor", 0)))
        r[actor_train.name] = r.pop(old_name)
    else:
        actor_train.model_name = ModelName("actor", 0)
        rollout.model_name = ModelName("actor", 0)

    critic_inf_topo = (r[critic_inf.name]["device_mesh"], r[critic_inf.name]["num_pp"],
                       r[critic_inf.name]["num_dp"], r[critic_inf.name]["num_mp"])
    critic_train_topo = (r[critic_train.name]["device_mesh"], r[critic_train.name]["num_pp"],
                         r[critic_train.name]["num_dp"], r[critic_train.name]["num_mp"])
    old_name = critic_train.name
    if critic_inf_topo != critic_train_topo:
        critic_train.pre_hooks.append(SyncParamHook(source=ModelName("critic", 0)))
        critic_train.model_name = ModelName("critic", 1)
        critic_train.post_hooks.append(SyncParamHook(target=ModelName("critic", 0)))
        r[critic_train.name] = r.pop(old_name)
    else:
        critic_train.model_name = ModelName("critic", 0)
        critic_inf.model_name = ModelName("critic", 0)

    rew_inf.post_hooks.append(OffloadHook())
    ref_inf.post_hooks.append(OffloadHook())

    rpc_dict = {rpc.name: rpc for rpc in model_rpcs}
    # import pprint
    # pprint.pprint(rpc_dict)

    rpc_alloc_dict = {}
    for rpc_name, alloc_info in r.items():
        if rpc_name in ["end_time", "mem_cost"]:
            continue
        rpc = rpc_dict[rpc_name]
        ps = ModelParallelStrategy(
            num_pp=alloc_info["num_pp"],
            num_dp=alloc_info["num_dp"],
            num_mp=alloc_info["num_mp"],
        )
        # use_sequence_parallel = False\
        #     if rpc.model_name.role in ["reward", "ref"] else (ps.num_mp > 1)
        use_sequence_parallel = ps.num_mp > 1
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
        with open(dump_dir, "wb") as f:
            pickle.dump(rpc_alloc_dict, f)
        with open(log_dir, "w") as f:
            import pprint
            pprint.pprint(rpc_alloc_dict, stream=f)

    return rpc_alloc_dict


def make_rpc_exe_list(rpcs: List[ModelRPC],
                      device_mesh: DeviceMesh,
                      num_gen_tokens: int,
                      n_ppo_minibatches: int,
                      if_print: bool = False,
                      log_dir: Optional[str] = None) -> List[RPCExecution]:
    rpc_exe_list = []
    log_flag = False
    for rpc in rpcs:
        # flash_mqat_config = load_model_config(rpc)
        feasible = enumerate_rpc_executions(rpc,
                                            device_mesh,
                                            num_gen_tokens=num_gen_tokens,
                                            n_ppo_minibatches=n_ppo_minibatches)
        rpc_exe_list.extend(feasible)

        if log_dir is not None:
            mode = "w" if not log_flag else "a"
            with open(log_dir, mode) as f:
                f.write(f"{rpc.name} feasible: {len(feasible)}\n")
                feasible.sort(key=lambda x: x.time_cost)
                # feasible = feasible[:30]
                for i, rpc_exe in enumerate(feasible):
                    f.write(f"{i}: time_cost: {rpc_exe.time_cost/(1e3)} ms, {rpc_exe.time_cost} "
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
                print(f"{i}: time_cost: {rpc_exe.time_cost/(1e3)} ms, {rpc_exe.time_cost} "
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
        # model_size_dict[rpc.model_name.role] = estimate_model_size(flash_mqat_config)
        model_size_dict[rpc.model_name.role] = rpc.model_type.size

        if if_print:
            print(f"model_name: {rpc.model_name.role}, "
                  f"model_size: {rpc.model_type.size}")
    return model_size_dict


def make_rpc_list(rpcs: List[ModelRPC], if_print: bool = False) -> List[RPC]:
    rpc_list = []
    for rpc in rpcs:
        rpc_list.append(RPC.from_config(rpc))
        if if_print:
            print(f"rpc_name: {rpc.name}, model_name: {rpc.model_name}, "
                  f"interface_type: {rpc.interface_type}")
    return rpc_list


def dump_search_settings(rpcs: List[ModelRPC], device_mesh: DeviceMesh, num_gen_tokens: int,
                         n_ppo_minibatches: int):
    dump_dir = "/home/meizy/model_device_mapping_search/test_case/"
    rpc_exe_list = make_rpc_exe_list(rpcs,
                                     device_mesh,
                                     num_gen_tokens=num_gen_tokens,
                                     n_ppo_minibatches=n_ppo_minibatches,
                                     log_dir=dump_dir + "rpc_exe_list.txt",
                                     if_print=True)
    rpc_list = make_rpc_list(rpcs, if_print=True)
    graph = build_graph(rpcs, 5, 1, if_print=True)
    comm_stats_ = comm_stats(if_print=True)
    model_size_dict = make_model_size_dict(rpcs, if_print=True)
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


def handpicked_model_device_mapping(
    device_mesh: ClusterDeviceMesh,
    model_rpcs: List[ModelRPC],
    model_configs: Dict[str, FlashMQATConfig],
    nodelist: Optional[str] = None,
    profile_layers: bool = False,
    num_gen_tokens: int = 256,
    n_ppo_minibatches: int = 1,
    mode: Literal["data_pipe", "model_pipe"] = "model_pipe",
) -> Dict[str, RPCAllocation]:
    n_nodes = device_mesh.n_nodes

    rollout, rew_inf, ref_inf, critic_inf, actor_train, critic_train = model_rpcs
    rew_inf.post_hooks.append(OffloadHook())
    ref_inf.post_hooks.append(OffloadHook())

    mapping = np.array([[1, 1, 1, 1, 1, 1, 1, 1]] * n_nodes)

    if mode == "model_pipe":
        parallel_config = ParallelismConfig(model_parallel_size=8,
                                            pipeline_parallel_size=n_nodes,
                                            data_parallel_size=1,
                                            use_sequence_parallel=True)
    elif mode == "data_pipe":
        parallel_config = ParallelismConfig(model_parallel_size=1,
                                            pipeline_parallel_size=n_nodes,
                                            data_parallel_size=8,
                                            use_sequence_parallel=True)
    elif mode == "full_model":
        parallel_config = ParallelismConfig(model_parallel_size=8 * n_nodes,
                                            pipeline_parallel_size=1,
                                            data_parallel_size=1,
                                            use_sequence_parallel=True)

    return {
        rollout.name: RPCAllocation(
            rpc=rollout,
            mapping=mapping,
            train_eval_config=ModelTrainEvalConfig(
                type="llama",
                path=MODEL_TYPE_TO_PATH[rollout.model_type],
                base_model_path=MODEL_TYPE_TO_PATH[rollout.model_type],
                gradient_checkpointing=True,
                parallel=parallel_config,
                optimizer=OptimizerConfig(type="adam", offload=False),
            ),
        ),
        rew_inf.name: RPCAllocation(
            rpc=rew_inf,
            mapping=mapping,
            train_eval_config=ModelTrainEvalConfig(
                type="llama",
                path=MODEL_TYPE_TO_PATH[rollout.model_type],
                base_model_path=MODEL_TYPE_TO_PATH[rollout.model_type],
                parallel=parallel_config,
            ),
        ),
        ref_inf.name: RPCAllocation(
            rpc=ref_inf,
            mapping=mapping,
            train_eval_config=ModelTrainEvalConfig(
                type="llama",
                path=MODEL_TYPE_TO_PATH[rollout.model_type],
                base_model_path=MODEL_TYPE_TO_PATH[rollout.model_type],
                parallel=parallel_config,
            ),
        ),
        critic_inf.name: RPCAllocation(
            rpc=critic_inf,
            mapping=mapping,
            train_eval_config=ModelTrainEvalConfig(
                type="llama",
                path=MODEL_TYPE_TO_PATH[rollout.model_type],
                base_model_path=MODEL_TYPE_TO_PATH[rollout.model_type],
                gradient_checkpointing=True,
                parallel=parallel_config,
                optimizer=OptimizerConfig(type="adam", offload=False),
            ),
        ),
        critic_train.name: RPCAllocation(
            rpc=critic_train,
            mapping=mapping,
            train_eval_config=ModelTrainEvalConfig(
                type="llama",
                path=MODEL_TYPE_TO_PATH[rollout.model_type],
                base_model_path=MODEL_TYPE_TO_PATH[rollout.model_type],
                gradient_checkpointing=True,
                parallel=parallel_config,
                optimizer=OptimizerConfig(type="adam", offload=False),
            ),
        ),
        actor_train.name: RPCAllocation(
            rpc=actor_train,
            mapping=mapping,
            train_eval_config=ModelTrainEvalConfig(
                type="llama",
                path=MODEL_TYPE_TO_PATH[rollout.model_type],
                base_model_path=MODEL_TYPE_TO_PATH[rollout.model_type],
                gradient_checkpointing=True,
                parallel=parallel_config,
                optimizer=OptimizerConfig(type="adam", offload=False),
            ),
        ),
    }


model_pipe_device_mapping = functools.partial(handpicked_model_device_mapping, mode="model_pipe")
data_pipe_device_mapping = functools.partial(handpicked_model_device_mapping, mode="data_pipe")
full_model_device_mapping = functools.partial(handpicked_model_device_mapping, mode="full_model")


def test_model_device_mapping(
    device_mesh: ClusterDeviceMesh,
    model_rpcs: List[ModelRPC],
    model_configs: Dict[str, FlashMQATConfig],
    nodelist: Optional[str] = None,
    profile_layers: bool = False,
    num_gen_tokens: int = 256,
    n_ppo_minibatches: int = 1,
    mode: Literal["data_pipe", "model_pipe"] = "model_pipe",
) -> Dict[str, RPCAllocation]:
    n_nodes = device_mesh.n_nodes

    rollout, rew_inf, ref_inf, critic_inf, actor_train, critic_train = model_rpcs

    actor_train.pre_hooks.append(SyncParamHook(source=ModelName("actor", 0)))
    actor_train.model_name = ModelName("actor", 1)
    actor_train.post_hooks.append(SyncParamHook(target=ModelName("actor", 0)))
    rew_inf.post_hooks.append(OffloadHook())
    ref_inf.post_hooks.append(OffloadHook())

    mapping = np.array([[1, 1, 1, 1, 1, 1, 1, 1]] * n_nodes)

    parallel_config = ParallelismConfig(model_parallel_size=8,
                                        pipeline_parallel_size=n_nodes,
                                        data_parallel_size=1,
                                        use_sequence_parallel=True)

    rollout_config = ParallelismConfig(model_parallel_size=2,
                                       data_parallel_size=8,
                                       pipeline_parallel_size=n_nodes // 2)

    return {
        rollout.name: RPCAllocation(
            rpc=rollout,
            mapping=mapping,
            train_eval_config=ModelTrainEvalConfig(
                type="llama",
                path=MODEL_TYPE_TO_PATH[rollout.model_type],
                base_model_path=MODEL_TYPE_TO_PATH[rollout.model_type],
                gradient_checkpointing=True,
                parallel=rollout_config,
            ),
        ),
        rew_inf.name: RPCAllocation(
            rpc=rew_inf,
            mapping=mapping,
            train_eval_config=ModelTrainEvalConfig(
                type="llama",
                path=MODEL_TYPE_TO_PATH[rollout.model_type],
                base_model_path=MODEL_TYPE_TO_PATH[rollout.model_type],
                parallel=parallel_config,
            ),
        ),
        ref_inf.name: RPCAllocation(
            rpc=ref_inf,
            mapping=mapping,
            train_eval_config=ModelTrainEvalConfig(
                type="llama",
                path=MODEL_TYPE_TO_PATH[rollout.model_type],
                base_model_path=MODEL_TYPE_TO_PATH[rollout.model_type],
                parallel=parallel_config,
            ),
        ),
        critic_inf.name: RPCAllocation(
            rpc=critic_inf,
            mapping=mapping,
            train_eval_config=ModelTrainEvalConfig(
                type="llama",
                path=MODEL_TYPE_TO_PATH[rollout.model_type],
                base_model_path=MODEL_TYPE_TO_PATH[rollout.model_type],
                gradient_checkpointing=True,
                parallel=parallel_config,
                optimizer=OptimizerConfig(type="adam", offload=False),
            ),
        ),
        critic_train.name: RPCAllocation(
            rpc=critic_train,
            mapping=mapping,
            train_eval_config=ModelTrainEvalConfig(
                type="llama",
                path=MODEL_TYPE_TO_PATH[rollout.model_type],
                base_model_path=MODEL_TYPE_TO_PATH[rollout.model_type],
                gradient_checkpointing=True,
                parallel=parallel_config,
                optimizer=OptimizerConfig(type="adam", offload=False),
            ),
        ),
        actor_train.name: RPCAllocation(
            rpc=actor_train,
            mapping=mapping,
            train_eval_config=ModelTrainEvalConfig(
                type="llama",
                path=MODEL_TYPE_TO_PATH[rollout.model_type],
                base_model_path=MODEL_TYPE_TO_PATH[rollout.model_type],
                gradient_checkpointing=True,
                parallel=parallel_config,
                optimizer=OptimizerConfig(type="adam", offload=False),
            ),
        ),
    }


def profile_search(rank):
    # bs_list = [128, 256, 512]
    # seqlen_list = [896, 384, 128]
    model_sizes = [7, 13, 34, 70]
    bs_list = [128, 256, 512]
    seqlen_list = [896, 384, 128]
    time_limits = [1800, 1800, 1800, 1800]
    n_nodes_list = [1, 2, 4, 8]
    modes = ["actor", "critic", "both"]
    for mode in modes:
        # for bs, seqlen in zip(bs_list, seqlen_list):
        bs = bs_list[rank]
        seqlen = seqlen_list[rank]
        for model_size, n_nodes, time_limit in zip(model_sizes, n_nodes_list, time_limits):
            # nodelist = "QH-com20"
            cluster_device_mesh = ClusterDeviceMesh(n_nodes=n_nodes, n_gpus_per_node=8, mem=80)
            if mode == "actor":
                rpcs = ppo_rpcs_example(model_size, 7, bs, seqlen)
                exp_name = f"ams-{model_size}_cms-7_bs-{bs}_seqlen-{seqlen}_n-{n_nodes}"
            elif mode == "critic":
                rpcs = ppo_rpcs_example(7, model_size, bs, seqlen)
                exp_name = f"ams-7_cms-{model_size}_bs-{bs}_seqlen-{seqlen}_n-{n_nodes}"
            elif mode == "both":
                n_nodes *= 2
                rpcs = ppo_rpcs_example(model_size, model_size, bs, seqlen)
                exp_name = f"ams-{model_size}_cms-{model_size}_bs-{bs}_seqlen-{seqlen}_n-{n_nodes}"

            node_start = 40
            node_end = node_start + n_nodes - 1
            nodelist = f"QH-com[{node_start:02d}-{node_end:02d}]"

            dump_dir = f"profile_result/search/{exp_name}.pkl"
            print(exp_name, nodelist)
            device_mesh = make_device_mesh_from_name(nodelist)

            rpc_exe_list = make_rpc_exe_list(rpcs,
                                             device_mesh,
                                             num_gen_tokens=seqlen,
                                             n_ppo_minibatches=4,
                                             log_dir=None,
                                             if_print=False)
            rpc_list = make_rpc_list(rpcs, if_print=False)
            graph = build_graph(rpcs, 5, 1, if_print=False)
            cost_table = pickle.load(open("profile_result/param_sync_cost_table_parallel.pkl", "rb"))
            model_size_dict = make_model_size_dict(rpcs, if_print=False)

            rs = mdm_search.mcmc_search_time_profile(rpc_list, rpc_exe_list, graph, cost_table,
                                                     model_size_dict, 0.0001, time_limit)
            import pprint
            pprint.pprint(rs)

            with open(dump_dir, "wb") as f:
                pickle.dump(rs, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int)
    args = parser.parse_args()
    profile_search(rank=args.rank)

    # experiment = config_package.make_experiment(args.expr_name)

    # os.environ.setdefault("IS_REMOTE", "0")
    # bs = 128
    # seqlen = 896
    # size = 7
    # n_nodes = 1
    # # node_start = 20
    # # node_end = node_start + n_nodes - 1
    # # nodelist = f"QH-com[{node_start:02d}-{node_end:02d}]"
    # nodelist = "QH-com20"
    # cluster_device_mesh = ClusterDeviceMesh(n_nodes=n_nodes, n_gpus_per_node=8, mem=80)
    # rpcs = ppo_rpcs_example(size, size, bs, seqlen)
    # device_mesh = make_device_mesh_from_name(nodelist)

    # # from reallm.profiler.device_mesh import find_sub_device_meshes
    # # import pprint
    # # pprint.pprint(find_sub_device_meshes(device_mesh))

    # dump_search_settings(rpcs, device_mesh, num_gen_tokens=seqlen, n_ppo_minibatches=4)
    # optimal_device_mapping(cluster_device_mesh, rpcs, None, nodelist)

    # 7b 13b 2x8:
    # [0, 2, 2, 1, 11, 3, ]
    # [0, 2, 0, 8, 33, 1, ] = 75s

    # 7b 34b 4x8
    # [0, 0, 9, 1, 4, 3, ] 87441235
    # [0, 0, 9, 1, 1, 3, ] 86986420
    # print_model_device_mapping_by_index(rpcs, device_mesh, [0, 0, 9, 1, 1, 3, ])
    # profile_search()
