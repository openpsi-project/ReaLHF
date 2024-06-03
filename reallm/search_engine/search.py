from typing import Dict, List, Literal, Optional
import argparse
import functools
import os
import pickle
import re

import numpy as np

from reallm.search_engine.enumerate import build_graph, enumerate_rpc_executions
from reallm.search_engine.estimate import get_param_realloc_stats, load_model_config

try:
    import reallm._C.mdm_search as mdm_search
except ModuleNotFoundError:
    mdm_search = None

from reallm.api.core.dfg import ModelName, ModelRPC, OffloadHook, SyncParamHook
from reallm.api.core.model_api import ModelFamily, ReaLModelConfig
from reallm.api.quickstart.device_mesh import DeviceMesh, make_device_mesh_from_name, RPCAllocation
from reallm.api.quickstart.model import ModelTrainEvalConfig, OptimizerConfig, ParallelismConfig
from reallm.api.quickstart.search import RPCExecution, RPCInstance
import reallm.base.constants as constants


def search_rpc_allocations(
    device_mesh: DeviceMesh,
    model_rpcs: Dict[str, ModelRPC],
    num_gen_tokens: int = 256,
    n_ppo_minibatches: int = 1,
    seq_len: int = 256,
) -> Dict[str, RPCAllocation]:
    from_file = os.environ.get("IS_REMOTE", "0") == "1"
    dump_dir = os.path.join(constants.LOG_ROOT, constants.experiment_name(), constants.trial_name(),
                            "device_mapping.pkl")
    log_dir = os.path.join(constants.LOG_ROOT, constants.experiment_name(), constants.trial_name(),
                           "device_mapping")
    rs_dir = os.path.join(constants.LOG_ROOT, constants.experiment_name(), constants.trial_name(),
                          "raw_search_result")
    rpc_exe_dir = os.path.join(constants.LOG_ROOT, constants.experiment_name(), constants.trial_name(),
                               "rpc_exe_info")

    if from_file:
        with open(dump_dir, "rb") as f:
            return pickle.load(f)
    else:
        os.makedirs(os.path.dirname(dump_dir), exist_ok=True)

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

    n_nodes = device_mesh.n_nodes
    table = {}
    for rpc_name, rpc in model_rpcs.items():
        print(f"Getting param realloc stats for {rpc.model_type} at {rpc.model_path}")
        t = get_param_realloc_stats(rpc.model_type, rpc.model_path, n_nodes, True)
        table.update(t)

    print(len(t))
    rpc_exe_list = make_rpc_exe_list(
        model_rpcs,
        device_mesh,
        num_gen_tokens=num_gen_tokens,
        n_ppo_minibatches=n_ppo_minibatches,
        seq_len=seq_len,
        # log_dir=rpc_exe_dir,
        if_print=True)
    rpc_list = list(model_rpcs.values())
    graph = build_graph(rpc_list, 5, 1, if_print=True)
    model_size_dict = make_model_size_dict(model_rpcs, if_print=True)
    # rpc_dict = {rpc.name: rpc for rpc in model_rpcs}

    n_nodes = device_mesh.n_nodes
    search_time = 120  # 60 * n_nodes

    rs: List[Dict[str, List]] = mdm_search.multi_mcmc_search(
        rpc_list,
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
    r = rs[-1]
    print(r)

    # hack, only suitable for configs in experiments/autoexp/auto_ppo.py
    # rollout, rew_inf, ref_inf, critic_inf, actor_train, critic_train = model_rpcs
    # rollout_topo = (r[rollout.name]["device_mesh"], r[rollout.name]["num_pp"], r[rollout.name]["num_dp"],
    #                 r[rollout.name]["num_mp"])
    # actor_train_topo = (r[actor_train.name]["device_mesh"], r[actor_train.name]["num_pp"],
    #                     r[actor_train.name]["num_dp"], r[actor_train.name]["num_mp"])
    # old_name = actor_train.name
    # optim_model_name = []
    # if rollout_topo != actor_train_topo:
    #     actor_train.pre_hooks.append(SyncParamHook(source=ModelName("actor", 0)))
    #     actor_train.model_name = ModelName("actor", 1)
    #     actor_train.post_hooks.append(SyncParamHook(target=ModelName("actor", 0)))
    #     r[actor_train.name] = r.pop(old_name)
    # else:
    #     actor_train.model_name = ModelName("actor", 0)
    #     rollout.model_name = ModelName("actor", 0)

    # critic_inf_topo = (r[critic_inf.name]["device_mesh"], r[critic_inf.name]["num_pp"],
    #                    r[critic_inf.name]["num_dp"], r[critic_inf.name]["num_mp"])
    # critic_train_topo = (r[critic_train.name]["device_mesh"], r[critic_train.name]["num_pp"],
    #                      r[critic_train.name]["num_dp"], r[critic_train.name]["num_mp"])
    # old_name = critic_train.name
    # if critic_inf_topo != critic_train_topo:
    #     critic_train.pre_hooks.append(SyncParamHook(source=ModelName("critic", 0)))
    #     critic_train.model_name = ModelName("critic", 1)
    #     critic_train.post_hooks.append(SyncParamHook(target=ModelName("critic", 0)))
    #     r[critic_train.name] = r.pop(old_name)
    # else:
    #     critic_train.model_name = ModelName("critic", 0)
    #     critic_inf.model_name = ModelName("critic", 0)

    # rew_inf.post_hooks.append(OffloadHook())
    # ref_inf.post_hooks.append(OffloadHook())

    # rpc_dict = {rpc.name: rpc for rpc in model_rpcs}
    # # import pprint
    # # pprint.pprint(rpc_dict)

    # rpc_alloc_dict = {}
    # for rpc_name, alloc_info in r.items():
    #     if rpc_name in ["end_time", "mem_cost"]:
    #         continue
    #     rpc = rpc_dict[rpc_name]
    #     parallel = ParallelismConfig(
    #         pipeline_parallel_size=alloc_info["num_pp"],
    #         data_parallel_size=alloc_info["num_dp"],
    #         model_parallel_size=alloc_info["num_mp"],
    #     )
    #     # use_sequence_parallel = False\
    #     #     if rpc.model_name.role in ["reward", "ref"] else (ps.num_mp > 1)
    #     use_sequence_parallel = parallel.model_parallel_size > 1
    #     gradient_checkpointing = True
    #     if rpc.model_name in [actor_train.model_name, critic_train.model_name]:
    #         optim_config = OptimizerConfig(type="adam", offload=False)
    #     else:
    #         optim_config = OptimizerConfig()
    #     # optim_config = OptimizerConfig(type="adam", offload=False)\
    #     #     if rpc.interface_type == ModelInterfaceType.TRAIN_STEP else OptimizerConfig()
    #     # print(alloc_info["device_mesh"])
    #     sub_device_mesh = make_device_mesh_from_name(alloc_info["device_mesh"])
    #     train_eval_config = ModelTrainEvalConfig(
    #         type=rpc.model_type._class,
    #         path=rpc.model_path,
    #         base_model_path=rpc.model_path,
    #         gradient_checkpointing=gradient_checkpointing,
    #         parallel=parallel,
    #         optimizer=optim_config,
    #     )
    #     rpc_alloc = RPCAllocation(
    #         rpc=rpc,
    #         device_mesh=sub_device_mesh,
    #         train_eval_config=train_eval_config,
    #     )
    #     rpc_alloc_dict[rpc_name] = rpc_alloc

    # if not from_file:
    #     with open(dump_dir, "wb") as f:
    #         pickle.dump(rpc_alloc_dict, f)
    #     with open(log_dir, "w") as f:
    #         import pprint
    #         pprint.pprint(rpc_alloc_dict, stream=f)

    # return rpc_alloc_dict


def make_rpc_exe_list(rpcs: Dict[str, ModelRPC],
                      device_mesh: DeviceMesh,
                      num_gen_tokens: int,
                      n_ppo_minibatches: int,
                      seq_len: int,
                      if_print: bool = False,
                      log_dir: Optional[str] = None) -> List[RPCExecution]:
    rpc_exe_list = []
    log_flag = False
    for rpc_name, rpc in rpcs.items():
        # flash_mqat_config = load_model_config(rpc)
        feasible = enumerate_rpc_executions(rpc,
                                            device_mesh,
                                            seq_len=seq_len,
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


def make_model_size_dict(rpcs: Dict[str, ModelRPC], if_print: bool = False) -> Dict[str, int]:
    model_size_dict = {}

    for rpc_name, rpc in rpcs.items():
        if rpc.model_name.role in model_size_dict:
            continue
        # model_configs = load_model_config(rpc)
        # model_size_dict[rpc.model_name.role] = estimate_model_size(flash_mqat_config)
        model_size_dict[rpc.model_name.role] = rpc.model_type.size

        if if_print:
            print(f"model_name: {rpc.model_name.role}, "
                  f"model_size: {rpc.model_type.size}")
    return model_size_dict


# def make_rpc_list(rpcs: List[ModelRPC], if_print: bool = False) -> List[ModelRPC]:
#     rpc_list = []
#     for rpc in rpcs:
#         rpc_list.append(ModelRPC)
#         if if_print:
#             print(f"rpc_name: {rpc.name}, model_name: {rpc.model_name}, "
#                   f"interface_type: {rpc.interface_type}")
#     return rpc_list


def dump_search_settings(rpcs: Dict[str, ModelRPC], device_mesh: DeviceMesh, num_gen_tokens: int,
                         n_ppo_minibatches: int):
    dump_dir = "/home/meizy/model_device_mapping_search/test_case/"
    rpc_exe_list = make_rpc_exe_list(rpcs,
                                     device_mesh,
                                     num_gen_tokens=num_gen_tokens,
                                     n_ppo_minibatches=n_ppo_minibatches,
                                     log_dir=dump_dir + "rpc_exe_list.txt",
                                     if_print=True)
    rpc_list = list(rpcs.values())
    graph = build_graph(rpcs, 5, 1, if_print=True)
    model_size_dict = make_model_size_dict(rpcs, if_print=True)
    with open(dump_dir + "rpc_list.pkl", "wb") as f:
        pickle.dump(rpc_list, f)
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
    device_mesh: DeviceMesh,
    model_rpcs: List[ModelRPC],
    model_configs: Dict[str, ReaLModelConfig],
    nodelist: Optional[str] = None,
    profile_layers: bool = False,
    num_gen_tokens: int = 256,
    n_ppo_minibatches: int = 1,
    mode: Literal["data_pipe", "model_pipe"] = "model_pipe",
) -> Dict[str, RPCAllocation]:
    n_nodes = device_mesh.cluster_n_nodes

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
                path=rollout.model_path,
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
                path=rew_inf.model_path,
                parallel=parallel_config,
            ),
        ),
        ref_inf.name: RPCAllocation(
            rpc=ref_inf,
            mapping=mapping,
            train_eval_config=ModelTrainEvalConfig(
                type="llama",
                path=ref_inf.model_path,
                parallel=parallel_config,
            ),
        ),
        critic_inf.name: RPCAllocation(
            rpc=critic_inf,
            mapping=mapping,
            train_eval_config=ModelTrainEvalConfig(
                type="llama",
                path=critic_inf.model_path,
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
                path=critic_train.model_path,
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
                path=actor_train.model_path,
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
    device_mesh: DeviceMesh,
    model_rpcs: List[ModelRPC],
    model_configs: Dict[str, ReaLModelConfig],
    nodelist: Optional[str] = None,
    profile_layers: bool = False,
    num_gen_tokens: int = 256,
    n_ppo_minibatches: int = 1,
    mode: Literal["data_pipe", "model_pipe"] = "model_pipe",
) -> Dict[str, RPCAllocation]:
    n_nodes = device_mesh.n_nodes

    rollout, rew_inf, ref_inf, critic_inf, actor_train, critic_train = model_rpcs

    assert n_nodes == 4
    actor_train.pre_hooks.append(SyncParamHook(source=ModelName("actor", 0)))
    actor_train.model_name = ModelName("actor", 1)
    actor_train.post_hooks.append(SyncParamHook(target=ModelName("actor", 0)))

    critic_train.pre_hooks.append(SyncParamHook(source=ModelName("critic", 0)))
    critic_train.model_name = ModelName("critic", 1)
    critic_train.post_hooks.append(SyncParamHook(target=ModelName("critic", 0)))
    rew_inf.post_hooks.append(OffloadHook())
    ref_inf.post_hooks.append(OffloadHook())

    # mapping = np.array([[1, 1, 1, 1, 1, 1, 1, 1]] * n_nodes)
    rollout_mapping = np.array([[1, 1, 1, 1, 1, 1, 1, 1]] * 2 + [[0, 0, 0, 0, 0, 0, 0, 0]] * 2)
    rollout_config = ParallelismConfig(model_parallel_size=1, pipeline_parallel_size=2, data_parallel_size=8)

    actor_train_mapping = np.array([[1, 1, 1, 1, 1, 1, 1, 1]] * 2 + [[0, 0, 0, 0, 0, 0, 0, 0]] * 2)
    actor_train_config = ParallelismConfig(model_parallel_size=1,
                                           pipeline_parallel_size=8,
                                           data_parallel_size=2)

    critic_train_mapping = np.array([[0, 0, 0, 0, 0, 0, 0, 0]] * 2 + [[1, 1, 1, 1, 1, 1, 1, 1]] * 2)
    critic_train_config = ParallelismConfig(model_parallel_size=4,
                                            pipeline_parallel_size=4,
                                            data_parallel_size=1,
                                            use_sequence_parallel=True)

    critic_inf_mapping = np.array([[0, 0, 0, 0, 0, 0, 0, 0]] * 2 + [[1, 1, 1, 1, 1, 1, 1, 1]] * 2)
    critic_inf_config = ParallelismConfig(model_parallel_size=2,
                                          pipeline_parallel_size=4,
                                          data_parallel_size=2,
                                          use_sequence_parallel=True)

    rew_inf_mapping = np.array([[1, 1, 1, 1, 1, 1, 1, 1]] * 2 + [[0, 0, 0, 0, 0, 0, 0, 0]] * 2)
    rew_inf_config = ParallelismConfig(model_parallel_size=2,
                                       pipeline_parallel_size=8,
                                       data_parallel_size=1,
                                       use_sequence_parallel=True)

    ref_inf_mapping = np.array([[1, 1, 1, 1, 1, 1, 1, 1]] * 2 + [[0, 0, 0, 0, 0, 0, 0, 0]] * 2)
    ref_inf_config = ParallelismConfig(model_parallel_size=4,
                                       pipeline_parallel_size=1,
                                       data_parallel_size=4,
                                       use_sequence_parallel=True)

    return {
        rollout.name: RPCAllocation(
            rpc=rollout,
            mapping=rollout_mapping,
            train_eval_config=ModelTrainEvalConfig(
                type="llama",
                path=rollout.model_path,
                gradient_checkpointing=True,
                parallel=rollout_config,
            ),
        ),
        rew_inf.name: RPCAllocation(
            rpc=rew_inf,
            mapping=rew_inf_mapping,
            train_eval_config=ModelTrainEvalConfig(
                type="llama",
                path=rew_inf.model_path,
                parallel=rew_inf_config,
            ),
        ),
        ref_inf.name: RPCAllocation(
            rpc=ref_inf,
            mapping=ref_inf_mapping,
            train_eval_config=ModelTrainEvalConfig(
                type="llama",
                path=ref_inf.model_path,
                parallel=ref_inf_config,
            ),
        ),
        critic_inf.name: RPCAllocation(
            rpc=critic_inf,
            mapping=critic_inf_mapping,
            train_eval_config=ModelTrainEvalConfig(
                type="llama",
                path=critic_inf.model_path,
                gradient_checkpointing=True,
                parallel=critic_inf_config,
                # optimizer=OptimizerConfig(type="adam", offload=False),
            ),
        ),
        critic_train.name: RPCAllocation(
            rpc=critic_train,
            mapping=critic_train_mapping,
            train_eval_config=ModelTrainEvalConfig(
                type="llama",
                path=critic_train.model_path,
                gradient_checkpointing=True,
                parallel=critic_train_config,
                optimizer=OptimizerConfig(type="adam", offload=False),
            ),
        ),
        actor_train.name: RPCAllocation(
            rpc=actor_train,
            mapping=actor_train_mapping,
            train_eval_config=ModelTrainEvalConfig(
                type="llama",
                path=actor_train.model_path,
                gradient_checkpointing=True,
                parallel=actor_train_config,
                optimizer=OptimizerConfig(type="adam", offload=False),
            ),
        ),
    }


# def profile_search(rank):
# bs_list = [128, 256, 512]
# seqlen_list = [896, 384, 128]
# model_sizes = [7, 13, 34, 70]
# bs_list = [128, 256, 512]
# seqlen_list = [896, 384, 128]
# time_limits = [120, 120, 120, 120]
# n_nodes_list = [1, 2, 4, 8]
# modes = ["actor"]  # , "critic", "both"]
# for rd in range(10)[rank::3]:
#     for mode in modes:
#         # for bs, seqlen in zip(bs_list, seqlen_list):
#         bs = bs_list[rank]
#         seqlen = seqlen_list[rank]
#         for model_size, n_nodes, time_limit in zip(model_sizes, n_nodes_list, time_limits):
#             # nodelist = "QH-com20"
#             cluster_device_mesh = ClusterDeviceMesh(n_nodes=n_nodes, n_gpus_per_node=8, mem=80)
#             if mode == "actor":
#                 rpcs = ppo_rpcs_example(model_size, 7, bs, seqlen)
#                 exp_name = f"ams-{model_size}_cms-7_bs-{bs}_seqlen-{seqlen}_n-{n_nodes}_r-{rd}"
#             elif mode == "critic":
#                 rpcs = ppo_rpcs_example(7, model_size, bs, seqlen)
#                 exp_name = f"ams-7_cms-{model_size}_bs-{bs}_seqlen-{seqlen}_n-{n_nodes}_r-{rd}"
#             elif mode == "both":
#                 n_nodes *= 2
#                 rpcs = ppo_rpcs_example(model_size, model_size, bs, seqlen)
#                 exp_name = f"ams-{model_size}_cms-{model_size}_bs-{bs}_seqlen-{seqlen}_n-{n_nodes}_r-{rd}"

#             node_start = 40
#             node_end = node_start + n_nodes - 1
#             nodelist = f"QH-com[{node_start:02d}-{node_end:02d}]"

#             dump_dir = f"profile_result/search_new/{exp_name}.pkl"
#             print(exp_name, nodelist)
#             device_mesh = make_device_mesh_from_name(nodelist)

#             rpc_exe_list = make_rpc_exe_list(rpcs,
#                                              device_mesh,
#                                              num_gen_tokens=seqlen,
#                                              n_ppo_minibatches=4,
#                                              log_dir=None,
#                                              if_print=False)
#             rpc_list = make_rpc_list(rpcs, if_print=False)
#             graph = build_graph(rpcs, 5, 1, if_print=False)
#             table = pickle.load(open("profile_result/param_sync_table_parallel.pkl", "rb"))
#             model_size_dict = make_model_size_dict(rpcs, if_print=False)

#             rs = mdm_search.mcmc_search_time_profile(rpc_list, rpc_exe_list, graph, table,
#                                                      model_size_dict, 0.0001, time_limit)
#             import pprint
#             pprint.pprint(rs)

#             with open(dump_dir, "wb") as f:
#                 pickle.dump(rs, f)

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
    exp = PPOConfig(experiment_name="test",
                    trial_name="test",
                    global_train_bs=256,
                    global_gen_bs=256,
                    n_nodes=1,
                    n_gpus_per_node=8,
                    nodelist="QH-com48",
                    actor=actor,
                    critic=critic,
                    ref=ref,
                    rew=rew,
                    ppo=ppo_cfg)
    device_mesh = make_device_mesh_from_name(exp.nodelist, exp.nodelist)
    print(device_mesh.global_mesh_name)
    search_rpc_allocations(
        device_mesh=device_mesh,
        model_rpcs=exp.rpcs,
        num_gen_tokens=256,
        n_ppo_minibatches=4,
        seq_len=256,
    )
