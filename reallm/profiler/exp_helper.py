from typing import Dict, List, Literal, Optional
import argparse
import functools
import os
import pickle
import pprint

import numpy as np

from reallm.api.core.config import MODEL_TYPE_TO_PATH
from reallm.api.core.dfg import ModelInterfaceType, ModelName, ModelRPC, OffloadHook, SyncParamHook
from reallm.api.quickstart.device_mesh import ClusterDeviceMesh, RPCAllocation
from reallm.api.quickstart.model import ModelTrainEvalConfig, OptimizerConfig, ParallelismConfig
from reallm.impl.model.nn.real_llm_base import FlashMQATConfig
from reallm.profiler.device_mesh import DeviceMesh, make_device_mesh_from_name, ModelParallelStrategy
from reallm.profiler.enumerate import build_graph, enumerate_rpc_executions
from reallm.profiler.estimate import (comm_stats, estimate_model_size, estimate_rpc_memory, estimate_rpc_time,
                                      load_model_config)
from reallm.profiler.experiments import ppo_rpcs_example
from reallm.profiler.profile_layers import profile_rpcs
from reallm.profiler.rpc import RPC, RPCExecution
import reallm.api.core.system_api as config_package
# import reallm.experiments.autoexp.auto_ppo
import reallm.base.constants
import reallm.profiler.cppsearch.mdm_search as mdm_search


def file_to_allocation_pkl(  # model_rpcs: List[ModelRPC],
        nodelist: str, from_fp: str, to_fp: str, node_map: Dict[int, int] = None):
    load_path = os.path.join(from_fp, "raw_search_result")
    with open(load_path, "r") as f:
        rs = eval(f.read())
    dm_load_path = os.path.join(from_fp, "device_mapping.pkl")
    with open(dm_load_path, "rb") as f:
        old_dm = pickle.load(f)
    # pprint.pprint(old_dm)
    model_rpcs = {k: rpc_alloc.rpc for k, rpc_alloc in old_dm.items()}

    r = rs[-1]
    pprint.pprint(r)

    if node_map is not None:
        for k, v in r.items():
            if not isinstance(v, dict):
                continue
            for from_node, to_node in node_map.items():
                v["device_mesh"] = v["device_mesh"].replace(str(from_node), str(to_node))
    # old_r = rs[-1]
    # r = {}
    # for k, v in old_r.items():
    #     # k: str
    #     kk = k.replace("1", "0")
    #     r[kk] = v
    pprint.pprint(r)

    # print(r)
    search_device_mesh = make_device_mesh_from_name(nodelist)
    # hack, only suitable for configs in reallm.experiments/autoexp/auto_ppo.py
    for k, v in model_rpcs.items():
        v: ModelRPC
        print(v)
        if v.model_name.role == "actor":
            if v.interface_type == ModelInterfaceType.GENERATE:
                rollout = v
            elif v.interface_type == ModelInterfaceType.TRAIN_STEP:
                actor_train = v
        elif v.model_name.role == "critic":
            if v.interface_type == ModelInterfaceType.INFERENCE:
                critic_inf = v
            elif v.interface_type == ModelInterfaceType.TRAIN_STEP:
                critic_train = v
        elif v.model_name.role == "reward":
            rew_inf = v
        elif v.model_name.role == "ref":
            ref_inf = v

    # rollout, rew_inf, ref_inf, critic_inf, actor_train, critic_train = model_rpcs

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

    rpc_dict = {rpc.name: rpc for rpc in model_rpcs.values()}
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

    dump_path = os.path.join(to_fp, "device_mapping.pkl")

    pprint.pprint(rpc_alloc_dict)
    with open(dump_path, "wb") as f:
        pickle.dump(rpc_alloc_dict, f)


if __name__ == "__main__":
    # expr_name = "sosp-a34s384g256t256-s"
    # experiment = config_package.make_experiment(expr_name)
    from_fp = "/lustre/aigc/llm/logs/meizy/sosp-a34s384g256t256-s/20240408-0"
    fp = "/lustre/aigc/llm/logs/meizy/sosp-a34s384g256t256-s/timemark/"
    # rpcs = ppo_rpcs_example(34, 7, 256, 384)
    nodelist = "QH-com[25-28]"
    node_map = {m: m - 16 for m in range(41, 45)}
    # file_to_allocation_pkl(rpcs, nodelist, fp)
    file_to_allocation_pkl(nodelist, from_fp, fp, node_map)
