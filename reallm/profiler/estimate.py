# estimate a fucntion-call level execution time for device mesh enumerate pruning
# assume one batch of data passes through all rpcs once
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Optional
import argparse
import dataclasses
import enum
import getpass
import itertools
import json
import os
import pprint

from reallm.api.core.config import MODEL_TYPE_TO_PATH
from reallm.api.core.dfg import ModelInterfaceType
from reallm.api.quickstart.device_mesh import ClusterDeviceMesh, RPCAllocation
from reallm.impl.model.nn.real_llm_api import ReaLModel
from reallm.impl.model.nn.real_llm_base import FlashMQATConfig
from reallm.profiler.device_mesh import *
from reallm.profiler.experiments import ProfileExperiment
from reallm.profiler.rpc import CommStats
from reallm.profiler.utils import make_stats_key
import reallm.api.core.system_api as config_package
import reallm.base.cluster

# import reallm.base.logging as logging

IF_PRINT = False


def log_debug(*args, **kwargs):
    if IF_PRINT:
        print(*args, **kwargs)


PROFILE_RESULT_PATH = os.path.join(
    reallm.base.cluster.spec.fileroot,
    "logs",
    getpass.getuser(),
    "profile",
    "profile",
    "profile_result",
)
PROFILE_NSYNC_RESULT_PATH = os.path.join(
    reallm.base.cluster.spec.fileroot,
    "logs",
    getpass.getuser(),
    "profile",
    "profile_nsync",
    "profile_result",
)


def parse_layer_stats_file_name(fn):
    _, num_mp = fn.split(".")[0].split("-")
    return int(num_mp)


def parse_comm_stats_file_name(fn):
    # example: summary-QH-com40-0.json
    local_gpu_id = int(fn.split(".")[0].split("-")[-1])
    host_name = fn.split(".")[0].split("-")[1:-1]
    host_name = "-".join(host_name)
    return host_name, local_gpu_id


def parse_layer_stats_files(stats_file_dir):
    res = {}
    for fn in os.listdir(stats_file_dir):
        fn: str
        if not fn.startswith("summary"):
            continue
        num_mp = parse_layer_stats_file_name(fn)
        stats = json.load(open(os.path.join(stats_file_dir, fn)))
        res[num_mp] = stats
    return res


def parse_comm_stats_files(stats_file_dir):
    res = defaultdict(list)
    for fn in os.listdir(stats_file_dir):
        fn: str
        if not fn.startswith("summary"):
            continue
        host_name, local_gpu_id = parse_comm_stats_file_name(fn)
        stats = json.load(open(os.path.join(stats_file_dir, fn)))
        for op_name, v in stats.items():
            if "send" or "recv" in op_name:
                data_size = 2 * 1024 * 1024 * 1024
            elif "offload" in op_name:
                data_size = 20 * 1024 * 1024
            res[op_name].append(data_size / v['mean'])

    for k, v in res.items():
        res[k] = int(mean(v)) if len(v) > 0 else 0
    return res


def find_nearest_key(k, d):
    keys = list(d.keys())
    op_keys = []
    for ik in keys:
        if k[0] == ik[0]:
            op_keys.append(ik)
    op_keys.sort()
    for i in range(len(op_keys)):
        if k < op_keys[i]:
            return op_keys[i]
    return op_keys[-1]


# def fitting_key(k, d):
#     op_name, bs, seqlen = k
#     keys = list(d.keys())
#     op_keys = []
#     for ik in keys:
#         if k[0] == ik[0]:
#             op_keys.append(ik)
#     key_map = {}
#     for k in op_keys:
#         if op_name == "fwd_gen_1":
#             if bs not in key_map.values():
#                 key_map[k] = bs
#         else:
#             if bs * seqlen not in key_map.values():
#                 key_map[k] = bs * seqlen
#     kv_tuples = sorted(key_map.items(), key=lambda x: x[1])
#     v_tuple = (k, bs) if op_name == "fwd_gen_1" else (k, bs * seqlen)
#     v = v_tuple[1]
#     kv_tuples.append(v_tuple)
#     kv_tuples.sort(key=lambda x: x[1])
#     left = kv_tuples.index(v_tuple) - 1
#     right = kv_tuples.index(v_tuple) + 1
#     if left < 0:
#         right_key = kv_tuples[right][0]
#         right_value = kv_tuples[right][1]
#         return d[right_key] * v / right_value
#     if right >= len(kv_tuples):
#         left_key = kv_tuples[left][0]
#         left_value = kv_tuples[left][1]
#         return d[left_key] * v / left_value
#     left_key = kv_tuples[left][0]
#     left_value = kv_tuples[left][1]
#     right_key = kv_tuples[right][0]
#     right_value = kv_tuples[right][1]
#     return d[left_key] + (d[right_key] - d[left_key]) * (v - left_value) / (right_value - left_value)

# def compute_inst_cost(op_cost, num_layers, num_pp, op_name, bs, seqlen, p=False):
#     op_key = (op_name, bs, seqlen)
#     if op_key not in op_cost["embedding_layer"]:
#         real_op_key = find_nearest_key(op_key, op_cost["embedding_layer"])
#     else:
#         real_op_key = op_key
#     if op_key in op_cost["embedding_layer"]:
#         embedding_layer_cost = op_cost["embedding_layer"][real_op_key]
#         flash_mqat_block_0_cost = op_cost["flash_mqat_block_0"][real_op_key]
#         head_cost = op_cost["head"][real_op_key]
#     else:
#         embedding_layer_cost = fitting_key(op_key, op_cost["embedding_layer"])
#         flash_mqat_block_0_cost = fitting_key(op_key, op_cost["flash_mqat_block_0"])
#         head_cost = fitting_key(op_key, op_cost["head"])
#     cost = (embedding_layer_cost + num_layers * flash_mqat_block_0_cost\
#             + head_cost) / num_pp
#     return cost


def compute_inst_cost(op_cost, num_layers, num_pp, op_name, bs, seqlen):
    op_key = (op_name, bs, seqlen)
    if op_key not in op_cost["embedding_layer"]:
        real_op_key = find_nearest_key(op_key, op_cost["embedding_layer"])
    else:
        real_op_key = op_key

    embedding_layer_cost = op_cost["embedding_layer"][real_op_key]
    flash_mqat_block_0_cost = op_cost["flash_mqat_block_0"][real_op_key]
    head_cost = op_cost["head"][real_op_key]
    cost = (embedding_layer_cost + num_layers * flash_mqat_block_0_cost\
            + head_cost) / num_pp

    # if op_name == "fwd_gen_1":
    #     print(f"op key {op_key} real op key {real_op_key}")
    #     print(f"{cost} = ({embedding_layer_cost} + {num_layers} * {flash_mqat_block_0_cost} + {head_cost})/{num_pp}")

    log_debug(f"op key {op_key} real op key {real_op_key}")
    log_debug(
        f"{cost} = ({embedding_layer_cost} + {num_layers} * {flash_mqat_block_0_cost} + {head_cost}) / {num_pp}"
    )

    # if op_name == "fwd_gen_1":
    #     if bs > 32:
    #         cost = cost * bs / real_op_key[1]
    if op_name != "opt" and op_name != "fwd_gen_1":
        cost = cost * bs * seqlen / (real_op_key[1] * real_op_key[2])
        log_debug(f"cost {cost} = {cost} * {bs * seqlen} / {real_op_key[1] * real_op_key[2]}")

    return cost


def comm_inst_cost(comm_stats, size, comm_type):
    # print(comm_stats, size, comm_type)
    return size / comm_stats[comm_type]  # micro seconds


def estimate_instruction_cost(
        layer_stats_path: str,  # path to layer stats file
        comm_stats_path: str,
        num_layers: int,  # model configuration, num transformers layer
        parallel_strategy: ModelParallelStrategy,
        hidden_dim: int,
        batch_size: int,
        seq_len: int,
        n_ppo_minibatches: int = 1):
    comm_stats = parse_comm_stats_files(comm_stats_path)
    layer_stats = parse_layer_stats_files(layer_stats_path)
    num_mp = parallel_strategy.num_mp
    num_pp = parallel_strategy.num_pp
    num_dp = parallel_strategy.num_dp
    num_gpus = parallel_strategy.num_dp * num_mp * num_pp
    layer_stats = layer_stats[num_mp]
    op_cost = {}
    layer_names = ["embedding_layer", "flash_mqat_block_0", "head"]
    for k, v in layer_stats.items():
        layer_name, op_name, bs, seqlen = k.split("-")
        bs, seqlen = int(bs), int(seqlen)
        if layer_name not in op_cost:
            op_cost[layer_name] = {}
        op_cost[layer_name][(op_name, bs, seqlen)] = v["mean"]

    for layer_name in layer_names:
        assert layer_name in op_cost

    train_mbs = batch_size / (2 * num_pp * num_dp * n_ppo_minibatches)\
        if num_pp > 1 else batch_size / (num_dp * n_ppo_minibatches)
    gen_mbs = batch_size / (num_pp * num_dp)
    log_debug(f"in estimate_instruction_cost, train_mbs {train_mbs} gen_mbs {gen_mbs} ")
    log_debug(f"batch size {batch_size} num_pp {num_pp} num_dp {num_dp} num_mp {num_mp} "
              f"train mbs {train_mbs} gen_mbs {gen_mbs}")

    # pprint.pprint(op_cost, indent=4)
    inst_cost = {}
    inst_keys = ["gen_fwd_0", "gen_fwd_1", "inf_fwd", "train_fwd", "train_bwd", "train_opt"]
    op_names = ["fwd_gen_0", "fwd_gen_1", "fwd", "fwd", "bwd", "opt"]

    for inst_key, op_name in zip(inst_keys, op_names):
        mbs = train_mbs if "train" in inst_key else gen_mbs
        # print(f"inst key {inst_key}")
        inst_cost[inst_key] = compute_inst_cost(op_cost, num_layers, num_pp, op_name, mbs, seq_len)

    # inst_cost["gen_fwd_0"] = compute_inst_cost(op_cost, num_layers, num_pp, "fwd_gen_0", gen_mbs, seq_len)
    # inst_cost["gen_fwd_1"] = compute_inst_cost(op_cost, num_layers, num_pp, "fwd_gen_1", gen_mbs, seq_len)
    # inst_cost["inf_fwd"] = compute_inst_cost(op_cost, num_layers, num_pp, "fwd", gen_mbs, seq_len)
    # inst_cost["train_fwd"] = compute_inst_cost(op_cost, num_layers, num_pp, "fwd", train_mbs, seq_len)
    # inst_cost["train_bwd"] = compute_inst_cost(op_cost, num_layers, num_pp, "bwd", train_mbs, seq_len)
    # inst_cost["train_opt"] = compute_inst_cost(op_cost, num_layers, num_pp, "opt", train_mbs, seq_len)

    comm_type = "remote_send" if num_gpus // num_pp >= 8 else "local_send"
    # print(type(hidden_dim), type(bs), type(seq_len))
    inst_cost["act_p2p"] = comm_inst_cost(comm_stats, 2 * hidden_dim * train_mbs * seq_len, comm_type)
    inst_cost["grad_p2p"] = comm_inst_cost(comm_stats, 2 * hidden_dim * train_mbs * seq_len, comm_type)
    inst_cost["gen_act_p2p"] = comm_inst_cost(comm_stats, 2 * hidden_dim * gen_mbs, comm_type)
    return inst_cost


def _estimate_rpc_cost(
        inst_cost,
        parallel_strategy: ModelParallelStrategy,
        model_interface_type: ModelInterfaceType,
        # model function call args
        num_gen_tokens: int,
        use_gradient_checkpointing: bool,
        n_ppo_minibatches: int = 1):
    num_pp = parallel_strategy.num_pp
    num_dp = parallel_strategy.num_dp
    num_mp = parallel_strategy.num_mp
    log_debug(f"ModelInterfaceType {model_interface_type} num_pp {num_pp} num_dp {num_dp} num_mp {num_mp}")
    if model_interface_type == ModelInterfaceType.INFERENCE:
        num_micro_batches = num_pp
        compute_cost = inst_cost["inf_fwd"] * (num_pp + num_micro_batches - 1)
        comm_cost = inst_cost["act_p2p"] * (num_pp + num_micro_batches - 2) * 2
        log_debug(f"{inst_cost['inf_fwd']} * {num_pp + num_micro_batches - 1}")
        if num_mp > 1:  # and num_pp * num_dp > 1:
            compute_cost = compute_cost * (1 - num_mp * 0.03)
            log_debug(f"num_dp num_mp coefficent {1- num_mp * 0.03}")
    elif model_interface_type == ModelInterfaceType.TRAIN_STEP:
        # TODO: add reduce grads, add ppo micro batches
        num_micro_batches = num_pp * 2 if num_pp > 1 else 1
        compute_cost = (inst_cost["train_fwd"] + inst_cost["train_bwd"]) * (num_pp + num_micro_batches -
                                                                            1) + inst_cost["train_opt"]
        log_debug(f"({inst_cost['train_fwd']} + {inst_cost['train_bwd']}) * {num_pp + num_micro_batches - 1}")
        if use_gradient_checkpointing:
            compute_cost += inst_cost["train_fwd"] * (num_pp + num_micro_batches - 1)
            log_debug(f"Gradient ckpt overhead {inst_cost['train_fwd']} * {num_pp + num_micro_batches - 1}")
        comm_cost = (inst_cost["grad_p2p"] + inst_cost["act_p2p"]) * (num_pp + num_micro_batches - 2) * 2
        compute_cost = compute_cost * n_ppo_minibatches
        comm_cost = comm_cost * n_ppo_minibatches
        if num_pp * num_dp <= 1:
            compute_cost = compute_cost * (1 - num_mp * 0.04)
        if num_mp > 1:  # and num_pp * num_dp > 1:
            compute_cost = compute_cost * (1 - num_mp * 0.03)
            log_debug(f"num_dp num_mp coefficent {1- num_mp * 0.03}")
    elif model_interface_type == ModelInterfaceType.GENERATE:
        num_micro_batches = num_pp
        num_gen_tokens = num_gen_tokens
        compute_cost = inst_cost["gen_fwd_0"] * (num_pp + num_micro_batches - 1) +\
                       inst_cost["gen_fwd_1"] * (num_gen_tokens - 1) * num_micro_batches
        log_debug(f"{inst_cost['gen_fwd_0']} * {num_pp + num_micro_batches - 1} + "
                  f"{inst_cost['gen_fwd_1']} * {(num_gen_tokens - 1) * num_micro_batches}")

        if num_dp * num_mp > 1:
            compute_cost = compute_cost * (1 - min(num_dp * num_mp, 8) * 0.03)
            log_debug(f"num_dp coefficent {1 - min(num_dp * num_mp, 8) * 0.03}")

        # comm_cost = inst_cost["act_p2p"] * (num_pp + num_micro_batches - 2) * 2 +\
        #             inst_cost["gen_act_p2p"] * (num_gen_tokens - 1) * num_micro_batches * 2
        comm_cost = 0

    # dirty heuristic
    if num_pp > 8:
        compute_cost = compute_cost * (1 + num_pp * 0.01)
        log_debug(f"num_pp coefficent {1 + num_pp * 0.01}")

    log_debug(f"Final cost {compute_cost} + {comm_cost} = {compute_cost + comm_cost}")

    # print(parallel_strategy)
    # print(f"gen_fwd_0 {inst_cost['gen_fwd_0']}*{num_pp+num_micro_batches-1}={inst_cost['gen_fwd_0'] * (num_pp + num_micro_batches - 1)}"
    #       f"gen_fwd_1 {inst_cost['gen_fwd_1']}*{(num_gen_tokens - 1) * num_micro_batches}={inst_cost['gen_fwd_1'] * (num_gen_tokens - 1) * num_micro_batches}"
    #       f"final {compute_cost}")
    # print(f"{model_interface_type} compute cost {compute_cost} comm cost {comm_cost}")
    # pprint.pprint(inst_cost, indent=4)
    return compute_cost + comm_cost


def load_model_config(rpc: ModelRPC) -> FlashMQATConfig:
    return getattr(ReaLModel,
                   f"config_from_{rpc.model_type._class}")(model_path=MODEL_TYPE_TO_PATH[rpc.model_type])


def estimate_rpc_time(rpc: ModelRPC,
                      parallel_strategy: ModelParallelStrategy,
                      num_gen_tokens=256,
                      use_gradient_checkpointing=False,
                      n_ppo_minibatches: int = 1,
                      bs=None,
                      seq_len=None):
    # print(f"estimating rpc cost for {rpc.name}")
    model_type = rpc.model_type
    layer_stats_path = os.path.join(PROFILE_RESULT_PATH, str(model_type))
    # comm_stats_path = os.path.join(PROFILE_RESULT_PATH, exp.device_mesh_name)
    comm_stats_path = os.path.join(PROFILE_RESULT_PATH, "default_comm")
    model_config = load_model_config(rpc)

    if bs == None or seq_len == None:
        bs = rpc.min_n_seqs
        seq_len = rpc.max_n_tokens // bs
    # ps = ModelParallelStrategy.from_config(rpc_alloc.train_eval_config.parallel)
    p = (rpc.interface_type == ModelInterfaceType.GENERATE)
    inst_cost = estimate_instruction_cost(layer_stats_path,
                                          comm_stats_path,
                                          model_config.n_layers,
                                          parallel_strategy,
                                          model_config.hidden_dim,
                                          bs,
                                          seq_len,
                                          n_ppo_minibatches=n_ppo_minibatches)
    # print(inst_cost)
    return _estimate_rpc_cost(inst_cost,
                              parallel_strategy,
                              rpc.interface_type,
                              num_gen_tokens=num_gen_tokens,
                              use_gradient_checkpointing=use_gradient_checkpointing,
                              n_ppo_minibatches=n_ppo_minibatches)


# def comm_stats(exp: ProfileExperiment, if_print=False):
#     comm_stats_path = os.path.join(PROFILE_RESULT_PATH, "default_comm")
#     comm_stats = parse_comm_stats_files(comm_stats_path)
#     r = CommStats(**comm_stats)
#     if if_print:
#         print(r)
#     return r


def comm_stats(if_print=False):
    # use default comm stats of cluster
    r = CommStats(
        # between GPUs on the same node
        local_send=170000,  # unit: bytes per micro seconds, 170 GB/s
        local_recv=170000,
        # IB between GPUs on different nodes
        remote_send=20000,  # unit: bytes per micro seconds 20 GB/s
        remote_recv=20000,
        offload_store=0,  # not used
        offload_load=0,  # not used
    )
    if if_print:
        print(r)
    return r


def estimate_model_size(model_config: FlashMQATConfig):
    h = model_config.hidden_dim
    i = model_config.intermediate_dim
    v = model_config.vocab_size
    L = model_config.n_layers
    # for llama actor only
    n_params = 3 * v * h + (3 * h * i + 4 * h * h) * L
    return 2 * n_params


def estimate_rpc_memory(rpc: ModelRPC,
                        parallel_strategy: ModelParallelStrategy,
                        batch_size: int,
                        seq_len: int,
                        offload: bool = False,
                        gradient_checkpointing: bool = False,
                        offload_optimizer: bool = False,
                        n_ppo_minibatches: int = 1,
                        gen_len: int = 128):
    interface_type = rpc.interface_type
    model_config = load_model_config(rpc)

    h = model_config.hidden_dim
    i = model_config.intermediate_dim
    v = model_config.vocab_size
    s = seq_len
    gs = gen_len
    b = batch_size
    L = model_config.n_layers
    # for llama actor only
    n_params = 3 * v * h + (3 * h * i + 4 * h * h) * L
    param_mem = 2 * n_params
    grad_mem = 2 * n_params
    optimizer_mem = 20 * n_params if not offload_optimizer else 0

    num_pp = parallel_strategy.num_pp
    num_mp = parallel_strategy.num_mp
    num_dp = parallel_strategy.num_dp
    # print(f"Parallel strategy: num_pp: {num_pp}, num_mp: {num_mp}, num_dp: {num_dp}")
    # zero1, pp and mp divide evenly
    # enable sequence parallel
    if interface_type == ModelInterfaceType.TRAIN_STEP:
        # gradient checkpointing is always enabled for flash attn
        static_mem = (param_mem + grad_mem) // (num_pp * num_mp) +\
                     optimizer_mem // (num_pp * num_dp * num_mp)
        micro_bs = b // (2 * num_pp * num_dp) if num_pp > 0 else b // (num_dp)
        active_mem = (micro_bs * s * h * num_pp * 2) * 2 * L // (num_pp * num_mp)
        # enabled gradient ckpt
        # print(f"train static_mem: {static_mem/(1024*1024*1024):02f} GB, "
        #       f"active_mem: {active_mem/(1024*1024*1024):02f} GB, "
        #       f"total: {(static_mem + active_mem)/(1024*1024*1024):02f} GB")
        return static_mem + active_mem, static_mem
    elif interface_type == ModelInterfaceType.INFERENCE:
        static_mem = int(2 * param_mem // (num_pp * num_mp))
        # active_mem = 0  # no tensor need to be stored in inference
        # print(f"inference static_mem: {static_mem/(1024*1024*1024):02f} GB")
        # if num_dp > 4:
        #     static_mem = static_mem * 1.25
        if offload:
            return static_mem, 0  # assume offload
        else:
            return static_mem, static_mem
    elif interface_type == ModelInterfaceType.GENERATE:
        static_mem = int(2 * param_mem // (num_pp * num_mp))
        # if num_dp > 4 and num_dp * num_mp * num_pp <= 16:
        #     static_mem = static_mem * 1.1
        active_mem = 2 * (2 * b * (gs + s) * h) * L // (num_pp * num_mp * num_dp)  # kv cache
        # print(f"generate static_mem: {static_mem/(1024*1024*1024):02f} GB, "
        #       f"kv_cache_mem: {kv_cache_mem/(1024*1024*1024):02f} GB, "
        #       f"total: {(static_mem + kv_cache_mem)/(1024*1024*1024):02f} GB")
        return static_mem + active_mem, static_mem


def main(args):
    exp: ProfileExperiment = config_package.make_experiment(args.expr_name)
    rpcs = exp.rpcs
    rpc_allocations = exp.rpc_allocations
    # device_mesh = exp.device_mesh_name

    bs_list = [128, 256, 512]
    seq_len_list = [1024, 512, 256]
    # bs_list = [32, 64]
    # seq_len_list = [128, 256]
    bs_seqlen_list = list(zip(bs_list, seq_len_list))
    r = {}
    for rpc, rpc_alloc in zip(rpcs, rpc_allocations):
        rpc_name = rpc.name
        p = ModelParallelStrategy.from_config(rpc_alloc.train_eval_config.parallel)
        for bs, seq_len in bs_seqlen_list:
            rpc_cost = estimate_rpc_time(rpc,
                                         p,
                                         use_gradient_checkpointing=True,
                                         bs=bs,
                                         seq_len=seq_len,
                                         num_gen_tokens=seq_len - 128)
            stats_key = make_stats_key(rpc_name, bs, seq_len)
            r[stats_key] = rpc_cost
            # print(f"RPC {stats_key} cost: {rpc_cost}")

    # pprint.pprint(r, indent=4)
    return r


def example(args):
    global IF_PRINT
    IF_PRINT = True
    if args.model_size == 7:
        n_nodes = 1
    elif args.model_size == 13:
        n_nodes = 2
    elif args.model_size == 34:
        n_nodes = 4
    elif args.model_size == 70:
        n_nodes = 8

    expr_name = f"profile-s{args.model_size}p{n_nodes}m1d8"

    exp: ProfileExperiment = config_package.make_experiment(expr_name)
    rpcs = exp.rpcs
    rollout, inf, train = rpcs

    bs = 128
    seq_len = 128

    p1 = ModelParallelStrategy(num_pp=1, num_mp=4, num_dp=8)
    rpc_cost = estimate_rpc_time(rollout,
                                 p1,
                                 use_gradient_checkpointing=True,
                                 num_gen_tokens=896,
                                 bs=bs,
                                 seq_len=seq_len,
                                 n_ppo_minibatches=4)
    mem_cost, static_mem = estimate_rpc_memory(rollout, p1, bs, seq_len)
    print(f"rollout {p1} rpc cost {rpc_cost} mem cost {mem_cost/(1024**3):.2f} GB")

    print("*" * 100)
    p2 = ModelParallelStrategy(num_pp=1, num_mp=4, num_dp=6)
    rpc_cost = estimate_rpc_time(rollout,
                                 p2,
                                 use_gradient_checkpointing=True,
                                 num_gen_tokens=896,
                                 bs=bs,
                                 seq_len=seq_len,
                                 n_ppo_minibatches=4)
    mem_cost, static_mem = estimate_rpc_memory(rollout, p2, bs, seq_len)
    print(f"rollout {p2} rpc cost {rpc_cost} mem cost {mem_cost/(1024**3):.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a profiling experiment.")
    parser.add_argument(
        "-s",
        "--model_size",
        type=int,
        default=7,
    )
    # parser.add_argument(
    #     "-f",
    #     "--trial_name",
    #     type=str,
    #     default=None,
    # )
    args = parser.parse_args()

    # r = main(args)
    example(args)
