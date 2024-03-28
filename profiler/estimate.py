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

from profiler.device_mesh import *
from profiler.experiments import ProfileExperiment
from profiler.rpc import CommStats
from profiler.utils import make_stats_key

from api.config.config_base import MODEL_TYPE_TO_PATH
from api.config.dfg import ModelInterfaceType
from experiments.autoexp.device_mapping import RPCAllocation
from impl.model.nn.flash_mqat.flash_mqat_api import FlashMQATModel
from impl.model.nn.flash_mqat.flash_mqat_base import FlashMQATConfig
import api.config.config_system as config_package
import base.cluster

PROFILE_RESULT_PATH = os.path.join(
    base.cluster.spec.fileroot,
    "logs",
    getpass.getuser(),
    "profile",
    "profile",
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
    # print(f"{cost} = ({embedding_layer_cost} + {num_layers} * {flash_mqat_block_0_cost} + {head_cost})/{num_pp}")
    # print(f"op key {op_key} real op key {real_op_key}")

    if op_name != "opt" and op_name != "fwd_gen_1":
        cost = cost * bs * seqlen / (real_op_key[1] * real_op_key[2])

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
):
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

    train_mbs = batch_size // (2 * num_pp * num_dp)
    gen_mbs = batch_size // (num_pp * num_dp)
    # print(f"in estimate_instruction_cost, train_mbs {train_mbs} gen_mbs {gen_mbs}")
    # print(f"batch size {batch_size} num_pp {num_pp} num_dp {num_dp} num_mp {num_mp}"
    #       f"train mbs {train_mbs} gen_mbs {gen_mbs}")

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
        use_gradient_checkpointing: bool):
    num_pp = parallel_strategy.num_pp
    if model_interface_type == ModelInterfaceType.INFERENCE:
        num_micro_batches = num_pp
        compute_cost = inst_cost["inf_fwd"] * (num_pp + num_micro_batches - 1)
        comm_cost = inst_cost["act_p2p"] * (num_pp + num_micro_batches - 2) * 2
    elif model_interface_type == ModelInterfaceType.TRAIN_STEP:
        # TODO: add reduce grads, add ppo micro batches
        num_micro_batches = num_pp * 2
        compute_cost = (inst_cost["train_fwd"] + inst_cost["train_bwd"]) * (num_pp + num_micro_batches - 1) +\
                        inst_cost["train_opt"]
        if use_gradient_checkpointing:
            compute_cost += inst_cost["train_fwd"] * (num_pp + num_micro_batches - 1)
        comm_cost = (inst_cost["grad_p2p"] + inst_cost["act_p2p"]) * (num_pp + num_micro_batches - 2) * 2
    elif model_interface_type == ModelInterfaceType.GENERATE:
        num_micro_batches = num_pp
        num_gen_tokens = num_gen_tokens
        compute_cost = inst_cost["gen_fwd_0"] * (num_pp + num_micro_batches - 1) +\
                       inst_cost["gen_fwd_1"] * (num_gen_tokens - 1) * num_micro_batches
        comm_cost = inst_cost["act_p2p"] * (num_pp + num_micro_batches - 2) * 2 +\
                    inst_cost["gen_act_p2p"] * (num_gen_tokens - 1) * num_micro_batches * 2
    # print(f"{model_interface_type} compute cost {compute_cost} comm cost {comm_cost}")
    # pprint.pprint(inst_cost, indent=4)
    return compute_cost + comm_cost


def load_model_config(rpc: ModelRPC) -> FlashMQATConfig:
    return getattr(FlashMQATModel,
                   f"config_from_{rpc.model_type._class}")(model_path=MODEL_TYPE_TO_PATH[rpc.model_type])


def estimate_rpc_cost(rpc: ModelRPC,
                      rpc_alloc: RPCAllocation,
                      num_gen_tokens=256,
                      use_gradient_checkpointing=False,
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
    ps = ModelParallelStrategy.from_config(rpc_alloc.train_eval_config.parallel)
    inst_cost = estimate_instruction_cost(layer_stats_path, comm_stats_path, model_config.n_layers, ps,
                                          model_config.hidden_dim, bs, seq_len)
    # print(inst_cost)
    return _estimate_rpc_cost(inst_cost,
                              ps,
                              rpc.interface_type,
                              num_gen_tokens=num_gen_tokens,
                              use_gradient_checkpointing=use_gradient_checkpointing)


def comm_stats(exp: ProfileExperiment, if_print=False):
    comm_stats_path = os.path.join(PROFILE_RESULT_PATH, "default_comm")
    comm_stats = parse_comm_stats_files(comm_stats_path)
    r = CommStats(**comm_stats)
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


def estimate_function_call_memory(rpc: ModelRPC,
                                  rpc_alloc: RPCAllocation,
                                  batch_size: int,
                                  seq_len: int,
                                  gradient_checkpointing: bool = False,
                                  offload_optimizer: bool = False):
    interface_type = rpc.interface_type
    model_config = load_model_config(rpc)
    parallel_strategy = ModelParallelStrategy.from_config(rpc_alloc.train_eval_config.parallel)

    h = model_config.hidden_dim
    i = model_config.intermediate_dim
    v = model_config.vocab_size
    s = seq_len
    b = batch_size
    L = model_config.n_layers
    # for llama actor only
    n_params = 3 * v * h + (3 * h * i + 4 * h * h) * L
    param_mem = 2 * n_params
    grad_mem = 2 * n_params
    optimizer_mem = 16 * n_params

    num_pp = parallel_strategy.num_pp
    num_mp = parallel_strategy.num_mp
    num_dp = parallel_strategy.num_dp
    # print(f"Parallel strategy: num_pp: {num_pp}, num_mp: {num_mp}, num_dp: {num_dp}")
    # zero1, pp and mp divide evenly
    # enable sequence parallel
    if interface_type == ModelInterfaceType.TRAIN_STEP:
        static_mem = (param_mem + grad_mem) // (num_pp * num_mp) +\
                     optimizer_mem // (num_pp * num_dp * num_mp)
        micro_bs = b // (2 * num_pp * num_dp)
        active_mem = (micro_bs * s * h * num_pp * 2) * 2 * L // (num_pp * num_mp)
        # enabled gradient ckpt
        # print(f"train static_mem: {static_mem/(1024*1024*1024):02f} GB, "
        #       f"active_mem: {active_mem/(1024*1024*1024):02f} GB, "
        #       f"total: {(static_mem + active_mem)/(1024*1024*1024):02f} GB")
        return static_mem + active_mem, static_mem
    elif interface_type == ModelInterfaceType.INFERENCE:
        static_mem = param_mem // (num_pp * num_mp)
        active_mem = 0  # no tensor need to be stored in inference
        # print(f"inference static_mem: {static_mem/(1024*1024*1024):02f} GB")
        return static_mem, static_mem
    elif interface_type == ModelInterfaceType.GENERATE:
        static_mem = param_mem // (num_pp * num_mp)
        active_mem = 2 * (2 * b * s * h) * L // (num_pp * num_mp * num_dp)  # kv cache
        # print(f"generate static_mem: {static_mem/(1024*1024*1024):02f} GB, "
        #       f"kv_cache_mem: {kv_cache_mem/(1024*1024*1024):02f} GB, "
        #       f"total: {(static_mem + kv_cache_mem)/(1024*1024*1024):02f} GB")
        return static_mem + active_mem, static_mem


def main(args):
    exp: ProfileExperiment = config_package.make_experiment(args.expr_name)
    rpcs = exp.rpcs
    rpc_allocations = exp.rpc_allocations
    # device_mesh = exp.device_mesh_name

    bs_list = [256, 512]
    seq_len_list = [128, 256, 512]
    # bs_list = [32, 64]
    # seq_len_list = [128, 256]
    bs_seqlen_list = list(itertools.product(bs_list, seq_len_list))
    r = {}
    for rpc, rpc_alloc in zip(rpcs, rpc_allocations):
        rpc_name = rpc.name
        for bs, seq_len in bs_seqlen_list:
            rpc_cost = estimate_rpc_cost(rpc,
                                         rpc_alloc,
                                         use_gradient_checkpointing=False,
                                         bs=bs,
                                         seq_len=seq_len)
            stats_key = make_stats_key(rpc_name, bs, seq_len)
            r[stats_key] = rpc_cost
            # print(f"RPC {stats_key} cost: {rpc_cost}")

    # pprint.pprint(r, indent=4)
    return r


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a profiling experiment.")
    parser.add_argument(
        "-e",
        "--expr_name",
        type=str,
        default=None,
    )
    # parser.add_argument(
    #     "-f",
    #     "--trial_name",
    #     type=str,
    #     default=None,
    # )
    args = parser.parse_args()

    r = main(args)
