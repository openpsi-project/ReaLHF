# estimate a fucntion-call level execution time for device mesh enumerate pruning
# assume one batch of data passes through all rpcs once
from collections import defaultdict
from statistics import mean
from typing import Dict, List, Optional
import dataclasses
import enum
import getpass
import json
import os

from profiler.device_mesh import *
from profiler.experiments import ProfileExperiment

from api.dfg import ModelInterfaceType
from impl.model.nn.flash_mqat.flash_mqat_base import FlashMQATConfig
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
    res = {"crosshost_p2p": [], "localhost_p2p": [], "offload": []}
    for fn in os.listdir(stats_file_dir):
        fn: str
        if not fn.startswith("summary"):
            continue
        host_name, local_gpu_id = parse_comm_stats_file_name(fn)
        stats = json.load(open(os.path.join(stats_file_dir, fn)))
        for k, v in stats.items():
            host, op_name, src, dst = k.split("|")
            if dst != "MEM":
                src = int(src)
                dst = int(dst)
                if src // 8 == dst // 8:
                    res["localhost_p2p"].append(2 * 1024 * 1024 * 1024 / v["mean"])
                else:
                    res["crosshost_p2p"].append(2 * 1024 * 1024 * 1024 / v["mean"])
            else:
                res["offload"].append(20 * 1024 * 1024 / v["mean"])

    for k, v in res.items():
        res[k] = mean(v)
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
        op_key = find_nearest_key(op_key, op_cost["embedding_layer"])
    return (op_cost["embedding_layer"][op_key] +\
           num_layers * op_cost["flash_mqat_block_0"][op_key] +\
           op_cost["head"][op_key]) / num_pp # micro seconds


def comm_inst_cost(comm_stats, size, comm_type):
    # print(comm_stats, size, comm_type)
    return size / comm_stats[comm_type]  # micro seconds


def estimate_instruction_cost(
    layer_stats_path: str,  # path to layer stats file
    comm_stats_path: str,
    num_layers: str,  # model configuration, num transformers layer
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
    # print(f"batch size {batch_size} num_pp {num_pp} num_dp {num_dp} num_mp {num_mp}"
    #       f"train mbs {train_mbs} gen_mbs {gen_mbs}")

    inst_cost = {}
    inst_cost["gen_fwd_0"] = compute_inst_cost(op_cost, num_layers, num_pp, "fwd_gen_0", gen_mbs, seq_len)
    inst_cost["gen_fwd_1"] = compute_inst_cost(op_cost, num_layers, num_pp, "fwd_gen_1", gen_mbs, seq_len)
    inst_cost["inf_fwd"] = compute_inst_cost(op_cost, num_layers, num_pp, "fwd", gen_mbs, seq_len)
    inst_cost["train_fwd"] = compute_inst_cost(op_cost, num_layers, num_pp, "fwd", train_mbs, seq_len)
    inst_cost["train_bwd"] = compute_inst_cost(op_cost, num_layers, num_pp, "bwd", train_mbs, seq_len)
    inst_cost["train_opt"] = compute_inst_cost(op_cost, num_layers, num_pp, "opt", train_mbs, seq_len)

    comm_type = "crosshost_p2p" if num_gpus // num_pp > 8 else "localhost_p2p"
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
        num_gen_tokens=256):
    num_pp = parallel_strategy.num_pp
    if model_interface_type == ModelInterfaceType.INFERENCE:
        num_micro_batches = num_pp
        return inst_cost["inf_fwd"] * (num_pp + num_micro_batches - 1) +\
               inst_cost["act_p2p"] * (num_pp + num_micro_batches - 2) * 2
    elif model_interface_type == ModelInterfaceType.TRAIN_STEP:
        # TODO: add reduce grads, add ppo micro batches
        num_micro_batches = num_pp * 2
        return (inst_cost["train_fwd"] + inst_cost["train_bwd"]) * (num_pp + num_micro_batches - 1) +\
               inst_cost["train_opt"] +\
               (inst_cost["grad_p2p"] + inst_cost["act_p2p"]) * (num_pp + num_micro_batches - 2) * 2
    elif model_interface_type == ModelInterfaceType.GENERATE:
        num_micro_batches = num_pp
        num_gen_tokens = num_gen_tokens
        return inst_cost["gen_fwd_0"] * (num_pp + num_micro_batches - 1) +\
               inst_cost["act_p2p"] * (num_pp + num_micro_batches - 2) * 2 +\
               inst_cost["gen_fwd_1"] * (num_gen_tokens - 1) * num_micro_batches +\
               inst_cost["gen_act_p2p"] * (num_gen_tokens - 1) * num_micro_batches * 2


def load_model_config(model_path):
    config_json_path = os.path.join(model_path, "flash_mqat_config.json")
    config_json = json.load(open(config_json_path, "r"))
    return FlashMQATConfig(**config_json)


def estimate_rpc_cost(exp: ProfileExperiment, rpc: ModelRPC, model_config: FlashMQATConfig,
                      parallel_strategy: ModelParallelStrategy):
    # print(f"estimating rpc cost for {model_rpc_name(rpc)}")
    model_type = exp.model_names_to_types[rpc.model_name]
    layer_stats_path = os.path.join(PROFILE_RESULT_PATH, model_type)
    # comm_stats_path = os.path.join(PROFILE_RESULT_PATH, exp.device_mesh_name)
    comm_stats_path = os.path.join(PROFILE_RESULT_PATH, "QH-com[40-43]")
    bs = rpc.min_n_seqs
    seq_len = rpc.max_n_tokens // bs
    inst_cost = estimate_instruction_cost(layer_stats_path, comm_stats_path, model_config.n_layers,
                                          parallel_strategy, model_config.hidden_dim, bs, seq_len)
    # print(inst_cost)
    return _estimate_rpc_cost(inst_cost, parallel_strategy, rpc.interface_type)


def estimate_model_device_mapping_cost(exp: ProfileExperiment, mdm: ModelDeviceMapping) -> float:
    assert mdm.model_device_mapping is not None
    assert mdm.model_parallel_strategy is not None
    # TODO: change to auto

    model_configs = {
        model_type: load_model_config(model_path)
        for model_type, model_path in zip(exp.model_types, exp.model_paths)
    }

    layer_stats_paths = {
        model_type: os.path.join(PROFILE_RESULT_PATH, model_type)
        for model_type in exp.model_types
    }

    # comm_stats_path = os.path.join(PROFILE_RESULT_PATH, exp.device_mesh_name)
    comm_stats_path = os.path.join(PROFILE_RESULT_PATH, "QH-com[40-43]")

    costs = {}
    for rpc_name, rpc in mdm.model_rpc_mapping.items():
        batch_size = rpc.min_n_seqs
        seq_len = rpc.max_n_tokens // batch_size
        model_type = exp.model_names_to_types[rpc.model_name]
        inst_cost = estimate_instruction_cost(layer_stats_paths[model_type], comm_stats_path,
                                              model_configs[model_type].n_layers,
                                              mdm.model_parallel_strategy[model_rpc_name(rpc)],
                                              model_configs[model_type].hidden_dim, batch_size, seq_len)
        costs[rpc_name] = _estimate_rpc_cost(inst_cost, mdm.model_parallel_strategy[model_rpc_name(rpc)],
                                             rpc.interface_type)
    return costs


def main():
    profiler_experiment = ProfileExperiment()  # n_nodes = 4, nodelist = QH-com[40-43]
    model_rpc_names = [model_rpc_name(rpc) for rpc in profiler_experiment.model_rpcs]
    model_rpc_mapping = dict(zip(model_rpc_names, profiler_experiment.model_rpcs))
    model_device_mapping = {
        rpc_name: DeviceMesh(device_mesh_name="QH-com[40-43]",
                             n_nodes=4,
                             n_gpus=32,
                             node_names=["QH-com40", "QH-com41", "QH-com42", "QH-com43"])
        for rpc_name in model_rpc_names
    }
    model_parallel_strategy = {
        rpc_name: ModelParallelStrategy(num_mp=1, num_pp=4, num_dp=1)
        for rpc_name in model_rpc_names
    }
    mdm = ModelDeviceMapping(model_names=profiler_experiment.model_names,
                             model_rpc_names=model_rpc_names,
                             model_rpc_mapping=model_rpc_mapping,
                             model_device_mapping=model_device_mapping,
                             model_parallel_strategy=model_parallel_strategy)
    print(estimate_model_device_mapping_cost(profiler_experiment, mdm))


if __name__ == "__main__":
    main()
