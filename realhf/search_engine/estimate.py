# Estimate a fucntion-call level execution time for device mesh enumerate pruning
# assume one batch of data passes through all rpcs once
import argparse
import getpass
import itertools
import os
import pickle
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

import realhf.base.cluster
import realhf.base.constants as constants
import realhf.base.logging as logging
from realhf.api.core.dfg import MFCDef, ModelFamily, ModelInterfaceType
from realhf.api.core.model_api import ReaLModelConfig
from realhf.api.quickstart.model import ParallelismConfig
from realhf.search_engine.param_realloc import estimate_param_realloc_time_cost
from realhf.search_engine.utils import load_model_config

logger = logging.getLogger("estimate", "benchmark")

PROFILE_RESULT_PATH = os.path.join(
    realhf.base.cluster.spec.fileroot,
    "logs",
    getpass.getuser(),
    "profile",
    "profile",
    "layer_stats",
)


def get_param_realloc_stats(
    model_family: ModelFamily,
    model_path: str,
    n_nodes: int,
    use_cache: bool = True,
):
    non_critic = ModelFamily(model_family._class, model_family.size, False)
    table_path = os.path.join(
        constants.PROFILER_CACHE_PATH,
        "param_realloc",
        f"prtc_{non_critic}_n{n_nodes}.pkl",
    )
    if not os.path.exists(table_path):
        print(
            f"Calculating estimation of param realloc time cost for {model_family} at {model_path}"
        )
        estimate_param_realloc_time_cost(n_nodes, {non_critic: model_path})

    print(f"Loading param realloc stats from {table_path}")
    return pickle.load(open(table_path, "rb"))


def get_organized_op_stats(
    model_family: ModelFamily, model_path: str, use_cache: bool = True
):
    non_critic = ModelFamily(model_family._class, model_family.size, False)
    # parse raw stats into list of OpInfo used for estimation
    cache_path = os.path.join(
        constants.PROFILER_CACHE_PATH, "organized_stats", f"{non_critic}.pkl"
    )
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    raw_result_path = os.path.join(
        constants.PROFILER_CACHE_PATH, "layer_stats", str(non_critic)
    )
    if not os.path.exists(raw_result_path):
        from realhf.apps.main import _main_profile_layers

        _main_profile_layers(non_critic, model_path)

    raw_stats_list = []
    for fn in os.listdir(raw_result_path):
        if not fn.startswith("layer-stats"):
            continue
        num_mp = int(fn.replace(".pkl", "").split("_")[1])
        rank = int(fn.replace(".pkl", "").split("_")[2])
        with open(os.path.join(raw_result_path, fn), "rb") as f:
            stats = pickle.load(f)

        if isinstance(stats, dict):
            stats = pd.DataFrame(stats)
        elif isinstance(stats, pd.DataFrame):
            pass
        else:
            raise ValueError(f"Unsupported stats type {type(stats)}")

        stats["num_mp"] = num_mp
        stats["rank"] = rank
        raw_stats_list.append(stats)

    raw_stats = pd.concat(raw_stats_list)
    bs_list = raw_stats["bs"].unique()
    seq_len_list = raw_stats["seq_len"].unique()
    op_name_list = raw_stats["op_name"].unique()
    layer_name_list = raw_stats["layer_name"].unique()
    num_mp_list = raw_stats["num_mp"].unique()
    organized_stats = defaultdict(list)

    for op_name, bs, seq_len, layer_name, num_mp in itertools.product(
        op_name_list, bs_list, seq_len_list, layer_name_list, num_mp_list
    ):
        filter_cond = (
            (raw_stats["op_name"] == op_name)
            & (raw_stats["bs"] == bs)
            & (raw_stats["seq_len"] == seq_len)
            & (raw_stats["layer_name"] == layer_name)
            & (raw_stats["num_mp"] == num_mp)
        )
        avg_time_ns = raw_stats[filter_cond]["time_ns"].mean()
        x = int(bs) if op_name == "fwd_gen_1" else int(bs * seq_len)

        organized_stats["op_name"].append(op_name)
        organized_stats["layer_name"].append(layer_name)
        organized_stats["bs"].append(bs)
        organized_stats["seq_len"].append(seq_len)
        organized_stats["num_mp"].append(num_mp)
        organized_stats["avg_time_ns"].append(avg_time_ns)
        organized_stats["x"].append(x)

    df = pd.DataFrame(organized_stats)
    if use_cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(df, f)

    return df


def computation_instruction_time_cost(
    op_stats: pd.DataFrame,
    op_name: str,
    num_layers: int,
    parallel_strategy: ParallelismConfig,
    bs: int,
    seqlen: int,
):
    # inst cost unit: ns
    layer_names = ["embedding_layer", "block_0", "head"]
    num_pp = parallel_strategy.pipeline_parallel_size
    num_mp = parallel_strategy.model_parallel_size
    op_stats = op_stats[
        (op_stats["op_name"] == op_name) & (op_stats["num_mp"] == num_mp)
    ]

    op_cost = {}
    embed_stats = op_stats[op_stats["layer_name"] == "embedding_layer"]
    if embed_stats[
        (embed_stats["bs"] == bs) & (embed_stats["seq_len"] == seqlen)
    ].empty:
        # do linear interpolation for data points that does not exist
        for layer_name in layer_names:
            layer_stats = op_stats[op_stats["layer_name"] == layer_name]
            assert layer_stats[
                (layer_stats["bs"] == bs) & (layer_stats["seq_len"] == seqlen)
            ].empty
            assert not layer_stats.empty, (
                layer_name,
                op_name,
                num_mp,
                op_stats,
            )
            xs = layer_stats["x"]
            ys = layer_stats["avg_time_ns"]
            x = int(bs) if op_name == "fwd_gen_1" else int(bs * seqlen)
            y = np.interp(x, xs, ys)
            if max(xs) < x or min(xs) > x:
                logger.warning(
                    f"Interpolated value outside profiling range, "
                    f"parallel strategy {parallel_strategy}: "
                    f"{x} in {sorted(list(set(xs)))}"
                )
                # estimate using largest or smallest value
                if max(xs) < x:
                    y = ys.max() * (x / xs[ys.idxmax()])
                else:
                    y = ys.min() * (x / xs[ys.idxmin()])
            op_cost[layer_name] = y
    else:
        for layer_name in layer_names:
            assert not op_stats[op_stats["layer_name"] == layer_name].empty
            required_stats = op_stats[
                (op_stats["layer_name"] == layer_name)
                & (op_stats["bs"] == bs)
                & (op_stats["seq_len"] == seqlen)
            ]
            assert required_stats.shape[0] == 1
            op_cost[layer_name] = required_stats["avg_time_ns"].values[0]

    embedding_layer_cost = op_cost["embedding_layer"]
    block_0_cost = op_cost["block_0"]
    head_cost = op_cost["head"]
    cost = (embedding_layer_cost + num_layers * block_0_cost + head_cost) / num_pp
    return cost


def communication_instruction_time_cost(comm_stats, size, comm_type):
    return size / comm_stats[comm_type]  # unit: ns


def estimate_instruction_time_costs(
    model_family: ModelFamily,
    model_path: str,
    num_layers: int,  # model configuration, num transformers layer
    parallel_strategy: ParallelismConfig,
    hidden_dim: int,
    batch_size: int,
    seq_len: int,
    n_ppo_minibatches: int = 1,
):
    comm_stats = default_communication_stats()
    op_stats = get_organized_op_stats(model_family, model_path, use_cache=True)

    num_mp = parallel_strategy.model_parallel_size
    num_pp = parallel_strategy.pipeline_parallel_size
    num_dp = parallel_strategy.data_parallel_size
    num_gpus = num_dp * num_mp * num_pp

    train_mbs = (
        batch_size / (2 * num_pp * num_dp * n_ppo_minibatches)
        if num_pp > 1
        else batch_size / (num_dp * n_ppo_minibatches)
    )
    gen_mbs = batch_size / (num_pp * num_dp)

    # pprint.pprint(op_cost, indent=4)
    inst_keys = [
        "gen_fwd_0",
        "gen_fwd_1",
        "inf_fwd",
        "train_fwd",
        "train_bwd",
        "train_opt",
    ]
    op_names = ["fwd_gen_0", "fwd_gen_1", "fwd", "fwd", "bwd", "opt"]
    inst_stats = {}

    for inst_key, op_name in zip(inst_keys, op_names):
        mbs = train_mbs if "train" in inst_key else gen_mbs
        inst_stats[inst_key] = computation_instruction_time_cost(
            op_stats, op_name, num_layers, parallel_strategy, mbs, seq_len
        )

    comm_type = "remote_send" if num_gpus // num_pp >= 8 else "local_send"

    inst_stats["act_p2p"] = communication_instruction_time_cost(
        comm_stats, 2 * hidden_dim * train_mbs * seq_len, comm_type
    )
    inst_stats["grad_p2p"] = communication_instruction_time_cost(
        comm_stats, 2 * hidden_dim * train_mbs * seq_len, comm_type
    )
    inst_stats["gen_act_p2p"] = communication_instruction_time_cost(
        comm_stats, 2 * hidden_dim * gen_mbs, comm_type
    )
    return inst_stats


def _estimate_rpc_time_cost(
    inst_stats,
    parallel_strategy: ParallelismConfig,
    model_interface_type: ModelInterfaceType,
    # model function call args
    num_gen_tokens: int,
    gradient_checkpointing: bool,
    n_ppo_minibatches: int = 1,
):
    # TODO: improve/remove heuristic
    num_pp = parallel_strategy.pipeline_parallel_size
    num_dp = parallel_strategy.data_parallel_size
    num_mp = parallel_strategy.model_parallel_size
    if model_interface_type == ModelInterfaceType.INFERENCE:
        num_micro_batches = num_pp
        compute_cost = inst_stats["inf_fwd"] * (num_pp + num_micro_batches - 1)
        comm_cost = inst_stats["act_p2p"] * (num_pp + num_micro_batches - 2) * 2
        if num_mp > 1:  # and num_pp * num_dp > 1:
            compute_cost = compute_cost * (1 - num_mp * 0.03)
    elif model_interface_type == ModelInterfaceType.TRAIN_STEP:
        # TODO: add reduce grads, add ppo micro batches
        num_micro_batches = num_pp * 2 if num_pp > 1 else 1
        compute_cost = (inst_stats["train_fwd"] + inst_stats["train_bwd"]) * (
            num_pp + num_micro_batches - 1
        ) + inst_stats["train_opt"]
        if gradient_checkpointing:
            compute_cost += inst_stats["train_fwd"] * (num_pp + num_micro_batches - 1)
        comm_cost = (
            (inst_stats["grad_p2p"] + inst_stats["act_p2p"])
            * (num_pp + num_micro_batches - 2)
            * 2
        )
        compute_cost = compute_cost * n_ppo_minibatches
        comm_cost = comm_cost * n_ppo_minibatches
        if num_pp * num_dp <= 1:
            compute_cost = compute_cost * (1 - num_mp * 0.04)
        if num_mp > 1:  # and num_pp * num_dp > 1:
            compute_cost = compute_cost * (1 - num_mp * 0.03)
    elif model_interface_type == ModelInterfaceType.GENERATE:
        num_micro_batches = num_pp
        num_gen_tokens = num_gen_tokens
        compute_cost = (
            inst_stats["gen_fwd_0"] * (num_pp + num_micro_batches - 1)
            + inst_stats["gen_fwd_1"] * (num_gen_tokens - 1) * num_micro_batches
        )

        if num_dp * num_mp > 1:
            compute_cost = compute_cost * (1 - min(num_dp * num_mp, 8) * 0.03)
        comm_cost = 0

    # dirty heuristic
    if num_pp > 8:
        compute_cost = compute_cost * (1 + num_pp * 0.01)

    # FIXME: disable comm cost for its not accurate
    comm_cost = 0

    return compute_cost + comm_cost


def estimate_rpc_time_cost(
    rpc: MFCDef,
    parallel_strategy: ParallelismConfig,
    bs: int,
    seq_len: int,
    num_gen_tokens: int = 256,
    gradient_checkpointing: bool = False,
    n_ppo_minibatches: int = 1,
):
    # time unit: miliseconds
    # FIXME: n_ppo_minibatches > 1 will result in bad estimation
    # when batch size is large enough, n_ppo_minibatches > 1 will not affect the estimation
    n_ppo_minibatches = 1
    model_type = rpc.model_type
    model_path = rpc.model_path
    model_config = load_model_config(rpc.model_type._class, rpc.model_path)

    inst_cost = estimate_instruction_time_costs(
        model_type,
        model_path,
        model_config.n_layers,
        parallel_strategy,
        model_config.hidden_dim,
        bs,
        seq_len,
        n_ppo_minibatches=n_ppo_minibatches,
    )
    return (
        _estimate_rpc_time_cost(
            inst_cost,
            parallel_strategy,
            rpc.interface_type,
            num_gen_tokens=num_gen_tokens,
            gradient_checkpointing=gradient_checkpointing,
            n_ppo_minibatches=n_ppo_minibatches,
        )
    ) / 1e6


def default_communication_stats(if_print=False):
    # use default comm stats of cluster
    r = dict(
        # between GPUs on the same node
        local_send=170,  # unit: GB/s
        local_recv=170,
        # IB between GPUs on different nodes
        remote_send=20,  # unit: GB/s
        remote_recv=20,
    )
    if if_print:
        print(r)
    return r


def estimate_model_size(model_config: ReaLModelConfig):
    h = model_config.hidden_dim
    i = model_config.intermediate_dim
    v = model_config.vocab_size
    L = model_config.n_layers
    # for llama actor only
    n_params = 3 * v * h + (3 * h * i + 4 * h * h) * L
    return 2 * n_params


def estimate_rpc_memory_cost(
    rpc: MFCDef,
    parallel_strategy: ParallelismConfig,
    batch_size: int,
    seq_len: int,
    offload: bool = False,
    gradient_checkpointing: bool = False,
    offload_optimizer: bool = False,
    n_ppo_minibatches: int = 1,
    num_gen_tokens: int = 128,
):
    # TODO: improve heuristic
    interface_type = rpc.interface_type
    model_config = load_model_config(rpc.model_type._class, rpc.model_path)

    h = model_config.hidden_dim
    i = model_config.intermediate_dim
    v = model_config.vocab_size
    s = seq_len
    gs = num_gen_tokens
    b = batch_size
    L = model_config.n_layers
    # for llama actor only
    n_params = 3 * v * h + (3 * h * i + 4 * h * h) * L
    param_mem = 2 * n_params
    grad_mem = 2 * n_params
    optimizer_mem = 20 * n_params if not offload_optimizer else 0

    num_pp = parallel_strategy.pipeline_parallel_size
    num_mp = parallel_strategy.model_parallel_size
    num_dp = parallel_strategy.data_parallel_size
    # zero1, pp and mp divide evenly
    # enable sequence parallel
    if interface_type == ModelInterfaceType.TRAIN_STEP:
        # gradient checkpointing is always enabled for flash attn
        static_mem = (param_mem + grad_mem) // (num_pp * num_mp) + optimizer_mem // (
            num_pp * num_dp * num_mp
        )
        micro_bs = b // (2 * num_pp * num_dp) if num_pp > 0 else b // (num_dp)
        if gradient_checkpointing:
            active_mem = (micro_bs * s * h * num_pp * 2) // (num_pp * num_mp)
        else:
            # FIXME: calculate other memory entries
            active_mem = (micro_bs * s * h * num_pp * 2) * 2 * L // (num_pp * num_mp)
        return static_mem + active_mem, static_mem
    elif interface_type == ModelInterfaceType.INFERENCE:
        static_mem = int(2 * param_mem // (num_pp * num_mp))
        # if num_dp > 4:
        #     static_mem = static_mem * 1.25
        if offload:
            return static_mem, 0  # assume offload
        else:
            return static_mem, static_mem
    elif interface_type == ModelInterfaceType.GENERATE:
        static_mem = int(2 * param_mem // (num_pp * num_mp))
        if num_dp > 4 and num_dp * num_mp * num_pp <= 16:
            static_mem = static_mem * 1.25
        if num_mp == 0 and num_pp == 0:
            static_mem = static_mem * 1.25
        active_mem = (
            2 * (2 * b * (gs + s) * h) * L // (num_pp * num_mp * num_dp)
        )  # kv cache
        return static_mem + active_mem, static_mem


def example(rpcs):
    if args.model_size == 7:
        n_nodes = 1
    elif args.model_size == 13:
        n_nodes = 2
    elif args.model_size == 34:
        n_nodes = 4
    elif args.model_size == 70:
        n_nodes = 8

    expr_name = f"profile-s{args.model_size}p{n_nodes}m1d8"

    rollout, inf, train = rpcs

    bs = 128
    seq_len = 1024

    p1 = ParallelismConfig(
        pipeline_parallel_size=1, model_parallel_size=4, data_parallel_size=8
    )
    rpc_cost = estimate_rpc_time_cost(
        train,
        p1,
        gradient_checkpointing=True,
        num_gen_tokens=896,
        bs=bs,
        seq_len=seq_len,
        n_ppo_minibatches=4,
    )
    mem_cost, static_mem = estimate_rpc_memory_cost(rollout, p1, bs, seq_len)
    print(f"{p1} rpc cost {rpc_cost:.2f} seconds mem cost {mem_cost/(1024**3):.2f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a profiling experiment.")
    parser.add_argument(
        "-s",
        "--model_size",
        type=int,
        default=7,
    )
    args = parser.parse_args()

    example(args)
