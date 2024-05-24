from typing import *
import argparse
import collections
import itertools
import json
import os
import pickle

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import transformers

from reallm.api.core.model_api import (MODEL_FAMILY_TO_PATH, ModelFamily, ModelName,
                                       REAL_MODEL_CONFIG_CONVERTER)
from reallm.base.monitor import (caculuate_llama_forward_flops, calculate_llama_gen_flops,
                                 calculate_llama_train_flops, CUDAKernelTime)


def round_to_nearest_tenth(num):
    """
    Round a float to the nearest 0.1.

    Args:
        num (float): The number to be rounded.

    Returns:
        float: The rounded number.
    """
    if isinstance(num, (float, int)):
        return round(num * 100) / 100
    else:
        return np.array([round(x * 100) / 100 for x in num])


def compute_rlhf_pflops(
    actor_size: int,
    critic_size: int,
    ref_size: int,
    rw_size: int,
    batch_size: int,
    prompt_len: int,
    gen_len: int,
    avg_time: float,
):
    mconfigs = {}
    for name, model_size in [
        ("actor", actor_size),
        ("critic", critic_size),
        ("ref", ref_size),
        ("rw", rw_size),
    ]:
        hf_model_type = "llama" if model_size != 34 else "codellama"
        path = MODEL_FAMILY_TO_PATH[ModelFamily(hf_model_type, model_size, True)]
        hf_config = transformers.AutoConfig.from_pretrained(path)
        mconfig = REAL_MODEL_CONFIG_CONVERTER[hf_model_type](hf_config)
        mconfigs[name] = mconfig
    assert (prompt_len + gen_len) * batch_size == 2**17, (batch_size, prompt_len, gen_len)
    flops = 0
    flops += calculate_llama_gen_flops(
        batch_size,
        [prompt_len] * batch_size,
        gen_len,
        num_layers=mconfigs["actor"].n_layers,
        hidden_size=mconfigs["actor"].hidden_dim,
        intermediate_size=mconfigs["actor"].intermediate_dim,
        vocab_size=mconfigs["actor"].vocab_size,
    )
    for name in ["critic", "ref", "rw"]:
        flops += caculuate_llama_forward_flops(
            batch_size,
            [prompt_len + gen_len] * batch_size,
            num_layers=mconfigs[name].n_layers,
            hidden_size=mconfigs[name].hidden_dim,
            intermediate_size=mconfigs[name].intermediate_dim,
            vocab_size=mconfigs[name].vocab_size,
        )
    for name in ["actor", "critic"]:
        flops += calculate_llama_train_flops(
            4,
            batch_size,
            [prompt_len + gen_len] * batch_size,
            num_layers=mconfigs[name].n_layers,
            hidden_size=mconfigs[name].hidden_dim,
            intermediate_size=mconfigs[name].intermediate_dim,
            vocab_size=mconfigs[name].vocab_size,
        )
    return flops / 1e15 / avg_time


def compute_rlhf_gen_pflops(
    actor_size: int,
    batch_size: int,
    prompt_len: int,
    gen_len: int,
    avg_time: float,
):
    mconfigs = {}
    for name, model_size in [("actor", actor_size)]:
        hf_model_type = "llama" if model_size != 34 else "codellama"
        path = MODEL_FAMILY_TO_PATH[ModelFamily(hf_model_type, model_size, True)]
        hf_config = transformers.AutoConfig.from_pretrained(path)
        mconfig = REAL_MODEL_CONFIG_CONVERTER[hf_model_type](hf_config)
        mconfigs[name] = mconfig
    assert (prompt_len + gen_len) * batch_size == 2**17, (batch_size, prompt_len, gen_len)
    return (calculate_llama_gen_flops(
        batch_size,
        [prompt_len] * batch_size,
        gen_len,
        num_layers=mconfigs["actor"].n_layers,
        hidden_size=mconfigs["actor"].hidden_dim,
        intermediate_size=mconfigs["actor"].intermediate_dim,
        vocab_size=mconfigs["actor"].vocab_size,
    ) / 1e15 / avg_time)


def compute_rlhf_inf_pflops(
    model_size: int,
    batch_size: int,
    seqlen: int,
    avg_time: float,
):
    hf_model_type = "llama" if model_size != 34 else "codellama"
    path = MODEL_FAMILY_TO_PATH[ModelFamily(hf_model_type, model_size, True)]
    hf_config = transformers.AutoConfig.from_pretrained(path)
    mconfig = REAL_MODEL_CONFIG_CONVERTER[hf_model_type](hf_config)
    flops = caculuate_llama_forward_flops(
        batch_size,
        [seqlen] * batch_size,
        num_layers=mconfig.n_layers,
        hidden_size=mconfig.hidden_dim,
        intermediate_size=mconfig.intermediate_dim,
        vocab_size=mconfig.vocab_size,
    )
    return flops / 1e15 / avg_time


def compute_rlhf_train_pflops(
    model_size: int,
    batch_size: int,
    seqlen: int,
    avg_time: float,
):
    return 4 * compute_rlhf_inf_pflops(model_size, batch_size, seqlen, avg_time)


def get_model_sizes(main_model_size: int, case: int):
    if case == 0:
        return (main_model_size, main_model_size, 7, 7)
    elif case == 1:
        return (7, 7, main_model_size, main_model_size)
    elif case == 2:
        return (main_model_size, main_model_size, main_model_size, main_model_size)
    else:
        raise NotImplementedError()


def get_n_gpus(main_model_size: int, case):
    default_ngpus = (8 if main_model_size == 7 else
                     16 if main_model_size == 13 else 32 if main_model_size == 34 else 64)
    return default_ngpus if case <= 1 else 2 * default_ngpus


def amend_baseline_data(all_data: List, baseline_name: str):
    if baseline_name == "DeepSpeedChat":
        with open("/lustre/fw/sosp24/dschat_res.pkl", "rb") as f:
            data: pd.DataFrame = pickle.load(f)
    elif baseline_name == "OpenRLHF":
        with open("/lustre/fw/sosp24/openrlhf_res.pkl", "rb") as f:
            data: pd.DataFrame = pickle.load(f)
    else:
        raise NotImplementedError()

    for actor_size, critic_size in itertools.product([7, 13, 34, 70], [7, 13, 34, 70]):
        seqlen = 128
        main_model_size = max(actor_size, critic_size)
        if actor_size != critic_size and actor_size > 7 and critic_size > 7:
            continue
        if critic_size == 7:
            case = 0
        elif actor_size == 7:
            case = 1
        else:
            case = 2
        if case == 0:
            ref_size = actor_size
            rew_size = 7
        elif case == 1:
            ref_size = 7
            rew_size = critic_size
        else:
            assert actor_size == critic_size > 7
            ref_size = rew_size = actor_size
        n_gpus = get_n_gpus(main_model_size, case)
        bs = 2**17 // (seqlen + 128)
        df = data[(data["a"] == actor_size)
                  & (data["c"] == critic_size)
                  & (data["s"] == seqlen)
                  & (data["n_gpus"] == n_gpus)]
        assert len(df) == 1, len(df)
        all_data.append(
            dict(
                actor_size=f"{actor_size}B",
                critic_size=critic_size,
                x=f"{actor_size}B+{critic_size}B",
                overall_time=np.mean(df["avg_t"]),
                n_gpus=n_gpus,
                actor_gen_time=np.mean(df["avg_gt"]),
                actor_gen_pflops=compute_rlhf_gen_pflops(
                    actor_size,
                    bs,
                    prompt_len=128,
                    gen_len=seqlen,
                    avg_time=np.mean(df["avg_gt"]),
                ),
                critic_inf_time=np.mean(df["avg_cit"]),
                critic_inf_pflops=compute_rlhf_inf_pflops(
                    critic_size,
                    bs,
                    seqlen + 128,
                    avg_time=np.mean(df["avg_cit"]),
                ),
                ref_inf_time=np.mean(df["avg_rfit"]),
                ref_inf_pflops=compute_rlhf_inf_pflops(
                    ref_size,
                    bs,
                    seqlen + 128,
                    avg_time=np.mean(df["avg_rfit"]),
                ),
                rew_inf_time=np.mean(df["avg_rit"]),
                rew_inf_pflops=compute_rlhf_inf_pflops(
                    rew_size,
                    bs,
                    seqlen + 128,
                    avg_time=np.mean(df["avg_rit"]),
                ),
                actor_train_time=np.mean(df["avg_att"]),
                actor_train_pflops=compute_rlhf_train_pflops(
                    actor_size,
                    bs,
                    seqlen + 128,
                    avg_time=np.mean(df["avg_att"]),
                ),
                critic_train_time=np.mean(df["avg_ctt"]),
                critic_train_pflops=compute_rlhf_train_pflops(
                    critic_size,
                    bs,
                    seqlen + 128,
                    avg_time=np.mean(df["avg_ctt"]),
                ),
                System=baseline_name,
            ))
    return all_data


def amend_ours_data(all_data: List, data: pd.DataFrame, mode):
    name = "ReaL (Ours)" if mode == "s" else "ReaL-Heuristic"
    data = data[data["mode"] == mode]
    for actor_size, critic_size in itertools.product([7, 13, 34, 70], [7, 13, 34, 70]):
        seqlen = 128
        main_model_size = max(actor_size, critic_size)
        if actor_size != critic_size and actor_size > 7 and critic_size > 7:
            continue
        if critic_size == 7:
            case = 0
        elif actor_size == 7:
            case = 1
        else:
            case = 2
        if case == 0:
            ref_size = actor_size
            rew_size = 7
        elif case == 1:
            ref_size = 7
            rew_size = critic_size
        else:
            assert actor_size == critic_size > 7
            ref_size = rew_size = actor_size
        n_gpus = get_n_gpus(main_model_size, case)
        bs = 2**17 // (seqlen + 128)
        df = data[(data["actor_model_size"] == actor_size)
                  & (data["critic_model_size"] == critic_size)
                  & (data["seqlen"] == seqlen)
                  & (data["n_nodes"] == n_gpus // 8)]
        assert len(df) == 1, len(df)
        logpath = df["log_path"].tolist()[0]
        handle_type2time = collections.defaultdict(list)
        handle_type2device_count = {}
        if mode == "s":
            actor_topos = []
            critic_topos = []
            with open(os.path.join(logpath, "device_mapping.pkl"), "rb") as f:
                device_mapping = pickle.load(f)
            for k, v in device_mapping.items():
                role = k.split("ModelName(role='")[1].split("'")[0]
                handle_name = k.split("@")[1]
                p = v.train_eval_config.parallel.pipeline_parallel_size
                m = v.train_eval_config.parallel.model_parallel_size
                d = v.train_eval_config.parallel.data_parallel_size
                handle_type2device_count[(role, handle_name)] = p * m * d
                topo = (p, m, d)
                if role == "actor":
                    actor_topos.append((topo, v.mapping))
                elif role == "critic":
                    critic_topos.append((topo, v.mapping))
        else:
            m = 8
            p = n_gpus // m
            d = 1
            for role, handle_name in [
                ("actor", "generate"),
                ("actor", "train_step"),
                ("critic", "inference"),
                ("critic", "train_step"),
                ("ref", "inference"),
                ("reward", "inference"),
            ]:
                handle_type2device_count[(role, handle_name)] = p * m * d
        for fn in os.listdir(logpath):
            if not fn.startswith("model_worker"):
                continue
            with open(os.path.join(logpath, fn), "r") as f:
                for line in f.readlines():
                    if "Model worker" in line and "handle request" in line:
                        role = line.split("#ModelName(role='")[1].split("'")[0]
                        handle_type = line.split("*")[1]
                        t = float(line.split("$")[1])
                        handle_type2time[(role, handle_type)].append(t)
        for k in handle_type2time:
            handle_type2time[k] = handle_type2time[k][5:]
        all_data.append(
            dict(
                actor_size=f"{actor_size}B",
                critic_size=critic_size,
                x=f"{actor_size}B+{critic_size}B",
                overall_time=np.mean(df["time"]),
                n_gpus=n_gpus,
                total_gpu_time=n_gpus * np.mean(df["time"]),
                actor_gen_device_count=handle_type2device_count[("actor", "generate")],
                actor_gen_time=np.mean(handle_type2time[("actor", "generate")]),
                actor_gen_pflops=compute_rlhf_gen_pflops(
                    actor_size,
                    bs,
                    prompt_len=128,
                    gen_len=seqlen,
                    avg_time=np.mean(handle_type2time[("actor", "generate")]),
                ),
                critic_inf_device_count=handle_type2device_count[("critic", "inference")],
                critic_inf_time=np.mean(handle_type2time[("critic", "inference")]),
                critic_inf_pflops=compute_rlhf_inf_pflops(
                    critic_size,
                    bs,
                    seqlen + 128,
                    avg_time=np.mean(handle_type2time[("critic", "inference")]),
                ),
                ref_inf_device_count=handle_type2device_count[("ref", "inference")],
                ref_inf_time=np.mean(handle_type2time[("ref", "inference")]),
                ref_inf_pflops=compute_rlhf_inf_pflops(
                    ref_size,
                    bs,
                    seqlen + 128,
                    avg_time=np.mean(handle_type2time[("ref", "inference")]),
                ),
                rew_inf_device_count=handle_type2device_count[("reward", "inference")],
                rew_inf_time=np.mean(handle_type2time[("reward", "inference")]),
                rew_inf_pflops=compute_rlhf_inf_pflops(
                    rew_size,
                    bs,
                    seqlen + 128,
                    avg_time=np.mean(handle_type2time[("reward", "inference")]),
                ),
                actor_train_device_count=handle_type2device_count[("actor", "train_step")],
                actor_train_time=np.mean(handle_type2time[("actor", "train_step")]),
                actor_train_pflops=compute_rlhf_train_pflops(
                    actor_size,
                    bs,
                    seqlen + 128,
                    avg_time=np.mean(handle_type2time[("actor", "train_step")]),
                ),
                critic_train_device_count=handle_type2device_count[("critic", "train_step")],
                critic_train_time=np.mean(handle_type2time[("critic", "train_step")]),
                critic_train_pflops=compute_rlhf_train_pflops(
                    critic_size,
                    bs,
                    seqlen + 128,
                    avg_time=np.mean(handle_type2time[("critic", "train_step")]),
                ),
                System=name,
            ))
    return all_data


def main():
    all_data = []
    # all_data = amend_baseline_data(all_data, "DeepSpeedChat")
    # all_data = amend_baseline_data(all_data, "OpenRLHF")
    # amend our System's result
    with open("/lustre/meizy/res_df.pkl", "rb") as f:
        data = pickle.load(f)
    all_data = amend_ours_data(all_data, data, "m")
    all_data = amend_ours_data(all_data, data, "s")

    # Convert data to DataFrame
    df = pd.DataFrame(all_data)

    # Set style
    sns.set_style("whitegrid")

    # Create subplots
    fig = plt.figure(layout="constrained", figsize=(9, 12))
    gs = GridSpec(8, 5, figure=fig)
    # width = 0.75
    # plt.xticks(rotation=45)

    cmap = sns.color_palette(n_colors=4)[2:]

    xlabel_fontsize = 10
    ylabel_fontsize = 12
    legend_fontsize = 12
    title_fontsize = 16

    colormap = {
        "actor": "#DAE8FC",
        "critic": "#F8CECC",
        "ref": "#FFE6CC",
        "reward": "#D5E8D4",
        "comm": "#E1D5E7",
        "memory_layout": "#FFFF88",
    }
    edge_colormap = {
        "actor": "#6C8EBF",
        "critic": "#B85450",
        "ref": "#D79B00",
        "reward": "#82B366",
        "comm": "#9673A6",
        "memory_layout": "#36393D",
    }

    # Plot for each seqlen setting
    width = 0.35

    # settings = []
    # _model_sizes = [7, 13, 34, 70]
    # for _a, _c in itertools.product(_model_sizes, _model_sizes):
    #     settings.append(f"{_a}B+{_c}B")
    settings = ["7B+7B", "7B+70B", "70B+7B", "70B+70B"]

    for i, setting in enumerate(settings):
        group = df[(df["x"] == setting)]
        _keys = [
            "overall_time",
            "actor_gen_time",
            "critic_inf_time",
            "ref_inf_time",
            "rew_inf_time",
            "actor_train_time",
            "critic_train_time",
        ]
        _labels = [
            "Total",
            "Actor Gen.",
            "Critic Inf.",
            "Ref. Inf.",
            "Reward Inf.",
            "Actor Train",
            "Critic Train",
        ]
        new_df = []
        for system in group["System"].unique():
            for _k, _l in zip(_keys, _labels):
                d = dict(
                    System=system,
                    setting=setting,
                    label=_l,
                    time=group[group["System"] == system][_k].values.item(),
                )
                new_df.append(d)
        new_df = pd.DataFrame(new_df)

        ax = fig.add_subplot(gs[i, :1])
        sns.barplot(
            ax=ax,
            data=new_df[new_df["label"] == "Total"],
            hue="System",
            y="time",
            x="label",
            palette=cmap,
            hatch="/",
            edgecolor="black",
            linewidth=1.0,
        )
        ax.set_ylabel(f"{setting}\nTime (s)", fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)
        ax.set_xlabel(None)
        ax.get_legend().remove()

        ax = fig.add_subplot(gs[i, 1:])
        sns.barplot(
            ax=ax,
            data=new_df[new_df["label"] != "Total"],
            hue="System",
            y="time",
            x="label",
            palette=cmap,
            hatch="-",
            edgecolor="black",
            linewidth=1.0,
        )
        ax.set_xlabel(None)
        ax.set_ylabel("Time (s)", fontsize=10)
        if i != 0:
            ax.get_legend().remove()

        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)

    fig.suptitle("RLHF Iteration Wall-Time Breakdown for Different Actor/Crtic Sizes", fontsize=16)

    _keys = [
        "n_gpus",
        "actor_gen_device_count",
        "critic_inf_device_count",
        "ref_inf_device_count",
        "rew_inf_device_count",
        "actor_train_device_count",
        "critic_train_device_count",
    ]
    _labels = [
        "Total",
        "Actor Gen.",
        "Critic Inf.",
        "Ref. Inf.",
        "Reward Inf.",
        "Actor Train",
        "Critic Train",
    ]
    device_df = []
    for s in settings:
        d = dict(Setting=s)
        for _k, _l in zip(_keys, _labels):
            d[_l] = df[(df["x"] == s) & (df["System"] == "ReaL (Ours)")][_k].values.item()
        device_df.append(d)
    device_df = pd.DataFrame(device_df)
    print(device_df.to_latex(index=False))
    # for i, (ax, key, ylabel) in enumerate(
    #     zip(
    #         axes.flatten(),
    #         [
    #             "actor_gen_time",
    #             "critic_inf_time",
    #             # "ref_inf_pflops",
    #             # "rew_inf_pflops",
    #             "actor_train_time",
    #             "critic_train_time",
    #         ],
    #         [
    #             "Actor Generation",
    #             "Critic Inference",
    #             # "Ref. Inference",
    #             # "Reward Inference",
    #             "Actor Training",
    #             "Critic Training",
    #         ],
    #     )
    # ):

    #     sns.barplot(
    #         ax=ax,
    #         data=df,
    #         hue="System",
    #         y=key,
    #         x="x",
    #         palette=cmap,
    #
    #     )
    #     if i != 0:
    #         ax.get_legend().remove()
    #     else:
    #         ax.legend(loc="upper left", fontsize=9.5, ncol=4)
    #     ax.set_xlabel(None)
    #     # ax.set_yscale("log")
    #     # ax.set_yticks([0.1, 1.0])
    #     ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=xlabel_fontsize)
    #     ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)
    # fig.suptitle("Function Call Throughput (Log PetaFLOP/s)", fontsize=title_fontsize)
    # fig.supxlabel("Actor Size + Critic Size", fontsize=14)

    # Adjust layout
    plt.tight_layout()

    plt.savefig("assets/figures/v_funcall.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
