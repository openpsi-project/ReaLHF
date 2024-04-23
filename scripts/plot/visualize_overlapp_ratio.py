from typing import *
import argparse
import collections
import itertools
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import transformers

from reallm.api.core.config import MODEL_TYPE_TO_PATH, ModelName, ModelType
from reallm.api.core.model_api import FLASH_MODEL_CONFIG_CONVERTER
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
        path = MODEL_TYPE_TO_PATH[ModelType(hf_model_type, model_size, True)]
        hf_config = transformers.AutoConfig.from_pretrained(path)
        mconfig = FLASH_MODEL_CONFIG_CONVERTER[hf_model_type](hf_config)
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


def amend_ours_data(all_data: List, data: pd.DataFrame, mode):
    with open("/lustre/fw/sosp24/reallocation-cost-exp.json", "r") as f:
        reallocation_cost_table = json.load(f)

    name = "ReaL (Ours)" if mode == "s" else "ReaL-Heuristic"
    data = data[data["mode"] == mode]
    for actor_size, critic_size in itertools.product([7, 13, 34, 70], [7, 13, 34, 70]):
        seqlen = 896
        main_model_size = max(actor_size, critic_size)
        # HACK:
        if (actor_size == 70 and critic_size == 70) or (actor_size != 7 and critic_size != 7):
            continue
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
        print(logpath)
        handle_type2time = collections.defaultdict(list)
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
                actor_size=actor_size,
                critic_size=critic_size,
                overall_time=np.mean(df["time"]),
                n_gpus=n_gpus,
                gen_time=np.mean(handle_type2time[("actor", "generate")]),
                actor_train_time=np.mean(handle_type2time[("actor", "train_step")]),
                critic_train_time=np.mean(handle_type2time[("critic", "train_step")]),
                critic_inf_time=np.mean(handle_type2time[("critic", "inference")]),
                ref_inf_time=np.mean(handle_type2time[("ref", "inference")]),
                rew_inf_time=np.mean(handle_type2time[("reward", "inference")]),
                pflops=compute_rlhf_pflops(
                    actor_size,
                    critic_size,
                    ref_size,
                    rew_size,
                    batch_size=bs,
                    prompt_len=128,
                    gen_len=seqlen,
                    avg_time=np.mean(df["time"]),
                ),
                System=name,
            ))
    return all_data


def main():
    all_data = []
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
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    # width = 0.75
    # plt.xticks(rotation=45)

    cmap = sns.color_palette(n_colors=5)

    xlabel_fontsize = 12
    ylabel_fontsize = 18
    legend_fontsize = 12
    title_fontsize = 18

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
    width = 0.7
    group = df[(df["System"] == "ReaL (Ours)")]
    settings = [f"{a}B+{c}B" for a, c in zip(group["actor_size"], group["critic_size"])]
    gen_time = round_to_nearest_tenth(group["gen_time"].to_numpy())
    train_time = round_to_nearest_tenth(group["actor_train_time"].to_numpy() +
                                        group["critic_train_time"].to_numpy())
    inf_time = round_to_nearest_tenth(group["critic_inf_time"].to_numpy() + group["ref_inf_time"].to_numpy() +
                                      group["rew_inf_time"].to_numpy())
    overlap_ratio = (gen_time + train_time + inf_time -
                     group["overall_time"].to_numpy()) / group["overall_time"].to_numpy()
    sns.barplot(x=settings,
                y=overlap_ratio,
                ax=ax,
                color=cmap[3],
                lw=2,
                edgecolor="black",
                hatch=r"\\",
                width=width)
    ax.set_ylim((0, 0.32))
    ax.set_xticklabels(settings, rotation=0, fontsize=xlabel_fontsize)
    ax.set_yticklabels([round_to_nearest_tenth(x) for x in ax.get_yticks()], fontsize=ylabel_fontsize)
    ax.set_ylabel("Overlap Ratio", fontsize=ylabel_fontsize)
    ax.set_xlabel("Actor Size + Critic Size", fontsize=ylabel_fontsize)
    ax.set_title("Overlapped Computation of ReaL", fontsize=title_fontsize)

    # Adjust layout
    plt.tight_layout()

    plt.savefig("assets/figures/voverlap.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
