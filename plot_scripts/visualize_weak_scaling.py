from typing import *
import argparse
import itertools
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import transformers

from api.config.config_base import MODEL_TYPE_TO_PATH, ModelType
from api.config.config_flash_model import FLASH_MODEL_CONFIG_CONVERTER
from base.monitor import caculuate_llama_forward_flops, calculate_llama_gen_flops, calculate_llama_train_flops


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


def amend_baseline_data(all_data: List, baseline_name: str):
    if baseline_name == "DeepSpeedChat":
        with open("/lustre/fw/sosp24/dschat_res.pkl", "rb") as f:
            data: pd.DataFrame = pickle.load(f)
    elif baseline_name == "OpenRLHF":
        with open("/lustre/fw/sosp24/openrlhf_res.pkl", "rb") as f:
            data: pd.DataFrame = pickle.load(f)
    else:
        raise NotImplementedError()

    for case in range(3):
        for main_model_size, seqlen in itertools.product([7, 13, 34, 70], [128, 384, 896]):
            subplot_y = 0 if seqlen == 128 else 1 if seqlen == 384 else 2
            actor_size, ref_size, critic_size, rew_size = get_model_sizes(main_model_size, case)
            n_gpus = get_n_gpus(main_model_size, case)
            bs = 2**17 // (seqlen + 128)
            df = data[(data["a"] == actor_size)
                      & (data["c"] == critic_size)
                      & (data["s"] == seqlen)
                      & (data["n_gpus"] == n_gpus)]
            if len(df) == 0:
                all_data.append(
                    dict(
                        subplot_x=case,
                        subplot_y=subplot_y,
                        ngpus=n_gpus,
                        pflops=None,
                        cih=None,
                        cil=None,
                        System=baseline_name,
                        t=None,
                    ))
                continue
            assert len(df) == 1, df
            d = df.to_dict(orient="records")[0]
            p = compute_rlhf_pflops(
                actor_size,
                critic_size,
                ref_size,
                rew_size,
                batch_size=bs,
                prompt_len=128,
                gen_len=seqlen,
                avg_time=d["avg_t"],
            )
            cih = compute_rlhf_pflops(
                actor_size,
                critic_size,
                ref_size,
                rew_size,
                batch_size=bs,
                prompt_len=128,
                gen_len=seqlen,
                avg_time=d["cil"],
            )
            cil = compute_rlhf_pflops(
                actor_size,
                critic_size,
                ref_size,
                rew_size,
                batch_size=bs,
                prompt_len=128,
                gen_len=seqlen,
                avg_time=d["cih"],
            )
            all_data.append(
                dict(
                    subplot_x=case,
                    subplot_y=subplot_y,
                    ngpus=n_gpus,
                    pflops=p,
                    cih=cih,
                    cil=cil,
                    System=baseline_name,
                    t=df['avg_t'].values.item(),
                ))
    return all_data


def t_score_ci(m, std, size):
    # Calculate mean and standard deviation
    mean = m
    std_dev = std

    # Define confidence level (e.g., 95%)
    confidence_level = 0.95

    # Degrees of freedom (n-1 for a sample)
    degrees_of_freedom = size - 1

    # Calculate the critical value based on the confidence level and degrees of freedom
    t_score = scipy.stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)

    # Calculate the margin of error
    margin_of_error = t_score * (std_dev / np.sqrt(size))

    # Calculate the confidence interval
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    return lower_bound, upper_bound


def amend_ours_data(all_data: List, data: pd.DataFrame, mode):
    name = "ReaL (Ours)" if mode == "s" else "ReaL-Heuristic"
    data = data[data["mode"] == mode]
    for case in range(3):
        for main_model_size, seqlen in itertools.product([7, 13, 34, 70], [128, 384, 896]):
            subplot_y = 0 if seqlen == 128 else 1 if seqlen == 384 else 2
            actor_size, ref_size, critic_size, rew_size = get_model_sizes(main_model_size, case)
            n_gpus = get_n_gpus(main_model_size, case)
            bs = 2**17 // (seqlen + 128)
            df = data[(data["actor_model_size"] == actor_size)
                      & (data["critic_model_size"] == critic_size)
                      & (data["seqlen"] == seqlen)
                      & (data["n_nodes"] == n_gpus // 8)]
            assert len(df) == 1, df
            d = df.to_dict(orient="records")[0]
            p = compute_rlhf_pflops(
                actor_size,
                critic_size,
                ref_size,
                rew_size,
                batch_size=bs,
                prompt_len=128,
                gen_len=seqlen,
                avg_time=d["time"],
            )
            cil_t, cih_t = t_score_ci(d["time"], d["std"], d["n"])
            cih = compute_rlhf_pflops(
                actor_size,
                critic_size,
                ref_size,
                rew_size,
                batch_size=bs,
                prompt_len=128,
                gen_len=seqlen,
                avg_time=cil_t,
            )
            cil = compute_rlhf_pflops(
                actor_size,
                critic_size,
                ref_size,
                rew_size,
                batch_size=bs,
                prompt_len=128,
                gen_len=seqlen,
                avg_time=cih_t,
            )
            all_data.append(
                dict(
                    subplot_x=case,
                    subplot_y=subplot_y,
                    ngpus=n_gpus,
                    pflops=p,
                    cih=cih,
                    cil=cil,
                    t=d['time'],
                    System=name,
                ))
    return all_data


def main():
    all_data = []
    all_data = amend_baseline_data(all_data, "DeepSpeedChat")
    all_data = amend_baseline_data(all_data, "OpenRLHF")
    # amend our System's result
    with open("/lustre/meizy/res_df.pkl", "rb") as f:
        data = pickle.load(f)
    all_data = amend_ours_data(all_data, data, "m")
    all_data = amend_ours_data(all_data, data, "s")

    # Convert data to DataFrame
    df = pd.DataFrame(all_data)
    # print(df.to_string(index=False))
    # input()

    # Set style
    sns.set_style("whitegrid")

    # Create subplots
    fig, axes = plt.subplots(3, 3, figsize=(12, 5))
    width = 0.75

    _n_gpus = df[df["System"] == "ReaL (Ours)"]["ngpus"].values
    assert (_n_gpus == df[df["System"] == "ReaL-Heuristic"]["ngpus"].values).all()
    heuristic_flops = df[df["System"] == "ReaL-Heuristic"]["pflops"].values
    real_flops = df[df["System"] == "ReaL (Ours)"]["pflops"].values
    rel = (real_flops - heuristic_flops) / heuristic_flops
    print(
        f"Relative improvement over heuristic plan: {((real_flops - heuristic_flops) / heuristic_flops * _n_gpus).sum() / (_n_gpus.sum())}, min {rel.min()}, max {rel.max()}"
    )
    _real_flops_7b34b = df[(df["System"] == "ReaL (Ours)")
                           & (df["subplot_x"] == 1)
                           & (df["subplot_y"] == 1)
                           & (df["ngpus"] == 32)]["pflops"].values
    _heuristic_flops_7b34b = df[(df["System"] == "ReaL-Heuristic")
                                & (df["ngpus"] == 32)
                                & (df["subplot_x"] == 1)
                                & (df["subplot_y"] == 1)]["pflops"].values
    _dschat_flops_7b34b = df[(df["System"] == "DeepSpeedChat")
                             & (df["ngpus"] == 32)
                             & (df["subplot_x"] == 1)
                             & (df["subplot_y"] == 1)]["pflops"].values
    _openrlhf_flops_7b34b = df[(df["System"] == "OpenRLHF") & (df["ngpus"] == 32) & (df["subplot_x"] == 1) &
                               (df["subplot_y"] == 1)]["pflops"].values
    print(
        f"Relative improvement in 7b+34b genlen 896 #GPU 32 case: ",
        (_real_flops_7b34b - _heuristic_flops_7b34b) / _heuristic_flops_7b34b,
        (_real_flops_7b34b - _dschat_flops_7b34b) / _dschat_flops_7b34b,
        (_real_flops_7b34b - _openrlhf_flops_7b34b) / _openrlhf_flops_7b34b,
    )

    dschat_flops = df[df["System"] == "DeepSpeedChat"]["pflops"].values
    mask = np.logical_not(np.isnan(dschat_flops))
    rel = (real_flops - dschat_flops) / dschat_flops
    _mean = np.where(mask, rel * _n_gpus, 0.0).sum() / ((_n_gpus * mask).sum())
    _min = np.where(mask, rel, 100.0).min()
    _max = np.where(mask, rel, 0.0).max()
    print(f"Relative improvement over DeepSpeedChat: {_mean}, min {_min}, max {_max}")

    openrlhf_flops = df[df["System"] == "OpenRLHF"]["pflops"].values
    mask = np.logical_not(np.isnan(openrlhf_flops))
    rel = (real_flops - openrlhf_flops) / openrlhf_flops
    _mean = np.where(mask, rel * _n_gpus, 0.0).sum() / ((_n_gpus * mask).sum())
    _min = np.where(mask, rel, 100.0).min()
    _max = np.where(mask, rel, 0.0).max()
    print(f"Relative improvement over openrlhf: {_mean}, min {_min}, max {_max}")

    _d = df[(df['subplot_x'] == 2) & (df['subplot_y'] == 0) & (df['ngpus'] == 128)]
    _complete_train_hours = _d['t'].values * 1600 / 3600
    print(f"Complete training hours for {_d['System']}: ", _complete_train_hours)

    # Plot for each seqlen setting
    for subplot_x, subplot_y in itertools.product(range(3), range(3)):
        case = subplot_x
        ax = axes[subplot_x, subplot_y]
        group = df[(df["subplot_x"] == case) & (df["subplot_y"] == subplot_y)]

        sns.barplot(
            x="ngpus",
            y="pflops",
            data=group,
            ax=ax,
            hue="System",
            width=width,
            palette=sns.color_palette(n_colors=4),
        )

        n_gpus = group["ngpus"].unique().tolist()
        systems = group["System"].unique().tolist()
        width_per_bar = width / len(systems)

        # plot error bars
        # offsets = [width_per_bar * systems.index(s) + 0.5 * width_per_bar for s in group["System"]]
        # errors = np.array([
        #         [mean - lower, upper - mean]
        #         for mean, lower, upper in zip(group['pflops'], group['cil'], group['cih'])
        #     ]).swapaxes(0, 1)
        # print(np.mean(errors[0] / group['pflops']))
        # ax.errorbar(
        #     [n_gpus.index(x) - width / 2 + offset for x, offset in zip(group["ngpus"], offsets)],
        #     group['pflops'],
        #     yerr=np.array([
        #         [mean - lower, upper - mean]
        #         for mean, lower, upper in zip(group['pflops'], group['cil'], group['cih'])
        #     ]).swapaxes(0, 1),
        #     fmt="none",
        #     ecolor="black",
        #     capsize=5,
        # )

        missing_points = group[group["pflops"].isnull()]
        offsets = [width_per_bar * systems.index(s) + 0.5 * width_per_bar for s in missing_points["System"]]
        # Plot missing data points with red cross
        if not missing_points.empty:
            ax.plot(
                [n_gpus.index(x) - width / 2 + offset for x, offset in zip(missing_points["ngpus"], offsets)],
                [0.1 * ax.get_ylim()[1]] * len(missing_points),
                "rx",
                markersize=10,
                mew=3,
            )

        # ax.set_yscale("log")
        if subplot_x == 2:
            ax.set_xlabel("Number of GPUs", fontsize=16)
        else:
            ax.set_xlabel(None)
        if subplot_y == 0 and subplot_x == 1:
            ax.set_ylabel("Throughput PetaFLOP/s", fontsize=16)
        else:
            ax.set_ylabel(None)
        if subplot_y == 2:
            ax2 = ax.twinx()
            rightylabel = ["Scale Actor", "Scale Critic", "Scale Both"][subplot_x]
            ax2.set_ylabel(rightylabel, fontsize=16)
            ax2.set_yticks([])

        if not (subplot_y == 0):
            ax.get_legend().remove()
        else:
            ax.legend(fontsize=8)

    for j, title in enumerate(["Gen.Len.=128", "Gen.Len.=384", "Gen.Len.=896"]):
        axes[0, j].set_title(title, fontsize=18)

    axes[0, 0].set_ylim(0, 3.5)
    axes[0, 0].set_yticks([0, 1, 2, 3])
    axes[0, 1].set_ylim(0, 2.5)
    axes[1, 1].set_ylim(0, 4)
    axes[1, 1].set_yticks([0, 1, 2, 3, 4])
    axes[1, 2].set_ylim(0, 3.5)
    axes[1, 2].set_yticks([0, 1, 2, 3])
    axes[2, 0].set_ylim(0, 7)
    axes[2, 0].set_yticks([0, 2, 4, 6])
    axes[2, 1].set_ylim(0, 4)
    axes[2, 1].set_yticks([0, 1, 2, 3, 4])
    # for i, title in enumerate(['Scale Actor', 'Scale Critic', 'Scale Both']):
    #     axes[i, 0].text(13.5, 2, title, fontsize=16, rotation=270, ha='left', va='baseline')

    # Adjust layout
    plt.tight_layout()

    plt.savefig("plot_scripts/figures/vws.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
