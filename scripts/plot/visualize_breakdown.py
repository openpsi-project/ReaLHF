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
        handle_type2time = collections.defaultdict(list)
        handle_type2device_count = {}
        handle_type2gpu_time = {}
        if mode == "s":
            actor_topos = []
            critic_topos = []
            with open(os.path.join(logpath, "device_mapping.pkl"), "rb") as f:
                device_mapping = pickle.load(f)
            for k, v in device_mapping.items():
                role = k.split("ModelName(role='")[1].split("'")[0]
                handle_name = k.split("@")[1]
                if role == "actor":
                    model_size = actor_size
                elif role == "critic":
                    model_size = critic_size
                elif role == "ref":
                    model_size = ref_size
                elif role == "reward":
                    model_size = rew_size
                else:
                    raise NotImplementedError(role)
                p = v.train_eval_config.parallel.pipeline_parallel_size
                m = v.train_eval_config.parallel.model_parallel_size
                d = v.train_eval_config.parallel.data_parallel_size
                if handle_name == "train_step":
                    _phn = "train"
                elif handle_name == "inference":
                    _phn = "inf"
                else:
                    _phn = "gen"
                _profile_log_path = (
                    f"/lustre/aigc/llm/logs/fw/profile-s{model_size}p{p}m{m}d{d}-{_phn}/cudakernel/")
                kernel_time = CUDAKernelTime(0, 0, 0, 0)
                for _j in range(p * m * d):
                    with open(os.path.join(_profile_log_path, f"kernel_time{_j}.pkl"), "rb") as f:
                        kernel_time = kernel_time + pickle.load(f)
                handle_type2gpu_time[(role, handle_name)] = kernel_time
                handle_type2device_count[(role, handle_name)] = p * m * d
                topo = (p, m, d)
                if role == "actor":
                    actor_topos.append((topo, v.mapping))
                elif role == "critic":
                    critic_topos.append((topo, v.mapping))
            assert len(actor_topos) == 2
            assert len(critic_topos) == 2
            param_realloc_time = 0
            if actor_topos[0][0] != actor_topos[1][0]:
                from_topo, from_mapping = actor_topos[0]
                to_topo, to_mapping = actor_topos[1]
                world_size = np.logical_or(from_mapping, to_mapping).sum()
                key1 = str((actor_size, world_size, from_topo, to_topo))
                key2 = str((actor_size, world_size, to_topo, from_topo))
                if key1 in reallocation_cost_table:
                    param_realloc_time += reallocation_cost_table[key1]["mem_shift_time_ns"] / 1e9 * world_size
                elif key2 in reallocation_cost_table:
                    param_realloc_time += reallocation_cost_table[key2]["mem_shift_time_ns"] / 1e9 * world_size
                else:
                    raise RuntimeError()
            if critic_topos[0][0] != critic_topos[1][0]:
                from_topo, from_mapping = critic_topos[0]
                to_topo, to_mapping = critic_topos[1]
                world_size = np.logical_or(from_mapping, to_mapping).sum()
                key1 = str((critic_size, world_size, from_topo, to_topo))
                key2 = str((critic_size, world_size, to_topo, from_topo))
                if key1 in reallocation_cost_table:
                    param_realloc_time += reallocation_cost_table[key1]["mem_shift_time_ns"] / 1e9 * world_size
                elif key2 in reallocation_cost_table:
                    param_realloc_time += reallocation_cost_table[key2]["mem_shift_time_ns"] / 1e9 * world_size
                else:
                    raise RuntimeError()
        else:
            param_realloc_time = 0
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
                if handle_name == "train_step":
                    _phn = "train"
                elif handle_name == "inference":
                    _phn = "inf"
                else:
                    _phn = "gen"
                _profile_log_path = (
                    f"/lustre/aigc/llm/logs/fw/profile-s{actor_size}p{p}m{m}d{d}-{_phn}/cudakernel/")
                kernel_time = CUDAKernelTime(0, 0, 0, 0)
                for _j in range(p * m * d):
                    with open(os.path.join(_profile_log_path, f"kernel_time{_j}.pkl"), "rb") as f:
                        kernel_time = kernel_time + pickle.load(f)
                handle_type2gpu_time[(role, handle_name)] = kernel_time
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
        # print(handle_type2time.keys())
        for _key in [
            ("actor", "generate"),
            ("actor", "train_step"),
            ("critic", "train_step"),
            ("critic", "inference"),
            ("ref", "inference"),
            ("reward", "inference"),
        ]:
            factor = handle_type2gpu_time[_key].total_secs / np.mean(
                handle_type2time[_key]) / handle_type2device_count[_key]
            handle_type2gpu_time[_key] = handle_type2gpu_time[_key] / factor
        compute_time = comm_time = misc_time = 0
        for x in [
                handle_type2gpu_time[("actor", "generate")],
                handle_type2gpu_time[("actor", "train_step")],
                handle_type2gpu_time[("critic", "train_step")],
                handle_type2gpu_time[("critic", "inference")],
                handle_type2gpu_time[("ref", "inference")],
                handle_type2gpu_time[("reward", "inference")],
        ]:
            x: CUDAKernelTime
            compute_time += (x.compute + x.mem) / 1e6
            comm_time += x.comm / 1e6
            misc_time += x.misc / 1e6
        all_data.append(
            dict(
                actor_size=actor_size,
                critic_size=critic_size,
                overall_time=np.mean(df["time"]),
                n_gpus=n_gpus,
                total_gpu_time=n_gpus * np.mean(df["time"]),
                gpu_compute_time=compute_time,
                gpu_comm_time=comm_time,
                gpu_misc_time=misc_time,
                reallocation_time=param_realloc_time,
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
    width = 0.35
    # factor = df[(df["System"] == "ReaL-Heuristic")]['gpu_compute_time'].to_numpy() + df[(df["System"] == "ReaL-Heuristic")]['gpu_comm_time'].to_numpy()
    factor = df[(df["System"] == "ReaL-Heuristic")]['total_gpu_time'].to_numpy()
    for i, system in enumerate(["ReaL (Ours)", "ReaL-Heuristic"]):
        group = df[(df["System"] == system)]
        # hatch = "/" if i == 0 else "-"
        hatch = None

        settings = [f"{a}B+{c}B" for a, c in zip(group["actor_size"], group["critic_size"])]

        total_time = group["total_gpu_time"].to_numpy()
        print(total_time / factor)

        compute_time = group["gpu_compute_time"].to_numpy()
        comm_time = group["gpu_comm_time"].to_numpy()
        realloc_time = group["reallocation_time"].to_numpy()
        print(f"Average allocation cost: {(realloc_time / total_time).mean() * 100:.2}%")

        idle_time = total_time - compute_time - comm_time - realloc_time

        # factor = compute_time
        # factor = 1
        # normalize
        compute_time = compute_time / factor
        comm_time = comm_time / factor
        realloc_time = realloc_time / factor
        idle_time = idle_time / factor
        # overlap_ratio = np.where(overlap_ratio < 0, np.zeros_like(overlap_ratio), overlap_ratio)

        if i == 0:
            barpos = np.arange(len(settings)) - width / 2 - 0.02
        else:
            barpos = np.arange(len(settings)) + width / 2 + 0.02
        ax.bar(
            barpos,
            compute_time,
            bottom=None,
            label="Compute",
            color=cmap[0],
            width=width,
            linewidth=1.0,
            edgecolor="black",
            hatch=hatch,
        )
        ax.bar(
            barpos,
            comm_time,
            bottom=compute_time,
            label="Comm.",
            color=cmap[1],
            width=width,
            linewidth=1.0,
            edgecolor="black",
            hatch=hatch,
        )
        ax.bar(
            barpos,
            realloc_time,
            bottom=compute_time + comm_time,
            label="Realloc.",
            color=cmap[3],
            width=width,
            linewidth=1.0,
            edgecolor="black",
            hatch=hatch,
        )
        ax.bar(
            barpos,
            idle_time,
            bottom=compute_time + comm_time + realloc_time,
            label="Kernel Launch Overhead & Idle",
            width=width,
            color=cmap[2],
            linewidth=1.0,
            edgecolor="black",
            hatch=hatch,
        )
        for xpos, t1, t2 in zip(barpos, compute_time, comm_time):
            ax.text(xpos, t1 / 2, f"{(float(t1) * 100):.0f}%", ha='center', color='white', weight='bold')
            ax.text(xpos,
                    t1 + t2 / 2,
                    f"{(float(t2) * 100):.0f}%",
                    ha='center',
                    color='white',
                    weight='bold',
                    va='center')
        if i == 0:
            ax.set_xticks(np.arange(len(settings)))
            ax.set_xticklabels(settings, rotation=0, fontsize=xlabel_fontsize)
        # ax.set_yticklabels([int(x) for x in ax.get_yticks()], fontsize=ylabel_fontsize)
        ax.set_ylabel("Normalized GPU Time", fontsize=ylabel_fontsize)
        ax.set_xlabel("Actor Size + Critic Size", fontsize=ylabel_fontsize)
        ax.set_ylim(0, 1.2)
        if i == 0:
            ax.legend(fontsize=legend_fontsize, ncol=4, loc="upper center")
            ax.set_title("GPU Time Breakdown, ReaL (Left) vs Heuristic (Right)", fontsize=title_fontsize)

    # group = df[(df["System"] == "ReaL (Ours)")]
    # ax = all_axes[1, 0]
    # settings = [f"{a}B+{c}B" for a, c in zip(group["actor_size"], group["critic_size"])]
    # gen_time = round_to_nearest_tenth(group["gen_time"].to_numpy())
    # train_time = round_to_nearest_tenth(
    #     group["actor_train_time"].to_numpy() + group["critic_train_time"].to_numpy()
    # )
    # inf_time = round_to_nearest_tenth(
    #     group["critic_inf_time"].to_numpy()
    #     + group["ref_inf_time"].to_numpy()
    #     + group["rew_inf_time"].to_numpy()
    # )
    # overlap_ratio = (gen_time + train_time + inf_time - group["overall_time"].to_numpy()) / group[
    #     "overall_time"
    # ].to_numpy()
    # sns.barplot(x=settings, y=overlap_ratio, ax=ax, color=cmap[3], lw=2, edgecolor="black", hatch=r"\\")
    # ax.set_ylim((0, 0.35))
    # ax.set_xticklabels(settings, rotation=45, fontsize=xlabel_fontsize)
    # ax.set_yticklabels([round_to_nearest_tenth(x) for x in ax.get_yticks()], fontsize=ylabel_fontsize)
    # ax.set_ylabel("Overlap Ratio", fontsize=ylabel_fontsize)
    # ax.set_title("Overlapped Computation of ReaL", fontsize=title_fontsize)

    # ax = all_axes[1, 1]
    # real_pflops = df[(df["System"] == "ReaL (Ours)")]["pflops"].to_numpy()
    # heuristic_pflops = df[(df["System"] == "ReaL-Heuristic")]["pflops"].to_numpy()
    # accleration = (real_pflops - heuristic_pflops) / heuristic_pflops

    # print(accleration.mean())
    # sns.barplot(x=settings, y=accleration, ax=ax, color=cmap[4], edgecolor="black", lw=2, hatch="//")
    # ax.set_xticklabels(settings, rotation=45, fontsize=xlabel_fontsize)
    # ax.set_yticklabels([round_to_nearest_tenth(x) for x in ax.get_yticks()], fontsize=ylabel_fontsize)
    # ax.set_ylabel("Acceleration Ratio", fontsize=ylabel_fontsize)
    # ax.set_title("End-to-End Acceleration over Heuristic", fontsize=title_fontsize)

    # Adjust layout
    plt.tight_layout()

    plt.savefig("assets/figures/vbreakdown.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
