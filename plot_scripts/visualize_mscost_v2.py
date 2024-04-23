from typing import *
import collections
import dataclasses
import itertools
import json
import math
import os
import pickle

from scipy.interpolate import make_interp_spline
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from api.config.config_base import ModelName
from tests.misc.est_mscost_v2 import compute_cost
from tests.utils import get_llama7b_flash_config
import base.topology


def get_n_gpus(main_model_size: int, case):
    default_ngpus = (8 if main_model_size == 7 else
                     16 if main_model_size == 13 else 32 if main_model_size == 34 else 64)
    return default_ngpus if case <= 1 else 2 * default_ngpus


def get_mem_shift_settings():
    with open("/lustre/meizy/res_df.pkl", "rb") as f:
        data = pickle.load(f)

    settings_per_exp = []
    data: pd.DataFrame = data[data["mode"] == "s"]
    for actor_size, critic_size in itertools.product([7, 13, 34, 70], [7, 13, 34, 70]):
        seqlen = 896
        main_model_size = max(actor_size, critic_size)
        # HACK:
        if actor_size == 70 and critic_size == 70:
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
        with open(os.path.join(logpath, "device_mapping.pkl"), "rb") as f:
            device_mapping = pickle.load(f)
        actor_topos = []
        critic_topos = []
        for k, v in device_mapping.items():
            role = k.split("ModelName(role='")[1].split("'")[0]
            handle_name = k.split("@")[1]
            if role not in ["actor", "critic"]:
                continue
            topo = (
                v.train_eval_config.parallel.pipeline_parallel_size,
                v.train_eval_config.parallel.model_parallel_size,
                v.train_eval_config.parallel.data_parallel_size,
            )
            if role == "actor":
                actor_topos.append((topo, v.mapping))
            else:
                critic_topos.append((topo, v.mapping))
        assert len(actor_topos) == 2
        assert len(critic_topos) == 2
        if actor_topos[0][0] != actor_topos[1][0]:
            from_topo, from_mapping = actor_topos[0]
            to_topo, to_mapping = actor_topos[1]
            world_size = np.logical_or(from_mapping, to_mapping).sum()
            actor_repara = (actor_size, world_size, from_topo, to_topo)
        else:
            actor_repara = None
        if critic_topos[0][0] != critic_topos[1][0]:
            from_topo, from_mapping = critic_topos[0]
            to_topo, to_mapping = critic_topos[1]
            world_size = np.logical_or(from_mapping, to_mapping).sum()
            critic_repara = (critic_size, world_size, from_topo, to_topo)
        else:
            critic_repara = None
        settings_per_exp.append((actor_size, actor_repara, critic_size, critic_repara, np.mean(df["time"])))

    return settings_per_exp


def get_mconfig_from_size(model_size):
    import transformers

    from api.config.config_base import MODEL_TYPE_TO_PATH, ModelType
    from api.config.config_flash_model import FLASH_MODEL_CONFIG_CONVERTER

    hf_model_type = "llama" if model_size != 34 else "codellama"
    hf_config = transformers.AutoConfig.from_pretrained(MODEL_TYPE_TO_PATH[ModelType(
        hf_model_type, model_size, False)])
    return FLASH_MODEL_CONFIG_CONVERTER[hf_model_type](hf_config)


def main():
    settings = get_mem_shift_settings()
    dump_file = "/lustre/fw/sosp24/reallocation-cost-exp.json"
    with open(dump_file, "r") as f:
        profile_table = json.load(f)

    plot_data = []
    for actor_size, actor_repara, critic_size, critic_repara, time in settings:
        if actor_repara is None:
            actor_t = actor_local_volume = actor_remote_volume = 0.0
        else:
            a, w, from_topo, to_topo = actor_repara
            key1 = str((a, w, from_topo, to_topo))
            key2 = str((a, w, to_topo, from_topo))
            _, lv1, rv1, *_ = compute_cost(
                world_size=w,
                from_model_name=ModelName("actor", 0),
                to_model_name=ModelName("actor", 1),
                from_topo=base.topology.PipeModelDataParallelTopology(num_pp=from_topo[0],
                                                                      num_mp=from_topo[1],
                                                                      num_dp=from_topo[2]),
                to_topo=base.topology.PipeModelDataParallelTopology(num_pp=to_topo[0],
                                                                    num_mp=to_topo[1],
                                                                    num_dp=to_topo[2]),
                bw=200.0,
                set_interval_cost=0.03,
                model_config=get_mconfig_from_size(a),
            )
            _, lv2, rv2, *_ = compute_cost(
                world_size=w,
                from_model_name=ModelName("actor", 1),
                to_model_name=ModelName("actor", 0),
                from_topo=base.topology.PipeModelDataParallelTopology(num_pp=to_topo[0],
                                                                      num_mp=to_topo[1],
                                                                      num_dp=to_topo[2]),
                to_topo=base.topology.PipeModelDataParallelTopology(num_pp=from_topo[0],
                                                                    num_mp=from_topo[1],
                                                                    num_dp=from_topo[2]),
                bw=200.0,
                set_interval_cost=0.03,
                model_config=get_mconfig_from_size(a),
            )
            actor_local_volume = lv1 + lv2
            actor_remote_volume = rv1 + rv2
            if key1 in profile_table:
                actor_t = profile_table[key1]["mem_shift_time_ns"]
            elif key2 in profile_table:
                actor_t = profile_table[key2]["mem_shift_time_ns"]
            else:
                assert w >= 40
                # HACK:
                actor_t = 1

        if critic_repara is None:
            critic_t = critic_local_volume = critic_remote_volume = 0.0
        else:
            c, w, from_topo, to_topo = critic_repara
            key1 = str((c, w, from_topo, to_topo))
            key2 = str((c, w, to_topo, from_topo))
            _, lv1, rv1, *_ = compute_cost(
                world_size=w,
                from_model_name=ModelName("critic", 0),
                to_model_name=ModelName("critic", 1),
                from_topo=base.topology.PipeModelDataParallelTopology(num_pp=from_topo[0],
                                                                      num_mp=from_topo[1],
                                                                      num_dp=from_topo[2]),
                to_topo=base.topology.PipeModelDataParallelTopology(num_pp=to_topo[0],
                                                                    num_mp=to_topo[1],
                                                                    num_dp=to_topo[2]),
                bw=200.0,
                set_interval_cost=0.03,
                model_config=get_mconfig_from_size(c),
            )
            _, lv2, rv2, *_ = compute_cost(
                world_size=w,
                from_model_name=ModelName("critic", 1),
                to_model_name=ModelName("critic", 0),
                from_topo=base.topology.PipeModelDataParallelTopology(num_pp=to_topo[0],
                                                                      num_mp=to_topo[1],
                                                                      num_dp=to_topo[2]),
                to_topo=base.topology.PipeModelDataParallelTopology(num_pp=from_topo[0],
                                                                    num_mp=from_topo[1],
                                                                    num_dp=from_topo[2]),
                bw=200.0,
                set_interval_cost=0.03,
                model_config=get_mconfig_from_size(c),
            )
            critic_local_volume = lv1 + lv2
            critic_remote_volume = rv1 + rv2
            if key1 in profile_table:
                critic_t = profile_table[key1]["mem_shift_time_ns"]
            elif key2 in profile_table:
                critic_t = profile_table[key2]["mem_shift_time_ns"]
            else:
                # HACK
                assert w >= 40
                critic_t = 1
        plot_data.append(
            dict(
                actor_size=actor_size,
                critic_size=critic_size,
                realloc_time=(actor_t + critic_t) / 1e9,
                realloc_cost=(actor_t + critic_t) / 1e9 / time,
                local_volume=actor_local_volume + critic_local_volume,
                remote_volume=actor_remote_volume + critic_remote_volume,
            ))

    df = pd.DataFrame(plot_data)
    cmap = sns.color_palette(n_colors=8)

    # Set style
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))

    ax = axes
    xlabel = [f"{a}B+{c}B" for a, c in zip(df["actor_size"], df["critic_size"])]
    sns.barplot(
        ax=ax,
        x=xlabel,
        y=df["realloc_cost"],
    )
    # width = 0.35
    # hatch = None
    # barpos = np.arange(len(xlabel)) - width * 0.5
    # ax.bar(
    #     barpos,
    #     df["local_volume"] * 2 * 8 / 1e9,
    #     bottom=None,
    #     label="Local Communication",
    #     color=cmap[0],
    #     width=width,
    #     linewidth=1.0,
    #     edgecolor="black",
    #     hatch=hatch,
    # )
    # barpos = np.arange(len(xlabel)) + width * 0.5
    # ax.bar(
    #     barpos,
    #     df["remote_volume"] * 2 * 8 / 1e9,
    #     bottom=None,
    #     label="Remote Communication",
    #     color=cmap[1],
    #     width=width,
    #     linewidth=1.0,
    #     edgecolor="black",
    #     hatch=hatch,
    # )
    # ax.set_xticks(np.arange(len(df)))
    # ax.set_xticklabels(xlabel, rotation=0, ha="right")
    # ax.legend()

    plt.tight_layout()
    plt.savefig("plot_scripts/figures/v_realloc.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
