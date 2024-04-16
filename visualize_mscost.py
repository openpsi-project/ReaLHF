import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
import math
import dataclasses
from typing import *
import json
import matplotlib.colors as colors
import os
import pickle
from est_mscost_v2 import compute_cost
from api.config.config_base import ModelName
import base.topology
from tests.utils import get_llama7b_flash_config


def abs_ratio(x, y):
    if x >= y:
        return x / y
    else:
        return y / x


@dataclasses.dataclass
class Entry:
    world_size: int
    dp_mp_pp1: Tuple[int]
    dp_mp_pp2: Tuple[int]
    comm_volume: float
    time_cost: float
    estimate_cost: float
    max_bcast_cnt: int
    total_local_volume: float
    total_remote_volume: float
    max_comm_volume: float


def build_plot_data():
    if os.path.exists("mscost_plot_data.jsonl"):
        with open("mscost_plot_data.jsonl", "r") as f:
            return [json.loads(line) for line in f]

    with open("memshift_cost.jsonl", "r") as f:
        data = [json.loads(line) for line in f]

    new_data: Dict[Tuple[Tuple, Tuple], Entry] = {}
    for d in data:
        if "world_size" not in d:
            d["world_size"] = 8
        else:
            d["world_size"] = int(d["world_size"])

        if d["world_size"] not in [8, 32]:
            continue

        est_cost, total_local_volume, total_remote_volume, max_bcast_cnt, max_comm_volume = compute_cost(
            world_size=d["world_size"],
            from_model_name=ModelName("Actor", 0),
            to_model_name=ModelName("Actor", 1),
            from_topo=base.topology.PipeModelDataParallelTopology(
                num_pp=d["from_pp_size"], num_mp=d["from_mp_size"], num_dp=d["from_dp_size"]
            ),
            to_topo=base.topology.PipeModelDataParallelTopology(
                num_pp=d["to_pp_size"], num_mp=d["to_mp_size"], num_dp=d["to_dp_size"]
            ),
            model_config=get_llama7b_flash_config(),
            bw=200.0,
            set_interval_cost=0.03,
        )
        key1 = (d["world_size"], d["from_dp_size"], d["from_mp_size"], d["from_pp_size"])
        key2 = (d["world_size"], d["to_dp_size"], d["to_mp_size"], d["to_pp_size"])
        new_data[(key1, key2)] = Entry(
            world_size=d["world_size"],
            dp_mp_pp1=(d["from_dp_size"], d["from_mp_size"], d["from_pp_size"]),
            dp_mp_pp2=(d["to_dp_size"], d["to_mp_size"], d["to_pp_size"]),
            comm_volume=d["comm_volume"] * 2 * 8 / 1024**3,  # Gb
            time_cost=d["mem_shift_time_ns"] / 1e9,
            estimate_cost=est_cost,
            max_bcast_cnt=max_bcast_cnt,
            total_local_volume=total_local_volume,
            total_remote_volume=total_remote_volume,
            max_comm_volume=max_comm_volume,
        )

    plot_data = {}
    for from_dp_mp_pp, to_dp_mp_pp in new_data.keys():
        if (to_dp_mp_pp, from_dp_mp_pp) not in new_data:
            continue
        if (to_dp_mp_pp, from_dp_mp_pp) in plot_data:
            continue

        d1 = new_data[(from_dp_mp_pp, to_dp_mp_pp)]
        d2 = new_data[(to_dp_mp_pp, from_dp_mp_pp)]
        assert d1.world_size == d2.world_size
        d = dict(
            world_size=d1.world_size,
            comm_volume=d1.comm_volume + d2.comm_volume,
            cost=d1.time_cost + d2.time_cost,
            est_cost=d1.estimate_cost + d2.estimate_cost,
            bcast_cnt=d1.max_bcast_cnt + d2.max_bcast_cnt,
            local_comm_ratio=(d1.total_local_volume + d2.total_local_volume)
            / (
                d1.total_local_volume
                + d2.total_local_volume
                + d1.total_remote_volume
                + d2.total_remote_volume
            ),
            total_remote_volume=d1.total_remote_volume + d2.total_remote_volume,
            total_local_volume=d1.total_local_volume + d2.total_local_volume,
            dp_mp_pp1=d1.dp_mp_pp1,
            dp_mp_pp2=d1.dp_mp_pp2,
            overlapped=int(np.prod(d1.dp_mp_pp1) == np.prod(d1.dp_mp_pp2) == d1.world_size),
            max_comm_volume=max(d1.max_comm_volume, d2.max_comm_volume),
        )
        plot_data[(from_dp_mp_pp, to_dp_mp_pp)] = d
    data = list(plot_data.values())

    with open("mscost_plot_data.jsonl", "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
    return data


def main():
    data = build_plot_data()

    df = pd.DataFrame(data)
    # Set style
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 1, figsize=(4, 3))
    cmap = sns.color_palette("viridis", as_cmap=True)

    group = df[df["world_size"] == 32]
    # for j, ((world_size, group), ax) in enumerate(zip(grouped, axes)):
    ax = axes
    x_key = "total_remote_volume"
    points = ax.scatter(
        group[x_key] * 2 * 8 / 1e9,
        group["cost"],
        c=group["max_comm_volume"] * 2 * 8 / 1e9,
        norm=colors.LogNorm(),
        s=30,
        cmap=cmap,
    )
    cbar = fig.colorbar(points)
    cbar.ax.set_yticks([8,16,32,64])
    cbar.ax.set_yticklabels([8,16,32,64])
    cbar.set_label('Max # Broadcast calls', rotation=270, labelpad=20)
    ax.set_xlabel("Inter-Node Communication Volume (Gb)", fontsize=12)
    ax.set_ylabel("Reallocation Time (s)", fontsize=12)
    title = "Reallocation Cost on 4 Nodes"
    ax.set_title(title, fontsize=12)

    # axes[0].set_xlim((-20, 620))
    # axes[0].set_xticks([0, 100, 200, 300, 400, 500, 600])
    # axes[0].set_ylim((-0.05, 2.0))
    # axes[1].set_xlim((-20, 420))
    # axes[1].set_ylim((-0.05, 0.9))

    # axes[0][0].set_ylim(-0.0, 0.5)
    # axes[0][0].set_title("Single-Node Overlapped Reallocation", fontsize=14)
    # # axes[0][1].set_ylim(-0.0, 1.0)
    # axes[0][1].set_title("Multi-Node Overlapped Reallocation", fontsize=14)
    # axes[1][0].set_title("Single-Node Disjoint Reallocation", fontsize=14)
    # # axes[1][1].set_ylim(-0.0, 1.0)
    # axes[1][1].set_title("Multi-Node Disjoint Reallocation", fontsize=14)

    plt.tight_layout()
    plt.savefig("vmsc.pdf")


if __name__ == "__main__":
    main()
