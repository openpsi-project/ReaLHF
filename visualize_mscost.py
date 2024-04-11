import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
import math


def abs_ratio(x, y):
    if x >= y:
        return x / y
    else:
        return y / x


def main():
    with open("memshift_cost.jsonl", "r") as f:
        data = [json.loads(line) for line in f]

    new_data = []
    xx = []
    for d in data:
        if "world_size" not in d:
            d["world_size"] = 8
        else:
            d["world_size"] = int(d["world_size"])

        if d["world_size"] not in [8, 32]:
            continue

        from_size = d["from_dp_size"] * d["from_mp_size"] * d["from_pp_size"]
        to_size = d["to_dp_size"] * d["to_mp_size"] * d["to_pp_size"]
        d["overlapped"] = from_size == to_size == d["world_size"]
        d["dp_ratio"] = d["to_dp_size"] // d["from_dp_size"]
        d["mp_ratio"] = d["to_mp_size"] // d["from_mp_size"]
        d["pp_ratio"] = abs(math.log(d["to_pp_size"] / d["from_pp_size"], 2))
        # d['max_ratio'] = max(d['dp_ratio'], d['mp_ratio'], d['pp_ratio'])
        # d['partition_closeness'] = abs_ratio(d['from_mp_size'], d['to_mp_size']) * abs_ratio(d['from_pp_size'], d['to_pp_size'])
        if d["comm_volume"] == 0:
            continue
        # d["cost"] = d["mem_shift_time_ns"] / d["fwd_time_ns"]
        d["cost"] = d["mem_shift_time_ns"] / 1e9
        d["comm_volume"] = d["comm_volume"] * 2 * 8 / 1024**3  # Gb
        if d["world_size"] == 8 and d["to_dp_size"] == 8:
            xx.append(d)
        new_data.append(d)

    xx = sorted(xx, key=lambda x: x["cost"])

    data = new_data
    print(len(data))
    df = pd.DataFrame(data)
    # Set style
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    # Group data by world_size
    grouped = df[df["overlapped"] == True].groupby("world_size")
    for i, ((name, group), ax) in enumerate(zip(grouped, axes[0])):
        sns.scatterplot(data=group, x="comm_volume", y="cost", ax=ax, hue="to_dp_size", palette="bright")
        ax.set_xlabel("Total Communication Volume (Gb)", fontsize=12)
        ax.set_ylabel("Reallocation Time (s)", fontsize=12)
        if i > 0:
            ax.get_legend().remove()
        else:
            ax.legend(title=r"$dp_2$", fontsize=10, loc=(0.6, 0.3))
    axes[0][0].set_ylim(-0.0, 0.5)
    axes[0][0].set_title("Single-Node Overlapped Reallocation", fontsize=14)
    axes[0][1].set_ylim(-0.0, 1.0)
    axes[0][1].set_title("Multi-Node Overlapped Reallocation", fontsize=14)

    grouped = df[df["overlapped"] == False].groupby("world_size")
    for i, ((name, group), ax) in enumerate(zip(grouped, axes[1])):
        sns.scatterplot(data=group, x="comm_volume", y="cost", ax=ax, hue="to_dp_size", palette="bright")
        ax.set_xlabel("Total Communication Volume (Gb)", fontsize=12)
        ax.set_ylabel("Reallocation Time (s)", fontsize=12)
        if i > 0:
            ax.get_legend().remove()
        else:
            ax.legend(title=r"$dp_2$", fontsize=10, loc=(0.6, 0.3))
    axes[1][0].set_ylim(-0.0, 0.5)
    axes[1][0].set_title("Single-Node Disjoint Reallocation", fontsize=14)
    axes[1][1].set_ylim(-0.0, 1.0)
    axes[1][1].set_title("Multi-Node Disjoint Reallocation", fontsize=14)
    
    plt.tight_layout()
    plt.savefig("vmsc.png")


if __name__ == "__main__":
    main()
