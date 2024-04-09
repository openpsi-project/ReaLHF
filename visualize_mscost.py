import json
import math

from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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

        if d["from_mp_size"] * d["from_dp_size"] * d["from_pp_size"] != d["world_size"]:
            continue
        if d["to_mp_size"] * d["to_dp_size"] * d["to_pp_size"] != d["world_size"]:
            continue

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

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    # Group data by world_size
    grouped = df.groupby("world_size")

    # Plot smooth lines for each group
    # for name, group in grouped:
    #     x = group['comm_volume']
    #     y = group['cost']
    #     print(group)

    #     # Interpolate the data for smooth lines
    #     x_new = np.linspace(x.min(), x.max(), 300)
    #     spl = make_interp_spline(x, y)
    #     y_smooth = spl(x_new)

    #     # Plot the smooth line
    #     plt.plot(x_new, y_smooth, label=f"World Size {name}", lw=2)

    # # sns.scatterplot(data=df, x="comm_volume", y="cost", hue="world_size", lw=2, palette='coolwarm')
    for (name, group), ax in zip(grouped, axes):
        subgroups = group.groupby("to_dp_size")
        for subgname, subgroup in subgroups:
            # sns.regplot(data=subgroup, x="comm_volume", y="cost", ax=ax, label=f"dp_ratio={subgname}")
            sns.scatterplot(data=subgroup, x="comm_volume", y="cost", ax=ax)

    plt.tight_layout()
    plt.savefig("vmsc.png")


if __name__ == "__main__":
    main()
