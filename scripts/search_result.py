from collections import defaultdict
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

data_path = "/lustre/meizy/res_df.pkl"

df_dict = defaultdict(list)

# markers = []

with open(data_path, "rb") as f:
    data = pd.read_pickle(f)

    for i, d in data.iterrows():
        time = d["time"]
        bs = d["batch_size"]
        log_path = d["log_path"]
        fn = os.path.join(log_path, "raw_search_result")
        if "nx2" in log_path:
            continue
        try:
            with open(fn, "r") as f:
                result = eval(f.read())
                estimated = result[-1]["end_time"]
                estimated = estimated / (5 * 1e6)
                df_dict["time"].append(time)
                df_dict["estimated"].append(estimated)
                df_dict["n_nodes"].append(d["n_nodes"])
                print(
                    f"Logpath {fn}: time: {time:.2f}, estimated: {estimated:.2f}, err: {(time - estimated)/time:.4f}"
                )
        except (FileNotFoundError, KeyError) as e:
            print(f"Logpath {fn}: time: {time}, estimated: None")

df = pd.DataFrame(df_dict)
print(df)

cmap = sns.color_palette("viridis", n_colors=len(df['n_nodes'].unique()))
fig, axes = plt.subplots(1, 1, figsize=(4, 3))
legend_labels = {
    1: "8 GPUs",
    2: "16 GPUs",
    4: "32 GPUs",
    8: "64 GPUs",
    16: "128 GPUs",
}
colors = {n: cmap[i] for i, n in enumerate(sorted(df['n_nodes'].unique()))}
print(colors)
# for n_nodes in df['n_nodes'].unique():
#     subset = df[df['n_nodes'] == n_nodes]
# print(type(n_nodes))

# axes.set_xscale("log")
# axes.set_yscale("log")

for n_nodes in reversed(df['n_nodes'].unique()):
    subset = df[df['n_nodes'] == n_nodes]
    axes.scatter(subset["estimated"], subset["time"], color=colors[n_nodes], label=legend_labels[n_nodes])
    print(subset["time"])

axes.set_xlabel("Estimated Time (s)")
axes.set_ylabel("Real Time (s)")
xticks = range(20, 121, 20)
yticks = range(20, 121, 20)

axes.set_xticks(xticks)
axes.set_yticks(yticks)
axes.set_xlim(1, 140)
axes.set_ylim(1, 140)
axes.set_xticks([0, 10, 20, 50, 100])
axes.set_yticks([0, 10, 20, 50, 100])

axes.plot(range(120), range(120), color='black', linestyle='--', alpha=0.5)
axes.plot(range(120), [1.3 * i for i in range(120)], color='black', linestyle='--', alpha=0.5)

plt.legend(fontsize='small', borderpad=0.25)
plt.tight_layout()
plt.savefig("logs/estimate_error.pdf")

# print(data)
# fn = "/lustre/aigc/llm/logs/meizy/sosp-a13s896g128t128-s/20240407-1/raw_search_result"

# with open(fn, "r") as f:
#     data = eval(f.read())

# print(data)
# estimated = data[-1]["endtime"]
