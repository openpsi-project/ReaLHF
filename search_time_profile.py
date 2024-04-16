import os
import pickle

import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

result_dir = "profile_result/search/"
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
nodes = [1, 2, 4, 8]
bs_list = [128, 256, 512]
seqlen_list = [896, 384, 128]
max_time = 500

fig, axes = plt.subplots(1, 3, figsize=(8, 3))

for bs, seqlen, ax in zip(bs_list, seqlen_list, axes):
    for i, ams in enumerate([7, 13, 34, 70]):
        fn = f"ams-{ams}_cms-7_bs-{bs}_seqlen-{seqlen}_n-{nodes[i]}.pkl"
        try:
            d = pickle.load(open(os.path.join(result_dir, fn), "rb"))
            print(f"Loaded {fn}")
        except FileNotFoundError:
            print(f"File {fn} not found")
            continue
        x = []
        y = []
        for r in d:
            end_time = r["end_time"] / 1e6
            used_time = r["used_time"] / 1e3
            if used_time > max_time:
                continue
            # x.append(used_time)
            # y.append(end_time)
            if len(x) > 0:
                x.append(used_time)
                y.append(y[-1])
            x.append(used_time)
            y.append(end_time / (d[0]["end_time"] / 1e6))
            # print(r["end_time"], r["used_time"])

        x.append(max_time)
        y.append(y[-1])

        # draw a line plot
        ax.plot(x, y, color=colors[i], label=f"{nodes[i]} x 8 GPUs")
        ax.set_xlabel("Search Time (s)")
        ax.set_title(f"Batch Size = {bs}")
# plt.xlabel("Search Time (s)")
plt.ylabel("Best Searched Cost/Initial Cost")

plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"logs/stprofile-all.pdf")
