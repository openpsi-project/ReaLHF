from collections import defaultdict
from enum import Enum
from typing import *
import argparse
import dataclasses
import pickle

from matplotlib.patches import Rectangle
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

from reallm.api.core.config import MODEL_FAMILY_TO_PATH, ModelFamily, ModelName
from reallm.api.core.model_api import REAL_MODEL_CONFIG_CONVERTER
from reallm.base.monitor import CUDATimeMarkType, TimeMarkEntry


def main():
    # Choose a colormap
    colormap = matplotlib.colormaps["Spectral"]  # You can choose any colormap from the available ones

    all_time_marks: List[List[TimeMarkEntry]] = []
    for i in tqdm.tqdm(range(32), desc="loading time marks"):
        with open(f"/lustre/aigc/llm/logs/meizy/sosp-a7c34s896g128t128-t/timemark/time_marks{i}.pkl",
                  "rb") as f:
            time_marks: List[TimeMarkEntry] = pickle.load(f)
        all_time_marks.append(time_marks)

    # min_st_time = min(min(x.start_time for x in time_marks) for time_marks in all_time_marks)
    # max_ed_time = max(max(x.end_time for x in time_marks) for time_marks in all_time_marks)
    min_st_time = min(
        min((x.start_time
             if x.type_ == CUDATimeMarkType.forward and x.model_name.role == "actor" else float("inf"))
            for x in time_marks) for time_marks in all_time_marks)
    max_ed_time = max(
        max(x.end_time for x in time_marks if x.type_ == CUDATimeMarkType.comm)
        for time_marks in all_time_marks)
    all_time_marks = [[
        TimeMarkEntry(
            x.name,
            x.model_name,
            x.type_,
            (x.start_time - min_st_time) / 1e9,
            (x.end_time - min_st_time) / 1e9,
        ) for x in time_marks if x.start_time >= min_st_time and x.end_time <= max_ed_time
        and x.name not in ["receive_request", "post_response"]
    ] for time_marks in all_time_marks]

    glue_time = 0.05
    # merge near time marks
    new_all_time_marks = []
    for time_marks in all_time_marks:
        new_time_marks = []
        x = time_marks[0]
        for y in time_marks[1:]:
            if y.start_time - x.end_time < glue_time and x.model_name == y.model_name:
                x.end_time = y.end_time
            else:
                print(x, y)
                new_time_marks.append(x)
                x = y
        new_time_marks.append(x)
        new_all_time_marks.append(new_time_marks)
    all_time_marks = new_all_time_marks

    all_types = ["actor", "critic", "ref", "memory_layout", "reward", "comm"]
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
    all_types_labeled = [False for _ in all_types]

    xlabel_fontsize = 20

    # Prepare data for plotting
    data = [defaultdict(list) for _ in all_time_marks]
    for i, time_marks in enumerate(tqdm.tqdm(all_time_marks, "Preparing data for plotting...")):
        for entry in time_marks:
            if entry.model_name is None:
                if entry.type_ == CUDATimeMarkType.misc:
                    continue
                assert entry.type_ in [
                    CUDATimeMarkType.mem_layout,
                    CUDATimeMarkType.comm,
                ], entry
                type_ = entry.type_.value
            elif entry.model_name.role in ["actor", "critic", "reward", "ref"]:
                type_ = entry.model_name.role
            else:
                raise NotImplementedError(entry)
            data[i][type_].append((entry.start_time, entry.end_time - entry.start_time))

    # Plotting
    fig, ax = plt.subplots(figsize=(20, 7))

    y_ticks = []
    y_ticklabels = []

    for gpu_idx, entries in enumerate(tqdm.tqdm(reversed(data), "plotting...")):
        for entry_type, start_durations in entries.items():
            starts, durations = zip(*start_durations)
            color = colormap[entry_type]
            if not all_types_labeled[all_types.index(entry_type)]:
                label = entry_type
                all_types_labeled[all_types.index(entry_type)] = True
            else:
                label = None
            addtional_kwargs = {}
            # short_durations = [d for d in durations if d < 0.1]
            # long_durations = [d for d in durations if d >= 0.1]
            # short_starts = [s for s, d in zip(starts, durations) if d < 0.1]
            # long_starts = [s for s, d in zip(starts, durations) if d >= 0.1]
            # ax.barh(
            #     gpu_idx, short_durations, left=short_starts, color=color, linewidth=0.0, edgecolor="black"
            # )
            ax.barh(
                gpu_idx,
                durations,
                left=starts,
                color=color,
                label=label,
                linewidth=0.5,
                edgecolor=edge_colormap[entry_type],
            )
        y_ticks.append(gpu_idx)
        y_ticklabels.append(f"{31 - gpu_idx}")

    fig.patch.set_visible(False)
    # ax.axis('off')
    plt.tick_params(left=False)
    sns.despine(top=True, right=True)
    ax.set_yticks(y_ticks)
    handles, labels = ax.get_legend_handles_labels()
    _types = ["actor", "critic", "reward", "ref", "comm", "memory_layout"]
    _labels = [
        "Actor Workload",
        "Critic Workload",
        "Reward Workload",
        "Ref. Workload",
        "Data Communication",
        "Parameter Reallocation",
    ]
    order = [labels.index(t) for t in _types]
    ax.legend([handles[idx] for idx in order], _labels, fontsize=14)
    ax.set_yticklabels([f"GPU {x}" for x in y_ticklabels])
    ax.set_xticklabels(ax.get_xticks(), fontsize=xlabel_fontsize)
    ax.patch.set_visible(False)
    # ax.set_xlabel("Time (s)", fontsize=xlabel_fontsize)
    # ax.set_ylabel("GPU Index", fontsize=xlabel_fontsize)
    plt.setp(ax.spines.values(), visible=False)
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)
    # ax.set_xlim((0, 45))
    # ax.set_title("Job Execution Time Intervals")
    # ax.legend(loc="upper left")

    # Show plot
    plt.savefig(f"assets/figures/v_timelinev2.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
