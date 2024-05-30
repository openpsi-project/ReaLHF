from collections import defaultdict
from enum import Enum
from typing import *
import argparse
import dataclasses
import pickle

from matplotlib.patches import Rectangle
import matplotlib
import matplotlib.pyplot as plt
import tqdm

from reallm.api.core.model_api import (MODEL_FAMILY_TO_PATH, ModelFamily, ModelName,
                                       REAL_MODEL_CONFIG_CONVERTER)
from reallm.base.monitor import CUDATimeMarkType, TimeMarkEntry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig_type", "-x", type=str, required=True)
    args = parser.parse_args()

    # Choose a colormap
    colormap = matplotlib.colormaps["Spectral"]  # You can choose any colormap from the available ones

    fig_type = args.fig_type
    all_time_marks: List[List[TimeMarkEntry]] = []
    for i in tqdm.tqdm(range(8), desc="loading time marks"):
        with open(
                f"/lustre/aigc/llm/logs/fw/sosp-a0s896g64t64-{fig_type}/test20240405/time_marks{i}.pkl",
                "rb",
        ) as f:
            time_marks: List[TimeMarkEntry] = pickle.load(f)
        all_time_marks.append(time_marks)

    min_st_time = min(
        min(x.start_time for x in time_marks if x.type_ == CUDATimeMarkType.forward)
        for time_marks in all_time_marks)
    max_ed_time = max(
        max(x.end_time for x in time_marks
            if x.type_ == CUDATimeMarkType.mem_layout or x.type_ == CUDATimeMarkType.optim_step)
        for time_marks in all_time_marks)
    all_time_marks = [[
        TimeMarkEntry(
            x.name,
            x.model_name,
            x.type_,
            (x.start_time - min_st_time) / 1e9,
            (x.end_time - min_st_time) / 1e9,
        ) for x in time_marks if x.start_time >= min_st_time and x.end_time <= max_ed_time
    ] for time_marks in all_time_marks]

    all_types = [
        "actor",
        "misc_compute",
        "critic",
        "misc",
        "ref",
        "memory_layout",
        "reward",
        "comm",
        "optim_step",
    ]
    all_types_labeled = [False for _ in all_types]

    # Prepare data for plotting
    data = [defaultdict(list) for _ in all_time_marks]
    for i, time_marks in enumerate(all_time_marks):
        for entry in time_marks:
            if entry.model_name is None:
                assert entry.type_ in [
                    CUDATimeMarkType.mem_layout,
                    CUDATimeMarkType.comm,
                    CUDATimeMarkType.misc,
                ], entry
                type_ = entry.type_.value
            elif entry.model_name.role in ["actor", "critic", "reward", "ref"]:
                if entry.type_ in [CUDATimeMarkType.misc]:
                    type_ = "misc_compute"
                elif entry.type_ in [CUDATimeMarkType.optim_step]:
                    type_ = "optim_step"
                else:
                    type_ = entry.model_name.role
            else:
                raise NotImplementedError(entry)
            data[i][type_].append((entry.start_time, entry.end_time - entry.start_time))

    # Plotting
    fig, ax = plt.subplots(figsize=(20, 5))

    y_ticks = []
    y_ticklabels = []

    for gpu_idx, entries in enumerate(data):
        for entry_type, start_durations in entries.items():
            starts, durations = zip(*start_durations)
            color = colormap(all_types.index(entry_type) / len(all_types))
            if not all_types_labeled[all_types.index(entry_type)]:
                label = entry_type
                all_types_labeled[all_types.index(entry_type)] = True
            else:
                label = None
            addtional_kwargs = {}
            short_durations = [d for d in durations if d < 0.1]
            long_durations = [d for d in durations if d >= 0.1]
            short_starts = [s for s, d in zip(starts, durations) if d < 0.1]
            long_starts = [s for s, d in zip(starts, durations) if d >= 0.1]
            ax.barh(
                gpu_idx,
                short_durations,
                left=short_starts,
                color=color,
                linewidth=0.0,
                edgecolor="black",
            )
            ax.barh(
                gpu_idx,
                long_durations,
                left=long_starts,
                color=color,
                label=label,
                linewidth=0.5,
                edgecolor="black",
            )
        y_ticks.append(gpu_idx)
        y_ticklabels.append(f"GPU {gpu_idx}")

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels)
    ax.set_xlabel("Time")
    ax.set_ylabel("Job Type")
    ax.set_xlim((0, 45))
    ax.set_title("Job Execution Time Intervals")
    ax.legend(loc="upper left")

    # Show plot
    plt.savefig(f"assets/figures/vtl-{fig_type}.png")


if __name__ == "__main__":
    main()
