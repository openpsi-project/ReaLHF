import dataclasses
import datetime
import enum
import itertools
import os
import re
import subprocess
import warnings

import numpy as np

warnings.filterwarnings('ignore')

bs_seqlen = [(128, 896), (256, 384), (512, 128)]
model_sizes = [7, 13, 34, 70]
# np.warnings.filerwarnings("ignore")


@dataclasses.dataclass
class LogResult:
    mean: float
    max: float
    min: float
    std: float
    n: int

    def __lt__(self, other):
        return self.mean < other.mean

    def __le__(self, other):
        return self.mean <= other.mean

    def __gt__(self, other):
        return self.mean > other.mean

    def __ge__(self, other):
        return self.mean >= other.mean

    def __eq__(self, other):
        return self.mean == other.mean


def _parse_log(rootdir: str, model_size, seqlen, bs, mode, n_nodes):
    # throughput is (4*gen_bs*n_actor_gpus / e2e_time)
    record_times = []
    oom = False
    # print(">>>", os.path.join(rootdir, f"model_worker-0"))
    with open(os.path.join(rootdir, f"model_worker-0"), "r") as f:
        begin_line_no = 0
        lines = f.readlines()
        for line_no, line in enumerate(lines):
            if "SBATCH SCRIPT BEGIN" in line:
                begin_line_no = line_no
        for line_no, line in enumerate(lines):
            if line_no < begin_line_no:
                continue
            if "out of memory" in line:
                oom = True
                break

    with open(os.path.join(rootdir, f"master_worker-0"), "r") as f:
        begin_line_no = 0
        lines = f.readlines()
        for line_no, line in enumerate(lines):
            if "SBATCH SCRIPT BEGIN" in line:
                begin_line_no = line_no

        for line_no, line in enumerate(lines):
            if line_no < begin_line_no:
                continue
            if oom:
                break
            if "End to end" not in line:
                continue
            x = float(line.split("#End to end# execution time: *")[1].split("*s.")[0])
            record_times.append(x)
    if oom:
        print(f"{model_size}B mode {mode} seqlen={seqlen} bs={bs} OOM")
        return LogResult(-1, -1, -1, -1, -1)
    else:
        # assert len(record_times) >= 12, len(record_times)
        record_times = record_times[2:17]
        print(
            f"{model_size}B mode {mode} seqlen={seqlen} bs={bs} n_nodes={n_nodes} mean step time {np.mean(record_times):.2f}s"
        )
        if len(record_times) > 10:
            return LogResult(np.mean(record_times), np.max(record_times), np.min(record_times),
                             np.std(record_times), len(record_times))
        else:
            return LogResult(-1, -1, -1, -1, -1)


def parse_log():
    modes = ["s", "m"]
    # modes = ["s", "m"]
    trial_names = [
        "20240407-0", "20240408-0", "20240408-1", "20240408-2", "20240409-2", "20240409-3", "20240409-4",
        "20240410-1", "20240410-2", "20240411-2", "20240412-3", "20240413-1"
    ]
    ms_to_n_nodes = {
        (7, 7, 1): 1,
        (7, 7, 2): 2,
        (7, 13, 1): 2,
        (7, 34, 1): 4,
        (7, 70, 1): 8,
        (13, 7, 1): 2,
        (34, 7, 1): 4,
        (70, 7, 1): 8,
        (13, 13, 1): 4,
        (34, 34, 1): 8,
        (70, 70, 1): 16,
    }

    res = {}
    log_path = {}
    for model_size in model_sizes:
        for bs, seqlen in bs_seqlen:
            for mode in modes:
                rs = []
                lps = []
                for trial_name in trial_names:
                    n_nodes = ms_to_n_nodes[(model_size, 7, 1)]
                    key = (model_size, 7, bs, seqlen, mode, n_nodes)
                    exp_name = f"sosp-a{model_size}s{seqlen}g{bs}t{bs}-{mode}"
                    root_dir = f"/lustre/aigc/llm/logs/meizy/{exp_name}/{trial_name}"
                    if not os.path.exists(root_dir):
                        continue
                    print(f"exp_name: {exp_name}, trial_name: {trial_name}")
                    try:
                        r = _parse_log(root_dir, model_size, seqlen, bs, mode, n_nodes)
                        if r.mean > 0:
                            rs.append(r)
                            lps.append(root_dir)
                    except FileNotFoundError:
                        pass

                if len(rs) > 0:
                    res[key] = np.max(rs) if mode == "m" else np.min(rs)
                    log_path[key] = lps[np.argmax(rs)] if mode == "m" else lps[np.argmin(rs)]

    # import pprint
    # pprint.pprint(res)

    # import pickle
    # pickle.dump(res, open("res.pkl", "wb"))
    print("*" * 30)

    for model_size in model_sizes:
        for bs, seqlen in bs_seqlen:
            if model_size == 7:
                continue
            for mode in modes:
                rs = []
                lps = []
                for trial_name in trial_names:
                    n_nodes = ms_to_n_nodes[(7, model_size, 1)]
                    key = (7, model_size, bs, seqlen, mode, n_nodes)
                    exp_name = f"sosp-a7c{model_size}s{seqlen}g{bs}t{bs}-{mode}"
                    root_dir = f"/lustre/aigc/llm/logs/meizy/{exp_name}/{trial_name}"
                    if not os.path.exists(root_dir):
                        continue
                    print(f"exp_name: {exp_name}, trial_name: {trial_name}")
                    try:
                        r = _parse_log(root_dir, f"critic {model_size}", seqlen, bs, mode, n_nodes)
                        if r.mean > 0:
                            rs.append(r)
                            lps.append(root_dir)
                    except FileNotFoundError:
                        pass

                if len(rs) > 0:
                    res[key] = np.max(rs) if mode == "m" else np.min(rs)
                    log_path[key] = lps[np.argmax(rs)] if mode == "m" else lps[np.argmin(rs)]

    print("*" * 30)
    trial_names = [
        "20240410-1", "20240410-2", "20240411-2", "20240412-2", "20240412-3", "20240412-4", "20240413-0",
        "20240413-1"
    ]
    # trial_names = ["20240412-4"]
    for model_size in model_sizes:
        if model_size == 7:
            continue
        for bs, seqlen in bs_seqlen:
            for mode in modes:
                rs = []
                lps = []
                for trial_name in trial_names:
                    n_nodes = ms_to_n_nodes[(model_size, model_size, 1)]
                    key = (model_size, model_size, bs, seqlen, mode, n_nodes)
                    exp_name = f"sosp-a{model_size}c{model_size}s{seqlen}g{bs}t{bs}-{mode}"
                    root_dir = f"/lustre/aigc/llm/logs/meizy/{exp_name}/{trial_name}"
                    # print(exp_name)

                    if not os.path.exists(root_dir):
                        continue
                    print(f"exp_name: {exp_name}, trial_name: {trial_name}")
                    try:
                        r = _parse_log(root_dir, f"both {model_size}", seqlen, bs, mode, n_nodes)
                        if r.mean > 0:
                            rs.append(r)
                            lps.append(root_dir)
                    except FileNotFoundError:
                        pass

                if len(rs) > 0:
                    res[key] = np.max(rs) if mode == "m" else np.min(rs)
                    log_path[key] = lps[np.argmax(rs)] if mode == "m" else lps[np.argmin(rs)]

    print("*" * 30)

    trial_names = ["20240414-0"]
    # trial_names = ["20240412-4"]
    for bs, seqlen in bs_seqlen:
        for mode in modes:
            rs = []
            lps = []
            for trial_name in trial_names:
                n_nodes = ms_to_n_nodes[(7, 7, 2)]
                key = (7, 7, bs, seqlen, mode, n_nodes)
                exp_name = f"sosp-a7s{seqlen}g{bs}t{bs}nx2-{mode}"
                root_dir = f"/lustre/aigc/llm/logs/meizy/{exp_name}/{trial_name}"
                # print(exp_name)

                if not os.path.exists(root_dir):
                    continue
                print(f"exp_name: {exp_name}, trial_name: {trial_name}")
                try:
                    r = _parse_log(root_dir, f"both 7", seqlen, bs, mode, n_nodes)
                    if r.mean > 0:
                        rs.append(r)
                        lps.append(root_dir)
                except FileNotFoundError:
                    pass

            if len(rs) > 0:
                res[key] = np.max(rs) if mode == "m" else np.min(rs)
                log_path[key] = lps[np.argmax(rs)] if mode == "m" else lps[np.argmin(rs)]

    print("*" * 30)

    # from collections import defaultdict
    # print("*" * 30)
    # for ams, cms in itertools.product(model_sizes, model_sizes):
    #     for bs, seqlen in bs_seqlen:
    #         v = defaultdict(lambda: "/")
    #         for mode in modes:
    #             key = (ams, cms, bs, seqlen, mode, n_nodes)
    #             v[mode] = res[key] if key in res else LogResult(-1, -1, -1, -1, -1)
    #         if any([r.mean > 0 for r in v.values()]):
    #             print(
    #                 f"actor {ams}B critic {cms} B seqlen={seqlen} bs={bs} search {v['s'].mean:.2f} model+pipe {v['m'].mean:.2f}"
    #             )

    import pickle
    import pprint
    pprint.pprint(log_path)

    import collections

    import pandas
    df_dict = collections.defaultdict(list)
    for k, v in res.items():
        v: LogResult
        lp = log_path[k]
        ams, cms, bs, seqlen, mode, n_nodes = k
        df_dict["actor_model_size"].append(ams)
        df_dict["critic_model_size"].append(cms)
        df_dict["batch_size"].append(bs)
        df_dict["seqlen"].append(seqlen)
        df_dict["mode"].append(mode)
        df_dict["time"].append(v.mean)
        df_dict["std"].append(v.std)
        df_dict["max"].append(v.max)
        df_dict["min"].append(v.min)
        df_dict["n"].append(v.n)
        df_dict["n_nodes"].append(n_nodes)
        df_dict["log_path"].append(lp)
    df = pandas.DataFrame(df_dict)
    print(df)
    pickle.dump(df, open("/home/meizy/logs/sosp/res_df.pkl", "wb"))


if __name__ == "__main__":
    parse_log()
