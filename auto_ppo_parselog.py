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


def _parse_log(rootdir: str, model_size, seqlen, bs, mode):
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
        return -1
    else:
        # assert len(record_times) >= 12, len(record_times)
        record_times = record_times[2:17]
        print(
            f"{model_size}B mode {mode} seqlen={seqlen} bs={bs} mean step time {np.mean(record_times):.2f}s")
        return np.mean(record_times)


def parse_log():

    modes = ["s", "m"]
    # modes = ["s", "m"]
    trial_names = [
        "20240407-0", "20240408-0", "20240408-1", "20240408-2", "20240409-2", "20240409-3", "20240409-4",
        "20240410-1", "20240410-2", "20240411-2"
    ]
    res = {}
    log_path = {}
    for model_size in model_sizes:
        for bs, seqlen in bs_seqlen:
            for mode in modes:
                rs = []
                lps = []
                for trial_name in trial_names:
                    key = (model_size, 7, bs, seqlen, mode)
                    exp_name = f"sosp-a{model_size}s{seqlen}g{bs}t{bs}-{mode}"
                    root_dir = f"/lustre/aigc/llm/logs/meizy/{exp_name}/{trial_name}"
                    if not os.path.exists(root_dir):
                        continue
                    print(f"exp_name: {exp_name}, trial_name: {trial_name}")
                    try:
                        r = _parse_log(root_dir, model_size, seqlen, bs, mode)
                        if r > 0:
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
                    key = (7, model_size, bs, seqlen, mode)
                    exp_name = f"sosp-a7c{model_size}s{seqlen}g{bs}t{bs}-{mode}"
                    root_dir = f"/lustre/aigc/llm/logs/meizy/{exp_name}/{trial_name}"
                    if not os.path.exists(root_dir):
                        continue
                    print(f"exp_name: {exp_name}, trial_name: {trial_name}")
                    try:
                        r = _parse_log(root_dir, f"critic {model_size}", seqlen, bs, mode)
                        if r > 0:
                            rs.append(r)
                            lps.append(root_dir)
                    except FileNotFoundError:
                        pass

                if len(rs) > 0:
                    res[key] = np.max(rs) if mode == "m" else np.min(rs)
                    log_path[key] = lps[np.argmax(rs)] if mode == "m" else lps[np.argmin(rs)]

    print("*" * 30)
    trial_names = ["20240410-1", "20240410-2", "20240411-2"]
    for model_size in model_sizes:
        if model_size == 70 or model_size == 7:
            continue
        for bs, seqlen in bs_seqlen:
            for mode in modes:
                rs = []
                lps = []
                for trial_name in trial_names:
                    key = (model_size, model_size, bs, seqlen, mode)
                    exp_name = f"sosp-a{model_size}c{model_size}s{seqlen}g{bs}t{bs}-{mode}"
                    root_dir = f"/lustre/aigc/llm/logs/meizy/{exp_name}/{trial_name}"
                    # print(exp_name)

                    if not os.path.exists(root_dir):
                        continue
                    print(f"exp_name: {exp_name}, trial_name: {trial_name}")
                    try:
                        r = _parse_log(root_dir, f"both {model_size}", seqlen, bs, mode)
                        if r > 0:
                            rs.append(r)
                            lps.append(root_dir)
                    except FileNotFoundError:
                        pass

                if len(rs) > 0:
                    res[key] = np.max(rs) if mode == "m" else np.min(rs)
                    log_path[key] = lps[np.argmax(rs)] if mode == "m" else lps[np.argmin(rs)]

    print("*" * 30)

    from collections import defaultdict
    print("*" * 30)
    for ams, cms in itertools.product(model_sizes, model_sizes):
        for bs, seqlen in bs_seqlen:
            v = defaultdict(lambda: "/")
            for mode in modes:
                key = (ams, cms, bs, seqlen, mode)
                v[mode] = res[key] if key in res else -1

            if any([r > 0 for r in v.values()]):
                print(
                    f"actor {ams}B critic {cms} B seqlen={seqlen} bs={bs} search {v['s']:.2f} model+pipe {v['m']:.2f}"
                )

    import pickle
    import pprint
    pprint.pprint(log_path)

    import collections

    import pandas
    df_dict = collections.defaultdict(list)
    for k, v in res.items():
        lp = log_path[k]
        ams, cms, bs, seqlen, mode = k
        df_dict["actor_model_size"].append(ams)
        df_dict["critic_model_size"].append(cms)
        df_dict["batch_size"].append(bs)
        df_dict["seqlen"].append(seqlen)
        df_dict["mode"].append(mode)
        df_dict["time"].append(v)
        df_dict["log_path"].append(lp)
    df = pandas.DataFrame(df_dict)
    print(df)
    # pickle.dump(df, open("/home/meizy/logs/sosp/res_df.pkl", "wb"))


if __name__ == "__main__":
    parse_log()
