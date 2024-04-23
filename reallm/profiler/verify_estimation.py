from statistics import mean
import argparse
import datetime
import itertools
import json
import os
import time

import profiler.estimate

from reallm.profiler.utils import find_factors


def verify_compute():
    # trial_names = ["20240404", "20240403"]
    # date = "20240328"
    trial_names = ["20240415-0"]
    expr_names = []
    sizes = [7, 13, 34, 70]
    for size in sizes:
        if size == 7:
            n_nodes = 1
        elif size == 13:
            n_nodes = 2
        elif size == 34:
            n_nodes = 4
        elif size == 70:
            n_nodes = 8

        num_gpus = n_nodes * 8
        for num_mp in [1, 2, 4, 8]:
            remain = num_gpus // num_mp
            for num_dp in find_factors(remain):
                num_pp = remain // num_dp
                # if num_dp * num_mp > 8 or num_pp > 8:
                #     continue
                if num_pp <= 8:
                    expr_names.append(f"profile-s{size}p{num_pp}m{num_mp}d{num_dp}")

    rs = []
    prs = []
    # bs_list = [32, 64, 128, 256]
    # seq_len_list = [128, 256, 512, 1024]
    for expr_name, trial_name in itertools.product(expr_names, trial_names):
        fp = f"/lustre/aigc/llm/logs/meizy/{expr_name}/{trial_name}/rpc_profile_stats_0.json"
        pr = None
        print(f"estimate expr_name: {expr_name}")
        try:
            with open(fp, "r") as f:
                pr = json.load(f)
                prs.append(pr)
        except Exception as e:
            pass
        args = argparse.Namespace()
        setattr(args, "expr_name", expr_name)
        r = profiler.estimate.main(args)
        if pr is not None:
            for k, v in pr.items():
                # vv = int(mean(v) * 1e6)
                vv = mean(v)
                # print(f"key {k} pr {vv:.2f}")
                if k in r:
                    rr = r[k] / 1e6
                    print(f"key {k} pr {vv:.2f} r {rr:.2f} error {(rr - vv) / rr:.2f}")


def verify_param_sync():
    pass


if __name__ == "__main__":
    verify_compute()