from statistics import mean
import argparse
import datetime
import itertools
import json
import os
import time

from reallm.search_engine.utils import find_factors
import reallm.search_engine.estimate as estimate


def verify_compute():
    # trial_names = ["20240404", "20240403"]
    # date = "20240328"
    trial_names = ["20240415-0", "20240416-0", "20240417-0", "20240418-0"]
    expr_names = []
    sizes = [7, 13, 34, 70]
    expr_name_to_key = {}
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
                if num_dp * num_mp > 8 or num_pp > 8:
                    continue
                if num_pp <= 8:
                    expr_name = f"profile-s{size}p{num_pp}m{num_mp}d{num_dp}"
                    expr_names.append(expr_name)
                    expr_name_to_key[expr_name] = (size, num_pp, num_mp, num_dp)

    rs = []
    prs = []

    import pandas as pd
    data_modelsize = []
    data_bs = []
    data_seqlen = []
    data_error_rate = []
    # data_pp = []
    # data_mp = []
    # data_dp = []

    # bs_list = [32, 64, 128, 256]
    # seq_len_list = [128, 256, 512, 1024]
    for expr_name, trial_name in itertools.product(expr_names, trial_names):
        fp = f"/lustre/aigc/llm/logs/meizy/{expr_name}/{trial_name}/rpc_profile_stats_0.json"
        pr = None
        try:
            print(f"estimate expr_name: {expr_name} trial_name: {trial_name}")
            with open(fp, "r") as f:
                pr = json.load(f)
                prs.append(pr)
        except Exception as e:
            pass
        args = argparse.Namespace()
        setattr(args, "expr_name", expr_name)
        r = estimate.estimate_expr(args)
        if pr is not None:
            for k, v in pr.items():
                # vv = int(mean(v) * 1e6)
                vv = mean(v)
                # print(f"key {k} pr {vv:.2f}")
                if k in r:
                    rr = r[k]
                    bs = int(k.split("|")[1])
                    print(f"key {k} pr {vv:.2f} r {rr:.2f} error {(vv - rr) / vv:.2f}")
                    data_modelsize.append(expr_name_to_key[expr_name][0])
                    data_bs.append(bs)
                    # data_seqlen.append(expr_name_to_key[expr_name][2])
                    data_error_rate.append(abs(vv - rr) / vv)

    df = pd.DataFrame({
        "model_size": data_modelsize,
        "batch_size": data_bs,
        # "seq_len": data_seqlen,
        "error_rate": data_error_rate
    })
    ms_bs_to_df = {}

    model_size_list = [7, 13, 34]
    batch_size_list = [128, 256, 512]
    for model_size in model_size_list:
        for batch_size in batch_size_list:
            sub_df = df[(df["model_size"] == model_size) & (df["batch_size"] == batch_size)]
            sub_df.sort_values("error_rate", inplace=False)
            sub_df = sub_df[:18]
            print(f"model_size: {model_size}, batch_size: {batch_size}")
            print(sub_df)
            ms_bs_to_df[(model_size, batch_size)] = sub_df

    import pickle

    pickle.dump(ms_bs_to_df, open("logs/ms_bs_to_df.pkl", "wb"))


if __name__ == "__main__":
    verify_compute()