import argparse
import datetime
import enum
import itertools
import math
import os
import subprocess

bs_seqlen = [(256, 384), (128, 896), (512, 128)]
# model_sizes = [70]
# # [70]


def main(round=0):
    trial_name = datetime.datetime.now().strftime("%Y%m%d") + f"-{round}"
    # trial_name = f"20240410-{round}"

    # modes = ["f"]
    # model_sizes = [70]
    # for model_size in model_sizes:
    #     if model_size == 70:
    #         continue
    #     for mode in modes:
    #         for bs, seqlen in bs_seqlen:
    #             exp_name = f"sosp-a{model_size}c{model_size}s{seqlen}g{bs}t{bs}-{mode}"
    #             cmd = f"python3 -m apps.main start -e {exp_name} -f {trial_name}"
    #             os.system(cmd)
    #             print(f"running {cmd}")
    #             try:
    #                 subprocess.run(cmd, shell=True, timeout=3600)
    #             except TimeoutError:
    #                 subprocess.run(f"scancel -u meizy", shell=True)
    #                 pass

    # modes = ["s"]
    # model_sizes = [70, 34]
    # for model_size in model_sizes:
    #     for bs, seqlen in bs_seqlen:
    #         # if model_size == 34 and bs in [128, 256]:
    #         #     continue
    #         for mode in modes:
    #             exp_name = f"sosp-a7c{model_size}s{seqlen}g{bs}t{bs}-{mode}"
    #             cmd = f"python3 -m apps.main start -e {exp_name} -f {trial_name}"
    #             print(f"running {cmd}")
    #             # os.system(cmd)
    #             try:
    #                 subprocess.run(cmd, shell=True, timeout=3600)
    #             except TimeoutError:
    #                 # subprocess.run(f"scancel -u meizy", shell=True)
    #                 pass

    modes = ["s", "m"]
    model_sizes = [7]
    for model_size in model_sizes:
        for bs, seqlen in bs_seqlen:
            for mode in modes:
                # if model_size == 13 and bs == 128:
                #     continue
                exp_name = f"sosp-a{model_size}s{seqlen}g{bs}t{bs}nx2-{mode}"
                cmd = f"python3 -m apps.main start -e {exp_name} -f {trial_name}"
                print(f"running {cmd}")
                # os.system(cmd)
                try:
                    subprocess.run(cmd, shell=True, timeout=3600)
                except TimeoutError:
                    # subprocess.run(f"scancel -u meizy", shell=True)
                    pass


if __name__ == "__main__":
    for r in [0]:
        main(r)
