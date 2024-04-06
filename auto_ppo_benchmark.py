import argparse
import datetime
import enum
import itertools
import math
import os
import subprocess

bs_seqlen = [(128, 896), (256, 384), (512, 128)]
model_sizes = [7, 13, 34, 70]


def main(round=0):
    for model_size in model_sizes:
        for bs, seqlen in bs_seqlen:
            trial_name = datetime.datetime.now().strftime("%Y%m%d") + f"-{round}"
            exp_name = f"sosp-a{model_size}s{seqlen}g{bs}t{bs}"
            cmd = f"python3 -m apps.main start -e {exp_name} -f {trial_name}"
            # os.system(cmd)
            try:
                p = subprocess.run(cmd, shell=True, timeout=3600)
            except:
                pass


if __name__ == "__main__":
    for r in range(1, 5):
        main(r)
