import argparse
import enum
import itertools
import math
import os

from experiments.benchmark.system.sosp2024 import (_get_heuristic_device_partition,
                                                   interested_parallel_strategies)


def sweep_model_size(model_size: int, verbose_only: bool):
    for seqlen, bs in itertools.product([256, 512, 1024], [80, 64, 32, 24, 20, 16, 12, 8]):
        for ps in interested_parallel_strategies:
            if ps["model_size"] == model_size:
                mp_size, dp_size, pp_size = ps["actor"]
                ref_mp_size, ref_dp_size, ref_pp_size = ps["ref"]
                break
        _, device_partition, _ = _get_heuristic_device_partition(model_size)
        exp_name = f"sba-a{model_size}-{mp_size}x{dp_size}x{pp_size}-ref{ref_mp_size}x{ref_dp_size}x{ref_pp_size}-c7r7-{'-'.join(map(str, device_partition))}-s{seqlen}g{bs}"
        trial_name = "1"
        cmd = f"python3 -m apps.main start -e {exp_name} -f {trial_name}"
        if verbose_only:
            print(cmd)
        else:
            os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", "-x", type=int, nargs="+", choices=[7, 13, 34, 70], required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose_only", "-v", action="store_true")
    args = parser.parse_args()
    for x in args.model_size:
        sweep_model_size(x, args.verbose_only)
