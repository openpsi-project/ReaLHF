import argparse
import datetime
import time

from reallm.profiler.multi_host_main import main
from reallm.profiler.utils import find_factors

if __name__ == "__main__":
    date = datetime.datetime.now().strftime("%Y%m%d") + "-0"
    expr_names = []
    # sizes = [70, 34, 13, 7]
    sizes = [70, 34, 13]
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
                if num_pp <= 8:
                    expr_names.append(f"profile-s{size}p{num_pp}m{num_mp}d{num_dp}")

    for expr_name in expr_names:
        st = time.monotonic()
        print(f"running expr_name: {expr_name} at {date}")
        args = argparse.Namespace()
        setattr(args, "expr_name", expr_name)
        setattr(args, "trial_name", date)
        setattr(args, "trace", False)
        error = main(args, if_raise=False)
        print(f"expr_name: {expr_name} at {date} done, error: {error}, "
              f"timecost {time.monotonic() - st:.2f}")
