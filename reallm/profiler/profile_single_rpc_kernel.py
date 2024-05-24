from typing import *
import argparse
import collections
import datetime
import itertools
import os
import pickle
import time

import tqdm

from reallm.api.core.dfg import ModelInterfaceType
from reallm.profiler.multi_host_main import main
from reallm.profiler.utils import find_factors
import reallm.base.constants as constants


def get_n_gpus(main_model_size: int, case):
    default_ngpus = (8 if main_model_size == 7 else
                     16 if main_model_size == 13 else 32 if main_model_size == 34 else 64)
    return default_ngpus if case <= 1 else 2 * default_ngpus


def build_exp_names(mode: str) -> List:
    with open("/lustre/meizy/res_df.pkl", "rb") as f:
        data = pickle.load(f)

    # model size, interface type, pp, mp, dp
    rpcs_to_profile: Set[Tuple[int, str, int, int, int]] = set()
    for actor_size, critic_size in itertools.product([7, 13, 34, 70], [7, 13, 34, 70]):
        seqlen = 896
        main_model_size = max(actor_size, critic_size)
        # HACK:
        if actor_size == 70 and critic_size == 70:
            continue
        if actor_size != critic_size and actor_size > 7 and critic_size > 7:
            continue
        if critic_size == 7:
            case = 0
        elif actor_size == 7:
            case = 1
        else:
            case = 2
        if case == 0:
            ref_size = actor_size
            rew_size = 7
        elif case == 1:
            ref_size = 7
            rew_size = critic_size
        else:
            assert actor_size == critic_size > 7
            ref_size = rew_size = actor_size
        n_gpus = get_n_gpus(main_model_size, case)
        bs = 2**17 // (seqlen + 128)
        df = data[(data["actor_model_size"] == actor_size)
                  & (data["critic_model_size"] == critic_size)
                  & (data["seqlen"] == seqlen)
                  & (data["n_nodes"] == n_gpus // 8)
                  & (data["mode"] == mode)]
        assert len(df) == 1, df
        logpath = df["log_path"].tolist()[0]
        if mode == "s":
            with open(os.path.join(logpath, "device_mapping.pkl"), "rb") as f:
                device_mapping = pickle.load(f)
            for k, v in device_mapping.items():
                role = k.split("ModelName(role='")[1].split("'")[0]
                if role == "actor":
                    model_size = actor_size
                elif role == "critic":
                    model_size = critic_size
                elif role == "ref":
                    model_size = ref_size
                elif role == "reward":
                    model_size = rew_size
                else:
                    raise NotImplementedError(role)
                handle_name = k.split("@")[1]
                rpcs_to_profile.add((
                    model_size,
                    handle_name,
                    v.train_eval_config.parallel.pipeline_parallel_size,
                    v.train_eval_config.parallel.model_parallel_size,
                    v.train_eval_config.parallel.data_parallel_size,
                ))
        else:
            mp_size = 8
            pp_size = n_gpus // mp_size
            rpcs_to_profile.add((actor_size, "generate", pp_size, mp_size, 1))
            rpcs_to_profile.add((actor_size, "train_step", pp_size, mp_size, 1))
            rpcs_to_profile.add((critic_size, "inference", pp_size, mp_size, 1))
            rpcs_to_profile.add((critic_size, "train_step", pp_size, mp_size, 1))
            rpcs_to_profile.add((ref_size, "inference", pp_size, mp_size, 1))
            rpcs_to_profile.add((rew_size, "inference", pp_size, mp_size, 1))

    expr_names = []
    n_gpus = []
    for x in sorted(list(rpcs_to_profile), key=lambda x: x[2] * x[3] * x[4]):
        if x[1] == "generate":
            rpc_t = "gen"
        elif x[1] == "train_step":
            rpc_t = "train"
        elif x[1] == "inference":
            rpc_t = "inf"
        else:
            raise NotImplementedError(x)
        expr_names.append(f"profile-s{x[0]}p{x[2]}m{x[3]}d{x[4]}-{rpc_t}")
        n_gpus.append(x[2] * x[3] * x[4])
    return expr_names, n_gpus


if __name__ == "__main__":
    trial_name = "cudakernel"
    expr_names = []
    # sizes = [70, 34, 13, 7]
    expr_names, n_gpus = build_exp_names("m")
    a, b = build_exp_names("s")
    expr_names.extend(a)
    n_gpus.extend(b)

    expr_names = list((expr_names))
    n_gpus = list((n_gpus))

    for expr_name, gpu in tqdm.tqdm(zip(expr_names, n_gpus), total=len(expr_names)):
        if all([
                os.path.exists(os.path.join(constants.LOG_ROOT, expr_name, trial_name, f"kernel_time{i}.pkl"))
                for i in range(gpu)
        ]):
            continue
        st = time.monotonic()
        print(f"running expr_name: {expr_name} trial_name")
        args = argparse.Namespace()
        setattr(args, "expr_name", expr_name)
        setattr(args, "trial_name", trial_name)
        setattr(args, "trace", False)
        error = main(args, if_raise=False)
        print(f"expr_name: {expr_name} cuda kernel profiling done, error: {error}, "
              f"timecost {time.monotonic() - st:.2f}")
