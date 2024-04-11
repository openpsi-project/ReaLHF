from collections import defaultdict
import os
import pickle

import torch
import torch.distributed as dist

import base.constants

# This module is used to check consistency of parallel methods

ROOT_DIR = "/home/meizy/consistency_tmp"
STEP_ID = 0
MICRO_BATCH_ID = 0
STORED_NAMES = set()
MP_STORED_NAMES = dict()
PARALLEL_MODE = False
REPORT_FILE = "/home/meizy/logs/consistency_report.txt"


def report(report_fp, name, full_tensor, other_tensor, check_dim, atol, rtol):
    if full_tensor.shape != other_tensor.shape:
        print(f"tensor {name} shape mismatch {full_tensor.shape}!={other_tensor.shape} "
              f"stripping other tensor to full_tensor shape")
        split_size = [
            full_tensor.shape[check_dim], other_tensor.shape[check_dim] - full_tensor.shape[check_dim]
        ]
        other_tensor, _ = torch.split(other_tensor, split_size, dim=check_dim)
        # report_fp.write(f"tensor {name} shape mismatch {full_tensor.shape}!={other_tensor.shape}\n")
        # return
    if not torch.allclose(full_tensor, other_tensor, atol=atol, rtol=rtol):
        print(f"tensor {name} **MISMATCH** atol={atol} rtol={rtol}")
        report_fp.write(f"tensor {name} **MISMATCH** atol={atol} rtol={rtol}\n")
    else:
        print(f"tensor {name} **MATCH**")
        report_fp.write(f"tensor {name} **MATCH**\n")
    print(f"max: {(full_tensor - other_tensor).abs().max()} "
          f"mean: {(full_tensor - other_tensor).abs().mean()}")
    report_fp.write(f"tensor {name} **MATCH**:\n")
    report_fp.write(f"max: {(full_tensor - other_tensor).abs().max()}\n")
    if torch.is_floating_point(full_tensor):
        report_fp.write(f"mean: {(full_tensor - other_tensor).abs().mean()}\n")
    report_fp.write(f"full_tensor: {full_tensor}\n")
    report_fp.write(f"other_tensor: {other_tensor}\n")
    report_fp.write(f"full_tensor - other_tensor abs: {(full_tensor - other_tensor).abs()}\n")


def store_model_parallel(name: str, tensor: torch.Tensor, check_dim: int, is_mp: bool):
    MP_STORED_NAMES[name] = check_dim
    if is_mp:
        name = f"{name}:MP:{base.constants.model_parallel_rank()}"
    fn = os.path.join(ROOT_DIR, name)
    tensor = tensor.clone().detach().cpu()
    with open(fn, "wb") as f:
        pickle.dump(tensor, f)


def check_all_model_parallel(mp_world_size, atol=1e-5, rtol=1e-8):
    report_fp = open(REPORT_FILE, "w")
    stored_names = sorted(list(MP_STORED_NAMES))
    for name in stored_names:
        mp_names = [f"{name}:MP:{i}" for i in range(mp_world_size)]
        mp_tensors = []
        for mp_name in mp_names:
            with open(os.path.join(ROOT_DIR, mp_name), "rb") as f:
                mp_tensors.append(pickle.load(f))

        with open(os.path.join(ROOT_DIR, name), "rb") as f:
            full_tensor = pickle.load(f)

        check_dim = MP_STORED_NAMES[name]
        if check_dim is not "full":
            other_tensor = torch.cat(mp_tensors, dim=check_dim)
            try:
                report(report_fp, name, full_tensor, other_tensor, check_dim, atol, rtol)
            except Exception as e:
                print(f"{name} report fail")
                raise e
        else:
            for i in range(mp_world_size):
                try:
                    report(report_fp, mp_names[i], full_tensor, mp_tensors[i], atol, rtol)
                except Exception as e:
                    print(f"{mp_names[i]} report fail")
                    raise e
    report_fp.close()


def store(name: str, tensor: torch.Tensor):
    STORED_NAMES.add(name)
    if PARALLEL_MODE:
        name = f"{name}:{MICRO_BATCH_ID}"
    fn = os.path.join(ROOT_DIR, name)
    tensor = tensor.clone().detach().cpu()
    with open(fn, "wb") as f:
        pickle.dump(tensor, f)
    # try:
    #     print(f"rank {dist.get_rank()} tensor {name} saved")
    # except RuntimeError:
    #     print(f"rank base tensor {name} saved")


def check_all(atol=1e-5, rtol=1e-8):
    report_fp = open(REPORT_FILE, "w")
    stored_names = sorted(list(STORED_NAMES))
    for name in stored_names:
        mb_names = []
        for fn in os.listdir(ROOT_DIR):
            if fn.startswith(name) and not fn == name:
                mb_names.append(fn)
        mb_names = sorted(mb_names)
        # print(f"mb tensor names {mb_names}")
        # if not os.path.exists(os.path.join(ROOT_DIR, name)):
        #     print(f"base tensor {name} not found")
        #     continue
        with open(os.path.join(ROOT_DIR, name), "rb") as f:
            full_tensor = pickle.load(f)
        mb_tensors = []
        for mb_name in mb_names:
            with open(os.path.join(ROOT_DIR, mb_name), "rb") as f:
                mb_tensors.append(pickle.load(f))
        other_tensor = torch.cat(mb_tensors, dim=0)
        report(report_fp, name, full_tensor, other_tensor, atol, rtol)
    report_fp.close()


def set_parallel_mode(b):
    global PARALLEL_MODE
    PARALLEL_MODE = b


def clear():
    for fn in os.listdir(ROOT_DIR):
        # print(f"removing {fn}")
        os.remove(os.path.join(ROOT_DIR, fn))


def set_step_id(v=0):
    global STEP_ID
    STEP_ID = v


def inc_step_id():
    global STEP_ID
    STEP_ID += 1


def step_id():
    return STEP_ID


def set_micro_batch_id(v):
    global MICRO_BATCH_ID
    MICRO_BATCH_ID = v


def micro_batch_id():
    return MICRO_BATCH_ID
