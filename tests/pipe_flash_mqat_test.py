# model level generation test
# dp_size = pp_size = 2
import os
import sys

sys.path.append("../")
import argparse
import multiprocessing as mp
import random
import time

import deepspeed
import torch

mp.set_start_method('spawn', force=True)

import logging

from deepspeed.accelerator import get_accelerator

from base.namedarray import NamedArray
from impl.model.backend.pipe_engine import PipeDataParallelTopology
from tests.pipe_utils import *
import api.config as config_package
import api.model
import base.gpu_utils
import base.name_resolve as name_resolve
import base.names as names
import impl.model.nn.pipe_nn

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
DATE_FORMAT = "%Y%m%d-%H:%M:%S"
logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT, level="INFO")

logger = logging.getLogger("test")


def setup_gpu(worker_index):
    name_resolve.clear_subtree(names.trainer_ddp_peer(EXPR_NAME, TRIAL_NAME, MODEL_NAME))
    time.sleep(1)
    base.gpu_utils.isolate_cuda_device(MODEL_TYPE, worker_index, PIPE_DEGREE * DATA_DEGREE, EXPR_NAME,
                                       TRIAL_NAME)
    time.sleep(1)
    base.gpu_utils.reveal_ddp_identity(EXPR_NAME, TRIAL_NAME, MODEL_NAME, worker_index)
    time.sleep(1)
    world_size, ddp_rank, local_gpu_id = base.gpu_utils.setup_ddp(EXPR_NAME, TRIAL_NAME, MODEL_NAME,
                                                                  worker_index)
    device = torch.device('cuda', 0)
    deepspeed.init_distributed()
    return device


def print_rank(s, rank):
    time.sleep(0.5 * rank)
    print(s)


def main(worker_index=0):
    # os.environ["NCCL_IB_DISABLE"] = "1"
    # os.environ["NCCL_P2P_DISABLE"] = "1"
    device = setup_gpu(worker_index)
    rank = torch.distributed.get_rank()
    cuda_visible = os.environ["CUDA_VISIBLE_DEVICES"]
    logger.info(
        f"WORKER INDEX: {worker_index}; TORCH DIST RANK: {torch.distributed.get_rank()}; CUDA VISIBLE: {cuda_visible}"
    )
    dp_world_size = 1
    bs_per_device = 4

    a = torch.tensor([rank], dtype=torch.float16, device=device)
    print(f"{rank}: {a}")
    torch.distributed.all_reduce(a)
    print(f"{rank}: {a}")
    # model = get_pipe_model(model_path, device)
    # backend = get_pipe_backend()
    # ft_spec = get_finetune_spec()
    # interface = get_simple_interface()

    # backend.initialize(model, ft_spec)
    # logger.info(("gpu mem info: MemAllocated=*{}*GB, MaxMemAllocated=${}$GB".format(
    #     round(get_accelerator().memory_allocated() / 1024**3, 2),
    #     round(get_accelerator().max_memory_allocated() / 1024**3, 2),
    # )))


if __name__ == "__main__":
    # main()
    name_resolve.clear_subtree(names.trial_root(experiment_name=EXPR_NAME, trial_name=TRIAL_NAME))
    os.environ["DLLM_MODE"] = "LOCAL"
    ps = [mp.Process(target=main, args=(i,)) for i in range(3)]

    for p in ps:
        p.start()

    for p in ps:
        p.join()
