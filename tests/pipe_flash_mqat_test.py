# model level generation test
# dp_size = pp_size = 2
import os
import sys

sys.path.append("../")
import argparse
import multiprocessing as mp
import random
import time

# import deepspeed
import torch

mp.set_start_method('spawn', force=True)

import logging

from base.namedarray import NamedArray
import api.config as config_package
import base.gpu_utils
import base.name_resolve as name_resolve
import base.names as names

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
DATE_FORMAT = "%Y%m%d-%H:%M:%S"
logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT, level="INFO")

logger = logging.getLogger("test")

# model_path = "/lustre/meizy/backup_zy/model_saves/four_layers_starcoder"
# model_path = "/lustre/meizy/base_models/pipe_starcoder"
model_path = "/lustre/meizy/base_models/pipe_starcoder_6pp_3s"
# model_path = "/lustre/meizy/base_models/pipe_4l_starcoder"
# model_path = "/lustre/meizy/backup_zy/starcoder"
EXPR_NAME = "test"
TRIAL_NAME = "test"
MODEL_NAME = "pipedatamodel"
MODEL_TYPE = "model_worker"
PIPE_DEGREE = 6
DATA_DEGREE = 1


def setup_gpu(worker_index, b):
    name_resolve.clear_subtree(names.trainer_ddp_peer(EXPR_NAME, TRIAL_NAME, MODEL_NAME))
    b.wait()
    base.gpu_utils.isolate_cuda_device(MODEL_TYPE, worker_index, PIPE_DEGREE * DATA_DEGREE, EXPR_NAME,
                                       TRIAL_NAME)
    b.wait()
    base.gpu_utils.reveal_ddp_identity(EXPR_NAME, TRIAL_NAME, MODEL_NAME, worker_index)
    b.wait()
    world_size, ddp_rank, local_gpu_id = base.gpu_utils.setup_ddp(EXPR_NAME, TRIAL_NAME, MODEL_NAME,
                                                                  worker_index)
    device = torch.device('cuda', 0)
    import deepspeed
    deepspeed.init_distributed()
    return device


def print_rank(s, rank):
    time.sleep(0.5 * rank)
    print(s)


def main(worker_index, b):
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = setup_gpu(worker_index, b)
    from impl.model.utils.data import build_packed_inputs, unpack_tensor
    from tests.pipe_utils import (get_example_batch, get_finetune_spec, get_pipe_backend, get_pipe_model,
                                  get_simple_interface)

    # os.environ["NCCL_IB_DISABLE"] = "1"
    # os.environ["NCCL_P2P_DISABLE"] = "1"
    rank = torch.distributed.get_rank()
    cuda_visible = os.environ["CUDA_VISIBLE_DEVICES"]
    logger.info(
        f"WORKER INDEX: {worker_index}; TORCH DIST RANK: {torch.distributed.get_rank()}; CUDA VISIBLE: {cuda_visible}"
    )
    topo, model = get_pipe_model(model_path, device)

    dp_worldsize = 1
    dp_rank = 0
    # dp_rank = torch.distributed.get_rank() % 2

    backend = get_pipe_backend()
    ft_spec = get_finetune_spec()
    interface = get_simple_interface()

    backend.initialize(model, ft_spec)
    from deepspeed.accelerator import get_accelerator
    logger.info(("gpu mem info: MemAllocated=*{}*GB, MaxMemAllocated=${}$GB".format(
        round(get_accelerator().memory_allocated() / 1024**3, 2),
        round(get_accelerator().max_memory_allocated() / 1024**3, 2),
    )))

    print(f"rank {worker_index}: model initialized")
    input_ids, attention_mask = get_example_batch(model.tokenizer, device, 8, dp_rank, dp_worldsize)
    packed_input_ids, cu_seqlens, max_seqlen = build_packed_inputs(input_ids, attention_mask)
    prompt_mask = torch.zeros_like(packed_input_ids)
    data = NamedArray(
        packed_input_ids=packed_input_ids,
        cu_seqlens=cu_seqlens,
        prompt_mask=prompt_mask,
    )
    # print(f"rank {worker_index}: begin train_step")
    # outputs = interface.train_step(model, data)
    # print(f"rank {worker_index}: end train_step")
    # print(f"rank {worker_index}: train_step outputs: {outputs}")

    print(f"rank {worker_index}: begin inference")
    outputs = interface.inference(model, data)
    print(f"rank {worker_index}: end inference")
    print(f"rank {worker_index}: inference outputs: {outputs}")


if __name__ == "__main__":
    b = mp.Barrier(PIPE_DEGREE * DATA_DEGREE)
    # main()
    name_resolve.clear_subtree(names.trial_root(experiment_name=EXPR_NAME, trial_name=TRIAL_NAME))
    os.environ["DLLM_MODE"] = "LOCAL"
    ps = [mp.Process(target=main, args=(i, b)) for i in range(PIPE_DEGREE * DATA_DEGREE)]

    for p in ps:
        p.start()

    for p in ps:
        p.join()
