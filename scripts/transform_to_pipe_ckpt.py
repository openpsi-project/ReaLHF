import os
import sys

sys.path.append("../")

import multiprocessing as mp

import torch

import api.config as config_package
import api.model
import base.gpu_utils
import base.name_resolve as name_resolve
import base.names as names

mp.set_start_method('spawn', force=True)

FULL_MODEL_DIR = "/lustre/meizy/backup_zy/model_saves/four_layers_starcoder"
PIPE_MODEL_DIR = "/lustre/meizy/backup_zy/model_saves/pipe_4l_starcoder"
NUM_PIPE_STAGES = 4
TEST_EXPR_NAME = "test"
TEST_TRIAL_NAME = "test"
TEST_MODEL_NAME = "default"


def setup_gpu(worker_index, world_size, barrier):
    name_resolve.clear_subtree(names.trainer_ddp_peer(TEST_EXPR_NAME, TEST_TRIAL_NAME, TEST_MODEL_NAME))
    barrier.wait()
    base.gpu_utils.isolate_cuda_device("model_worker", worker_index, world_size, TEST_EXPR_NAME,
                                       TEST_TRIAL_NAME)
    barrier.wait()
    base.gpu_utils.reveal_ddp_identity(TEST_EXPR_NAME, TEST_TRIAL_NAME, TEST_MODEL_NAME, worker_index)
    barrier.wait()
    world_size, ddp_rank, local_gpu_id = base.gpu_utils.setup_ddp(TEST_EXPR_NAME, TEST_TRIAL_NAME,
                                                                  TEST_MODEL_NAME, worker_index)
    device = torch.device('cuda', 0)
    import deepspeed
    deepspeed.init_distributed()
    return device


def make_full_pipe_model(device):
    from impl.model.backend.ds_pipe_engine import PipeDataParallelTopology
    topology = PipeDataParallelTopology(num_pp=NUM_PIPE_STAGES, num_dp=1)
    model_config = config_package.Model(type_="starcoder_flash_mqat_pipe",
                                        args=dict(
                                            model_path=FULL_MODEL_DIR,
                                            topology=topology,
                                            dtype=torch.float16,
                                        ))
    model = api.model.make_model(model_config, name=TEST_MODEL_NAME, device=device)


def main(worker_index, barrier):
    device = setup_gpu(worker_index, world_size=NUM_PIPE_STAGES, barrier=barrier)
    pipe_module = make_full_pipe_model(device).module


if __name__ == "__main__":
    b = mp.Barrier(NUM_PIPE_STAGES)
    # main()
    name_resolve.clear_subtree(names.trial_root(experiment_name=TEST_EXPR_NAME, trial_name=TEST_TRIAL_NAME))
    os.environ["DLLM_MODE"] = "LOCAL"
    ps = [mp.Process(target=main, args=(i, b)) for i in range(NUM_PIPE_STAGES)]

    for p in ps:
        p.start()

    for p in ps:
        p.join()
