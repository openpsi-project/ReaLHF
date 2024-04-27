import functools
import itertools
import os
import time
import unittest

import torch
import torch.distributed
import torch.multiprocessing as mp

from reallm.api.core import model_api, system_api
from tests.utils import *
import reallm.base.constants as constants

MODEL_TYPE = "llama"
MODEL_SIZE = 7
BASE_MODEL_PATH = f"/lustre/public/pretrained_model_weights/Llama-2-{MODEL_SIZE}b-hf"


def make_model(device, from_type: str, load_dir=None):
    import reallm.api.core.model_api as model_api
    import reallm.impl.model.nn.real_llm_api

    model_config = system_api.Model(
        "real_model",
        args=dict(
            model_path=load_dir if load_dir is not None else BASE_MODEL_PATH,
            from_type=from_type,
            dtype="fp16",
            hf_model_type=MODEL_TYPE,
            tokenizer_path=BASE_MODEL_PATH,
            sequence_parallel=False,
            gradient_accumulation_fusion=False,
        ),
    )
    model = model_api.make_model(model_config, name=MODEL_NAME, device=device)
    return model


def build_hf_model(is_critic, rank, world_size, mp_size, pp_size, save_dir=None):
    device = setup_gpu(rank, world_size)
    init_global_constants(1, mp_size, pp_size)
    torch_dist_rank = torch.distributed.get_rank()
    cuda_visible = os.environ["CUDA_VISIBLE_DEVICES"]
    # print(
    #     f"PROCESS RANK: {rank}; \n" f"TORCH DIST RANK: {torch_dist_rank}; \n" f"CUDA VISIBLE: {cuda_visible}"
    # )
    os.environ["REAL_LOG_LOAD_TIME"] = "1"
    torch.cuda.synchronize()
    tik = time.perf_counter()
    model = make_model(device, from_type="hf_as_critic" if is_critic else "hf_as_actor")
    torch.cuda.synchronize()
    tok = time.perf_counter()

    print(f"rank={rank} Model build time: {tok - tik:.3f} seconds")
    if save_dir is not None:
        model.module.save(save_dir)

    return device, model


def load_model(save_critic, is_critic, rank, world_size, mp_size, pp_size, load_dir):
    device = setup_gpu(rank, world_size)
    init_global_constants(1, mp_size, pp_size)
    os.environ["REAL_LOG_LOAD_TIME"] = "1"
    torch.cuda.synchronize()
    tik = time.perf_counter()
    if is_critic and not save_critic:
        from_type = "actor_as_critic"
    else:
        from_type = "self"
    model = make_model(device, from_type=from_type, load_dir=load_dir)
    torch.cuda.synchronize()
    tok = time.perf_counter()

    print(f"rank={rank} Model build time: {tok - tik:.3f} seconds")

    return device, model


class ParallelReaLModelSaveLoadTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        clear_name_resolve()
        cls.baseline_model = None

    def init_tokenizer(self):

        self.tokenizer = model_api.load_hf_tokenizer(BASE_MODEL_PATH)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

    def _testLoadFromHF(self, is_critic: bool, mp_size: int, pp_size: int):
        clear_name_resolve()
        world_size = mp_size * pp_size
        setup_barrier(world_size)
        procs = [
            mp.Process(target=build_hf_model, args=(is_critic, i, world_size, mp_size, pp_size))
            for i in range(world_size)
        ]
        for p in procs:
            p.start()

        for p in procs:
            p.join()

    def testLoadFromHF(self):
        with constants.model_scope(MODEL_NAME):
            for is_critic, mp_size, pp_size in itertools.product([True, False], [1, 2], [1, 3]):
                self._testLoadFromHF(is_critic, mp_size, pp_size)

    def _testSaveThenLoad(
        self,
        save_critic: bool,
        load_critic: bool,
        save_mp_size: int,
        save_pp_size: int,
        load_mp_size: int,
        load_pp_size: int,
    ):
        clear_name_resolve()
        save_dir = os.path.join("/tmp/real_model_sl_test/")
        os.system(f"rm -rf {save_dir}")
        save_world_size = save_mp_size * save_pp_size
        setup_barrier(save_world_size)
        procs = [
            mp.Process(
                target=build_hf_model,
                args=(save_critic, i, save_world_size, save_mp_size, save_pp_size, save_dir),
            ) for i in range(save_world_size)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

        clear_name_resolve()
        load_world_size = load_mp_size * load_pp_size
        setup_barrier(load_world_size)
        procs = [
            mp.Process(
                target=load_model,
                args=(save_critic, load_critic, i, load_world_size, load_mp_size, load_pp_size, save_dir),
            ) for i in range(load_world_size)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()

    def testSaveThenLoad(self):
        with constants.model_scope(MODEL_NAME):
            for (
                    save_critic,
                    load_critic,
                    save_mp_size,
                    save_pp_size,
                    load_mp_size,
                    load_pp_size,
            ) in itertools.product([True, False], [False, True], [1, 2], [2, 4], [1, 2], [1, 3]):
                print(
                    ">>>>>>>>>>>>>",
                    save_mp_size,
                    save_pp_size,
                    load_mp_size,
                    load_pp_size,
                )
                if save_critic and not load_critic:
                    continue
                self._testSaveThenLoad(save_critic, load_critic, save_mp_size, save_pp_size, load_mp_size,
                                       load_pp_size)


if __name__ == "__main__":
    unittest.main(defaultTest="ParallelReaLModelSaveLoadTest.testSaveThenLoad")
