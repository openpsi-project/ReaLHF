from typing import *
import collections
import torch
import functools
import gc
import pynvml
import torch.profiler
import dataclasses
from impl.model.nn.flash_mqat.flash_mqat_api import FlashMQATModel, add_helper_functions
from tests.utils import init_global_constants, setup_gpu, clear_name_resolve, setup_barrier
import base.constants
from tests.utils import *
import time


def test_impl():
    mconfig = get_llama7b_flash_config()
    with base.constants.model_scope("default"):
        m = FlashMQATModel(mconfig, device=torch.device("cuda:0"), dtype=torch.float16)
        torch.cuda.synchronize()
        add_helper_functions(m)
        m.instantiate()
        print("After model creation", get_memory(0))

        m.async_offload()
        m.wait_for_offload()

        x = torch.randint(0, mconfig.vocab_size, (32, 256), dtype=torch.long, device="cuda")
        with torch.no_grad():
            y = m(input_ids=x)
            for _ in range(5):
                m(input_ids=x)
                # assert torch.allclose(m(input_ids=x), y)
                m.async_offload()
            m.wait_for_offload()
            del x
        clear_gpu_cache()
        print("After model offload", get_memory(0))


def test(idx, world_size, profile: bool):
    # PyTorch CUDA setup
    clear_name_resolve()
    setup_barrier(world_size)
    setup_gpu(idx, world_size)
    init_global_constants(num_dp=1, num_mp=1, num_pp=1, model_name="default")

    print("After setup memory", get_memory(0))

    test_impl()

    if profile:
        profiler = get_pytorch_profiler("offload.json")
        profiler.start()

    test_impl()

    if profile:
        profiler.__exit__(None, None, None)


if __name__ == "__main__":
    test(0, 1, True)
