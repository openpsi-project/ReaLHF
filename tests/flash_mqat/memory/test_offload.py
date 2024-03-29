from typing import *
import collections
import torch
import functools
import gc
import pynvml
import torch.profiler
import dataclasses
import base.constants
from tests.utils import (
    get_llama7b_flash_config,
    MODEL_NAME,
    get_memory,
    clear_gpu_cache,
    setup_gpu,
    init_global_constants,
    setup_barrier,
    clear_name_resolve,
    get_pytorch_profiler,
)
import time


def test_impl(world_size):
    from impl.model.nn.flash_mqat.flash_mqat_api import FlashMQATModel, add_helper_functions
    from impl.model.backend.pipe_inf import InferencePipelineEngine

    mconfig = get_llama7b_flash_config()
    with base.constants.model_scope(MODEL_NAME):
        m = FlashMQATModel(mconfig, device=torch.device("cuda:0"), dtype=torch.float16)
        # add_helper_functions(m)
        m.instantiate()
        m.load_from_hf("/lustre/public/pretrained_model_weights/Llama-2-7b-hf")
        torch.cuda.synchronize()
        original_state_dict = m.state_dict()
        engine = InferencePipelineEngine(m)
        print("After model creation", get_memory(0))

        m.async_offload()
        m.wait_for_offload()

        packed_input_ids = torch.randint(0, mconfig.vocab_size, (32 * 256,), dtype=torch.long, device="cuda")
        cu_seqlens = torch.linspace(0, 256 * 32, 33, dtype=torch.int32, device="cuda")
        seqlens_cpu = [256 for _ in range(32)]
        assert cu_seqlens[-1] == packed_input_ids.shape[0]
        with torch.no_grad():
            y = engine.forward(seqlens_cpu, packed_input_ids, cu_seqlens, num_micro_batches=world_size)
            for _ in range(5):
                engine.forward(seqlens_cpu, packed_input_ids, cu_seqlens, num_micro_batches=world_size)
                # assert torch.allclose(m(input_ids=x), y)
                new_state_dict = m.state_dict()
                for k in original_state_dict:
                    # print(k, original_state_dict[k])
                    assert torch.allclose(original_state_dict[k], new_state_dict[k]), (k, original_state_dict[k], new_state_dict[k])
                m.async_offload()
            m.wait_for_offload()
        clear_gpu_cache()
        print("After model offload", get_memory(0))


def test(idx, world_size, profile: bool):
    # PyTorch CUDA setup
    setup_gpu(idx, world_size)
    init_global_constants(num_dp=1, num_mp=1, num_pp=world_size)

    print("After setup memory", get_memory(0))

    test_impl(world_size)

    if profile:
        profiler = get_pytorch_profiler(f"offload{idx}.json")
        profiler.start()

    test_impl(world_size)

    if profile:
        profiler.__exit__(None, None, None)


if __name__ == "__main__":
    print(torch.cuda.is_initialized())
    import multiprocessing as mp

    clear_name_resolve()
    world_size = 2
    setup_barrier(world_size)
    procs = [mp.Process(target=test, args=(i, world_size, True)) for i in range(world_size)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
