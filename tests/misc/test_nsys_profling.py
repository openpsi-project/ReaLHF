from typing import *
import collections
import dataclasses
import functools
import gc
import time

import pynvml
import torch
import torch.distributed
import torch.profiler

from reallm.base.testing import (clear_gpu_cache, clear_name_resolve, get_llama7b_real_config, get_memory,
                                 init_global_constants, MODEL_NAME, setup_barrier, setup_gpu)
import reallm.base.constants as constants

COMPUTE_KERNEL_KEYS = [
    "elementwise_kernel",
    "gemm_",
    "aten::",
    "at::native::",
    "flash",
]

COMM_KERNEL_KEYS = [
    "c10d::",
    "nccl::",
    "ncclDevKernel",
]

MEM_KERNEL_KEYS = [
    "Memcpy",
    "Memset",
]

MISC_KERNEL_KEYS = [
    "at_cuda_detail",
    "CudaCodeGen",
]


@dataclasses.dataclass
class CUDAKernelTime:  # in us
    compute: int
    comm: int
    mem: int
    misc: int

    @classmethod
    def from_profiler(cls, p: torch.profiler._KinetoProfile):
        compute_time = comm_time = mem_time = misc_time = 0
        unknown_keys = []
        for x in p.key_averages():
            if x.device_type != torch.autograd.DeviceType.CUDA:
                continue
            if x.self_cuda_time_total <= 0:
                continue
            if "waitevent" in x.key.lower():
                print(x.key)
            if any(k in x.key for k in COMPUTE_KERNEL_KEYS):
                compute_time += x.self_cuda_time_total
            elif any(k in x.key for k in COMM_KERNEL_KEYS):
                comm_time += x.self_cuda_time_total
            elif any(k in x.key for k in MEM_KERNEL_KEYS):
                mem_time += x.self_cuda_time_total
            elif any(k in x.key for k in MISC_KERNEL_KEYS):
                misc_time += x.self_cuda_time_total
            else:
                unknown_keys.append(x)
        if unknown_keys:
            raise NotImplementedError(
                f"Unknown keys: {[(x.key, x.self_cuda_time_total) for x in unknown_keys]}")
        return cls(compute=compute_time, comm=comm_time, mem=mem_time, misc=misc_time)

    def __repr__(self):
        return f"CUDAKernelTime(compute={self.compute}us, comm={self.comm}us, mem={self.mem}us, misc={self.misc}us)"


def get_pytorch_profiler(save_fn: str):

    def trace_handler(p: torch.profiler._KinetoProfile):
        # for x in p.key_averages():
        #     print(x.key, x.self_cuda_time_total)
        print(CUDAKernelTime.from_profiler(p))
        # print(
        #     p.key_averages().table(
        #         sort_by="cuda_memory_usage", row_limit=20, max_name_column_width=30, max_src_column_width=30
        #     )
        # )
        # p.export_chrome_trace(save_fn)

    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        on_trace_ready=trace_handler,
    )


def test_impl(world_size, profile: bool):
    from reallm.impl.model.backend.pipe_inf import InferencePipelineEngine
    from reallm.impl.model.nn.real_llm_api import add_helper_functions, ReaLModel

    torch.distributed.barrier()
    mconfig = get_llama7b_real_config()
    with constants.model_scope(MODEL_NAME):
        m = ReaLModel(mconfig, device=torch.device("cuda:0"), dtype=torch.float16)
        m.instantiate()
        torch.cuda.synchronize()
        print("After model instantiation", get_memory(0))
        if constants.pipe_parallel_world_size() == 1:
            add_helper_functions(m)
            engine = m
        else:
            engine = InferencePipelineEngine(m)
        print("After model creation", get_memory(0))

        packed_input_ids = torch.randint(0, mconfig.vocab_size, (32 * 256,), dtype=torch.long, device="cuda")
        cu_seqlens = torch.linspace(0, 256 * 32, 33, dtype=torch.int32, device="cuda")
        assert cu_seqlens[-1] == packed_input_ids.shape[0]
        seqlens_cpu = [256] * 32
        if profile:
            profiler = get_pytorch_profiler("test_nsys.json")
            profiler.start()
            torch.cuda.synchronize()
            tik = time.perf_counter()
        with torch.no_grad():
            if constants.pipe_parallel_world_size() == 1:
                y = engine.forward(packed_input_ids=packed_input_ids, cu_seqlens=cu_seqlens, max_seqlen=256)
                for _ in range(5):
                    engine.forward(packed_input_ids=packed_input_ids, cu_seqlens=cu_seqlens, max_seqlen=256)
            else:
                y = engine.forward(seqlens_cpu, packed_input_ids, cu_seqlens, num_micro_batches=world_size)
                for _ in range(5):
                    engine.forward(seqlens_cpu, packed_input_ids, cu_seqlens, num_micro_batches=world_size)
                    # assert torch.allclose(m(input_ids=x), y)
        if profile:
            torch.cuda.synchronize()
            print(">>>>>>>>>>>>> recorded time ", time.perf_counter() - tik)
            profiler.__exit__(None, None, None)
        clear_gpu_cache()
        print("After model offload", get_memory(0))
    torch.distributed.barrier()


def test(idx, world_size):
    # PyTorch CUDA setup
    print(0)
    setup_gpu(idx, world_size)
    print(111)
    init_global_constants(num_dp=1, num_mp=2, num_pp=1)

    print("After setup memory", get_memory(0))

    test_impl(world_size, profile=False)

    test_impl(world_size, profile=True)

    torch.distributed.barrier()


if __name__ == "__main__":
    import multiprocessing as mp

    clear_name_resolve()
    world_size = 2
    setup_barrier(world_size)
    procs = [mp.Process(target=test, args=(i, world_size)) for i in range(world_size)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
