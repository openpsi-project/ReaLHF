from typing import *
import collections
import functools
import gc

from deepspeed.ops.adam.fused_adam import FusedAdam
import pynvml
import torch
import torch.profiler


def print_memory(idx):
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total_memory = memory_info.total / (1024**2)  # Convert bytes to megabytes
    used_memory = memory_info.used / (1024**2)
    print(idx, used_memory, total_memory)


def _dissect_param_hook(optimizer: torch.optim.Optimizer, args, kwargs):
    opt_state_cache = kwargs["opt_state_cache"]
    assert len(opt_state_cache) == 0
    named_parameters = kwargs["named_parameters"]
    assert isinstance(named_parameters, list)
    for g in optimizer.param_groups:
        g['params'] = None
    for pname, p in named_parameters:
        assert p in optimizer.state
        opt_state_cache[pname] = optimizer.state.pop(p)


def _compose_param_hook(optimizer: torch.optim.Optimizer, args, kwargs):
    opt_state_cache = kwargs["opt_state_cache"]
    named_parameters = kwargs["named_parameters"]
    assert isinstance(named_parameters, list)
    assert len(optimizer.param_groups) == 1
    optimizer.param_groups[0]['params'] = [p for pname, p in named_parameters]
    if opt_state_cache:
        for pname, p in named_parameters:
            optimizer.state[p] = opt_state_cache.pop(pname)


class HookableFusedAdam(FusedAdam):

    def step(self, *args, **kwargs):
        return super().step()


def trace_handler(p: torch.profiler._KinetoProfile):
    print(p.key_averages().table(sort_by="cuda_memory_usage",
                                 row_limit=20,
                                 max_name_column_width=30,
                                 max_src_column_width=30))
    # p.export_chrome_trace(os.path.join(dirname, f"rank{rank}.json"))


def setup_pytorch():
    torch.cuda.init()
    torch.cuda.set_device(0)
    x = torch.randn(1, device="cuda", dtype=torch.float64, requires_grad=True)
    y = x * torch.randn(1000, device="cuda", dtype=torch.float64)
    y.mean().backward()
    del x, y
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


def test():
    # PyTorch CUDA setup
    # with torch.profiler.profile(
    #         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    #         # record_shapes=True,
    #         profile_memory=True,
    #         with_stack=True,
    #         on_trace_ready=trace_handler,
    #         # with_flops=True,
    # ) as prof:
    setup_pytorch()

    print_memory(0)
    # a = torch.empty(int(2e4), int(2e4), dtype=torch.float64, device="cuda")
    a = torch.nn.Linear(int(1e4), int(1e4), dtype=torch.float32, device="cuda", bias=False)
    optimizer = HookableFusedAdam(a.parameters())
    print_memory(1)

    optimizer.register_step_pre_hook(_compose_param_hook)
    optimizer.register_step_post_hook(_dissect_param_hook)
    a.weight = None

    opt_state_cache = {}

    for _ in range(5):
        a.weight = torch.nn.Parameter(data=torch.randn(int(1e4), int(1e4), dtype=torch.float32,
                                                       device="cuda"),
                                      requires_grad=True)
        a.weight.grad = None

        print_memory(2)

        a.weight.backward(torch.randn_like(a.weight))

        print_memory(3)
        opt_state_cache = {}
        optimizer.step(
            named_parameters=list(a.named_parameters()),
            opt_state_cache=opt_state_cache,
        )
        assert len(opt_state_cache) > 0
        torch.cuda.synchronize()
        a.weight = None
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        print_memory(4)


pynvml.nvmlInit()

test()

pynvml.nvmlShutdown()
