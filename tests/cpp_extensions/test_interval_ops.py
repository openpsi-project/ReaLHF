import os
import random
import time
import uuid
from typing import *

import numpy as np
import pytest
import torch

from realhf.impl.model.nn.flatten_param import (
    _set_intervals_py,
    _slice_intervals_py,
    set_intervals,
    slice_intervals,
)


def make_intervals(maxsize, n_intervals):
    assert maxsize // n_intervals > 1
    s = maxsize // n_intervals
    intervals = []
    interval_size = 0
    for i in range(n_intervals):
        intervals.append((i * s, i * s + s // 2))
        interval_size += s // 2
    return np.array(intervals, dtype=np.int64), interval_size


def maybe_synchronize_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@pytest.mark.parametrize("device", [torch.device("cuda"), torch.device("cpu")])
@pytest.mark.parametrize("n_intervals", list(reversed([1, 100, 10000, 100000])))
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
def test_get(n_intervals: int, dtype: torch.dtype, device: torch.device):
    if device == torch.device("cuda") and not torch.cuda.is_available():
        return

    input_tensor = torch.randn(int(1e8), device=device, dtype=dtype)
    intervals, _ = make_intervals(input_tensor.size(0), n_intervals)

    # warmup
    slice_intervals(input_tensor, intervals)
    _slice_intervals_py(input_tensor, intervals)

    maybe_synchronize_cuda()
    tik = time.perf_counter()
    for _ in range(10):
        output_tensor = slice_intervals(input_tensor, intervals)
    maybe_synchronize_cuda()
    t1 = time.perf_counter() - tik

    maybe_synchronize_cuda()
    tik = time.perf_counter()
    for _ in range(10):
        o2 = _slice_intervals_py(input_tensor, intervals)
    maybe_synchronize_cuda()
    t2 = time.perf_counter() - tik
    assert torch.allclose(output_tensor, o2)
    print(f"Success! C++ ext time: {t1:.4f}, PyTorch time: {t2:.4f}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="This test requires a GPU.")
@pytest.mark.parametrize(
    "n_intervals", list(reversed([1, 10, 100, 1000, 2048, 4096, 10000, 100000]))
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_set(n_intervals: int, dtype: torch.dtype):
    # NOTE: Since the set_intervals degenerate to the python implementation with CPU tensors,
    # We don't need to test it with CPU tensors.

    x = torch.randn(int(1e8), device="cuda", dtype=dtype)
    intervals, interval_size = make_intervals(x.size(0), n_intervals)
    src = torch.randn(interval_size, device="cuda", dtype=dtype)

    # warmup
    input_tensor1 = x.clone()
    set_intervals(src, input_tensor1, intervals)
    input_tensor2 = x.clone()
    _set_intervals_py(src, input_tensor2, intervals)

    input_tensor1 = x.clone()
    maybe_synchronize_cuda()
    tik = time.perf_counter()
    for _ in range(10):
        set_intervals(src, input_tensor1, intervals)
    maybe_synchronize_cuda()
    t1 = time.perf_counter() - tik

    input_tensor2 = x.clone()
    maybe_synchronize_cuda()
    tik = time.perf_counter()
    for _ in range(10):
        _set_intervals_py(src, input_tensor2, intervals)
    maybe_synchronize_cuda()
    t2 = time.perf_counter() - tik

    assert torch.allclose(input_tensor1, input_tensor2)
    print(f"Success! C++ ext time: {t1:.4f}, PyTorch time: {t2:.4f}")
