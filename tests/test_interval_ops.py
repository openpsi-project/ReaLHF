import torch
from torch.utils.cpp_extension import load

# slice_intervals_cuda = load(
#     'slice_intervals',
#     ['csrc/interval_op/interval_get.cu'],
#     verbose=True)
import random
import interval_op_cuda
import time


def test_set():
    # Usage example
    n_intervals = 100000
    input_tensor = torch.randn(int(1e8), device="cuda")
    offset = 0
    intervals = []
    max_interval_size = 0
    for i in range(n_intervals):
        start = random.randint(offset, offset + 10000)
        end = random.randint(start, start + 10000)
        offset = end
        if offset >= input_tensor.size(0):
            break
        intervals.append((start, end))
        max_interval_size = max(end - start, max_interval_size)
    # print(intervals)
    intervals_cuda = torch.tensor(intervals, device="cuda", dtype=torch.long)
    output_size = sum(j - i for i, j in intervals)
    interval_sizes = torch.tensor([j - i for i, j in intervals], device="cuda", dtype=torch.long)
    offsets = torch.nn.functional.pad(interval_sizes.cumsum(0)[:-1], (1, 0), value=0)

    # warmup
    input_tensor2 = input_tensor.clone()
    output_tensor = torch.randn(output_size, device="cuda")
    interval_op_cuda.set_intervals_cuda(
        output_tensor,
        input_tensor,
        intervals_cuda,
        interval_sizes,
        offsets,
        max_interval_size,
    )
    offset = 0
    for i, j in intervals:
        input_tensor2[i:j] = output_tensor[offset : offset + j - i]
        offset += j - i
    assert torch.allclose(input_tensor, input_tensor2)

    torch.cuda.synchronize()
    tik = time.perf_counter()
    for _ in range(5):
        interval_op_cuda.set_intervals_cuda(
            output_tensor,
            input_tensor,
            intervals_cuda,
            interval_sizes,
            offsets,
            max_interval_size,
        )

    torch.cuda.synchronize()
    t1 = time.perf_counter() - tik

    torch.cuda.synchronize()
    tik = time.perf_counter()
    for _ in range(5):
        offset = 0
        for i, j in intervals:
            input_tensor2[i:j] = output_tensor[offset : offset + j - i]
            offset += j - i
    torch.cuda.synchronize()
    t2 = time.perf_counter() - tik
    print(f"Success! Cuda ext time: {t1:.4f}, PyTorch time: {t2:.4f}")


def test_get():
    # Usage example
    n_intervals = 100000
    input_tensor = torch.randn(int(1e8), device="cuda")
    offset = 0
    intervals = []
    max_interval_size = 0
    for i in range(n_intervals):
        start = random.randint(offset, offset + 10000)
        end = random.randint(start, start + 10000)
        offset = end
        if offset >= input_tensor.size(0):
            break
        intervals.append((start, end))
        max_interval_size = max(end - start, max_interval_size)
    # print(intervals)
    intervals_cuda = torch.tensor(intervals, device="cuda", dtype=torch.long)
    output_size = sum(j - i for i, j in intervals)
    interval_sizes = torch.tensor([j - i for i, j in intervals], device="cuda", dtype=torch.long)
    offsets = torch.nn.functional.pad(interval_sizes.cumsum(0)[:-1], (1, 0), value=0)

    # warmup
    output_tensor = interval_op_cuda.slice_intervals_cuda(
        input_tensor,
        intervals_cuda,
        interval_sizes,
        offsets,
        max_interval_size,
        output_size,
    )
    o2 = torch.cat([input_tensor[i:j] for i, j in intervals], dim=0)

    torch.cuda.synchronize()
    tik = time.perf_counter()
    for _ in range(5):
        output_tensor = interval_op_cuda.slice_intervals_cuda(
            input_tensor, intervals_cuda, interval_sizes, offsets, max_interval_size, output_size
        )
    torch.cuda.synchronize()
    t1 = time.perf_counter() - tik

    torch.cuda.synchronize()
    tik = time.perf_counter()
    for _ in range(5):
        o2 = torch.cat([input_tensor[i:j] for i, j in intervals], dim=0)
    torch.cuda.synchronize()
    t2 = time.perf_counter() - tik
    assert torch.allclose(output_tensor, o2)
    print(f"Success! Cuda ext time: {t1:.4f}, PyTorch time: {t2:.4f}")


if __name__ == "__main__":
    test_set()
