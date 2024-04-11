import os

os.environ["PYTORCH_JIT"] = "0"

import random

import nvidia_dlprof_pytorch_nvtx
import torch

nvidia_dlprof_pytorch_nvtx.init()

with torch.autograd.profiler.emit_nvtx():
    for i in range(100):
        m = random.randint(200, 1000)
        n = random.randint(200, 1000)
        p = random.randint(200, 1000)

        A = torch.rand(m, n, dtype=torch.float32, device=torch.device('cuda:0'))
        B = torch.rand(n, p, dtype=torch.float32, device=torch.device('cuda:0'))

        C = torch.mm(A, B)

        print(f"Iteration {i+1}: Matrix A size {m} x {n}, Matrix B size {n} x {p}, Matrix C size {m} x {p}")
