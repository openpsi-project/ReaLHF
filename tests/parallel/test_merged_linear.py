"""
Bash script to run this example:

```bash
#!/bin/sh
CUDA_DEVICE_MAX_CONNECTIONS=1 \
OMP_NUM_THREADS=8 \
MASTER_ADDR=localhost \
MASTER_PORT=7777 \
torchrun --standalone --nnodes=1 --nproc-per-node=8 --module \
    tests.parallel.test_merged_linear
```

"""
import os
import time

import torch
import torch.distributed
import torch.profiler

import reallm.base.namedarray

batch_size = 8
seqlen = 4096
vocab_size = 32000

hidden_dim = 1024
head_dim = 128
n_q_heads = 16

use_bias = False
n_kv_heads = 16
sequence_parallel = False


def main(rank: int = None, world_size: int = None):
    if rank is None:
        rank = int(os.environ["RANK"])
    if world_size is None:
        world_size = int(os.environ["WORLD_SIZE"])

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    device = torch.device("cuda", 0)
    dtype = torch.float16

    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    import deepspeed

    deepspeed.init_distributed()

    from reallm.impl.model.parallelism.model_parallel.mappings import gather_from_tensor_model_parallel_region
    from reallm.impl.model.parallelism.model_parallel.modules import (
        ColumnParallelLinear, merged_linear_with_grad_accumulation_and_async_allreduce)

    from .test_model_parallel import init_global_constants

    NUM_MP = 8
    NUM_PP = NUM_DP = 1
    init_global_constants(NUM_DP, NUM_MP, NUM_PP)

    q_attn = ColumnParallelLinear(
        hidden_dim,
        head_dim * n_q_heads,
        bias=use_bias,
        async_tensor_model_parallel_allreduce=not sequence_parallel,
        sequence_parallel=sequence_parallel,
        gradient_accumulation_fusion=False,
        dtype=dtype,
        device=device,
    )
    if n_kv_heads // NUM_MP > 0:
        assert n_kv_heads % NUM_MP == 0
        kv_attn = ColumnParallelLinear(
            hidden_dim,
            head_dim * n_kv_heads,
            bias=use_bias,
            async_tensor_model_parallel_allreduce=not sequence_parallel,
            sequence_parallel=sequence_parallel,
            gradient_accumulation_fusion=False,
            dtype=dtype,
            device=device,
        )
    else:
        kv_attn = torch.nn.Linear(hidden_dim, head_dim * n_kv_heads, bias=use_bias).to(device=device,
                                                                                       dtype=dtype)
        torch.distributed.all_reduce(kv_attn.weight.data)
        if kv_attn.bias is not None:
            torch.distributed.all_reduce(kv_attn.bias.data)

    for _ in range(5):
        if sequence_parallel:
            assert batch_size % NUM_MP == 0
            x = torch.randn(batch_size // NUM_MP, hidden_dim, dtype=dtype, device=device, requires_grad=True)
        else:
            x = torch.randn(batch_size, hidden_dim, dtype=dtype, device=device, requires_grad=False)
            torch.distributed.all_reduce(x)
            x = x.requires_grad_()
        q1 = q_attn(x)
        kv1 = kv_attn(x)

        q2, kv2 = merged_linear_with_grad_accumulation_and_async_allreduce(
            x,
            False,
            not sequence_parallel,
            sequence_parallel,
            [True, isinstance(kv_attn, ColumnParallelLinear)],
            q_attn.weight,
            q_attn.bias,
            kv_attn.weight,
            kv_attn.bias,
        )

        q1 = gather_from_tensor_model_parallel_region(q1)
        q2 = gather_from_tensor_model_parallel_region(q2)
        assert torch.allclose(q1, q2, atol=2e-2), (q1 - q2).abs().max()
        assert torch.allclose(kv1, kv2, atol=2e-2), (kv1 - kv2).abs().max()

        q_attn.weight.grad = None
        if q_attn.bias is not None:
            q_attn.bias.grad = None
        kv_attn.weight.grad = None
        if kv_attn.bias is not None:
            kv_attn.bias.grad = None
        x.grad = None

        loss = q1.sum() + kv1.sum()
        loss.backward()

        grad_qw1 = q_attn.weight.grad.clone()
        if q_attn.bias is not None:
            grad_qb1 = q_attn.bias.grad.clone()
        grad_kvw1 = kv_attn.weight.grad.clone()
        if kv_attn.bias is not None:
            grad_kvb1 = kv_attn.bias.grad.clone()
        grad_x1 = x.grad.clone()

        q_attn.weight.grad = None
        if q_attn.bias is not None:
            q_attn.bias.grad = None
        kv_attn.weight.grad = None
        if kv_attn.bias is not None:
            kv_attn.bias.grad = None
        x.grad = None

        loss = q2.sum() + kv2.sum()
        loss.backward()

        grad_qw2 = q_attn.weight.grad.clone()
        if q_attn.bias is not None:
            grad_qb2 = q_attn.bias.grad.clone()
        grad_kvw2 = kv_attn.weight.grad.clone()
        if kv_attn.bias is not None:
            grad_kvb2 = kv_attn.bias.grad.clone()
        grad_x2 = x.grad.clone()

        assert torch.allclose(grad_kvw1, grad_kvw2, atol=2e-2), (
            torch.distributed.get_rank(),
            (grad_kvw1 - grad_kvw2).abs().max(),
        )
        assert torch.allclose(grad_qw1, grad_qw2, atol=2e-2), (
            torch.distributed.get_rank(),
            (grad_qw1 - grad_qw2).abs().max(),
        )
        assert torch.allclose(grad_x1, grad_x2, atol=2e-2), (
            torch.distributed.get_rank(),
            (grad_x1 - grad_x2).abs().max(),
        )
        if q_attn.bias is not None:
            assert torch.allclose(grad_qb1, grad_qb2, atol=2e-2), (grad_qb1 - grad_qb2).abs().max()
        if kv_attn.bias is not None:
            assert torch.allclose(grad_kvb1, grad_kvb2, atol=2e-2), (grad_kvb1 - grad_kvb2).abs().max()
    if torch.distributed.get_rank() == 0:
        print("success")


if __name__ == "__main__":
    main()
