"""
Bash script to run this example:

```bash
#!/bin/sh
python3 -m apps.remote reset_name_resolve -e test -f test
CUDA_DEVICE_MAX_CONNECTIONS=1 \
OMP_NUM_THREADS=8 \
MASTER_ADDR=localhost \
MASTER_PORT=7777 \
torchrun --standalone --nnodes=1 --nproc-per-node=8 --module \
    tests.torch_profile_example
```

"""
import os
import base.gpu_utils
import time
import torch
import torch.profiler
import torch.distributed
import base.namedarray

batch_size = 2
seqlen = 4096
vocab_size = 32000


def main(rank: int = None, world_size: int = None):
    if rank is None:
        rank = int(os.environ["RANK"])
    if world_size is None:
        world_size = int(os.environ["WORLD_SIZE"])
    WORLD_SIZE = world_size

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    device = torch.device("cuda", 0)

    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    import deepspeed

    deepspeed.init_distributed()

    from .parallel.test_model_parallel import (
        init_global_constants,
        NUM_DP,
        NUM_MP,
        NUM_PP,
        make_model,
        make_backend,
        make_finetune_spec,
        make_interface,
    )

    init_global_constants(NUM_DP, NUM_MP, NUM_PP)
    torch_dist_rank = torch.distributed.get_rank()
    cuda_visible = os.environ["CUDA_VISIBLE_DEVICES"]
    print(
        f"PROCESS RANK: {rank}; \n" f"TORCH DIST RANK: {torch_dist_rank}; \n" f"CUDA VISIBLE: {cuda_visible}"
    )

    model = make_model(device)
    backend = make_backend()
    ft_spec = make_finetune_spec(8192)
    interface = make_interface()

    backend.initialize(model, ft_spec)

    packed_input_ids = torch.randint(0, vocab_size, (batch_size * seqlen,), dtype=torch.long, device=device)
    cu_seqlens = torch.arange(batch_size + 1, dtype=torch.int32, device=device) * seqlen
    data = base.namedarray.NamedArray(packed_input_ids=packed_input_ids, cu_seqlens=cu_seqlens)

    model.module.eval()

    s = torch.profiler.schedule(skip_first=1, warmup=1, active=4, repeat=1, wait=0)

    def trace_handler(p):
        p.export_chrome_trace(f"./mp{rank}_trace.json")

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        schedule=s,
        on_trace_ready=trace_handler,
        with_flops=True,
    ) as prof:
        for _ in range(10):
            st = time.monotonic()
            res = interface.inference(model, data)
            print(f"rank {rank} mp inference time cost {time.monotonic() - st:.4f}")
            prof.step()


if __name__ == "__main__":
    main()
