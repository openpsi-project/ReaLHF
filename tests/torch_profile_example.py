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
import functools
import os
import time
import unittest

import torch
import torch.distributed
import torch.profiler

from reallm.api.core.config import MODEL_TYPE_TO_PATH, ModelType
from tests.utils import *
import reallm.api.core.system as config_package
import reallm.base.constants

## performance related config
PROFILE_INTERFACE_TYPE = "inference"
SHORTNAME = {"inference": "fwd", "train_step": "fwdbwd", "generate": "gen"}

NUM_MP = 1
NUM_PP = 4
NUM_DP = 2
NUM_SHARDS = 3
WORLD_SIZE = NUM_MP * NUM_DP * NUM_PP
MODEL_TYPE = "llama"

MODEL_PARALLEL_PATH = BASELINE_MODEL_PATH = MODEL_TYPE_TO_PATH[ModelType("llama", 7, False)]
BATCH_SIZE = 512
SEQ_LEN = 256
MIN_NEW_TOKENS = 256
MAX_NEW_TOKENS = 256

USE_GRADIENT_CHECKPOINTING = True
USE_BF16 = False
USE_SEQ_PARALLEL = NUM_MP > 1
GRADIENT_ACCUMULATION_FUSION = False
ASYNC_P2P = False
OFFLOAD_OPTIMIZER_STATE = False
OFFLOAD_PARAM = False


def make_backend():
    import reallm.api.core.model as model_api

    engine_type = "pipe" if NUM_PP > 1 else "deepspeed"
    args = dict(
        optimizer_name="adam",
        optimizer_config=dict(lr=1e-5, weight_decay=0.0, betas=(0.9, 0.95)),
        warmup_steps_proportion=0.0,
        min_lr_ratio=0.0,
        zero_stage=1,
        gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
        engine_type=engine_type,
        offload_optimizer_state=OFFLOAD_OPTIMIZER_STATE,
        offload_param=OFFLOAD_PARAM,
        enable_fp16=not USE_BF16,
        enable_bf16=USE_BF16,
        sequence_parallel=USE_SEQ_PARALLEL,
        enable_async_p2p_communication=ASYNC_P2P,
    )
    return model_api.make_backend(config_package.ModelBackend(
        type_="ds_train",
        args=args,
    ))


def make_interface():
    import reallm.profiler.interface

    import reallm.api.core.dfg
    import reallm.api.core.model as model_api
    return model_api.make_interface(api.core.dfg.ModelInterface(type_="flash_sft", args=dict()))


def make_model(device):
    import reallm.api.core.model as model_api
    import reallm.impl.model.nn.real_llm_api

    # from_type = "self" if NUM_PP == 1 else "empty_actor"
    # if NUM_MP == NUM_PP == 1:
    #     from_type = "random_actor"
    from_type = "hf_as_actor"

    model_config = config_package.Model(
        "flash_mqat",
        args=dict(
            model_path=MODEL_PARALLEL_PATH,
            from_type=from_type,
            dtype="bf16" if USE_BF16 else "fp16",
            hf_model_type=MODEL_TYPE,
            tokenizer_path=MODEL_PARALLEL_PATH,
            sequence_parallel=USE_SEQ_PARALLEL,
            gradient_accumulation_fusion=False,
        ),
    )

    model = model_api.make_model(model_config, name=MODEL_NAME, device=device)
    return model


def init_handles(rank):
    with reallm.base.constants.model_scope(MODEL_NAME):
        device = setup_gpu(rank, WORLD_SIZE)
        init_global_constants(NUM_DP, NUM_MP, NUM_PP)
        torch_dist_rank = torch.distributed.get_rank()
        cuda_visible = os.environ["CUDA_VISIBLE_DEVICES"]
        print(f"PROCESS RANK: {rank}; \n"
              f"TORCH DIST RANK: {torch_dist_rank}; \n"
              f"CUDA VISIBLE: {cuda_visible}")

        model = make_model(device)
        backend = make_backend()
        ft_spec = make_finetune_spec(BATCH_SIZE)
        interface = make_interface()
        # ft_spec = None
        # backend = None
        # interface = None

        model.module.instantiate()
        backend.initialize(model, ft_spec)
    return device, model, backend, interface


def main(rank: int = None, world_size: int = None):
    if rank is None:
        rank = int(os.environ["RANK"])
    if world_size is None:
        world_size = int(os.environ["WORLD_SIZE"])

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    device = torch.device("cuda", 0)
    os.environ["LOCAL_RANK"] = str(0)  # for the usage of deepspeed

    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    import deepspeed

    deepspeed.init_distributed()

    from .utils import init_global_constants

    init_global_constants(NUM_DP, NUM_MP, NUM_PP)

    # packed_input_ids = torch.randint(0, vocab_size, (batch_size * seqlen,), dtype=torch.long, device=device)
    # cu_seqlens = torch.arange(batch_size + 1, dtype=torch.int32, device=device) * seqlen
    # prompt_mask = torch.randint(0, 2, (batch_size * seqlen,), dtype=torch.bool, device=device)

    # if reallm.base.constants.model_parallel_rank() == 0:
    #     gpu_memory_mb("before model initialization")

    model = make_model(device)
    data = random_sample(BATCH_SIZE // NUM_DP, SEQ_LEN, 32000)
    if USE_GRADIENT_CHECKPOINTING:
        model.module.gradient_checkpointing_enable()
    # if reallm.base.constants.model_parallel_rank() == 0:
    #     gpu_memory_mb("after model initialization")
    backend = make_backend()
    ft_spec = make_finetune_spec(32)
    interface = make_interface()

    model.module.instantiate()
    model = backend.initialize(model, ft_spec)

    s = torch.profiler.schedule(skip_first=0, warmup=0, active=1, repeat=1, wait=0)

    dirname = f"./trace_result/{SHORTNAME[PROFILE_INTERFACE_TYPE]}_mp{NUM_MP}pp{NUM_PP}_local"
    os.makedirs(dirname, exist_ok=True)

    def trace_handler(p: torch.profiler._KinetoProfile):
        if reallm.base.constants.model_parallel_rank() == 0 and reallm.base.constants.data_parallel_rank(
        ) == 0:
            print(p.key_averages(group_by_input_shape=True).table(sort_by="cuda_memory_usage", row_limit=20))
            p.export_chrome_trace(os.path.join(dirname, f"rank{rank}.json"))

    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            schedule=s,
            on_trace_ready=trace_handler,
            with_flops=True,
    ) as prof:
        for _ in range(2):
            torch.cuda.synchronize()
            st = time.monotonic()

            if PROFILE_INTERFACE_TYPE != "generate":
                res = getattr(interface, PROFILE_INTERFACE_TYPE)(model, data)
            else:
                from reallm.impl.model.nn.flash_mqat.flash_generate import GenerationConfig
                gconfig = GenerationConfig(min_new_tokens=10, max_new_tokens=10)
                res = interface.generate(model, data, gconfig=gconfig)
        torch.cuda.synchronize()
        if (base.constants.model_parallel_rank() == 0
                and reallm.base.constants.pipe_parallel_rank() == NUM_PP - 1):
            if PROFILE_INTERFACE_TYPE == "generate":
                print(
                    f"generate {res['gen_tokens'].shape[1]} tokens * batch size {res['gen_tokens'].shape[0]}, "
                    f"time: {time.monotonic() - st}")
            else:
                print(f"{PROFILE_INTERFACE_TYPE} time: {time.monotonic() - st}")
        prof.step()


if __name__ == "__main__":
    with reallm.base.constants.model_scope(MODEL_NAME):
        main()
