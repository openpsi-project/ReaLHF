"""
Bash script to run this example:

```bash
#!/bin/sh
CUDA_DEVICE_MAX_CONNECTIONS=1 \
OMP_NUM_THREADS=8 \
MASTER_ADDR=localhost \
MASTER_PORT=7777 \
torchrun --standalone --nnodes=1 --nproc-per-node=8 --module \
    tests.torch_profile_example
```

"""
import os
import random
import time

import torch
import torch.distributed
import torch.profiler

import api.config as config_package
import base.gpu_utils
import base.namedarray

batch_size = 6
n_minibatches = 3
seqlen = 4096
vocab_size = 32000

MODEL_NAME = "default"

# parallelism config
NUM_MP = 8
NUM_PP = 1
NUM_DP = 1
NUM_SHARDS = 3
WORLD_SIZE = NUM_MP * NUM_DP * NUM_PP
MODEL_TYPE = "llama"
if MODEL_TYPE == "llama":
    if NUM_PP == 1:
        SUFFIX = f"_{NUM_MP}mp_{NUM_SHARDS}s"
    elif NUM_MP == 1:
        SUFFIX = f"_{NUM_PP}pp_{NUM_SHARDS}s"
    elif NUM_PP > 1:
        SUFFIX = f"_{NUM_PP}pp_{NUM_MP}mp_{NUM_SHARDS}s"
    # BASELINE_MODEL_PATH = "/home/meizy/models/test/Llama-2-4l"
    # MODEL_PARALLEL_PATH = f"/lustre/public/pretrained_model_weights/sharded/Llama-2-4l{SUFFIX}"
    BASELINE_MODEL_PATH = "/lustre/public/pretrained_model_weights/Llama-2-13b-hf"
    MODEL_PARALLEL_PATH = f"/lustre/public/pretrained_model_weights/sharded/Llama-2-13b-hf{SUFFIX}"

## performance related config
USE_GRADIENT_CHECKPOINTING = True
USE_BF16 = False
USE_SEQ_PARALLEL = True
GRADIENT_ACCUMULATION_FUSION = False


def make_finetune_spec(bs_per_device, total_train_epochs=1, total_train_steps=10, steps_per_epoch=10):
    import api.model

    finetune_spec = api.model.FinetuneSpec(
        total_train_epochs=total_train_epochs,
        total_train_steps=total_train_steps,
        steps_per_epoch=steps_per_epoch,
        batch_size_per_device=bs_per_device,
    )
    return finetune_spec


def make_backend():
    import api.model

    if NUM_PP == 1:
        return api.model.make_backend(
            config_package.ModelBackend(
                type_="ds_train",
                args=dict(
                    optimizer_name="adam",
                    optimizer_config=dict(lr=1e-5, weight_decay=0.0, betas=(0.9, 0.95)),
                    warmup_steps_proportion=0.0,
                    min_lr_ratio=0.0,
                    # TODO: test zero_stage = 2 or 3 later
                    gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
                    zero_stage=1,
                    enable_fp16=not USE_BF16,
                    enable_bf16=USE_BF16,
                ),
            ))
    elif NUM_PP > 1:
        return api.model.make_backend(
            config_package.ModelBackend(
                type_="ds_train",
                args=dict(
                    optimizer_name="adam",
                    optimizer_config=dict(lr=1e-5, weight_decay=0.0, betas=(0.9, 0.95)),
                    warmup_steps_proportion=0.0,
                    min_lr_ratio=0.0,
                    zero_stage=1,
                    engine_type="pipe",
                    gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
                    num_pipeline_stages=NUM_PP,
                    enable_fp16=not USE_BF16,
                    enable_bf16=USE_BF16,
                    sequence_parallel=USE_SEQ_PARALLEL,
                ),
            ))


def make_interface():
    import api.model

    return api.model.make_interface(
        config_package.ModelInterface(type_="flash_sft", args=dict(n_minibatches=n_minibatches)))


def make_model(device):
    import api.model
    import impl.model.nn.flash_mqat.flash_mqat_api

    model_config = config_package.Model(
        "flash_mqat_actor",
        args=dict(
            model_path=MODEL_PARALLEL_PATH,
            from_type=MODEL_TYPE,
            tokenizer_path=MODEL_PARALLEL_PATH,
            init_from_scratch=True,
            no_param_instantiation=True,
            dtype="bf16" if USE_BF16 else "fp16",
        ),
    )
    assert NUM_PP > 1 or NUM_MP > 1, "can not test model without mp or dp"
    if NUM_PP == 1:
        model_config.wrappers += [
            config_package.ModelWrapper(
                "model_parallel",
                args=dict(
                    model_path=MODEL_PARALLEL_PATH,
                    sequence_parallel=USE_SEQ_PARALLEL,
                    gradient_accumulation_fusion=GRADIENT_ACCUMULATION_FUSION,
                    is_critic=False,
                    init_critic_from_actor=False,
                    init_from_scratch=False,
                ),
            )
        ]
    elif NUM_MP == 1:
        model_config.wrappers += [
            config_package.ModelWrapper(
                "pipe",
                args=dict(
                    model_path=MODEL_PARALLEL_PATH,
                    num_pp=NUM_PP,
                    num_dp=NUM_DP,
                    is_critic=False,
                    init_critic_from_actor=False,
                    init_from_scratch=False,
                ),
            )
        ]
    elif NUM_PP > 1:
        model_config.wrappers += [
            config_package.ModelWrapper(
                "model_pipe_parallel",
                args=dict(
                    model_path=MODEL_PARALLEL_PATH,
                    num_pp=NUM_PP,
                    num_mp=NUM_MP,
                    num_dp=NUM_DP,
                    sequence_parallel=USE_SEQ_PARALLEL,
                    gradient_accumulation_fusion=GRADIENT_ACCUMULATION_FUSION,
                    is_critic=False,
                    init_critic_from_actor=False,
                    init_from_scratch=False,
                ),
            )
        ]

    model = api.model.make_model(model_config, name=MODEL_NAME, device=device)
    return model


def random_sentence(min_len=50, max_len=100):
    words = ["the", "quick", "brown", "fox", "jumped", "over", "the", "lazy", "dog"]
    sentence_length = random.randint(min_len, max_len)
    return " ".join(random.choices(words, k=sentence_length))


def make_input(tokenizer, device, s):
    from flash_attn.bert_padding import unpad_input

    tokenizer.padding_side = "left"
    prompts = tokenizer(s, return_tensors="pt", padding=True)

    input_ids, attention_mask = prompts["input_ids"], prompts["attention_mask"]
    input_ids = input_ids[:, :seqlen].to(device)
    attention_mask = attention_mask[:, :seqlen].to(device)

    packed_input_ids, _, cu_seqlens, max_seqlen = unpad_input(input_ids, attention_mask)
    prompt_mask = torch.zeros_like(packed_input_ids)
    data = base.namedarray.NamedArray(
        packed_input_ids=packed_input_ids,
        cu_seqlens=cu_seqlens.int(),
        prompt_mask=prompt_mask.bool(),
    )
    return data


def make_batch(tokenizer, device, seed=373):
    import base.constants

    dp_rank = base.constants.data_parallel_rank()
    dp_worldsize = base.constants.data_parallel_world_size()
    random.seed(seed)
    whole_batch = [random_sentence(min_len=seqlen, max_len=seqlen) for _ in range(batch_size)]
    dp_batch = whole_batch[batch_size // dp_worldsize * dp_rank:batch_size // dp_worldsize * (dp_rank + 1)]
    return make_input(tokenizer, device, dp_batch)


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

    from .parallel.utils import init_global_constants

    init_global_constants(NUM_DP, NUM_MP, NUM_PP)

    model = make_model(device)
    backend = make_backend()
    ft_spec = make_finetune_spec(512)
    interface = make_interface()

    model = backend.initialize(model, ft_spec)

    # packed_input_ids = torch.randint(0, vocab_size, (batch_size * seqlen,), dtype=torch.long, device=device)
    # cu_seqlens = torch.arange(batch_size + 1, dtype=torch.int32, device=device) * seqlen
    # prompt_mask = torch.randint(0, 2, (batch_size * seqlen,), dtype=torch.bool, device=device)
    data = make_batch(model.tokenizer, device)

    s = torch.profiler.schedule(skip_first=1, warmup=1, active=2, repeat=1, wait=0)

    def trace_handler(p):
        p.export_chrome_trace(f"./trace_result/fwdbwd_mp{NUM_MP}pp{NUM_PP}_local/mp{rank}_trace.json")

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
            torch.cuda.synchronize()
            st = time.monotonic()
            res = interface.train_step(model, data)
            torch.cuda.synchronize()
            prof.step()


if __name__ == "__main__":
    main()
