"""
Bash script to run this example:

```bash
#!/bin/sh
python3 -m apps.remote reset_name_resolve -e test -f test
FLASH_MQAT_USE_TE=1 \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
OMP_NUM_THREADS=8 \
MASTER_ADDR=localhost \
MASTER_PORT=7777 \
torchrun --standalone --nnodes=1 --nproc-per-node=8 --module \
    tests.torch_profile_example
```

"""
import json
import os
import random
import time

import torch
import torch.distributed
import torch.profiler

from base.monitor import gpu_memory_mb
import api.config as config_package
import base.constants
import base.gpu_utils
import base.namedarray

seqlen = 1024
vocab_size = 32000

batch_size_tokens = 1024 * 64
batch_size = batch_size_tokens // seqlen

MODEL_NAME = "default"

# parallelism config
NUM_MP = 1
NUM_PP = 4
NUM_DP = 2
assert batch_size >= NUM_DP
WORLD_SIZE = NUM_MP * NUM_DP * NUM_PP
MODEL_TYPE = "codellama"
MODEL_PARALLEL_PATH = "/lustre/public/pretrained_model_weights/sharded/CodeLlama-34b-hf_2pp_2mp_3s"
MODEL_PARALLEL_PATH = "/lustre/public/pretrained_model_weights/sharded/CodeLlama-34b-hf_4pp_3s"
BASE_MODEL_PATH = "/lustre/public/pretrained_model_weights/CodeLlama-34b-hf/"

## performance related config
PROFILE_INTERFACE_TYPE = "generate"
SHORTNAME = {"inference": "fwd", "train_step": "fwdbwd", "generate": "gen"}
USE_GRADIENT_CHECKPOINTING = True
USE_BF16 = False
USE_SEQ_PARALLEL = False
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

    if PROFILE_INTERFACE_TYPE == "train_step":
        return api.model.make_backend(
            config_package.ModelBackend(
                type_="ds_train",
                args=dict(
                    optimizer_name="adam",
                    optimizer_config=dict(lr=1e-5, weight_decay=0.0, betas=(0.9, 0.95)),
                    warmup_steps_proportion=0.0,
                    min_lr_ratio=0.0,
                    zero_stage=1 if NUM_PP > 1 else 2,
                    engine_type="pipe" if NUM_PP > 1 else "deepspeed",
                    gradient_checkpointing=USE_GRADIENT_CHECKPOINTING,
                    num_pipeline_stages=NUM_PP,
                    enable_fp16=not USE_BF16,
                    enable_bf16=USE_BF16,
                    sequence_parallel=USE_SEQ_PARALLEL,
                    num_pipeline_micro_batches=NUM_PP,
                ),
            ))
    else:
        return api.model.make_backend(
            config_package.ModelBackend(
                type_="ds_inference",
                args=dict(
                    zero_stage=0,
                    engine_type="pipe" if NUM_PP > 1 else "deepspeed",
                    num_pipeline_stages=NUM_PP,
                    enable_fp16=not USE_BF16,
                    enable_bf16=USE_BF16,
                    sequence_parallel=USE_SEQ_PARALLEL,
                    num_pipeline_micro_batches=NUM_PP,
                ),
            ))


def make_interface():
    import api.model

    return api.model.make_interface(config_package.ModelInterface(type_="flash_sft", args=dict()))


def make_model(device):
    import api.model
    import impl.model.nn.flash_mqat.flash_mqat_api

    model_config = config_package.Model(
        "flash_mqat",
        args=dict(
            model_path=MODEL_PARALLEL_PATH,
            from_type="self" if NUM_PP == 1 else "empty_actor",
            dtype="bf16" if USE_BF16 else "fp16",
            hf_model_type=MODEL_TYPE,
            tokenizer_path=BASE_MODEL_PATH,
            sequence_parallel=USE_SEQ_PARALLEL,
            gradient_accumulation_fusion=GRADIENT_ACCUMULATION_FUSION,
        ),
    )
    assert NUM_PP > 1 or NUM_MP > 1, "can not test model without mp or dp"
    if NUM_PP > 1:
        model_config.wrappers += [
            config_package.ModelWrapper(
                "pipe_flash_mqat",
                args=dict(
                    model_path=MODEL_PARALLEL_PATH,
                    partition_method="parameters_balanced",
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
    whole_batch = [random_sentence(min_len=seqlen + 100, max_len=seqlen + 100) for _ in range(batch_size)]
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

    from .utils import init_global_constants

    init_global_constants(NUM_DP, NUM_MP, NUM_PP)

    # packed_input_ids = torch.randint(0, vocab_size, (batch_size * seqlen,), dtype=torch.long, device=device)
    # cu_seqlens = torch.arange(batch_size + 1, dtype=torch.int32, device=device) * seqlen
    # prompt_mask = torch.randint(0, 2, (batch_size * seqlen,), dtype=torch.bool, device=device)

    if base.constants.model_parallel_rank() == 0:
        gpu_memory_mb("before model initialization")
    model = make_model(device)
    # cnt = 0
    # params_to_mem = {}
    # for k, m in model.module.named_parameters():
    #     if m.is_cuda:
    #         if m.dtype == torch.float16:
    #             factor = 2
    #         elif m.dtype == torch.float32:
    #             factor = 4
    #         else:
    #             raise NotImplementedError(m.dtype)
    #         cnt += m.numel() * factor
    #         params_to_mem[k] = m.numel() * factor
    # print(cnt / 1024**2)
    # pretty_dict = json.dumps({k: v / 1024**2 for k, v in params_to_mem.items()}, indent=4)
    # print(pretty_dict)
    data = make_batch(model.tokenizer, device)
    if USE_GRADIENT_CHECKPOINTING:
        model.module.gradient_checkpointing_enable()
    if base.constants.model_parallel_rank() == 0:
        gpu_memory_mb("after model initialization")
    backend = make_backend()
    ft_spec = make_finetune_spec(512)
    interface = make_interface()

    model = backend.initialize(model, ft_spec)

    s = torch.profiler.schedule(skip_first=1, warmup=1, active=2, repeat=1, wait=0)

    dirname = f"./trace_result/{SHORTNAME[PROFILE_INTERFACE_TYPE]}_mp{NUM_MP}pp{NUM_PP}_local"
    os.makedirs(dirname, exist_ok=True)

    def trace_handler(p: torch.profiler._KinetoProfile):
        if base.constants.model_parallel_rank() == 0 and base.constants.data_parallel_rank() == 0:
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
        for _ in range(10):
            torch.cuda.synchronize()
            st = time.monotonic()

            if PROFILE_INTERFACE_TYPE != "generate":
                res = getattr(interface, PROFILE_INTERFACE_TYPE)(model, data)
            else:
                from impl.model.nn.flash_mqat.flash_generate import GenerationConfig

                gconfig = GenerationConfig(min_new_tokens=1, max_new_tokens=10)
                res = interface.generate(model, data, gconfig)
            torch.cuda.synchronize()
            if (base.constants.model_parallel_rank() == 0
                    and base.constants.pipe_parallel_rank() == NUM_PP - 1):
                print(
                    f"generate {res['gen_tokens'].shape[1]} tokens * batch size {res['gen_tokens'].shape[0]}, "
                    f"time: {time.monotonic() - st}")
            prof.step()


if __name__ == "__main__":
    main()
