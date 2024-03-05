import dataclasses
import os
import random

import torch
import torch.multiprocessing as mp

import base.gpu_utils
import base.name_resolve as name_resolve
import base.names as names

EXPR_NAME = "test"
TRIAL_NAME = "test"
MODEL_NAME = "pipedatamodel"
WORKER_TYPE = "model_worker"

BARRIER = None


def setup_barrier(world_size):
    global BARRIER
    BARRIER = mp.Barrier(world_size)


def setup_gpu(rank, world_size):
    os.environ["DLLM_MODE"] = "LOCAL"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    BARRIER.wait()
    base.gpu_utils.isolate_cuda_device(WORKER_TYPE, rank, world_size, EXPR_NAME, TRIAL_NAME)
    print(f"rank {rank} isolated cuda device")
    BARRIER.wait()
    base.gpu_utils.reveal_ddp_identity(EXPR_NAME, TRIAL_NAME, rank)
    print(f"rank {rank} revealed ddp identity")
    BARRIER.wait()
    info = base.gpu_utils.setup_ddp(EXPR_NAME, TRIAL_NAME, rank)
    world_size = info.world_size
    device = torch.device('cuda', 0)
    print(f"rank {rank} setup ddp")
    import deepspeed
    deepspeed.init_distributed()
    print(f"rank {rank} setup deepspeed")
    return device


def clear_name_resolve():
    name_resolve.clear_subtree(names.trial_root(experiment_name=EXPR_NAME, trial_name=TRIAL_NAME))


def make_finetune_spec(bs_per_device, total_train_epochs=1, total_train_steps=10, steps_per_epoch=10):
    import api.model
    finetune_spec = api.model.FinetuneSpec(
        total_train_epochs=total_train_epochs,
        total_train_steps=total_train_steps,
        steps_per_epoch=steps_per_epoch,
        batch_size_per_device=bs_per_device,
    )
    return finetune_spec


def random_sentence(min_len=100, max_len=128):
    words = ["the", "quick", "brown", "fox", "jumped", "over", "the", "lazy", "dog"]
    sentence_length = random.randint(min_len, max_len)
    return " ".join(random.choices(words, k=sentence_length))
    # return "Output less than 50 words:"


def make_input(tokenizer, device, s):
    tokenizer.padding_side = "left"
    prompts = tokenizer(s, return_tensors="pt", padding=True)

    input_ids, attention_mask = prompts["input_ids"], prompts["attention_mask"]
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    # print(f"make input input_ids.shape {input_ids.shape}")

    return input_ids, attention_mask


def make_batch(tokenizer, device, batch_size, dp_rank, dp_worldsize, seed=373):
    random.seed(seed)
    whole_batch = [random_sentence() for _ in range(batch_size)]
    dp_batch = whole_batch[batch_size // dp_worldsize * dp_rank:batch_size // dp_worldsize * (dp_rank + 1)]
    return make_input(tokenizer, device, dp_batch)


def init_global_constants(num_dp, num_mp, num_pp, model_name=None):
    if model_name is None:
        model_name = MODEL_NAME
    from base.topology import PipelineParallelGrid, PipeModelDataParallelTopology
    topo = PipeModelDataParallelTopology(num_dp=num_dp, num_mp=num_mp, num_pp=num_pp)
    ws = num_dp * num_mp * num_pp
    import torch.distributed as dist
    wg = dist.new_group(ranks=range(ws))
    import base.constants
    base.constants.set_model_name(MODEL_NAME)
    base.constants.set_parallelism_group(model_name=MODEL_NAME, pgroup=wg)
    grid = PipelineParallelGrid(process_group=wg, topology=topo)
    base.constants.set_grid(model_name=MODEL_NAME, grid=grid)
    base.constants.set_experiment_trial_names(EXPR_NAME, TRIAL_NAME)


def init_data(tokenizer, device, batch_size, seed, dp_rank=None, num_dp=None):
    from flash_attn.bert_padding import unpad_input

    import base.constants
    import base.namedarray
    if dp_rank == None:
        assert num_dp == None
        dp_rank = base.constants.data_parallel_rank()
        num_dp = base.constants.data_parallel_world_size()
    input_ids, attention_mask = make_batch(tokenizer, device, batch_size, dp_rank % num_dp, num_dp, seed=seed)
    packed_input_ids, _, cu_seqlens, max_seqlen = unpad_input(input_ids, attention_mask)
    prompt_mask = torch.zeros_like(packed_input_ids)
    data = base.namedarray.NamedArray(
        packed_input_ids=packed_input_ids,
        cu_seqlens=cu_seqlens,
        prompts=input_ids,
        prompt_mask=prompt_mask.bool(),
        prompt_att_mask=attention_mask,
    )
    return data


@dataclasses.dataclass
class StatsEntry:
    cost: float


class ProfileStatsTable:
    pass
