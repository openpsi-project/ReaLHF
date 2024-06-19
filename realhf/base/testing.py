from typing import Callable, Dict
import dataclasses
import gc
import multiprocessing as mp
import os
import queue
import random
import time
import traceback

import pynvml
import torch
import torch.distributed as dist

from realhf.api.core import config
from realhf.api.core.config import ModelFamily
from realhf.base import (
    constants,
    gpu_utils,
    name_resolve,
    namedarray,
    names,
    topology,
)
from realhf.base.topology import ParallelGrid, PipeModelDataParallelTopology

# mp.set_start_method("spawn", force=True)  # Otherwise a CUDA reinitialization error will be thrown

MODEL_NAME = "default"
_DEFAULT_EXPR_NAME = "test"
_DEFAULT_TRIAL_NAME = "test"


class StandaloneTestingProcess(mp.Process):
    """Aims for defining this class:
    + Removing duplicate setup GPU codes in each test.

    Note that `init_global_constants` should be called in `func`.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        barrier: mp.Barrier,
        err_queue: mp.Queue,
        func: Callable,
        *args,
        expr_name: str = None,
        trial_name: str = None,
        **kwargs,
    ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.err_queue = err_queue
        self.barrier = barrier

        self.expr_name = (
            expr_name if expr_name is not None else _DEFAULT_EXPR_NAME
        )
        self.trial_name = (
            trial_name if trial_name is not None else _DEFAULT_TRIAL_NAME
        )

        self.func = func
        self.args = args
        self.kwargs = kwargs

    def _run(self):
        return self.func(*self.args, **self.kwargs)

    def run(self) -> None:
        assert not torch.cuda.is_initialized()
        os.environ["REAL_MODE"] = "LOCAL"
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

        # isolate cuda devices
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.rank)
        os.environ["GPU_DEVICES_ISOLATED"] = str(1)
        torch.cuda.set_device(0)

        self.barrier.wait()

        # init process group
        gpu_utils.reveal_ddp_identity(
            self.expr_name, self.trial_name, self.rank
        )
        self.barrier.wait()
        from realhf.impl.model.comm.global_comm import setup_global_comm

        setup_global_comm(self.expr_name, self.trial_name, self.rank)
        # NOTE: The import must be here.
        import deepspeed

        deepspeed.init_distributed()

        # setup some useful constants
        constants.set_experiment_trial_names(self.expr_name, self.trial_name)

        # misc setup
        pynvml.nvmlInit()
        pytorch_memory_burnin(self.rank)

        try:
            self._run()
        except Exception as e:
            print(traceback.format_exc(), flush=True)
            self.err_queue.put(e)
            raise e


def init_global_constants(
    num_dp,
    num_mp,
    num_pp,
    topo=None,
    model_name=None,
    msid2mwid=None,
    sequence_parallel=False,
    gradient_checkpointing=True,
):
    model_name = model_name if model_name is not None else MODEL_NAME

    if topo is None:
        topo = PipeModelDataParallelTopology(
            num_dp=num_dp,
            num_mp=num_mp,
            num_pp=num_pp,
            sequence_parallel=sequence_parallel,
            gradient_checkpointing=gradient_checkpointing,
        )
        ws = num_dp * num_mp * num_pp
    else:
        ws = topo.world_size()

    with constants.model_scope(model_name):
        constants.set_rank_mapping(model_name, topo, msid2mwid=msid2mwid)
        wg_ranks = [constants.to_global_pg_rank(i) for i in range(ws)]
        wg = topology.new_or_get_group(ranks=wg_ranks)

        constants.set_parallelism_group(
            model_name=model_name, pgroup=wg, ranks=wg_ranks
        )
        grid = ParallelGrid(process_group=wg, topology=topo)
        constants.set_grid(model_name=model_name, grid=grid)


class LocalMultiProcessTest:
    """Aims for defining this class:
    1. Defining a barrier and a queue for all sub-processes.
    2. Error handling after launch.
    """

    def __init__(
        self,
        world_size: int,
        func: Callable,
        *args,
        expr_name: str = None,
        trial_name: str = None,
        **kwargs,
    ):
        self.barrier = mp.Barrier(world_size)
        self.err_queue = mp.Queue(world_size)
        self.processes = [
            StandaloneTestingProcess(
                rank,
                world_size,
                self.barrier,
                self.err_queue,
                func,
                *args,
                expr_name=expr_name,
                trial_name=trial_name,
                **kwargs,
            )
            for rank in range(world_size)
        ]

    def launch(self):
        assert not torch.cuda.is_initialized()
        [p.start() for p in self.processes]
        assert not torch.cuda.is_initialized()
        while any([p.is_alive() for p in self.processes]):
            try:
                err = self.err_queue.get_nowait()
                [p.terminate() for p in self.processes]
                raise err
            except queue.Empty:
                time.sleep(0.1)
        [p.join() for p in self.processes]


def clear_name_resolve(expr_name=None, trial_name=None):
    expr_name = expr_name if expr_name is not None else _DEFAULT_EXPR_NAME
    trial_name = trial_name if trial_name is not None else _DEFAULT_TRIAL_NAME
    name_resolve.clear_subtree(
        names.trial_root(experiment_name=expr_name, trial_name=trial_name)
    )


def make_finetune_spec(
    bs_per_device,
    total_train_epochs=1,
    total_train_steps=10,
    steps_per_epoch=10,
    max_seq_len=1024,
):
    import realhf.api.core.model_api as model_api

    finetune_spec = model_api.FinetuneSpec(
        total_train_epochs=total_train_epochs,
        total_train_steps=total_train_steps,
        steps_per_epoch=steps_per_epoch,
    )
    return finetune_spec


def random_sentence(min_len=100, max_len=128):
    words = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumped",
        "over",
        "the",
        "lazy",
        "dog",
    ]
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
    dp_batch = whole_batch[
        batch_size
        // dp_worldsize
        * dp_rank : batch_size
        // dp_worldsize
        * (dp_rank + 1)
    ]
    return make_input(tokenizer, device, dp_batch)


def init_data(tokenizer, device, batch_size, seed, dp_rank=None, num_dp=None):
    from realhf.impl.model.utils.padding import unpad_input

    if dp_rank == None:
        assert num_dp == None
        dp_rank = constants.data_parallel_rank()
        num_dp = constants.data_parallel_world_size()
    input_ids, attention_mask = make_batch(
        tokenizer, device, batch_size, dp_rank % num_dp, num_dp, seed=seed
    )
    packed_input_ids, _, cu_seqlens, max_seqlen = unpad_input(
        input_ids, attention_mask
    )
    prompt_mask = torch.zeros_like(packed_input_ids)
    data = namedarray.NamedArray(
        packed_input_ids=packed_input_ids,
        cu_seqlens=cu_seqlens,
        prompts=input_ids,
        prompt_mask=prompt_mask.bool(),
        prompt_att_mask=attention_mask,
    )
    return data


def random_sample(bs, seq_len, vocab_size):
    import torch

    from realhf.impl.model.utils.padding import unpad_input

    input_ids = torch.randint(0, vocab_size, (bs, seq_len), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    packed_input_ids, _, cu_seqlens, max_seqlen = unpad_input(
        input_ids, attention_mask
    )
    prompt_mask = torch.zeros_like(packed_input_ids)
    data = namedarray.NamedArray(
        packed_input_ids=packed_input_ids,
        cu_seqlens=cu_seqlens,
        prompts=input_ids,
        prompt_mask=prompt_mask.bool(),
        prompt_att_mask=attention_mask,
    )
    return data


def pytorch_memory_burnin(rank):
    torch.cuda.set_device(0)
    torch.cuda.init()
    x = torch.randn(1, device="cuda", dtype=torch.float64, requires_grad=True)
    y = x * torch.randn(1000, device="cuda", dtype=torch.float64)
    y.mean().backward()
    del x, y
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


def clear_gpu_cache():
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


def get_memory(rank):
    handle = pynvml.nvmlDeviceGetHandleByIndex(rank)
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    # total_memory = memory_info.total / (1024**2)  # Convert bytes to megabytes
    used_memory = memory_info.used / (1024**2)
    return used_memory


def get_pytorch_profiler(save_fn: str):

    def trace_handler(p: torch.profiler._KinetoProfile):
        # print(
        #     p.key_averages().table(
        #         sort_by="cuda_memory_usage", row_limit=20, max_name_column_width=30, max_src_column_width=30
        #     )
        # )
        p.export_chrome_trace(save_fn)

    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        on_trace_ready=trace_handler,
        with_flops=True,
    )


def get_llama_config(size):
    from realhf.api.core.model_api import ReaLModelConfig

    if size == 7:
        size_args = dict(
            n_layers=40,
            n_kv_heads=32,
            head_dim=128,
            hidden_dim=4096,
            intermediate_dim=11008,
            n_positions=4096,
        )
    elif size == 13:
        size_args = dict(
            n_layers=40,
            n_kv_heads=40,
            head_dim=128,
            hidden_dim=5120,
            intermediate_dim=13824,
            n_positions=4096,
        )
    elif size == 34:
        size_args = dict(
            n_layers=48,
            n_kv_heads=8,
            head_dim=128,
            hidden_dim=8192,
            intermediate_dim=22016,
            n_positions=16384,
        )
    elif size == 70:
        size_args = dict(
            n_layers=80,
            n_kv_heads=8,
            head_dim=128,
            hidden_dim=8192,
            intermediate_dim=28672,
            n_positions=32768,
        )
    else:
        raise ValueError(f"size {size} not supported")

    return ReaLModelConfig(
        vocab_size=32000,
        activation_function="silu",
        use_attention_bias=False,
        layer_norm_type="rms",
        mlp_type="llama",
        apply_rotary=True,
        **size_args,
    )
