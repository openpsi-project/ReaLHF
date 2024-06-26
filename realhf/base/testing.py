import dataclasses
import gc
import multiprocessing as mp
import os
import pickle
import queue
import random
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import pynvml
import torch
import torch.utils.data
import transformers

import realhf.api.core.data_api as data_api
import realhf.api.core.model_api as model_api
import realhf.api.core.system_api as system_api
from realhf.api.core.config import ModelFamily
from realhf.base import constants, gpu_utils, name_resolve, namedarray, names, topology
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

        self.expr_name = expr_name if expr_name is not None else _DEFAULT_EXPR_NAME
        self.trial_name = trial_name if trial_name is not None else _DEFAULT_TRIAL_NAME

        self.func = func
        self.args = args
        self.kwargs = kwargs

    def _run(self):
        return self.func(*self.args, **self.kwargs)

    def run(self) -> None:
        assert not torch.cuda.is_initialized()

        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
        os.environ["REAL_MODE"] = "LOCAL"
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

        # isolate cuda devices
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.rank)
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(self.rank+2) if self.rank in [0, 1] else str(self.rank+4)
        os.environ["GPU_DEVICES_ISOLATED"] = str(1)
        torch.cuda.set_device(0)

        self.barrier.wait()

        # init process group
        gpu_utils.reveal_ddp_identity(self.expr_name, self.trial_name, self.rank)
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
    max_prompt_len=None,
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
            max_prompt_len=max_prompt_len,
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
        self.expr_name = expr_name
        self.trial_name = trial_name
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
        clear_name_resolve(self.expr_name, self.trial_name)
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


def make_packed_input_batches(dataset: torch.utils.data.Dataset, batch_size: int):
    batches = [
        [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        for i in range(0, len(dataset), batch_size)
    ]
    batches = [data_api.gather_sequences(batch) for batch in batches]
    return batches


def make_huggingface_generate_input_batches(
    dataset: List[str],
    tokenizer: transformers.PreTrainedTokenizerFast,
    max_length: int,
    batch_size: int,
):
    batches = [
        [dataset[j]["prompt"] for j in range(i, i + batch_size)]
        for i in range(0, len(dataset), batch_size)
    ]
    tokenizer.padding_side = "left"
    batches = [
        tokenizer(
            batch,
            return_tensors="pt",
            max_length=max_length,
            padding=True,
            truncation=True,
            return_attention_mask=True,
        )
        for batch in batches
    ]
    return batches


def make_random_input_batches(shape: Tuple[int], n_batches: int, vocab_size: int):
    batches = [torch.randint(0, vocab_size, shape) for _ in range(n_batches)]
    return batches


def prepare(
    model_family: ModelFamily,
    model_path: str,
    backend_config: system_api.ModelBackend,
    dataset_config: Optional[system_api.Dataset] = None,
    device: str = "cuda",
):
    from realrlhf.impl.model.nn.real_llm_api import ReaLModel, make_real_model

    import realrlhf.impl.dataset
    import realrlhf.impl.model

    with constants.model_scope(MODEL_NAME):
        if dataset_config is not None:
            dataset = data_api.make_dataset(
                dataset_config,
                seed=1,
                ddp_rank=constants.data_parallel_rank(),
                world_size=constants.data_parallel_world_size(),
                tokenizer_or_tokenizer_name=model_path,
                experiment_name=_DEFAULT_EXPR_NAME,
                trial_name=_DEFAULT_TRIAL_NAME,
            )
        else:
            dataset = None
        model: model_api.Model = make_real_model(
            MODEL_NAME,
            device=device,
            model_path=model_path,
            is_critic=False,
            init_critic_from_actor=False,
            dtype="fp16",
            hf_model_family=model_family._class,
        )
        module: ReaLModel = model.module
        module.instantiate()

        backend = model_api.make_backend(backend_config)
        backend.initialize(model, None)
        return model, dataset, backend


def shrink_mconfig(mconfig: model_api.ReaLModelConfig):
    mconfig.hidden_dim = 128
    mconfig.head_dim = 16
    mconfig.n_kv_heads = 1
    mconfig.intermediate_dim = 256
    mconfig.n_layers = 2
    return mconfig


def remove_file_cache(path: str):
    for f in os.listdir(path):
        if f.endswith(".pkl"):
            os.remove(os.path.join(path, f))


def test_result_file_name(
    identifier: str,
    model_family: model_api.ModelFamily,
    dp_rank: int,
):
    assert "-" not in identifier and "-" not in model_family._class
    return f"{identifier}-{model_family._class}-{model_family.size}-{dp_rank}.pkl"


def info_from_file_name(file_name: str):
    parts = file_name.split("-")
    assert len(parts) == 4
    identifier = parts[0]
    model_class = parts[1]
    model_size = int(parts[2])
    dp_rank = int(parts[3].split(".")[0])
    return identifier, model_class, model_size, dp_rank


def save_test_result(
    result: Any,
    path: str,
    model_family: model_api.ModelFamily,
    dp_rank: int,
    identifier: str,
):
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(
        path, test_result_file_name(identifier, model_family, dp_rank)
    )
    with open(save_path, "wb") as f:
        pickle.dump(result, f)


def check_generation_consistency(
    path: str, model_family: model_api.ModelFamily, identifiers: List[str]
):
    dp_rank_counter = {identifier: 0 for identifier in identifiers}
    for f in os.listdir(path):
        if not f.endswith(".pkl"):
            continue
        identifier, _, _, _ = info_from_file_name(f)
        if identifier in identifiers:
            dp_rank_counter[identifier] += 1

    results = {}
    for identifier in identifiers:
        tmp = []
        for i in range(dp_rank_counter[identifier]):
            p = test_result_file_name(identifier, model_family, i)
            load_path = os.path.join(path, p)
            t = pickle.load(open(load_path, "rb"))
            tmp.append(t)
        res = torch.cat(tmp, dim=0)
        print(identifier, res.shape)
        results[identifier] = res

    baseline_result = results[identifiers[0]]
    for identifier, result in results.items():
        if identifier == identifiers[0]:
            continue
        matched_seqs = 0
        matched_tokens = 0
        for i in range(len(baseline_result)):
            a = baseline_result[i]
            b = result[i]
            assert torch.is_tensor(a) and torch.is_tensor(b)
            assert a.dim() == 1 and b.dim() == 1, (a.shape, b.shape)
            gen_len = a.shape[0] if a.shape[0] < b.shape[0] else b.shape[0]
            b = b[:gen_len]
            a = a[:gen_len]
            for j in range(gen_len):
                if a[j] != b[j]:
                    print(f"Mismatch at sequence {i} position {j}")
                    break
                matched_tokens += 1
            else:
                # print(f"Batch {i} sequence {j} check passed")
                matched_seqs += 1
        print(
            f"{identifiers[0]} and {identifier} Matched {matched_seqs}/{len(baseline_result)} "
            f"sequences and {matched_tokens}/{len(baseline_result) * gen_len} tokens"
        )


COMPARE_TENSOR_PATH = "profile_result/compare_tensor/"
_TMP_TENSOR_STORAGE = {}
_COMPARE_RUN_IDENTIFIER = None


def set_compare_run_identifier(identifier: str):
    global _COMPARE_RUN_IDENTIFIER
    _COMPARE_RUN_IDENTIFIER = identifier


def store_tensor_to_compare(
    tensor: torch.Tensor,
    tensor_name: str,
):
    assert _COMPARE_RUN_IDENTIFIER is not None
    assert tensor_name not in _TMP_TENSOR_STORAGE
    _TMP_TENSOR_STORAGE[tensor_name] = tensor


def dump_stored_to_file():
    if len(_TMP_TENSOR_STORAGE) == 0:
        return
    os.makedirs(COMPARE_TENSOR_PATH, exist_ok=True)
    mp = constants.model_parallel_rank()
    pp = constants.pipe_parallel_rank()
    dp = constants.data_parallel_rank()
    fn = _COMPARE_RUN_IDENTIFIER + f"_dp{dp}_pp{pp}_mp{mp}.pkl"
    with open(os.path.join(COMPARE_TENSOR_PATH, fn), "wb") as f:
        pickle.dump(_TMP_TENSOR_STORAGE, f)


def compare_two_runs(run_id1, run_id2, mp=1, dp=1, pp=1):
    import itertools

    os.makedirs(os.path.join(COMPARE_TENSOR_PATH, "logs"), exist_ok=True)
    for mpr, dpr, ppr in itertools.product(range(mp), range(dp), range(pp)):
        log_path = os.path.join(
            COMPARE_TENSOR_PATH,
            f"logs/{run_id1}_{run_id2}_dp{dpr}_pp{ppr}_mp{mpr}.log",
        )
        fn1 = run_id1 + f"_dp{dpr}_pp{ppr}_mp{mpr}.pkl"
        fn2 = run_id2 + f"_dp{dpr}_pp{ppr}_mp{mpr}.pkl"
        ts1 = pickle.load(open(os.path.join(COMPARE_TENSOR_PATH, fn1), "rb"))
        ts2 = pickle.load(open(os.path.join(COMPARE_TENSOR_PATH, fn2), "rb"))
        assert ts1.keys() == ts2.keys()

        with open(log_path, "w") as f:
            print(f"Comparing {fn1} and {fn2}")
            f.write(f"dp{dp}_pp{pp}_mp{mp}:\n")

            match_count = 0
            for k in ts1.keys():
                match = torch.allclose(ts1[k], ts2[k], atol=1e-2, rtol=1e-3)
                print(f"tensor {k} shape {ts1[k].shape}")
                f.write(f"tensor {k} shape {ts1[k].shape}\n\n")
                if match:
                    print(f"tensor {k} matches")
                    f.write(f"tensor {k} matches\n\n")
                    f.write(f"max diff: {torch.max(torch.abs(ts1[k] - ts2[k]))}\n\n")
                    match_count += 1
                else:
                    # torch.set_printoptions(precision=4)
                    print(f"tensor {k} does not match")
                    f.write(
                        f"tensor {k} does not match\n"
                        f"max diff: {torch.max(torch.abs(ts1[k] - ts2[k]))}\n\n"
                    )

            f.write(f"Matched {match_count}/{len(ts1)} tensors")
            print(f"Matched {match_count}/{len(ts1)} tensors")
