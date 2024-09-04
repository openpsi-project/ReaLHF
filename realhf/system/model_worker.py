import collections
import contextlib
import gc
import itertools
import multiprocessing as mp
import os
import pickle
import queue
import socket
import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

import deepspeed
import numpy as np
import pynvml
import tabulate
import torch
import torch.distributed as dist
import torch.utils.data

import realhf.api.core.dfg as dfg
import realhf.api.core.system_api as system_api
import realhf.impl.model.comm.data_transfer as data_transfer_comm
import realhf.impl.model.comm.global_comm as global_comm
import realhf.impl.model.comm.param_realloc as param_realloc_comm
from realhf.api.core.config import ModelName
from realhf.base import (
    constants,
    gpu_utils,
    logging,
    recover,
    seeding,
    timeutil,
    topology,
)
from realhf.base.monitor import (
    CUDATimeMarkType,
    cuda_tmark,
    cuda_tmarked,
    dump_tmark_db,
    gpu_utilization_monitor,
)
from realhf.impl.model.nn.real_llm_api import ReaLModel
from realhf.impl.model.utils import cuda_graph
from realhf.system import request_reply_stream, worker_base

# NOTE: Register all implemented datasets and models.
import realhf.api.core.data_api as data_api  # isort:skip
import realhf.api.core.model_api as model_api  # isort:skip

logger = logging.getLogger("Model Worker", "colored")
blogger = logging.getLogger("benchmark")

TIME_RECORD_RPCS = [
    "generate",
    "inference",
    "train_step",
    "save",
    "evaluate",
    "initialize",
]


def get_pytorch_profiler(with_stack: bool, enabled: bool = True):
    if enabled:
        return torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=with_stack,
            with_flops=True,
        )
    else:
        return contextlib.nullcontext()


class NoRequestToHandle(Exception):
    pass


class ModelWorker(worker_base.Worker):
    _setup_counter = -1

    def _configure(self, cfg: system_api.ModelWorker):
        self._setup_counter += 1

        self.config = cfg
        self.model_names = [s.id.model_name for s in cfg.shards]
        self.shard_indices = [
            cfg.model_topos[s.id.model_name].get_rank(
                data=s.id.dp_rank, pipe=s.id.pp_rank, model=s.id.mp_rank
            )
            for s in cfg.shards
        ]

        self.__experiment_name = self.config.worker_info.experiment_name
        self.__trial_name = self.config.worker_info.trial_name

        self.data_consumers = self.config.model_rpcs[0].data_consumers

        self.__worker_index = cfg.worker_info.worker_index

        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
        torch.backends.cudnn.deterministic = cfg.cudnn_deterministic

        seeding.set_random_seed(cfg.seed)

        # Reveal process group identity of this worker to world.
        gpu_utils.reveal_pg_identity(
            self.__experiment_name, self.__trial_name, self.__worker_index
        )
        self.__dist_env_resolved = False

        self.__clear_cache_frequency = timeutil.FrequencyControl(
            frequency_steps=self.config.cuda_cache_clear_freq
        )

        r = self.config.worker_info

        # recover info
        self.__recover_run = os.environ.get("REAL_RECOVER_RUN", "0") == "1"
        self.__recover_info = (
            recover.load_recover_info() if self.__recover_run else None
        )
        self.__recover_states_root = os.path.join(
            constants.RECOVER_ROOT, self.__experiment_name, self.__trial_name
        )
        self.__epoch_since_recover = 0
        self.__recover_first_epoch_done = False

        return r

    @property
    def _mp_rank(self) -> int:
        return constants.model_parallel_rank()

    @property
    def _pp_rank(self) -> int:
        return constants.pipe_parallel_rank()

    @property
    def _dp_rank(self) -> int:
        return constants.data_parallel_rank()

    @property
    def _pp_size(self) -> int:
        return constants.pipe_parallel_world_size()

    @property
    def _mp_size(self) -> int:
        return constants.model_parallel_world_size()

    @property
    def _dp_size(self) -> int:
        return constants.data_parallel_world_size()

    @property
    def _is_dp_head(self) -> bool:
        return self._mp_rank == 0 and self._pp_rank == self._pp_size - 1

    @property
    def _model(self) -> model_api.Model:
        return self.__models[constants.model_name()]

    @property
    def _interface(self) -> model_api.ModelInterface:
        return self.__interfaces[constants.model_name()]

    @property
    def _eval_dataloader(self) -> torch.utils.data.DataLoader:
        return self.__eval_dataloaders[constants.model_name()]

    @property
    def _module(self) -> torch.nn.Module | ReaLModel:
        return self.__unwrapped_models[constants.model_name()]

    @property
    def _backend(self) -> model_api.ModelBackend:
        return self.__backends[constants.model_name()]

    def __lazy_setup(self):
        # Add an additional subscript pattern for source RPCs.
        self.__has_dataset = False
        self.__dataset_dp_size = self.__dataset_dp_rank = 0
        sub_patterns = [s.id for s in self.config.shards]
        src_rpc = [rpc for rpc in self.config.model_rpcs if rpc.is_src][0]
        self.__src_rpc_model_name = src_rpc.model_name
        for s in self.config.shards:
            _pp_size = s.id.topo.get_dim("pipe")
            if not (s.id.mp_rank == 0 and s.id.pp_rank == _pp_size - 1):
                continue
            if src_rpc.model_name == s.id.model_name:
                self.__has_dataset = True
                self.__dataset_dp_size = s.id.topo.get_dim("data")
                self.__dataset_dp_rank = s.id.dp_rank
                sub_patterns.append(f"__data{self.__dataset_dp_rank}__")
                break

        # Build stream connecting with master workers.
        self.__stream = request_reply_stream.make_worker_stream(
            self.config.worker_info,
            idx=self.__worker_index,
        )

        self.__pg_info = global_comm.setup_global_comm(
            expr_name=self.__experiment_name,
            trial_name=self.__trial_name,
            worker_index=self.__worker_index,
            model_topos=self.config.model_topos,
            msid2mwid=self.config.msid2mwid,
        )

        self.__data_transfer_info = data_transfer_comm.setup_data_transfer(
            model_topos=self.config.model_topos,
            msid2mwid=self.config.msid2mwid,
            data_transfer_pairs=self.config.data_transfer_pairs,
        )

        self.__param_realloc_info = param_realloc_comm.setup_param_realloc(
            model_topos=self.config.model_topos,
            msid2mwid=self.config.msid2mwid,
            param_realloc_pairs=self.config.sync_param_pairs,
        )

        # logger.info(f"SetUp Information - Model worker index {self.__worker_index} located at "
        #             f"{socket.gethostname()} GPU {self.__pg_info.local_gpu_id}.")

        deepspeed.init_distributed()
        # self.logger.info("deepspeed init distributed on model worker")
        self.__device = torch.device("cuda:0")

        for model_name_, topo_ in self.config.model_topos.items():
            rpcs = [
                rpc for rpc in self.config.model_rpcs if rpc.model_name == model_name_
            ]
            assert len(rpcs) >= 1
            is_trainable_model = any(
                [
                    rpc.interface_type == dfg.ModelInterfaceType.TRAIN_STEP
                    for rpc in rpcs
                ]
            )
            param_realloc_comm.set_trainable(model_name_, is_trainable_model)
            constants.set_rank_mapping(model_name_, topo_, self.config.msid2mwid)
            grid = topology.ParallelGrid(
                topology=topo_,
                rank_mapping=constants.rank_mapping_of_model(model_name_),
                process_group=self.__pg_info.model_groups[model_name_],
            )
            constants.set_grid(model_name_, grid)

        # Set up training dataset for source RPCs.
        if self.__has_dataset:
            datasets = [
                data_api.make_dataset(
                    d,
                    self.config.seed,
                    self.__dataset_dp_rank,
                    self.__dataset_dp_size,
                    self.config.tokenizer_name_or_path,
                    self.config.worker_info.experiment_name,
                    self.config.worker_info.trial_name,
                    cache_root=(
                        None
                        if not self.config.use_dataset_cache
                        else self.config.dataset_cahce_root
                    ),
                )
                for d in self.config.datasets
            ]
            if len(self.config.datasets) == 1:
                self.__dataset = datasets[0]
            else:
                self.__dataset = torch.utils.data.ConcatDataset(datasets)
            self.__dataloader = data_api.make_dataloader(
                self.config.dataloader, self.__dataset
            )
            self.__dataset_n_seqs = 0
            for tmp_sample in self.__dataloader:
                self.__dataset_n_seqs += tmp_sample.bs

            self.__data_generator = enumerate(self.__dataloader)
            self.__dataset_batch_counter = None

            self.__dataset_epoch = (
                0 if not self.__recover_run else self.__recover_info.recover_start.epoch
            )
            self.__cur_sample: data_api.SequenceSample | None = None

        self.__models: Dict[ModelName, model_api.Model] = dict()
        self.__model_is_handle: Dict[ModelName, bool] = dict()
        self.__interfaces: Dict[ModelName, model_api.ModelInterface] = dict()
        self.__eval_dataloaders: Dict[ModelName, torch.utils.data.DataLoader] = dict()

        self.__backends: Dict[ModelName, model_api.ModelBackend] = dict()
        self.__unwrapped_models: Dict[ModelName, torch.nn.Module | ReaLModel] = dict()

        self.__backend_initialized: Dict[ModelName, bool] = dict()

        for s in self.config.shards:
            with constants.model_scope(s.id.model_name):
                self.__backend_initialized[s.id.model_name] = False
                tik = time.perf_counter()
                if self.__recover_run:
                    model_path = os.path.join(
                        self.__recover_states_root, "ckpt", s.id.model_name.role
                    )
                    s.model.args["model_path"] = model_path
                    s.model.args["init_critic_from_actor"] = False

                if constants.parallelism_rank() == 0:
                    self.logger.info(f"Making model {s.id.model_name}...")

                self.__models[s.id.model_name] = model = model_api.make_model(
                    s.model, name=s.id.model_name, device=self.__device
                )
                self.__unwrapped_models[s.id.model_name] = model.module
                if s.should_instantiate:
                    if isinstance(model.module, ReaLModel):
                        model.module.instantiate()
                    self.__model_is_handle[s.id.model_name] = False
                else:
                    self.__model_is_handle[s.id.model_name] = True
                self.__backends[s.id.model_name] = model_api.make_backend(s.backend)
                interface_impl = [
                    rpc.interface_impl
                    for rpc in self.config.model_rpcs
                    if rpc.model_name == s.id.model_name
                ]
                assert all(x == interface_impl[0] for x in interface_impl)
                self.__interfaces[s.id.model_name] = model_api.make_interface(
                    interface_impl[0]
                )

                if s.eval_datasets is not None and s.eval_dataloader is not None:
                    eval_datasets = [
                        data_api.make_dataset(
                            d,
                            self.config.seed,
                            s.id.dp_rank,
                            s.id.topo.get_dim("data"),
                            self.__models[s.id.model_name].tokenizer,
                            self.config.worker_info.experiment_name,
                            self.config.worker_info.trial_name,
                            cache_root=(
                                None
                                if not self.config.use_dataset_cache
                                else self.config.dataset_cahce_root
                            ),
                        )
                        for d in s.eval_datasets
                    ]
                    if len(eval_datasets) > 1:
                        eval_dataset = torch.utils.data.ConcatDataset(eval_datasets)
                    else:
                        eval_dataset = eval_datasets[0]
                    eval_dataloader = data_api.make_dataloader(
                        s.eval_dataloader, eval_dataset
                    )
                else:
                    eval_dataloader = None
                self.__eval_dataloaders[s.id.model_name] = eval_dataloader

        self.__request_cache = {}
        self.__ack_cache = {}

        self.__request_queue = queue.Queue(maxsize=8)
        self.__reply_queue = queue.Queue(maxsize=8)
        self.__request_sample_size = dict()

        # Storing data loaded from the dataset and outputs of the
        # model function call.
        self.__data_storage: Dict[int, data_api.SequenceSample] = {}

        # When entering a new epoch, the loaded data with the same id
        # may have not consumed in the previous epoch. We store them
        # temporarily in the following list.
        self.__early_arrived_data: List[data_api.SequenceSample] = []

        self.__data_sent_worker_indices: Dict[int, Dict[str, Set]] = (
            collections.defaultdict(lambda: collections.defaultdict(set))
        )
        self.__data_received_worker_indices: Dict[int, Dict[str, Set]] = (
            collections.defaultdict(lambda: collections.defaultdict(set))
        )

        self.__compute_input_queues = {
            model_name: dict(
                train_step=queue.Queue(4),
                inference=queue.Queue(4),
                generate=queue.Queue(4),
                evaluate=queue.Queue(4),
            )
            for model_name in self.__models.keys()
        }

    def prefetch_from_dataset(self):
        if self.__cur_sample is None:
            try:
                self.__dataset_batch_counter, self.__cur_sample = next(
                    self.__data_generator
                )
            except StopIteration:
                self.__dataset_epoch += 1
                if self.__recover_run:
                    self.__epoch_since_recover += 1
                    if self.__epoch_since_recover > 1:
                        self.__recover_first_epoch_done = True
                self.__data_generator = enumerate(self.__dataloader)
                self.__dataset_batch_counter, self.__cur_sample = next(
                    self.__data_generator
                )

    def __handle_one_rpc_hook(self, hook: str, hook_data: Any):
        tik = time.perf_counter()
        if hook == "data_transfer":
            self.__data_transfer_among_workers(hook_data)
        elif hook == "param_realloc":
            from_model_name: ModelName = hook_data["from_model_name"]
            to_model_name: ModelName = hook_data["to_model_name"]
            from_topo: topology.PipeModelDataParallelTopology = hook_data["from_topo"]
            to_topo: topology.PipeModelDataParallelTopology = hook_data["to_topo"]
            to_model_config = hook_data["to_model_config"]
            if from_model_name in self.__unwrapped_models:
                m = self.__unwrapped_models[from_model_name]
            else:
                m = self.__unwrapped_models[to_model_name]
            if not isinstance(m, ReaLModel):
                raise ValueError(
                    f"Model {from_model_name} (type={type(m)}) is not a ReaLModel, "
                    f"so it can't use parameter realloction."
                )
            try:
                new_layers, new_param, _ = m.build_reparallelized_layers_async(
                    from_model_name=from_model_name,
                    to_model_name=to_model_name,
                    from_topo=from_topo,
                    to_topo=to_topo,
                    to_model_config=to_model_config,
                    pg_info=self.__param_realloc_info,
                )
            except RuntimeError as e:
                if from_model_name in self.__unwrapped_models:
                    logger.error(f"from model error: {from_model_name}")
                if to_model_name in self.__unwrapped_models:
                    logger.info(f"to model error: {to_model_name}")
                raise e
            if (
                from_model_name in self.__models
                and not param_realloc_comm.is_trainable(from_model_name)
            ):
                self.__model_is_handle[from_model_name] = True
            if to_model_name in self.__models and param_realloc_comm.is_trainable(
                from_model_name
            ):
                self.__unwrapped_models[to_model_name].patch_reparallelization(
                    (new_layers, new_param), eta=hook_data["eta"]
                )
                self.__model_is_handle[to_model_name] = False
        elif hook == "offload":
            # NOTE: Profiling (or cuda synchronization) will cause an overhead ~0.5s.
            # with cuda_tmarked("offload", CUDATimeMarkType.mem_layout):
            m = self.__unwrapped_models[hook_data["model_name"]]
            if not isinstance(m, ReaLModel):
                logger.warning(
                    f"Model {hook_data['model_name']} (type={type(m)}) is not a ReaLModel, "
                    f"so it can't use offload."
                )
                return
            if not m._offloaded:
                m.async_offload()
        else:
            raise NotImplementedError(f"Unknown hook {hook}.")
        blogger.debug(
            f"Model worker {self.__worker_index} handle "
            f"RPC hook {hook} CPU time {time.perf_counter() - tik:.4f}s."
        )

    def handle_all_pre_hooks(self):
        # drain request queues, handle all pending hooks, then recover the queue
        cache = []
        while True:
            try:
                request, data, handled, res = self.__request_queue.get_nowait()
                request: request_reply_stream.Payload
                if not handled:
                    while len(request.pre_hooks) > 0:
                        assert len(request.pre_hooks) == len(request.pre_hook_data)
                        assert not handled and res is None
                        self.__handle_one_rpc_hook(
                            request.pre_hooks.pop(0),
                            request.pre_hook_data.pop(0),
                        )
                cache.append((request, data, handled, res))
            except queue.Empty:
                break

        for c in cache:
            self.__request_queue.put_nowait(c)

    def model_poll_step(
        self,
        request: request_reply_stream.Payload,
        data: Any,
        handled: bool,
        res: Optional[Any],
    ) -> worker_base.PollResult:
        tik = time.perf_counter()
        # self.logger.info(f"Model worker {self.__worker_index} #{request.handler}# "
        #                  f"start handle request *{request.handle_name}*, "
        #                  f"request_id {request.request_id}.")

        if isinstance(request.handler, str):
            assert request.handler == f"__data{self.__dataset_dp_rank}__"
            handler_model_name = self.__src_rpc_model_name
        else:
            handler_model_name = request.handler.model_name

        assert not handled and res is None, (
            handled,
            res,
            len(request.post_hooks),
        )

        with constants.model_scope(handler_model_name):
            res = None
            if request.handle_name == "empty":
                # Empty request is used for executing hooks,
                # e.g., data transfer, parameter syncrhonization.
                pass
            ############## initialization ##############
            elif request.handle_name == "initialize":
                assert not self.__model_is_handle[request.handler.model_name]
                self.__models[request.handler.model_name] = self._backend.initialize(
                    self._model, data
                )
                self.__backend_initialized[request.handler.model_name] = True
                # Offload this model after initialization if any MFC requires offloading.
                for rpc in self.config.model_rpcs:
                    if rpc.model_name != request.handler.model_name:
                        continue
                    if all(
                        not isinstance(hook, dfg.OffloadHook)
                        for hook in rpc._post_hooks
                    ):
                        continue
                    self.__unwrapped_models[request.handler.model_name].async_offload()
                    break
            elif request.handle_name == "model_config":
                if isinstance(
                    self.__unwrapped_models[request.handler.model_name],
                    ReaLModel,
                ):
                    res = self.__unwrapped_models[request.handler.model_name].config
            ############## data loading ##############
            elif request.handle_name == "fetch":
                fetched_data = self.__cur_sample.unpack() + self.__early_arrived_data
                if self.__recover_run and not self.__recover_first_epoch_done:
                    fetched_data = list(
                        filter(
                            lambda x: x.ids[0]
                            not in self.__recover_info.hash_vals_to_ignore,
                            fetched_data,
                        )
                    )

                # NOTE: Data used in the previous epoch may still exist in the storage
                # before executing the "clear_data_cache" command. We store data with
                # the same ID in the next epoch temporarily in early_arrived_data
                # to prevent them from being cleared in the future requests.
                # These data will be gradually filled into the storage.
                self.__early_arrived_data = []
                data_loaded = []
                for x in fetched_data:
                    if x.ids[0] in self.__data_storage:
                        self.__early_arrived_data.append(x)
                    else:
                        self.__data_storage[x.ids[0]] = x
                        data_loaded.append(x)

                if len(data_loaded) > 0:
                    meta_sample = data_api.SequenceSample.gather(data_loaded).meta()
                else:
                    meta_sample = None
                res = data_api.DataBatchMeta(
                    dp_rank=self._dp_rank,
                    meta_sample=meta_sample,
                    epoch=self.__dataset_epoch,
                    is_final_batch=(
                        self.__dataset_batch_counter == len(self.__dataloader) - 1
                    ),
                )
                self.__cur_sample = None
            elif request.handle_name == "spec":
                res = self.__dataset_n_seqs
            elif request.handle_name == "clear_data_cache":
                with cuda_tmarked("clear_data_cache", CUDATimeMarkType.misc):
                    ids = request.data
                    for _id in ids:
                        if _id in self.__data_storage:
                            del self.__data_storage[_id]
                        if _id in self.__data_sent_worker_indices:
                            del self.__data_sent_worker_indices[_id]
                        if _id in self.__data_received_worker_indices:
                            del self.__data_received_worker_indices[_id]
                    gc.collect()
                    if (
                        self.config.cuda_cache_cleanliness
                        and self.__clear_cache_frequency.check()
                    ):
                        st = time.monotonic()
                        gc.collect()
                        torch.cuda.empty_cache()
                        gc.collect()
                        et = time.monotonic()
                        blogger.debug(
                            f"Model worker {self.__worker_index} cleared cache in {et-st:.4f}s"
                        )
                dump_tmark_db(self.__worker_index)
                res = request_reply_stream.NoResponse()
            ############## computation function calls ##############
            elif request.handle_name in ["inference", "generate", "train_step"]:
                res = self.__handle_model_function_calls(request, data)
            elif request.handle_name == "evaluate":
                assert not self.__model_is_handle[
                    request.handler.model_name
                ], request.handler.model_name
                res = self._interface.evaluate(
                    self._model, self._eval_dataloader
                )  # -> Dict
            elif request.handle_name == "save":
                if not self.__model_is_handle[request.handler.model_name]:
                    self._interface.save(self._model, data)  # -> None
            else:
                raise NotImplementedError(
                    f"Unknown request type: {request.handle_name}."
                )

            if (
                request.handle_name in TIME_RECORD_RPCS
                and self._is_dp_head
                and self._dp_rank == 0
            ):
                blogger.info(
                    f"Model worker #{handler_model_name}# handle request *{request.handle_name}*"
                    f" in ${time.perf_counter() - tik:.4f}$s"
                )

        # Handle all post hooks right after the main computation
        if len(request.post_hooks) > 0:
            assert len(request.post_hooks) == len(request.post_hook_data)
            for hook, hook_data in zip(request.post_hooks, request.post_hook_data):
                self.__handle_one_rpc_hook(hook, hook_data)

        self.__reply_queue.put_nowait((request, res))
        sample_count = data.bs if isinstance(data, data_api.SequenceSample) else 1
        self.__request_sample_size[request.request_id] = sample_count

    @contextlib.contextmanager
    def __maybe_profile_rpc(self, rpc: dfg.MFCDef):
        # Whether to enable profiling is controlled by the following environment variables.
        _enable_profiler = os.getenv("REAL_DUMP_TRACE", "0") == "1"
        _enable_memory_dump = os.getenv("REAL_DUMP_MEMORY", "0") == "1"
        if _enable_memory_dump:
            torch.cuda.memory._record_memory_history()

        # pfer ca be a null context if enable_profiler is False
        pfer = get_pytorch_profiler(with_stack=True, enabled=_enable_profiler)
        pfer.__enter__()
        # The pytorch profiler will call cuda synchronize for us.
        profiler_tik = time.perf_counter()

        try:
            yield self
        finally:
            # Dump profiler results.
            pfer.__exit__(None, None, None)

            def _get_subdir(name):
                subdir = os.path.join(
                    constants.LOG_ROOT,
                    constants.experiment_name(),
                    constants.trial_name(),
                    name,
                    f"setup{self._setup_counter}",
                )
                os.makedirs(subdir, exist_ok=True)
                return subdir

            if _enable_profiler:
                if self._dp_rank == 0 and self._is_dp_head:
                    blogger.info(
                        f"RPC {rpc.name} execution time "
                        f"w/o external data processing: {time.perf_counter() - profiler_tik:.2f} secs."
                    )
                    collect_tik = time.perf_counter()
                    blogger.info(
                        f"Collecting system metrics from the profiler. "
                        "This may take for a while..."
                    )

                pfer.export_chrome_trace(
                    os.path.join(
                        _get_subdir("trace"), f"{rpc.name}_r{dist.get_rank()}.json"
                    )
                )
                if self._dp_rank == 0 and self._is_dp_head:
                    blogger.info(
                        f"System metrics collected. Time consumption:"
                        f" {time.perf_counter() - collect_tik:.2f} secs."
                    )
            if _enable_memory_dump:
                torch.cuda.memory._dump_snapshot(
                    os.path.join(
                        _get_subdir("gpuMemory"), f"{rpc.name}_r{dist.get_rank()}.pkl"
                    )
                )

    def __handle_model_function_calls(
        self, request: request_reply_stream.Payload, data: Any
    ):
        # Check that the model is instantiated and is not empty.
        assert not self.__model_is_handle[
            request.handler.model_name
        ], request.handler.model_name

        input_queue = self.__compute_input_queues[request.handler.model_name][
            request.handle_name
        ]
        rpc: dfg.MFCDef = next(
            rpc for rpc in self.config.model_rpcs if rpc.name == request.data
        )

        data: data_api.SequenceSample = input_queue.get_nowait()

        if self.config.profile_mode:
            data = self._interface.mock(request.handle_name, self._model, data)

        if rpc.input_key_remap:
            data.remap_keys_(rpc.input_key_remap)

        with self.__maybe_profile_rpc(rpc):
            if request.handle_name == "inference":
                res = self._interface.inference(
                    self._model, data, n_mbs=rpc.n_mbs
                )  # -> SequenceSample
            elif request.handle_name == "train_step":
                res = self._interface.train_step(
                    self._model, data, n_mbs=rpc.n_mbs
                )  # -> Dict
            elif request.handle_name == "generate":
                res = self._interface.generate(
                    self._model, data, n_mbs=rpc.n_mbs
                )  # -> SequenceSample
            else:
                raise NotImplementedError(f"Unknown MFC type: {request.handle_name}.")

        if isinstance(res, data_api.SequenceSample) and rpc.output_key_remap:
            res.remap_keys_(rpc.output_key_remap)

        # Store data into storage.
        if self._is_dp_head and isinstance(res, data_api.SequenceSample):
            for x in res.unpack():
                # The input data must exist in the storage, otherwise
                # the model function call will not run.
                self.__data_storage[x.ids[0]].update_(x)

        # Only return meta data back to the master worker.
        if isinstance(res, data_api.SequenceSample):
            res = res.meta()

        # Monitoring info. There's an all-gather and an all-reduce
        # over the parallelism group in this function.
        self.__log_gpu_stats(request)
        return res

    @cuda_tmark("data_transfer", CUDATimeMarkType.comm)
    def __data_transfer_among_workers(self, hook_data: Dict[str, Any]):
        meta_sample = hook_data["meta_sample"]
        comm_plan = data_transfer_comm.derive_data_transfer_plan(
            keys=hook_data["keys"],
            global_ids=meta_sample.ids,
            consumer_name=hook_data["target"],
            consumer_mapping=hook_data["target_mapping"],
            producer_names=hook_data["producer_names"],
            producer_mappings=hook_data["producer_mappings"],
            data_transfer_info=self.__data_transfer_info,
        )

        data_transfer_comm.run_data_transfer(
            comm_plan=comm_plan,
            meta_samples={x.ids[0]: x for x in meta_sample.unpack()},
            storage=self.__data_storage,
            sent_worker_idx_table=self.__data_sent_worker_indices,
            received_worker_idx_table=self.__data_received_worker_indices,
        )

        if hook_data["target"] in self.__models:
            with constants.model_scope(hook_data["target"]):
                local_ids = [
                    meta_sample.ids[i]
                    for i in hook_data["target_mapping"][self._dp_rank]
                ]
            r = data_api.SequenceSample.gather(
                [self.__data_storage[_id] for _id in local_ids],
                keys=meta_sample.keys,
            )
            self.__compute_input_queues[hook_data["target"]][
                hook_data["handle_name"]
            ].put_nowait(r)

    @cuda_tmark("post_response", CUDATimeMarkType.misc)
    def maybe_post_responses(self):
        ready_to_post = []
        try:
            request, res = self.__reply_queue.get_nowait()
            ready_to_post.append((request, res))
        except queue.Empty:
            pass

        batch_size = sample_size = 0
        for request, res in ready_to_post:
            # For some requests, do not respond to the master worker.
            if isinstance(res, request_reply_stream.NoResponse):
                continue
            request: request_reply_stream.Payload
            reply = request_reply_stream.Payload(
                handler="master",
                request_id=request.request_id,
                handle_name=request.handle_name,
                data=res,
            )
            self.__stream.post(reply)
            # logger.info(f"handle_name {request.handle_name} Posted req id = {request.request_id}")
            sample_size += self.__request_sample_size.pop(request.request_id)
            batch_size += 1
        return worker_base.PollResult(sample_count=sample_size, batch_count=batch_size)

    def __maybe_receive_one_request(self):
        try:
            r: request_reply_stream.Payload = self.__stream.poll()
            if r.handle_name == "ack":
                self.__ack_cache[r.request_id] = r
            else:
                self.__stream.post(
                    request_reply_stream.Payload(
                        handler="master",
                        request_id=r.syn_reply_id,
                        handle_name="syn",
                    ),
                )
                self.__request_cache[r.ack_reply_id] = r
        except request_reply_stream.NoMessage:
            return

    @cuda_tmark("receive_request", CUDATimeMarkType.misc)
    def maybe_receive_requests(self):
        self.__maybe_receive_one_request()
        cur_ack_ids = list(self.__ack_cache.keys())
        for ack_id in cur_ack_ids:
            if ack_id in self.__request_cache:
                self.__ack_cache.pop(ack_id)
                req = self.__request_cache.pop(ack_id)
                self.__request_queue.put_nowait((req, req.data, False, None))

    def _poll(self):
        if not self.__dist_env_resolved:
            self.__lazy_setup()
            pynvml.nvmlInit()
            self.__nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(
                self.__pg_info.local_gpu_id
            )
            self.__dist_env_resolved = True

        if self.__has_dataset:
            self.prefetch_from_dataset()

        self.maybe_receive_requests()

        # Prioritize the reset request.
        for _ in range(self.__request_queue.qsize()):
            request, data, handled, res = self.__request_queue.get_nowait()
            if request.handle_name == "reset":
                return self.__experiment_complete_exit()
            self.__request_queue.put_nowait((request, data, handled, res))

        # NOTE: We ensure that all model workers have the same set of requests
        # at any time through a TCP-like protocol, i.e., req -> ack -> syn -> resp.
        # Each request is composed of pre-hooks, the main request, and post-hooks.
        # We execute all pre-hooks first because they involve data transfer
        # among workers. Executing them first avoids blocking MFCs that require
        # data from the same set of GPUs but are executed on disjoint GPUs.
        self.handle_all_pre_hooks()

        # Execute one MFC them immediately return the result, such that
        # we can correctly log the time consumption in the master worker.
        try:
            request, data, handled, res = self.__request_queue.get_nowait()
            self.model_poll_step(request, data, handled, res)
        except queue.Empty:
            pass

        r = self.maybe_post_responses()
        return r

    def __experiment_complete_exit(self):
        self.__stream.close()

        self.__unwrapped_models.clear()

        # Calling backend.destroy removes all hooks and releases the memory.
        for model_name, backend in self.__backends.items():
            backend.destroy(self.__models[model_name])

        self.__models.clear()
        self.__backends.clear()
        self.__interfaces.clear()
        self.__data_storage.clear()

        # Reset model worker states.
        self.__dist_env_resolved = False

        before_mem = pynvml.nvmlDeviceGetMemoryInfo(self.__nvml_handle).used

        constants.reset_run()
        topology.destroy_all_comm_groups()
        cuda_graph.destroy_all()

        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        gc.collect()

        # Record memory.
        after_mem = pynvml.nvmlDeviceGetMemoryInfo(self.__nvml_handle).used
        blogger.debug(
            f"GPU memory used upon experiment complete: "
            f"{before_mem/1024**2:.2f}MB -> {after_mem / 1024**2:.2f}MB"
        )

        self.__nvml_handle = None
        try:
            pynvml.nvmlShutdown()
        except pynvml.nvml.NVMLError_Uninitialized:
            pass
        self.pause()
        return worker_base.PollResult(sample_count=0, batch_count=0)

    def __recover_save(self):
        # store model and dataset states for recover
        if self.__dist_env_resolved:

            for model_name, model in self.__models.items():
                if self.__model_is_handle[model_name]:
                    continue
                constants._model_name = None  # force quit model_scope
                with constants.model_scope(model_name):
                    ckpt_save_dir = os.path.join(
                        self.__recover_states_root, "ckpt", model_name.role
                    )
                    # replace old recover ckpt
                    logger.info(
                        f"saving model {model_name} ckpt for recover at {ckpt_save_dir}. "
                        f"epoch {model.version.epoch}, epoch_step {model.version.epoch_step}, "
                        f"global step {model.version.global_step}"
                    )
                    if self.__has_dataset:
                        logger.info(
                            f"Dataset info: " f"dataset epoch {self.__dataset_epoch}"
                        )
                    self._interface.save(model, ckpt_save_dir)
                    logger.info(f"saving done.")

    def _exit_hook(self, exit_status: worker_base.WorkerServerStatus):
        logger.info(
            f"Model worker {self.__worker_index} exit with status {exit_status}."
        )
        if os.environ.get("REAL_SAVE_RECOVER_STATES", "0") == "0":
            return
        if exit_status == worker_base.WorkerServerStatus.ERROR:
            try:
                sleep_time = 600
                current_sleep_time = 0
                while current_sleep_time < sleep_time:
                    logger.info(
                        f"ERROR exit, waited {current_sleep_time} s for interruption ..."
                    )
                    time.sleep(10)
                    current_sleep_time += 10
            except KeyboardInterrupt:
                logger.info("Received SIGINT, starting recover save")

        self.__recover_save()

    def __log_gpu_stats(self, request: request_reply_stream.Payload):
        # Log GPU utilization and memory statistics.
        utilization = pynvml.nvmlDeviceGetUtilizationRates(self.__nvml_handle)  # bytes
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.__nvml_handle)  # bytes
        torch_mem_stats = torch.cuda.memory_stats(0)

        # All-gather hostname, gpu ID, and stats.
        hostname = socket.gethostname()
        hostname_len = len(hostname)
        assert hostname_len < 64, "hostname should not have more than 64 chars"
        # Encode hostnames into long.
        hostname_np = np.fromstring(
            hostname + "x" * (64 - len(hostname)), dtype=np.int64
        )
        local_mem_stats = torch.tensor(
            [hostname_len, self.__pg_info.local_gpu_id]
            + hostname_np.tolist()
            + [
                torch_mem_stats["allocated_bytes.all.peak"],
                torch_mem_stats["reserved_bytes.all.peak"],
                memory_info.used,
            ],
            dtype=torch.long,
            device="cuda",
        )  # length 2 + 8 + 3 = 13
        mem_stats = local_mem_stats.new_zeros(
            size=(
                dist.get_world_size(constants.parallelism_group()),
                local_mem_stats.shape[0],
            )
        )
        # All-gather memory stats.
        dist.all_gather_into_tensor(
            mem_stats, local_mem_stats, group=constants.parallelism_group()
        )
        mem_stats = mem_stats.cpu().numpy()

        # All-reduce utilization.
        gpu_compute_util = torch.tensor(
            utilization.gpu, dtype=torch.float32, device="cuda"
        )
        dist.all_reduce(gpu_compute_util, group=constants.parallelism_group())
        gpu_compute_util = gpu_compute_util.item() / dist.get_world_size(
            constants.parallelism_group()
        )

        def _decode_hostname(idx):
            hn_np = mem_stats[idx, 2 : 2 + 8]
            l = mem_stats[idx, 0]
            return hn_np.tobytes().decode("utf-8")[:l]

        def _decode_gpu_id(idx):
            return f"{_decode_hostname(idx)}:{mem_stats[idx, 1]}"

        max_used_gpu_id = _decode_gpu_id(np.argmax(mem_stats[:, -1]))
        max_reserved_gpu_id = _decode_gpu_id(np.argmax(mem_stats[:, -2]))
        max_tensor_gpu_id = _decode_gpu_id(np.argmax(mem_stats[:, -3]))

        # NOTE: We only log the peak memory because it's
        # the most important for detecting OOM issues.
        headers = [
            " ",
            "TotalMem",
            "PeakUsedMem",
            "PeakTensorMem",
            "PeakReservedMem",
            "MaxMemUtil",
            "AvgComputeUtil",
        ]
        line1 = [
            "Value",
            f"{memory_info.total / 1024**2:.2f}MB",
            f"{max(mem_stats[:, -1]) / 1024**2:.2f}MB",
            f"{max(mem_stats[:, -3]) / 1024**2:.2f}MB",
            f"{max(mem_stats[:, -2]) / 1024**2:.2f}MB",
            f"{max(mem_stats[:, -1]) / memory_info.total * 100:.2f}%",
            f"{gpu_compute_util:.2f}%",
        ]
        line2 = [
            "GPU ID",
            "-",
            max_used_gpu_id,
            max_tensor_gpu_id,
            max_reserved_gpu_id,
            max_used_gpu_id,
            "-",
        ]

        if self._dp_rank == 0 and self._is_dp_head:
            logger.info(
                f"Aggregated GPU memory stats after MFC `{request.handle_name}`"
                f" within model `{request.handler.model_name}`:\n"
                + tabulate.tabulate(
                    [headers, line1, line2], headers="firstrow", tablefmt="fancy_grid"
                )
            )
