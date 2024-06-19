from typing import Any, Dict, List, Optional, Set, Tuple
import collections
import gc
import itertools
import multiprocessing as mp
import os
import queue
import socket
import time
import uuid

import deepspeed
import numpy as np
import pynvml
import torch
import torch.distributed as dist
import torch.utils.data

from realhf.api.core.config import ModelName
from realhf.base import (
    constants,
    gpu_utils,
    logging,
    namedarray,
    numpy_utils,
    recover,
    seeding,
    timeutil,
    topology,
)
from realhf.base.monitor import (
    cuda_tmark,
    cuda_tmarked,
    CUDATimeMarkType,
    dump_tmark_db,
    gpu_utilization_monitor,
)
from realhf.impl.model.nn.real_llm_api import ReaLModel
from realhf.system import request_reply_stream, worker_base
import realhf.api.core.dfg as dfg
import realhf.api.core.system_api as system_api
import realhf.impl.model.comm.data_transfer as data_transfer_comm
import realhf.impl.model.comm.global_comm as global_comm
import realhf.impl.model.comm.param_realloc as param_realloc_comm

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


def get_pytorch_profiler(with_stack):
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    )


class NoRequestToHandle(Exception):
    pass


class ModelWorker(worker_base.Worker):

    def _configure(self, cfg: system_api.ModelWorker):
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

        self.config.model_rpcs, _ = dfg.build_graph(self.config.model_rpcs)
        self.data2required_rpc_names = self.config.model_rpcs[
            0
        ].data2required_rpc_names

        # NOTE: here worker_index is different from peer/ddp rank
        self.__worker_index = cfg.worker_info.worker_index

        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
        torch.backends.cudnn.deterministic = cfg.cudnn_deterministic

        seeding.set_random_seed(cfg.seed)

        # Reveal DDP identity of this worker to world.
        gpu_utils.reveal_ddp_identity(
            self.__experiment_name, self.__trial_name, self.__worker_index
        )
        self.__dist_env_resolved = False

        self.__clear_cache_frequency = timeutil.FrequencyControl(
            frequency_steps=self.config.cuda_cache_clear_freq
        )

        r = self.config.worker_info

        # log info
        self.__total_time = 0.01

        # recover info
        self.__recover_run = os.environ.get("RECOVER_RUN", "0") == "1"
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
                rpc
                for rpc in self.config.model_rpcs
                if rpc.model_name == model_name_
            ]
            assert len(rpcs) >= 1
            is_trainable_model = any(
                [
                    rpc.interface_type == dfg.ModelInterfaceType.TRAIN_STEP
                    for rpc in rpcs
                ]
            )
            param_realloc_comm.set_trainable(model_name_, is_trainable_model)
            constants.set_rank_mapping(
                model_name_, topo_, self.config.msid2mwid
            )
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
                self.__dataset_n_seqs += len(
                    data_api.split_sequences(tmp_sample)
                )

            self.__data_generator = enumerate(self.__dataloader)
            self.__dataset_batch_counter = None

            self.__dataset_epoch = (
                0
                if not self.__recover_run
                else self.__recover_info.recover_start.epoch
            )
            self.__cur_sample = None
            self.__fetched_sample_cache = []

        self.__models: Dict[ModelName, model_api.Model] = dict()
        self.__model_is_handle: Dict[ModelName, bool] = dict()
        self.__interfaces: Dict[ModelName, model_api.ModelInterface] = dict()
        self.__eval_dataloaders: Dict[
            ModelName, torch.utils.data.DataLoader
        ] = dict()

        self.__backends: Dict[ModelName, model_api.ModelBackend] = dict()
        self.__unwrapped_models: Dict[
            ModelName, torch.nn.Module | ReaLModel
        ] = dict()

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
                self.__backends[s.id.model_name] = model_api.make_backend(
                    s.backend
                )
                interface_impl = [
                    rpc.interface_impl
                    for rpc in self.config.model_rpcs
                    if rpc.model_name == s.id.model_name
                ]
                assert all(x == interface_impl[0] for x in interface_impl)
                self.__interfaces[s.id.model_name] = model_api.make_interface(
                    interface_impl[0]
                )

                if (
                    s.eval_datasets is not None
                    and s.eval_dataloader is not None
                ):
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
                        eval_dataset = torch.utils.data.ConcatDataset(
                            eval_datasets
                        )
                    else:
                        eval_dataset = eval_datasets[0]
                    eval_dataloader = data_api.make_dataloader(
                        s.eval_dataloader, eval_dataset
                    )
                else:
                    eval_dataloader = None
                self.__eval_dataloaders[s.id.model_name] = eval_dataloader

        self.__request_cache = []
        self.__ack_cache = {}

        self.__request_queue = queue.Queue(maxsize=8)
        self.__reply_queue = queue.Queue(maxsize=8)
        self.__request_sample_size = dict()

        # Storing model function call outputs. Model worker will serve as
        # the data producer when other model workers require this data.
        self.__data_owner_storage: Dict[int, Dict[str, torch.Tensor]] = (
            collections.defaultdict(dict)
        )
        self.__data_receive_cache: Dict[int, Dict[str, torch.Tensor]] = (
            collections.defaultdict(dict)
        )

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

        # A monitoring process.
        self.__gpu_util_mp = mp.Process(
            target=gpu_utilization_monitor, args=(self.__worker_index, 20, 7200)
        )
        # self.__gpu_util_mp.start()

    def __prefetch_from_dataset(self):
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
        if hook == "data_transfer":
            self.__data_transfer_among_workers(hook_data)
        elif hook == "param_realloc":
            from_model_name: ModelName = hook_data["from_model_name"]
            to_model_name: ModelName = hook_data["to_model_name"]
            from_topo: topology.PipeModelDataParallelTopology = hook_data[
                "from_topo"
            ]
            to_topo: topology.PipeModelDataParallelTopology = hook_data[
                "to_topo"
            ]
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
            if from_model_name in self.__models:
                self.__model_is_handle[from_model_name] = True
            if to_model_name in self.__models:
                self.__unwrapped_models[to_model_name].patch_reparallelization(
                    (new_layers, new_param)
                )
                self.__model_is_handle[to_model_name] = False
        elif hook == "offload":
            with cuda_tmarked("offload", CUDATimeMarkType.mem_layout):
                m = self.__unwrapped_models[hook_data["model_name"]]
                if not isinstance(m, ReaLModel):
                    raise ValueError(
                        f"Model {from_model_name} (type={type(m)}) is not a ReaLModel, "
                        f"so it can't use offload."
                    )
                m.async_offload()
        else:
            raise NotImplementedError(f"Unknown hook {hook}.")

    def __handle_all_pre_hooks(self):
        # drain request queues, handle all pending hooks, then recover the queue
        cache = []
        while True:
            try:
                request, data, handled, res = self.__request_queue.get_nowait()
                request: request_reply_stream.Payload
                if not handled:
                    while len(request.pre_hooks) > 0:
                        assert len(request.pre_hooks) == len(
                            request.pre_hook_data
                        )
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

    def __model_poll_step(
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

        # handle post hooks
        if handled and len(request.post_hooks) > 0:
            assert len(request.post_hooks) == len(request.post_hook_data)
            self.__handle_one_rpc_hook(
                request.post_hooks.pop(0), request.post_hook_data.pop(0)
            )
            self.__request_queue.put_nowait((request, data, True, res))
            return

        if handled and len(request.post_hooks) == 0:
            self.__reply_queue.put_nowait((request, res))
            # self.logger.info(f"Model worker {self.__worker_index} #{request.handler}# "
            #                          f"finish handling request *{request.handle_name}*, "
            #                          f"request_id {request.request_id}.")
            sample_count = (
                data.length(0) if isinstance(data, namedarray.NamedArray) else 1
            )
            self.__request_sample_size[request.request_id] = sample_count
            return

        assert not handled and res is None, (
            handled,
            res,
            len(request.post_hooks),
        )

        with constants.model_scope(handler_model_name):
            res = None
            if request.handle_name == "empty":
                # Empty request is used for executing hooks, e.g., data transfer, parameter syncrhonization.
                pass
            ############## initialization ##############
            elif request.handle_name == "initialize":
                assert not self.__model_is_handle[request.handler.model_name]
                self.__models[request.handler.model_name] = (
                    self._backend.initialize(self._model, data)
                )
                self.__backend_initialized[request.handler.model_name] = True
            elif request.handle_name == "model_config":
                if isinstance(
                    self.__unwrapped_models[request.handler.model_name],
                    ReaLModel,
                ):
                    res = self.__unwrapped_models[
                        request.handler.model_name
                    ].config
            ############## data loading ##############
            elif request.handle_name == "fetch":
                fetched_data = data_api.split_sequences(self.__cur_sample)
                if self.__recover_run and not self.__recover_first_epoch_done:
                    fetched_data = list(
                        filter(
                            lambda x: hash(x)
                            not in self.__recover_info.hash_vals_to_ignore,
                            fetched_data,
                        )
                    )

                seqlens = [x.metadata["seqlens"][0] for x in fetched_data]
                res = data_api.DataBatchMeta(
                    dp_rank=self._dp_rank,
                    seqlens=seqlens,
                    keys=list(self.__cur_sample.keys()),
                    epoch=self.__dataset_epoch,
                    is_final_batch=(
                        self.__dataset_batch_counter
                        == len(self.__dataloader) - 1
                    ),
                    hash_vals=[hash(x) for x in fetched_data],
                )
                self.__fetched_sample_cache += fetched_data
                self.__cur_sample = None
            elif request.handle_name == "store":
                buffer_indices = request.data
                assert len(buffer_indices) == len(
                    self.__fetched_sample_cache
                ), (
                    len(buffer_indices),
                    len(self.__fetched_sample_cache),
                )
                for buf_idx, x in zip(
                    buffer_indices, self.__fetched_sample_cache
                ):
                    for k, v in x.items():
                        assert v.device == torch.device("cpu")
                        self.__data_owner_storage[buf_idx][k] = v
                self.__fetched_sample_cache.clear()
            elif request.handle_name == "spec":
                res = self.__dataset_n_seqs
            elif request.handle_name == "clear_data_cache":
                with cuda_tmarked("clear_data_cache", CUDATimeMarkType.misc):
                    buf_indices = request.data
                    for buf_idx in buf_indices:
                        if buf_idx in self.__data_owner_storage:
                            del self.__data_owner_storage[buf_idx]
                        if buf_idx in self.__data_receive_cache:
                            del self.__data_receive_cache[buf_idx]
                        if buf_idx in self.__data_sent_worker_indices:
                            del self.__data_sent_worker_indices[buf_idx]
                        if buf_idx in self.__data_received_worker_indices:
                            del self.__data_received_worker_indices[buf_idx]
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

        self.__request_queue.put_nowait((request, data, True, res))

    def __handle_model_function_calls(
        self, request: request_reply_stream.Payload, data: Any
    ):
        assert not self.__model_is_handle[
            request.handler.model_name
        ], request.handler.model_name
        input_queue = self.__compute_input_queues[request.handler.model_name][
            request.handle_name
        ]
        data, buffer_indices, seqlens, output_key_remap = (
            input_queue.get_nowait()
        )
        data: namedarray.NamedArray
        data.register_metadata(seqlens=seqlens)
        if request.handle_name == "inference":
            res = self._interface.inference(self._model, data)  # -> NamedArray
        elif request.handle_name == "train_step":
            res = self._interface.train_step(self._model, data)  # -> Dict
        elif request.handle_name == "generate":
            res = self._interface.generate(self._model, data)  # -> NamedArray

        if res is not None and isinstance(res, namedarray.NamedArray):
            new_res = {}
            for k, v in res.items():
                if k in output_key_remap:
                    new_res[output_key_remap[k]] = v
                else:
                    new_res[k] = v
            new_res = namedarray.from_dict(new_res)
            new_res.register_metadata(**res.metadata)
            res = new_res, buffer_indices, seqlens

        utilization = pynvml.nvmlDeviceGetUtilizationRates(self.__nvml_handle)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.__nvml_handle)
        total_memory = memory_info.total / (
            1024**2
        )  # Convert bytes to megabytes
        used_memory = memory_info.used / (1024**2)
        memory_usage_percentage = (used_memory / total_memory) * 100
        logger.info(
            f"Worker Index {self.__worker_index}, GPU {self.__pg_info.local_gpu_id}: "
            f"Compute Utilization - {utilization.gpu}%, "
            f"Total Memory - {total_memory:.2f}MB, Used Memory - {used_memory:.2f}MB, "
            f"Memory Usage - {memory_usage_percentage:.2f}%"
        )
        return res

    @cuda_tmark("data_transfer", CUDATimeMarkType.comm)
    def __data_transfer_among_workers(self, hook_data: Dict[str, Any]):

        comm_plan = data_transfer_comm.derive_data_transfer_plan(
            keys=hook_data["keys"],
            global_buffer_indices=hook_data["buffer_indices"],
            global_seqlens=hook_data["seqlens"],
            consumer_name=hook_data["target"],
            consumer_mapping=hook_data["target_mapping"],
            producer_names=hook_data["producer_names"],
            producer_mappings=hook_data["producer_mappings"],
            data_transfer_info=self.__data_transfer_info,
        )

        data = dict()
        for step in comm_plan:
            if (
                isinstance(step, data_transfer_comm.DataTransferReceiverStep)
                and step.rank == dist.get_rank()
            ):
                if isinstance(
                    self.__unwrapped_models[hook_data["target"]], ReaLModel
                ):
                    vocab_size = self.__unwrapped_models[
                        hook_data["target"]
                    ].config.vocab_size
                else:
                    vocab_size = None
                buf_indices = step.buf_indices
                seqlens = step.seqlens
                if step.src == dist.get_rank():
                    for buf_idx in buf_indices:
                        self.__data_owner_storage[buf_idx][step.key] = (
                            self.__data_owner_storage[buf_idx][step.key].to(
                                self.__device
                            )
                        )
                    vs = torch.cat(
                        [
                            self.__data_owner_storage[buf_idx][step.key]
                            for buf_idx in buf_indices
                        ],
                        dim=0,
                    )
                else:
                    all_sent_dst_ranks = [
                        self.__data_received_worker_indices[buf_idx][step.key]
                        for buf_idx in buf_indices
                    ]
                    if all(
                        set(step.dst_ranks).issubset(set(sent_dst_ranks))
                        for sent_dst_ranks in all_sent_dst_ranks
                    ):
                        vs = torch.cat(
                            [
                                self.__data_receive_cache[buf_idx][step.key]
                                for buf_idx in buf_indices
                            ],
                            dim=0,
                        )
                    else:
                        total_len = 0
                        for seqlen in seqlens:
                            shape = data_api.get_shape_from_key_and_seqlen(
                                step.key, seqlen, vocab_size
                            )
                            total_len += int(np.prod(shape))
                        dtype = data_api.get_dtype_from_key(step.key)
                        buf = torch.zeros(
                            (total_len,),
                            dtype=dtype,
                            device=torch.cuda.current_device(),
                        )
                        # print(f"{dist.get_rank()} recv {step.key} from {step.src} with shape {buf.shape}")
                        dist.broadcast(buf, src=step.src, group=step.group)
                        vs = buf.clone().view(-1, *shape[1:])
                        for buf_idx in buf_indices:
                            self.__data_received_worker_indices[buf_idx][
                                step.key
                            ].union(step.dst_ranks)
                offset = 0
                for seqlen, buf_idx in zip(seqlens, buf_indices):
                    shape = data_api.get_shape_from_key_and_seqlen(
                        step.key, seqlen, vocab_size
                    )
                    v = vs[offset : offset + shape[0]]
                    offset += shape[0]
                    data[(buf_idx, step.key)] = (v, seqlen)
                    if (
                        step.key not in self.__data_owner_storage[buf_idx]
                        and step.key not in self.__data_receive_cache[buf_idx]
                    ):
                        self.__data_receive_cache[buf_idx][step.key] = v

            if (
                isinstance(step, data_transfer_comm.DataTransferSenderStep)
                and step.rank == dist.get_rank()
            ):
                buf_indices = step.buf_indices
                all_sent_dst_ranks = [
                    self.__data_sent_worker_indices[buf_idx][step.key]
                    for buf_idx in buf_indices
                ]
                if all(
                    set(step.dst_ranks).issubset(set(sent_dst_ranks))
                    for sent_dst_ranks in all_sent_dst_ranks
                ):
                    pass
                else:
                    for buf_idx in buf_indices:
                        self.__data_owner_storage[buf_idx][step.key] = (
                            self.__data_owner_storage[buf_idx][step.key].to(
                                self.__device
                            )
                        )
                    vs = torch.cat(
                        [
                            self.__data_owner_storage[buf_idx][step.key]
                            for buf_idx in buf_indices
                        ],
                        dim=0,
                    )
                    if vs.dtype != data_api.get_dtype_from_key(step.key):
                        raise ValueError(
                            f"Infered dtype of {step.key} ({data_api.get_dtype_from_key(step.key)})"
                            f" is not equal to the actual dtype ({vs.dtype}). "
                            "Is it correctly set in the dataset implementation?"
                        )
                    # print(f"{dist.get_rank()} send {step.key} to {step.dst_ranks} with shape {vs.shape}")
                    dist.broadcast(vs, src=step.rank, group=step.group)
                    for buf_idx in buf_indices:
                        self.__data_sent_worker_indices[buf_idx][
                            step.key
                        ].union(step.dst_ranks)

        if len(data) > 0:
            local_buffer_indices = sorted(
                list(set([buf_idx for buf_idx, _ in data.keys()]))
            )
            local_keys = list(set([key for _, key in data.keys()]))

            local_seqlens = []
            for buf_idx in local_buffer_indices:
                local_seqlens.append(data[(buf_idx, local_keys[0])][1])

            input_key_remap = hook_data["input_key_remap"]
            _data = []
            for buf_idx in local_buffer_indices:
                d = {}
                for k in local_keys:
                    v, seqlen = data[(buf_idx, k)]
                    if k in input_key_remap:
                        k = input_key_remap[k]
                    d[k] = v
                x = namedarray.from_dict(d)
                x.register_metadata(seqlens=[seqlen])
                _data.append(x)
            r = data_api.gather_sequences(_data)
            self.__compute_input_queues[hook_data["target"]][
                hook_data["handle_name"]
            ].put_nowait(
                (
                    r,
                    local_buffer_indices,
                    local_seqlens,
                    hook_data["output_key_remap"],
                )
            )

    def __post_one_response(self, request: request_reply_stream.Payload, res):
        if isinstance(res, tuple) and isinstance(res[0], namedarray.NamedArray):
            res, buffer_indices, seqlens = res
            reply = request_reply_stream.Payload(
                handler="master",
                request_id=request.request_id,
                handle_name=request.handle_name,
                data={
                    "keys": list(res.keys()),
                    "seqlens": res.metadata["seqlens"],
                    "buffer_indices": list(buffer_indices),
                },
            )
        else:
            reply = request_reply_stream.Payload(
                handler="master",
                request_id=request.request_id,
                handle_name=request.handle_name,
                data=res,
            )

        self.__stream.post(reply)
        # logger.info(f"handle_name {request.handle_name} Posted req id = {request.request_id}")

        if isinstance(request.handler, system_api.ModelShardID) and isinstance(
            res, namedarray.NamedArray
        ):
            with constants.model_scope(request.handler.model_name):
                if self._is_dp_head:
                    xs = data_api.split_sequences(res)
                    for buffer_idx, x in zip(buffer_indices, xs):
                        for k, v in x.items():
                            self.__data_owner_storage[buffer_idx][k] = v

    @cuda_tmark("post_response", CUDATimeMarkType.misc)
    def __maybe_post_responses(self):
        ready_to_post = []
        try:
            request, res = self.__reply_queue.get_nowait()
            ready_to_post.append((request, res))
        except queue.Empty:
            pass

        batch_size = sample_size = 0
        for request, res in ready_to_post:
            request: request_reply_stream.Payload
            self.__post_one_response(request, res)
            sample_size += self.__request_sample_size.pop(request.request_id)
            batch_size += 1
        return worker_base.PollResult(
            sample_count=sample_size, batch_count=batch_size
        )

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
                self.__request_cache.append(r)
        except request_reply_stream.NoMessage:
            return

    @cuda_tmark("receive_request", CUDATimeMarkType.misc)
    def __maybe_receive_requests(self):
        for _ in range(8):
            self.__maybe_receive_one_request()

        while len(self.__request_cache) > 0:
            request: request_reply_stream.Payload = self.__request_cache[0]
            while request.ack_reply_id not in self.__ack_cache:
                self.__maybe_receive_one_request()

            self.__ack_cache.pop(request.ack_reply_id)
            self.__request_cache.pop(0)

            self.__request_queue.put_nowait(
                (request, request.data, False, None)
            )

    def _poll(self):
        if not self.__dist_env_resolved:
            self.__lazy_setup()
            pynvml.nvmlInit()
            self.__nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(
                self.__pg_info.local_gpu_id
            )
            self.__dist_env_resolved = True

        if self.__has_dataset:
            self.__prefetch_from_dataset()

        st = time.monotonic()
        self.__maybe_receive_requests()

        # NOTE: We ensure that all model workers have the same set of requests
        # at any time through a TCP-like protocol, i.e., req -> ack -> syn -> resp.
        # Each request is composed of pre-hooks, the main request, and post-hooks.
        # We execute all pre-hooks first because they involve data transfer
        # among workers. Then, we round-robinly execute hooks and requests.
        # These are designed to prevent mutual blocking when different requests
        # are handled in different but intersected sets of model workers.
        # E.g., If we have a request A and a request B, the execution order will be
        # A.pre_hook -> B.pre_hook -> A -> B -> A.post_hook -> B.post_hook.
        self.__handle_all_pre_hooks()
        for _ in range(16):
            try:
                request, data, handled, res = self.__request_queue.get_nowait()
                self.__model_poll_step(request, data, handled, res)
            except queue.Empty:
                break

        r = self.__maybe_post_responses()

        t = time.monotonic() - st
        self.__total_time += t
        return r

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
                            f"Dataset info: "
                            f"dataset epoch {self.__dataset_epoch}"
                        )
                    self._interface.save(model, ckpt_save_dir)
                    logger.info(f"saving done.")

    def _exit_hook(self, exit_status: worker_base.WorkerServerStatus):
        logger.info(
            f"Model worker {self.__worker_index} exit with status {exit_status}."
        )
        if os.environ.get("SAVE_RECOVER_STATES", "0") == "0":
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
