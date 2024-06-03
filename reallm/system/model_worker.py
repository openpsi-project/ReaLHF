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
import torch
import torch.distributed as dist
import torch.utils.data

from reallm.api.core.config import ModelName
from reallm.base import constants, gpu_utils, logging, namedarray, numpy_utils, seeding, timeutil, topology
from reallm.base.monitor import (cuda_tmark, cuda_tmarked, CUDATimeMarkType, dump_tmark_db,
                                 gpu_utilization_monitor)
from reallm.impl.model.nn.real_llm_api import ReaLModel
from reallm.impl.model.nn.real_llm_parallel import pipeline_repartition_strategy
from reallm.system import request_reply_stream, worker_base
import reallm.api.core.dfg as dfg
import reallm.api.core.system_api as system_api
import reallm.impl.model.comm.data_transfer as data_transfer_comm
import reallm.impl.model.comm.global_comm as global_comm
import reallm.impl.model.comm.param_realloc as param_realloc_comm

# NOTE: Register all implemented datasets and models.
import reallm.api.core.data_api as data_api  # isort:skip
import reallm.api.core.model_api as model_api  # isort:skip

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


def get_pytorch_profiler():
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


def _get_shape_from_key_and_seqlen(k: str, seqlen: int):
    if k in [
            "input_lens",
            "prompt_lens",
            "seq_no_eos_mask",
            "rewards",
            "reward_score",
            "group_factor",
            "pos_input_lens",
    ]:
        shape = (1,)
    elif k in ["cu_seqlens", "prompt_cu_seqlens"]:
        shape = (2,)
    # FIXME: problem here if we use groups instead of pairs?
    elif k in ["seqlogp"]:
        shape = (1, 2)
    elif k in [
            "packed_seq",
            "prompt_mask",
            "packed_input_ids",
            "values",
            "packed_prompts",
    ]:
        shape = (seqlen,)
    elif k in [
            "packed_logprobs",
            "packed_ref_logprobs",
            "old_logp",
            "ref_logp",
            "advantages",
            "ppo_loss_mask",
            "kl_rewards",
            "returns",
    ]:
        shape = (seqlen - 1,)
    else:
        raise NotImplementedError(f"Unknown key {k} in packed data.")
    return shape


def _get_dtype_from_key(k: str):
    if k in [
            "seq_no_eos_mask",
            "ppo_loss_mask",
            "prompt_mask",
    ]:
        dtype = torch.bool
    elif k in [
            "reward_score",
            "packed_ref_logprobs",
            "old_logp",
            "ref_logp",
            "advantages",
            "kl_rewards",
            "returns",
            "values",
    ]:
        dtype = torch.float16
    elif k in [
            "input_lens",
            "prompt_lens",
            "cu_seqlens",
            "prompt_cu_seqlens",
            "pos_input_lens",
    ]:
        dtype = torch.int32
    elif k in ["packed_seq", "packed_input_ids", "packed_prompts"]:
        dtype = torch.int64
    elif k in ["rewards", "packed_logprobs", "group_factor", "seqlogp"]:
        dtype = torch.float32
    else:
        raise NotImplementedError(f"Unknown key {k} in packed data.")
    return dtype


class ModelWorker(worker_base.Worker):

    def _configure(self, cfg: system_api.ModelWorker):
        self.config = cfg
        self.model_names = [s.id.model_name for s in cfg.shards]
        self.shard_indices = [
            cfg.model_topos[s.id.model_name].get_rank(data=s.id.dp_rank,
                                                      pipe=s.id.pp_rank,
                                                      model=s.id.mp_rank) for s in cfg.shards
        ]

        self.__experiment_name = self.config.worker_info.experiment_name
        self.__trial_name = self.config.worker_info.trial_name

        self.config.model_rpcs, _ = dfg.build_graph(self.config.model_rpcs)
        self.data2required_rpc_names = self.config.model_rpcs[0].data2required_rpc_names

        # NOTE: here worker_index is different from peer/ddp rank
        self.__worker_index = cfg.worker_info.worker_index

        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
        torch.backends.cudnn.deterministic = cfg.cudnn_deterministic

        seeding.set_random_seed(cfg.seed)

        # Reveal DDP identity of this worker to world.
        gpu_utils.reveal_ddp_identity(self.__experiment_name, self.__trial_name, self.__worker_index)
        self.__dist_env_resolved = False

        self.__clear_cache_frequency = timeutil.FrequencyControl(
            frequency_steps=self.config.cuda_cache_clear_freq)

        r = self.config.worker_info

        # log info
        self.__total_time = 0.01

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
            constants.set_parallelism_group(
                model_name_,
                self.__pg_info.model_groups[model_name_],
            )
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
                    cache_root=(None
                                if not self.config.use_dataset_cache else self.config.dataset_cahce_root),
                ) for d in self.config.datasets
            ]
            if len(self.config.datasets) == 1:
                self.__dataset = datasets[0]
            else:
                self.__dataset = torch.utils.data.ConcatDataset(datasets)
            self.__dataloader = data_api.make_dataloader(self.config.dataloader, self.__dataset)
            self.__dataset_n_seqs = 0
            for tmp_sample in self.__dataloader:
                self.__dataset_n_seqs += len(data_api.split_sequences(tmp_sample))
            
            self.__data_generator = enumerate(self.__dataloader)
            self.__dataset_batch_counter = None

            self.__dataset_epoch = 0
            self.__cur_sample = None
            self.__fetched_sample_cache = []

        self.__models: Dict[ModelName, model_api.Model] = dict()
        self.__model_is_handle: Dict[ModelName, bool] = dict()
        self.__interfaces: Dict[ModelName, model_api.ModelInterface] = dict()
        self.__eval_dataloaders: Dict[ModelName, torch.utils.data.DataLoader] = dict()

        self.__backends: Dict[ModelName, model_api.ModelBackend] = dict()
        self.__unwrapped_models: Dict[ModelName, torch.nn.Module | ReaLModel] = dict()

        self.__backend_initialized: Dict[ModelName, bool] = dict()
        self.__profiler_launched = False

        for s in self.config.shards:
            with constants.model_scope(s.id.model_name):
                self.__backend_initialized[s.id.model_name] = False
                tik = time.perf_counter()
                self.__models[s.id.model_name] = model = model_api.make_model(s.model,
                                                                              name=s.id.model_name,
                                                                              device=self.__device)
                self.__unwrapped_models[s.id.model_name] = model.module
                if s.id.model_name.replica_id == 0:
                    assert isinstance(model.module, ReaLModel)
                    model.module.instantiate()
                    self.__model_is_handle[s.id.model_name] = False
                else:
                    self.__model_is_handle[s.id.model_name] = True
                self.__backends[s.id.model_name] = model_api.make_backend(s.backend)
                interface_impl = [
                    rpc.interface_impl for rpc in self.config.model_rpcs if rpc.model_name == s.id.model_name
                ]
                assert all(x == interface_impl[0] for x in interface_impl)
                self.__interfaces[s.id.model_name] = model_api.make_interface(interface_impl[0])

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
                            cache_root=(None if not self.config.use_dataset_cache else
                                        self.config.dataset_cahce_root),
                        ) for d in s.eval_datasets
                    ]
                    if len(eval_datasets) > 1:
                        eval_dataset = torch.utils.data.ConcatDataset(eval_datasets)
                    else:
                        eval_dataset = eval_datasets[0]
                    eval_dataloader = data_api.make_dataloader(s.eval_dataloader, eval_dataset)
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
        self.__data_owner_storage: Dict[int, Dict[str, torch.Tensor]] = (collections.defaultdict(dict))
        self.__data_receive_cache: Dict[int, Dict[str, torch.Tensor]] = (collections.defaultdict(dict))

        self.__data_sent_worker_indices: Dict[int, Dict[str, Set]] = (
            collections.defaultdict(lambda: collections.defaultdict(set)))
        self.__data_received_worker_indices: Dict[int, Dict[str, Set]] = (
            collections.defaultdict(lambda: collections.defaultdict(set)))

        self.__compute_input_queues = dict(
            train_step=queue.Queue(4),
            inference=queue.Queue(4),
            generate=queue.Queue(4),
            evaluate=queue.Queue(4),
        )

        # A monitoring process.
        self.__gpu_util_mp = mp.Process(target=gpu_utilization_monitor, args=(self.__worker_index, 20, 7200))
        # self.__gpu_util_mp.start()

    def __prefetch_from_dataset(self):
        if self.__cur_sample is None:
            try:
                self.__dataset_batch_counter, self.__cur_sample = next(self.__data_generator)
            except StopIteration:
                self.__dataset_epoch += 1
                self.__data_generator = enumerate(self.__dataloader)
                self.__dataset_batch_counter, self.__cur_sample = next(self.__data_generator)

    def __handle_one_rpc_hook(self, hook: str, hook_data: Any):
        if hook == "data_transfer":
            # torch.cuda.synchronize()
            tik = time.perf_counter()
            self.__data_transfer_among_workers(hook_data)
            # blogger.debug(
            #     f"data transfer CPU time: {time.perf_counter() - tik:.4f}s, "
            #     f"# remaining data in local storage: {sum([len([x for x in xs if isinstance(x, torch.Tensor) and x.device == self.__device]) for xs in self.__data_owner_storage.values()])}, "
            #     f"# remaining data in receiver cache: {sum([len(xs) for xs in self.__data_receive_cache.values()])}."
            # )
            # torch.cuda.synchronize()
        elif hook == "param_realloc":
            tik = time.perf_counter()
            # torch.cuda.synchronize()
            from_model_name: ModelName = hook_data["from_model_name"]
            to_model_name: ModelName = hook_data["to_model_name"]
            from_topo: topology.PipeModelDataParallelTopology = hook_data["from_topo"]
            to_topo: topology.PipeModelDataParallelTopology = hook_data["to_topo"]
            to_model_config = hook_data["to_model_config"]
            # profiler = get_pytorch_profiler()
            # profiler.start()
            if from_model_name in self.__unwrapped_models:
                m = self.__unwrapped_models[from_model_name]
            else:
                m = self.__unwrapped_models[to_model_name]
            assert isinstance(m, ReaLModel), type(m)
            new_layers, new_param, _ = m.build_reparallelized_layers_async(
                from_model_name=from_model_name,
                to_model_name=to_model_name,
                from_topo=from_topo,
                to_topo=to_topo,
                to_model_config=to_model_config,
                pg_info=self.__param_realloc_info,
            )
            if from_model_name in self.__models:
                self.__model_is_handle[from_model_name] = True
            if to_model_name in self.__models:
                self.__unwrapped_models[to_model_name].patch_reparallelization((new_layers, new_param))
                self.__model_is_handle[to_model_name] = False
            # FIXME: suppress this log
            blogger.debug(f"param_realloc CPU time: {time.perf_counter() - tik:.4f}s")
            # profiler.__exit__(None, None, None)
        elif hook == "offload":
            with cuda_tmarked("offload", CUDATimeMarkType.mem_layout):
                tik = time.perf_counter()
                m = self.__unwrapped_models[hook_data["model_name"]]
                assert isinstance(m, ReaLModel), type(m)
                m.async_offload()
                # blogger.debug(f"async_offload enqueue CUDA request time: {time.perf_counter() - tik:.4f}s")
        else:
            raise NotImplementedError(f"Unknown hook {hook}.")

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

        if len(request.pre_hooks) > 0:
            assert len(request.pre_hooks) == len(request.pre_hook_data)
            assert not handled and res is None
            self.__handle_one_rpc_hook(request.pre_hooks.pop(0), request.pre_hook_data.pop(0))
            self.__request_queue.put_nowait((request, data, False, None))
            return
        if handled and len(request.post_hooks) > 0:
            assert len(request.post_hooks) == len(request.post_hook_data)
            self.__handle_one_rpc_hook(request.post_hooks.pop(0), request.post_hook_data.pop(0))
            self.__request_queue.put_nowait((request, data, True, res))
            return
        if handled and len(request.post_hooks) == 0:
            self.__reply_queue.put_nowait((request, res))

            # self.logger.info(f"Model worker {self.__worker_index} #{request.handler}# "
            #                          f"finish handling request *{request.handle_name}*, "
            #                          f"request_id {request.request_id}.")
            sample_count = (data.length(0) if isinstance(data, namedarray.NamedArray) else 1)
            self.__request_sample_size[request.request_id] = sample_count
            return

        assert not handled and res is None

        with constants.model_scope(handler_model_name):
            res = None
            if request.handle_name == "empty":
                # Empty request is used for executing hooks, e.g., data transfer, parameter syncrhonization.
                pass
            ############## initialization ##############
            elif request.handle_name == "initialize":
                assert not self.__model_is_handle[request.handler.model_name]
                self.__models[request.handler.model_name] = self._backend.initialize(self._model, data)
                self.__backend_initialized[request.handler.model_name] = True
            elif request.handle_name == "model_config":
                assert isinstance(self.__unwrapped_models[request.handler.model_name], ReaLModel)
                res = self.__unwrapped_models[request.handler.model_name].config
            ############## data loading ##############
            elif request.handle_name == "fetch":
                fetched_data = data_api.split_sequences(self.__cur_sample)
                seqlens = [x.metadata["seqlens"][0] for x in fetched_data]
                res = data_api.DataBatchMeta(
                    dp_rank=self._dp_rank,
                    seqlens=seqlens,
                    keys=list(self.__cur_sample.keys()),
                    epoch=self.__dataset_epoch,
                    is_final_batch=(self.__dataset_batch_counter == len(self.__dataloader) - 1),
                )
                self.__fetched_sample_cache += fetched_data
                self.__cur_sample = None
            elif request.handle_name == "store":
                buffer_indices = request.data
                assert len(buffer_indices) == len(self.__fetched_sample_cache), (
                    len(buffer_indices),
                    len(self.__fetched_sample_cache),
                )
                for buf_idx, x in zip(buffer_indices, self.__fetched_sample_cache):
                    for k, v in x.items():
                        assert v.device == torch.device("cpu")
                        self.__data_owner_storage[buf_idx][k] = v
                self.__fetched_sample_cache.clear()
                res = None
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
                    if (self.config.cuda_cache_cleanliness and self.__clear_cache_frequency.check()):
                        st = time.monotonic()
                        gc.collect()
                        torch.cuda.empty_cache()
                        gc.collect()
                        et = time.monotonic()
                        blogger.debug(f"Model worker {self.__worker_index} cleared cache in {et-st:.4f}s")
                dump_tmark_db(self.__worker_index)
            ############## computation function calls ##############
            elif request.handle_name == "inference":
                assert not self.__model_is_handle[request.handler.model_name], request.handler.model_name
                data, buffer_indices, seqlens, output_key_remap = (
                    self.__compute_input_queues[request.handle_name].get_nowait())
                data: namedarray.NamedArray
                data.register_metadata(seqlens=seqlens)
                # if constants.model_name().role == "reward":
                #     profiler = get_pytorch_profiler()
                #     profiler.start()
                res = self._interface.inference(self._model, data)  # -> NamedArray
                # if constants.model_name().role == "reward":
                #     profiler.__exit__(None, None, None)
                #     profiler.export_chrome_trace(
                #         os.path.join(constants.LOG_ROOT, self.__experiment_name, self.__trial_name, f"reward_inf{self.__worker_index}.json")
                #     )
                # FIXME: refactor this repetitive code
                if res is not None:
                    new_res = {}
                    for k, v in res.items():
                        if k in output_key_remap:
                            new_res[output_key_remap[k]] = v
                        else:
                            new_res[k] = v
                    new_res = namedarray.from_dict(new_res)
                    new_res.register_metadata(**res.metadata)
                    assert "seqlens" in new_res.metadata
                    res = new_res, buffer_indices, seqlens
            elif request.handle_name == "train_step":
                assert not self.__model_is_handle[request.handler.model_name], request.handler.model_name
                data, _, seqlens, *_ = self.__compute_input_queues[request.handle_name].get_nowait()
                data.register_metadata(seqlens=seqlens)
                res = self._interface.train_step(self._model, data)  # -> Dict
            elif request.handle_name == "generate":
                assert not self.__model_is_handle[request.handler.model_name], request.handler.model_name
                data, buffer_indices, seqlens, output_key_remap = (
                    self.__compute_input_queues[request.handle_name].get_nowait())
                data.register_metadata(seqlens=seqlens)
                res = self._interface.generate(self._model, data)  # -> NamedArray
                if res is not None:
                    new_res = {}
                    for k, v in res.items():
                        if k in output_key_remap:
                            new_res[output_key_remap[k]] = v
                        else:
                            new_res[k] = v
                    new_res = namedarray.from_dict(new_res)
                    new_res.register_metadata(**res.metadata)
                    res = new_res, buffer_indices, seqlens
            elif request.handle_name == "evaluate":
                assert not self.__model_is_handle[request.handler.model_name], request.handler.model_name
                res = self._interface.evaluate(self._model, self._eval_dataloader)  # -> Dict
            elif request.handle_name == "save":
                assert not self.__model_is_handle[request.handler.model_name], request.handler.model_name
                res = self._interface.save(self._model, data)  # -> None
            else:
                raise NotImplementedError(f"Unknown request type: {request.handle_name}.")

            if (request.handle_name in TIME_RECORD_RPCS and self._is_dp_head and self._dp_rank == 0):
                blogger.info(f"Model worker #{handler_model_name}# handle request *{request.handle_name}*"
                             f" in ${time.perf_counter() - tik:.4f}$s")

        self.__request_queue.put_nowait((request, data, True, res))

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
            if (isinstance(step, data_transfer_comm.DataTransferReceiverStep)
                    and step.rank == dist.get_rank()):
                buf_indices = step.buf_indices
                seqlens = step.seqlens
                if step.src == dist.get_rank():
                    for buf_idx in buf_indices:
                        self.__data_owner_storage[buf_idx][step.key] = (
                            self.__data_owner_storage[buf_idx][step.key].to(self.__device))
                    vs = torch.cat(
                        [self.__data_owner_storage[buf_idx][step.key] for buf_idx in buf_indices],
                        dim=0,
                    )
                else:
                    all_sent_dst_ranks = [
                        self.__data_received_worker_indices[buf_idx][step.key] for buf_idx in buf_indices
                    ]
                    if all(
                            set(step.dst_ranks).issubset(set(sent_dst_ranks))
                            for sent_dst_ranks in all_sent_dst_ranks):
                        vs = torch.cat(
                            [self.__data_receive_cache[buf_idx][step.key] for buf_idx in buf_indices],
                            dim=0,
                        )
                    else:
                        total_len = 0
                        for seqlen in seqlens:
                            shape = _get_shape_from_key_and_seqlen(step.key, seqlen)
                            total_len += int(np.prod(shape))
                        dtype = _get_dtype_from_key(step.key)
                        buf = torch.zeros(
                            (total_len,),
                            dtype=dtype,
                            device=torch.cuda.current_device(),
                        )
                        # print(f"{dist.get_rank()} recv {step.key} from {step.src} with shape {buf.shape}")
                        dist.broadcast(buf, src=step.src, group=step.group)
                        vs = buf.clone().view(-1, *shape[1:])
                        for buf_idx in buf_indices:
                            self.__data_received_worker_indices[buf_idx][step.key].union(step.dst_ranks)
                offset = 0
                for seqlen, buf_idx in zip(seqlens, buf_indices):
                    shape = _get_shape_from_key_and_seqlen(step.key, seqlen)
                    v = vs[offset:offset + shape[0]]
                    offset += shape[0]
                    data[(buf_idx, step.key)] = (v, seqlen)
                    if (step.key not in self.__data_owner_storage[buf_idx]
                            and step.key not in self.__data_receive_cache[buf_idx]):
                        self.__data_receive_cache[buf_idx][step.key] = v

            if (isinstance(step, data_transfer_comm.DataTransferSenderStep) and step.rank == dist.get_rank()):
                buf_indices = step.buf_indices
                all_sent_dst_ranks = [
                    self.__data_sent_worker_indices[buf_idx][step.key] for buf_idx in buf_indices
                ]
                if all(
                        set(step.dst_ranks).issubset(set(sent_dst_ranks))
                        for sent_dst_ranks in all_sent_dst_ranks):
                    pass
                else:
                    for buf_idx in buf_indices:
                        self.__data_owner_storage[buf_idx][step.key] = (
                            self.__data_owner_storage[buf_idx][step.key].to(self.__device))
                    vs = torch.cat(
                        [self.__data_owner_storage[buf_idx][step.key] for buf_idx in buf_indices],
                        dim=0,
                    )
                    # print(f"{dist.get_rank()} send {step.key} to {step.dst_ranks} with shape {vs.shape}")
                    dist.broadcast(vs, src=step.rank, group=step.group)
                    for buf_idx in buf_indices:
                        self.__data_sent_worker_indices[buf_idx][step.key].union(step.dst_ranks)

        if len(data) > 0:
            local_buffer_indices = sorted(list(set([buf_idx for buf_idx, _ in data.keys()])))
            local_keys = list(set([key for _, key in data.keys()]))

            local_seqlens = []
            for buf_idx in local_buffer_indices:
                local_seqlens.append(data[(buf_idx, local_keys[0])][1])

            k = local_keys[0]
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
            self.__compute_input_queues[hook_data["handle_name"]].put_nowait(
                (r, local_buffer_indices, local_seqlens, hook_data["output_key_remap"]))

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

        if isinstance(request.handler, system_api.ModelShardID) and isinstance(res, namedarray.NamedArray):
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
                    ),)
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

            self.__request_queue.put_nowait((request, request.data, False, None))

    def _poll(self):
        if not self.__dist_env_resolved:
            self.__lazy_setup()
            self.__dist_env_resolved = True
            # self.tracer.start()
            # self.__profiler = torch.profiler.profile(
            #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            #     # with_stack=True,
            #     # with_flops=True,
            # )

        if self.__has_dataset:
            self.__prefetch_from_dataset()

        st = time.monotonic()
        self.__maybe_receive_requests()

        for _ in range(16):
            # TODO: we can have a smarter scheduling plan, the current plan is basically round-robin
            try:
                request, data, handled, res = self.__request_queue.get_nowait()
                if (request.handle_name in ["train_step", "inference", "generate"]
                        and not self.__profiler_launched):
                    self.tracer.start()
                    # self.__profiler.start()
                    self.__profiler_launched = True
                    self.__profiler_ctl = timeutil.FrequencyControl(frequency_seconds=10)
                self.__model_poll_step(request, data, handled, res)
            except queue.Empty:
                break

        r = self.__maybe_post_responses()

        # FIXME: add memory monitoring
        if r.batch_count > 0:
            # tik = time.perf_counter()
            # blogger.debug(("Model worker #{}#: MemAllocated=*{}*GB, MaxMemAllocated=${}$GB".format(
            #     ",".join(self.model_names),
            #     round(get_accelerator().memory_allocated() / 1024**3, 2),
            #     round(get_accelerator().max_memory_allocated() / 1024**3, 2),
            # )))
            # blogger.debug(f"monitoring overhead {time.perf_counter()-tik}s")
            # if os.environ.get("REAL_TRACE", "0") == "1":
            #     self.tracer.save()
            if self.__profiler_launched and self.__profiler_ctl.check():
                self.tracer.save()
                # import torch.autograd.profiler as prof
                # self.__profiler.stop()
                # prof.KinetoStepTracker.erase_step_count("ProfilerStep")
                # self.__profiler
                # self.__profiler.stop_trace()
                # self.__profiler_launched = False
                # self.__profiler.export_chrome_trace(self._tracer_output_file)

        t = time.monotonic() - st
        self.__total_time += t
        # blogger.debug(
        #     f"Model worker #{','.join(self.model_names)}# poll time: {t:.4f}s, engine poll time {pt:.4f}s, percent {pt/t:.4f}"
        # )
        return r
