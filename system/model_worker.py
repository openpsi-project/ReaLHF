from typing import Any, Dict, List, Optional, Tuple
import collections
import gc
import itertools
import multiprocessing as mp
import os
import queue
import socket
import time
import uuid

from deepspeed.accelerator import get_accelerator
import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data

from base.monitor import gpu_utilization_monitor, time_mark
from base.topology import ParallelGrid
from impl.model.backend.pipe_engine.stream_pipe_engine import EngineFuture, StreamPipeEngine
from impl.model.nn.flash_mqat.flash_mqat_api import FlashMQATModel
import api.config.config_system as config_system
import api.config.dfg
import api.data
import api.model
import base.constants
import base.datapack as datapack
import base.dataparallel as dataparallel
import base.gpu_utils as gpu_utils
import base.logging as logging
import base.namedarray as namedarray
import base.numpy_utils
import base.seeding as seeding
import base.timeutil
import system.request_reply_stream as request_reply_stream
import system.worker_base as worker_base

# Register all implemented datasets and models.
import impl.model  # isort:skip
import impl.dataset  # isort:skip

logger = logging.getLogger("Model Worker", "colored")
blogger = logging.getLogger("benchmark")


class NoRequestToHandle(Exception):
    pass


def _get_seqlens_from_batch_sample(sample: namedarray.NamedArray) -> np.ndarray | None:
    if "input_lens" in sample.keys():
        return sample["input_lens"].cpu().numpy()
    elif "cu_seqlens" in sample.keys():
        return (sample["cu_seqlens"][1:] - sample["cu_seqlens"][:-1]).cpu().numpy()
    # NOTE: The order matters. We should first try to get the length of the generated text rather than prompts.
    elif "prompt_lens" in sample.keys():
        return sample["prompt_lens"].cpu().numpy()
    elif "prompt_cu_seqlens" in sample.keys():
        return (sample["prompt_cu_seqlens"][1:] - sample["prompt_cu_seqlens"][:-1]).cpu().numpy()
    else:
        return None


def split_packed_batch_into_seqs(
    sample: namedarray.NamedArray,
    input_lens: Optional[torch.Tensor] = None,
    return_seqlens: bool = False,
) -> List[namedarray.NamedArray]:
    if input_lens is None:
        if "input_lens" in sample:
            input_lens = sample["input_lens"]
        elif "prompt_lens" in sample:
            input_lens = sample["prompt_lens"]
        elif "cu_seqlens" in sample:
            input_lens = sample["cu_seqlens"][1:] - sample["cu_seqlens"][:-1]
        elif "prompt_cu_seqlens" in sample:
            input_lens = sample["prompt_cu_seqlens"][1:] - sample["prompt_cu_seqlens"][:-1]

    partitions = [(i, i + 1) for i in range(input_lens.shape[0])]
    sample["input_lens"] = input_lens
    res = dataparallel.PackedParallelDataBroker.scatter_to(sample,
                                                           n_dp=len(input_lens),
                                                           partitions=partitions)
    if not return_seqlens:
        return res
    else:
        return res, input_lens


def _get_shape_from_key_and_seqlen(k: str, seqlen: int):
    if k in ["input_lens", "prompt_lens", "seq_no_eos_mask", "rewards", "reward_score", "group_factor"]:
        shape = (1,)
    elif k in ["cu_seqlens", "prompt_cu_seqlens"]:
        shape = (2,)
    elif k in ["packed_seq", "prompt_mask", "packed_input_ids", "values", "packed_prompts"]:
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
    elif k in ["input_lens", "prompt_lens", "cu_seqlens", "prompt_cu_seqlens"]:
        dtype = torch.int32
    elif k in ["packed_seq", "packed_input_ids", "packed_prompts"]:
        dtype = torch.int64
    elif k in ["rewards", "packed_logprobs", "group_factor"]:
        dtype = torch.float32
    else:
        raise NotImplementedError(f"Unknown key {k} in packed data.")
    return dtype


def _get_tensor_buffer_from_key_and_seqlen(k: str, seqlen: int):
    shape = _get_shape_from_key_and_seqlen(k, seqlen)
    dtype = _get_dtype_from_key(k)
    return base.constants.get_global_memory_buffer().get_tensor(shape, dtype, name="data_transfer")


class ModelWorker(worker_base.Worker):

    def __init__(self, server=None):
        super().__init__(server)

        self.__dist_env_resolved = False

        self.__stream = None

        # log info
        self.__total_time = 0.01
        self.__engine_poll_time = 0

    def _configure(self, cfg: config_system.ModelWorker):
        self.config = cfg
        self.model_names = [s.id.model_name for s in cfg.shards]
        self.shard_indices = [
            cfg.model_topos[s.id.model_name].get_rank(data=s.id.dp_rank,
                                                      pipe=s.id.pp_rank,
                                                      model=s.id.mp_rank) for s in cfg.shards
        ]

        self.__experiment_name = self.config.worker_info.experiment_name
        self.__trial_name = self.config.worker_info.trial_name

        self.config.model_rpcs, _ = api.config.dfg.build_graph(self.config.model_rpcs)
        self.data2required_rpc_names = self.config.model_rpcs[0].data2required_rpc_names

        # NOTE: here worker_index is different from peer/ddp rank
        self.__worker_index = cfg.worker_info.worker_index

        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
        torch.backends.cudnn.deterministic = cfg.cudnn_deterministic

        seeding.set_random_seed(cfg.seed)

        # Reveal DDP identity of this worker to world.
        gpu_utils.reveal_ddp_identity(self.__experiment_name, self.__trial_name, self.__worker_index)
        self.__dist_env_resolved = False

        self.__clear_cache_frequency = base.timeutil.FrequencyControl(
            frequency_steps=self.config.cuda_cache_clear_freq)

        r = self.config.worker_info
        return r

    @property
    def _is_stream_pipe(self) -> bool:
        return self.__interfaces[base.constants.model_name()].is_future_interface

    @property
    def _mp_rank(self) -> int:
        return base.constants.model_parallel_rank()

    @property
    def _pp_rank(self) -> int:
        return base.constants.pipe_parallel_rank()

    @property
    def _dp_rank(self) -> int:
        return base.constants.data_parallel_rank()

    @property
    def _pp_size(self) -> int:
        return base.constants.pipe_parallel_world_size()

    @property
    def _mp_size(self) -> int:
        return base.constants.model_parallel_world_size()

    @property
    def _dp_size(self) -> int:
        return base.constants.data_parallel_world_size()

    @property
    def _is_dp_head(self) -> bool:
        return self._mp_rank == 0 and self._pp_rank == self._pp_size - 1

    @property
    def _model(self) -> api.model.Model:
        return self.__models[base.constants.model_name()]

    @property
    def _interface(self) -> api.model.ModelInterface:
        return self.__interfaces[base.constants.model_name()]

    @property
    def _backend(self) -> api.model.ModelBackend:
        return self.__backends[base.constants.model_name()]

    @property
    def _engine(self):
        return self.__engines[base.constants.model_name()]

    @property
    def _eval_dataloader(self) -> torch.utils.data.DataLoader:
        return self.__eval_dataloaders[base.constants.model_name()]

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
            sub_patterns=sub_patterns,
        )

        self.__pg_info = gpu_utils.setup_ddp(
            expr_name=self.__experiment_name,
            trial_name=self.__trial_name,
            worker_index=self.__worker_index,
            model_topos=self.config.model_topos,
            msid2mwid=self.config.msid2mwid,
            mw_bcast_groups=self.config.mw_bcast_groups,
            param_sync_pairs=self.config.sync_param_pairs,
            data_transfer_pairs=self.config.data_transfer_pairs,
        )

        base.constants.set_experiment_trial_names(self.__experiment_name, self.__trial_name)

        # logger.info(f"SetUp Information - Model worker index {self.__worker_index} located at "
        #             f"{socket.gethostname()} GPU {self.__pg_info.local_gpu_id}.")

        deepspeed.init_distributed()
        # self.logger.info("deepspeed init distributed on model worker")
        self.__device = torch.device("cuda:0")

        for model_name_, topo_ in self.config.model_topos.items():
            base.constants.set_parallelism_group(
                model_name_,
                self.__pg_info.model_groups[model_name_],
            )
            base.constants.set_rank_mapping(model_name_, topo_, self.config.msid2mwid)
            grid = ParallelGrid(
                topology=topo_,
                rank_mapping=base.constants.rank_mapping_of_model(model_name_),
                process_group=self.__pg_info.model_groups[model_name_],
            )
            base.constants.set_grid(model_name_, grid)

        # Set up training dataset for source RPCs.
        if self.__has_dataset:
            datasets = [
                api.data.make_dataset(
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
            self.__max_seqlen = max(d.max_seqlen for d in datasets)
            self.__dataloader = api.data.make_dataloader(self.config.dataloader, self.__dataset)
            self.__data_generator = enumerate([])

            self.__dataset_epoch = -1
            self.__dataset_global_step = self.__dataset_epoch_step = 0
            self.__dict_sample = None

            self.__fetched_data = None

        self.__models: Dict[str, api.model.Model] = dict()
        self.__interfaces: Dict[str, api.model.ModelInterface] = dict()
        self.__backends: Dict[str, api.model.ModelBackend] = dict()
        self.__eval_dataloaders: Dict[str, torch.utils.data.DataLoader] = dict()
        self.__engines: Dict[str, Any] = dict()

        for s in self.config.shards:
            with base.constants.model_scope(s.id.model_name):
                self.__models[s.id.model_name] = api.model.make_model(s.model,
                                                                      name=s.id.model_name,
                                                                      device=self.__device)
                interface_impl = [
                    rpc.interface_impl for rpc in self.config.model_rpcs if rpc.model_name == s.id.model_name
                ]
                assert all(x == interface_impl[0] for x in interface_impl)
                self.__interfaces[s.id.model_name] = api.model.make_interface(interface_impl[0])
                self.__backends[s.id.model_name] = api.model.make_backend(s.backend)

            if s.eval_datasets is not None and s.eval_dataloader is not None:
                eval_datasets = [
                    api.data.make_dataset(
                        d,
                        self.config.seed,
                        s.id.dp_rank,
                        s.id.topo.get_dim("data"),
                        self.__models[s.id.model_name].tokenizer,
                        self.config.worker_info.experiment_name,
                        self.config.worker_info.trial_name,
                        cache_root=(None
                                    if not self.config.use_dataset_cache else self.config.dataset_cahce_root),
                    ) for d in s.eval_datasets
                ]
                if len(eval_datasets) > 1:
                    eval_dataset = torch.utils.data.ConcatDataset(eval_datasets)
                else:
                    eval_dataset = eval_datasets[0]
                eval_dataloader = api.data.make_dataloader(self.config.eval_dataloader, eval_dataset)
            else:
                eval_dataloader = None
            self.__eval_dataloaders[s.id.model_name] = eval_dataloader

        self.__request_queue = queue.Queue(maxsize=8)
        self.__reply_queue = queue.Queue(maxsize=8)
        self.__request_sample_size = dict()

        self.__request_storage = dict()  # mapping from request id to requests
        self.__future_storage = dict()  # mapping from request id to corresponding future
        self.__post_hook_data_storage = dict()
        self.__request_time = dict()

        # Storing model function call outputs. Model worker will serve as
        # the data producer when other model workers require this data.
        self.__data_owner_storage = dict()
        # Record which RPCs have received the data.
        # After all RPCs have received the data, remove it from storage.
        self.__data_send_record = collections.defaultdict(list)

        self.__compute_input_queues = dict(
            train_step=queue.Queue(4),
            inference=queue.Queue(4),
            generate=queue.Queue(4),
            evaluate=queue.Queue(4),
        )

        # A monitoring process.
        self.__gpu_util_mp = mp.Process(target=gpu_utilization_monitor,
                                        args=(self.__pg_info.local_gpu_id, 7200))
        # self.__gpu_util_mp.start()

    def __prefetch_from_dataset(self):
        if self.__dict_sample is None:
            try:
                self.__dataset_epoch_step, self.__dict_sample = next(self.__data_generator)
            except StopIteration:
                self.__dataset_epoch += 1
                self.__data_generator = enumerate(self.__dataloader)
                self.__dataset_epoch_step, self.__dict_sample = next(self.__data_generator)

    def __maybe_receive_request(self):
        recv_tik = time.perf_counter()
        try:
            request: request_reply_stream.Payload = self.__stream.poll()
        except request_reply_stream.NoMessage:
            return

        self.__request_storage[request.request_id] = request
        self.__request_queue.put_nowait((request, request.data))

    def __model_poll_step(self) -> worker_base.PollResult:
        tik = time.perf_counter()

        try:
            request, data = self.__request_queue.get_nowait()
            request: request_reply_stream.Payload
        except queue.Empty:
            return
        # self.logger.info(f"Model worker {self.__worker_index} #{request.handler}# "
        #                  f"start handle request *{request.handle_name}*, "
        #                  f"request_id {request.request_id}.")

        if isinstance(request.handler, str):
            assert request.handler == f"__data{self.__dataset_dp_rank}__"
            handler_model_name = self.__src_rpc_model_name
        else:
            handler_model_name = request.handler.model_name

        with base.constants.model_scope(handler_model_name):
            res = None
            if request.handle_name == "initialize":
                base.constants.set_max_seqlen(data.max_seqlen)  # used by generate cuda graph buffer
                self.__models[request.handler.model_name] = self._backend.initialize(self._model, data)
                self.__engines[request.handler.model_name] = self._model.module
                if self._is_stream_pipe:
                    assert isinstance(self._engine, StreamPipeEngine)
                m = self.__models[request.handler.model_name].module.module
                res = None if not isinstance(m, FlashMQATModel) else m.config
            ############## data loading ##############
            elif request.handle_name == "fetch":
                assert self.__fetched_data is None
                res = api.data.DataBatch(
                    data=self.__dict_sample,
                    epoch=self.__dataset_epoch,
                    epoch_step=self.__dataset_epoch_step,
                    global_step=self.__dataset_global_step,
                )
                self.__fetched_data = split_packed_batch_into_seqs(namedarray.from_dict(self.__dict_sample))
                self.__dict_sample = None
                self.__dataset_global_step += 1
            elif request.handle_name == "store":
                buffer_indices = request.data
                assert len(buffer_indices) == len(self.__fetched_data)
                for buf_idx, x in zip(buffer_indices, self.__fetched_data):
                    for k, v in x.items():
                        self.__data_owner_storage[(buf_idx, k)] = v.to(self.__device)
                self.__fetched_data = None
                res = None
            elif request.handle_name == "spec":
                if self.__dataloader.batch_size is not None:
                    batch_size = self.__dataloader.batch_size
                else:
                    # We assume that this is a packed dataset. Batch size equals to the number of tokens in the batch.
                    assert isinstance(self.__dataloader.dataset, torch.utils.data.IterableDataset)
                    batch_size = list(self.__dict_sample.values())[0].shape[0]
                res = api.model.FinetuneSpec(
                    total_train_epochs=-1,  # place-holder, to be filled by master worker
                    total_train_steps=-1,  # place-holder, to be filled by master worker
                    steps_per_epoch=len(self.__dataloader),
                    batch_size_per_device=batch_size,
                    max_seqlen=self.__max_seqlen,
                )
            ############## data transfer ##############
            elif request.handle_name == "data_transfer":
                with base.constants.model_scope_disabled():
                    self.__data_transfer_among_workers(request)
            ############## parameter synchronization ##############
            elif request.handle_name == "send_param":
                if self._dp_rank == 0:
                    self.__send_params_to_sync(request.data)
            elif request.handle_name == "recv_param":
                self.__recv_params_to_sync(request.data)
            ############## offload ##############
            elif request.handle_name in ["offload", "load_to_device"]:
                print(f">>>>>>> model worker {self.__worker_index} model name "
                      f"{request.handler.model_name} receive {request.handle_name} "
                      f"request, which is not implemented yet.")
            ############## computation function calls ##############
            elif request.handle_name == "inference":
                data, buffer_indices, seqlens, output_key_remap = self.__compute_input_queues[
                    request.handle_name].get_nowait()
                res = self._interface.inference(self._model, data)  # -> NamedArray
                if res is not None:
                    new_res = {}
                    for k, v in res.items():
                        if k in output_key_remap:
                            new_res[output_key_remap[k]] = v
                        else:
                            new_res[k] = v
                    new_res = namedarray.from_dict(new_res)
                    res = new_res, buffer_indices, seqlens
            elif request.handle_name == "train_step":
                data, *_ = self.__compute_input_queues[request.handle_name].get_nowait()
                res = self._interface.train_step(self._model, data)  # -> Dict
            elif request.handle_name == "generate":
                data, buffer_indices, seqlens, output_key_remap = self.__compute_input_queues[
                    request.handle_name].get_nowait()
                res = self._interface.generate(self._model, data)  # -> NamedArray
                if res is not None:
                    new_res = {}
                    for k, v in res.items():
                        if k in output_key_remap:
                            new_res[output_key_remap[k]] = v
                        else:
                            new_res[k] = v
                    new_res = namedarray.from_dict(new_res)
                    res = new_res, buffer_indices, seqlens
            elif request.handle_name == "evaluate":
                res = self._interface.evaluate(self._model, self._eval_dataloader)  # -> Dict
            elif request.handle_name == "save":
                res = self._interface.save(self._model, data)  # -> None
            else:
                raise NotImplementedError(f"Unknown request type: {request.handle_name}.")

            if self._is_stream_pipe and isinstance(res, tuple):
                # When using stream pipe engine and future interface,
                # there are two kinds of APIs in the interface, one is blocking API that returns the result directly,
                # the other is non-blocking API that returns a future object.
                # When handling non-blocking API, we only store the future and leave the job to the engine
                # and check/poll the result later.
                future, cache_data = res
                assert isinstance(future, EngineFuture)
                self.__future_storage[request.request_id] = future
                self.__post_hook_data_storage[request.request_id] = cache_data
                self.__request_time[request.request_id] = tik
                self.logger.info(
                    f"Model worker #{handler_model_name}# issued future request *{request.handle_name}*.")
            else:
                if self._is_dp_head and self._dp_rank == 0:
                    blogger.info(f"Model worker #{handler_model_name}# handle request *{request.handle_name}*"
                                 f" in ${time.perf_counter() - tik:.4f}$s")
                self.__reply_queue.put_nowait((request, res))

        # self.logger.info(f"Model worker {self.__worker_index} #{request.handler}# "
        #                          f"finish handling request *{request.handle_name}*, "
        #                          f"request_id {request.request_id}.")
        sample_count = data.length(0) if isinstance(data, namedarray.NamedArray) else 1
        self.__request_sample_size[request.request_id] = sample_count

    def __sync_param_repartition(self, other_model_name: str) -> Dict[int, List[int]]:
        """Get the right parameters to sync."""
        from impl.model.nn.flash_mqat.flash_mqat_base import (flash_model_embed_param_count,
                                                              flash_model_head_param_count,
                                                              flash_model_tblock_param_count)
        from impl.model.nn.flash_mqat.flash_mqat_parallel import (partition_pipeline_layers,
                                                                  pipeline_repartition_strategy)

        self_model_name = base.constants.model_name()
        other_pp_size = self.config.model_topos[other_model_name].get_dim("pipe")

        mconfig = self.__models[self_model_name].module.module.config
        src_layer_partition = partition_pipeline_layers(
            mconfig,
            base.constants.pipe_parallel_world_size(),
            flash_model_embed_param_count,
            flash_model_tblock_param_count,
            flash_model_head_param_count,
        )
        src_layer_mapping = {k: list(range(v[0], v[1])) for k, v in src_layer_partition.items()}

        dst_layer_partition = partition_pipeline_layers(
            mconfig,
            other_pp_size,
            flash_model_embed_param_count,
            flash_model_tblock_param_count,
            flash_model_head_param_count,
        )
        dst_layer_mapping = {k: list(range(v[0], v[1])) for k, v in dst_layer_partition.items()}

        repartition_strategy = pipeline_repartition_strategy(src_layer_mapping, dst_layer_mapping)
        rs = {k[1]: v for k, v in repartition_strategy.items() if k[0] == base.constants.pipe_parallel_rank()}
        return {k: sorted(rs[k]) for k in sorted(rs.keys())}

    def __send_params_to_sync(self, dst: str):
        from impl.model.nn.flash_mqat.flash_mqat_api import FlashMQATModel
        from impl.model.nn.flash_mqat.flash_mqat_parallel import mp_partition_flash_mqat_state_dict

        rs = self.__sync_param_repartition(dst)

        src = base.constants.model_name()
        src_mp_size = base.constants.model_parallel_world_size()
        dst_mp_size = self.config.model_topos[dst].get_dim("model")

        m: FlashMQATModel = self.__models[src].module.module
        state_dict = m.state_dict()

        _comm_handles = []
        for dst_pp_rank, global_layer_indices in rs.items():
            sub_sd = {k: v for k, v in state_dict.items() if int(k.split(".")[0]) in global_layer_indices}
            for k, v in sub_sd.items():

                if dst_mp_size > src_mp_size:
                    factor = dst_mp_size // src_mp_size
                    sds = mp_partition_flash_mqat_state_dict({k: v}, m.config, factor)
                    assert all(len(sd) == 1 for sd in sds)
                    dst_mp_ranks = [i + factor * base.constants.model_parallel_rank() for i in range(factor)]
                    params = [list(sd.values())[0] for sd in sds]
                else:
                    factor = src_mp_size // dst_mp_size
                    dst_mp_ranks = [base.constants.model_parallel_rank() // factor]
                    params = [v]

                assert len(dst_mp_ranks) == len(params)
                for dst_mp_rank, param in zip(dst_mp_ranks, params):
                    key = gpu_utils.ParamSyncPair(
                        src=src,
                        dst=dst,
                        src_mp_rank=base.constants.model_parallel_rank(),
                        src_pp_rank=base.constants.pipe_parallel_rank(),
                        dst_pp_rank=dst_pp_rank,
                        dst_mp_rank=dst_mp_rank,
                    )

                    handle = dist.broadcast(
                        param,
                        src=self.__pg_info.param_sync_src_ranks[key],
                        group=self.__pg_info.param_sync_groups[key],
                        async_op=True,
                    )
                    _comm_handles.append(handle)

        [h.wait() for h in _comm_handles]

    def __recv_params_to_sync(self, src: str):
        from impl.model.nn.flash_mqat.flash_mqat_api import FlashMQATModel
        from impl.model.nn.flash_mqat.flash_mqat_parallel import (mp_merge_flash_mqat_state_dict,
                                                                  mp_partition_flash_mqat_state_dict)

        rs = self.__sync_param_repartition(src)

        dst = base.constants.model_name()
        dst_mp_size = base.constants.model_parallel_world_size()
        src_mp_size = self.config.model_topos[src].get_dim("model")

        pp_recv_srcs = {}
        for _src_pp_rank, layer_indices in rs.items():
            pp_recv_srcs.update({i: _src_pp_rank for i in layer_indices})

        m: FlashMQATModel = self.__models[dst].module.module
        state_dict = m.state_dict()

        new_state_dict = {}
        for layer_idx in range(m.layer_idx_start, m.layer_idx_end):
            sub_sd = {k: v for k, v in state_dict.items() if int(k.split(".")[0]) == layer_idx}
            for k, v in sub_sd.items():
                src_pp_rank = pp_recv_srcs[layer_idx]

                if dst_mp_size >= src_mp_size:
                    factor = dst_mp_size // src_mp_size
                    src_mp_ranks = [base.constants.model_parallel_rank() // factor]
                    bufs = [
                        base.constants.get_global_memory_buffer().get_tensor(v.shape,
                                                                             v.dtype,
                                                                             name="param_sync_0")
                    ]
                else:
                    factor = src_mp_size // dst_mp_size
                    src_mp_ranks = [i + factor * base.constants.model_parallel_rank() for i in range(factor)]
                    shape = list(
                        mp_partition_flash_mqat_state_dict({
                            k: v
                        }, m.config, factor, mp_rank=0).values())[0].shape
                    bufs = [
                        base.constants.get_global_memory_buffer().get_tensor(shape,
                                                                             v.dtype,
                                                                             name=f"param_sync_{i}")
                        for i in range(factor)
                    ]

                _comm_handles = []
                assert len(src_mp_ranks) == len(bufs)
                for src_mp_rank, buf in zip(src_mp_ranks, bufs):
                    key = gpu_utils.ParamSyncPair(
                        src=src,
                        dst=dst,
                        src_mp_rank=src_mp_rank,
                        src_pp_rank=src_pp_rank,
                        dst_pp_rank=base.constants.pipe_parallel_rank(),
                        dst_mp_rank=base.constants.model_parallel_rank(),
                    )
                    handle = dist.broadcast(
                        buf,
                        src=self.__pg_info.param_sync_src_ranks[key],
                        group=self.__pg_info.param_sync_groups[key],
                        async_op=True,
                    )
                    _comm_handles.append(handle)

                [h.wait() for h in _comm_handles]

                new_v = mp_merge_flash_mqat_state_dict([{k: buf} for buf in bufs], m.config)
                new_state_dict.update(new_v)

        m.load_state_dict(new_state_dict)

    def __data_transfer_among_workers(self, request: request_reply_stream.Payload):
        from impl.model.nn.flash_mqat.flash_mqat_parallel import pipeline_repartition_strategy

        keys = request.data["keys"]
        target = request.data["target"]
        global_buffer_indices = request.buffer_indices
        global_seqlens = request.seqlens

        data = collections.defaultdict(list)
        for k in keys:
            local_buffer_indices = []
            local_seqlens = []

            producer_name = request.data["producer_names"][k]
            producer_mapping = request.data["producer_mappings"][(producer_name, k)]
            target_mapping = request.data["target_mapping"]
            if producer_name not in self.__models and target not in self.__models:
                continue

            # partition mapping starts from zero, which is different from buffer indices
            repart_strat = pipeline_repartition_strategy(producer_mapping, target_mapping)

            target_dp_rank = producer_dp_rank = None
            if target in self.__models:
                with base.constants.model_scope(target):
                    target_dp_rank = base.constants.data_parallel_rank()
            if producer_name in self.__models:
                with base.constants.model_scope(producer_name):
                    producer_dp_rank = base.constants.data_parallel_rank()
                    producer_mp_rank = base.constants.model_parallel_rank()
                    producer_pp_rank = base.constants.pipe_parallel_rank()
                    producer_pp_size = base.constants.pipe_parallel_world_size()
                producer_is_dp_head = producer_mp_rank == 0 and producer_pp_rank == producer_pp_size - 1

            for (dp_i, dp_j), comm_slots in repart_strat.items():
                if len(comm_slots) == 0:
                    continue
                if target_dp_rank == dp_j:
                    # receiver
                    group_key = gpu_utils.DataTransferPair(
                        src=producer_name,
                        src_dp_rank=dp_i,
                        dst=target,
                        dst_dp_rank=dp_j,
                    )
                    bcast_src = self.__pg_info.data_transfer_src_ranks[group_key]
                    group = self.__pg_info.data_transfer_groups[group_key]
                    for _i in comm_slots:
                        buf_idx = global_buffer_indices[_i]
                        seqlen = global_seqlens[_i]
                        if bcast_src == dist.get_rank():
                            v = self.__data_owner_storage[(buf_idx, k)]
                        else:
                            buf = _get_tensor_buffer_from_key_and_seqlen(k, seqlen)
                            dist.broadcast(buf, src=bcast_src, group=group)
                            v = buf.clone()
                        data[k].append(v)
                        local_buffer_indices.append(buf_idx)
                        local_seqlens.append(seqlen)
                if producer_dp_rank == dp_i and producer_is_dp_head:
                    # sender
                    group_key = gpu_utils.DataTransferPair(
                        src=producer_name,
                        src_dp_rank=dp_i,
                        dst=target,
                        dst_dp_rank=dp_j,
                    )
                    bcast_src = self.__pg_info.data_transfer_src_ranks[group_key]
                    group = self.__pg_info.data_transfer_groups[group_key]
                    for _i in comm_slots:
                        buf_idx = global_buffer_indices[_i]
                        v = self.__data_owner_storage[(buf_idx, k)]
                        assert v.dtype == _get_dtype_from_key(k), (k, v.dtype, _get_dtype_from_key(k))
                        assert v.shape == _get_shape_from_key_and_seqlen(
                            k, global_seqlens[_i]), (k, v.shape, global_seqlens[_i])
                        dist.broadcast(v, src=bcast_src, group=group)
                        # Mark data as sent and remove it from storage if all targets have received it.
                        rpc_name = request.data["rpc_name"]
                        if rpc_name not in self.__data_send_record[(buf_idx, k)]:
                            self.__data_send_record[(buf_idx, k)].append(rpc_name)
                        if not set(self.data2required_rpc_names[k]).difference(
                                self.__data_send_record[(buf_idx, k)]):
                            self.__data_owner_storage.pop((buf_idx, k))
                            self.__data_send_record.pop((buf_idx, k))

        # if target in self.__models:
        #     with base.constants.model_scope(target):
        #         dist.barrier(group=base.constants.parallelism_group())
        # for producer_name in request.data['producer_names'].keys():
        #     if producer_name in self.__models:
        #         with base.constants.model_scope(target):
        #             dist.barrier(group=base.constants.parallelism_group())

        if len(data) > 0:
            assert target_dp_rank is not None
            input_key_remap = request.data["input_key_remap"]
            _data = []
            l = len(list(data.values())[0])
            for i in range(l):
                d = {}
                for k, v in data.items():
                    if k in input_key_remap:
                        k = input_key_remap[k]
                    d[k] = v[i]
                _data.append(namedarray.from_dict(d))
            r = dataparallel.PackedParallelDataBroker.gather_from(_data)
            self.__compute_input_queues[request.data["handle_name"]].put_nowait(
                (r, local_buffer_indices, local_seqlens, request.data['output_key_remap']))

    def __maybe_engine_poll_step(self):
        for model_name in self.model_names:
            with base.constants.model_scope(model_name):
                if self._is_stream_pipe and self._engine is not None:
                    self._engine: StreamPipeEngine
                    self._engine.poll_one_step()

    def __post_one_response(self, request: request_reply_stream.Payload, res):
        if isinstance(res, tuple) and isinstance(res[0], namedarray.NamedArray):
            res, buffer_indices, seqlens = res
            new_seqlens = _get_seqlens_from_batch_sample(res)
            if new_seqlens is None:
                new_seqlens = np.array(seqlens)
            reply = request_reply_stream.Payload(
                handler="master",
                request_id=request.request_id,
                handle_name=request.handle_name,
                data=list(res.keys()),
                seqlens=list(new_seqlens),
                buffer_indices=list(buffer_indices),
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

        if isinstance(request.handler, config_system.ModelShardID) and isinstance(res, namedarray.NamedArray):
            with base.constants.model_scope(request.handler.model_name):
                if self._is_dp_head:
                    xs = split_packed_batch_into_seqs(res, torch.from_numpy(new_seqlens).cuda())
                    for buffer_idx, x in zip(buffer_indices, xs):
                        for k, v in x.items():
                            self.__data_owner_storage[(buffer_idx, k)] = v

    def __maybe_post_responses(self):
        ready_to_post = []
        try:
            request, res = self.__reply_queue.get_nowait()
            ready_to_post.append((request, res))
        except queue.Empty:
            pass

        for request_id, future in self.__future_storage.items():
            future: EngineFuture
            if future.done():
                request = self.__request_storage[request_id]
                data = self.__post_hook_data_storage.pop(request_id)
                with base.constants.model_scope(request.handler.model_name):
                    res = self._interface.execute_post_hook(request.handle_name, self._model, data, future)
                tik = self.__request_time.pop(request_id)
                blogger.info(
                    f"Model worker #{request.handler.model_name}# handle request *{request.handle_name}*"
                    f" in ${time.perf_counter() - tik:.4f}$s")
                ready_to_post.append((request, res))

        batch_size = sample_size = 0
        for request, res in ready_to_post:
            request: request_reply_stream.Payload
            self.__post_one_response(request, res)

            if isinstance(request.handler, config_system.ModelShardID):
                with base.constants.model_scope(request.handler.model_name):
                    if self._is_stream_pipe:
                        self.__request_storage.pop(request.request_id)
                        if request.request_id in self.__future_storage:
                            self.__future_storage.pop(request.request_id)
            sample_size += self.__request_sample_size.pop(request.request_id)
            batch_size += 1
        return worker_base.PollResult(sample_count=sample_size, batch_count=batch_size)

    def _poll(self):
        if not self.__dist_env_resolved:
            self.__lazy_setup()
            self.__dist_env_resolved = True
            self.tracer.start()

        if self.__has_dataset:
            self.__prefetch_from_dataset()

        st = time.monotonic()
        self.__maybe_receive_request()

        self.__model_poll_step()

        poll_st = time.monotonic()
        self.__maybe_engine_poll_step()
        pt = time.monotonic() - poll_st
        r = self.__maybe_post_responses()

        if r.batch_count > 0:
            if self.config.cuda_cache_cleanliness and self.__clear_cache_frequency.check():
                # following huggingface trl # ALWAYS COST 0.3+ SEC
                st = time.monotonic()
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()
                et = time.monotonic()
                blogger.debug(f"Model worker {self.__worker_index} cleared cache in {et-st:.4f}s")

            tik = time.perf_counter()
            # blogger.debug(("Model worker #{}#: MemAllocated=*{}*GB, MaxMemAllocated=${}$GB".format(
            #     ",".join(self.model_names),
            #     round(get_accelerator().memory_allocated() / 1024**3, 2),
            #     round(get_accelerator().max_memory_allocated() / 1024**3, 2),
            # )))
            blogger.debug(f"monitoring overhead {time.perf_counter()-tik}s")
            if os.environ.get("DLLM_TRACE", "0") == "1":
                self.tracer.save()
                for model_name in self.model_names:
                    with base.constants.model_scope(model_name):
                        if self._is_stream_pipe:
                            assert isinstance(self._engine, StreamPipeEngine)
                            blogger.debug(f"Tracer for controller saving ... ")
                            self._engine.save_tracer()

        t = time.monotonic() - st
        self.__total_time += t
        self.__engine_poll_time += pt
        # blogger.debug(
        #     f"Model worker #{','.join(self.model_names)}# poll time: {t:.4f}s, engine poll time {pt:.4f}s, percent {pt/t:.4f}"
        # )
        if r.batch_count > 0:
            blogger.debug(
                f"Total time {self.__total_time:.4f}s, engine poll time {self.__engine_poll_time:.4f}s, "
                f"percent {self.__engine_poll_time/self.__total_time:.4f}")
        return r
