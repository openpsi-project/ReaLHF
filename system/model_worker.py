from typing import Any, Dict
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
from base.constants import GlobalMemoryBuffer
import api.config as config
import api.data
import api.model
import base.constants
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


class ModelWorker(worker_base.Worker):

    def __init__(self, server=None):
        super().__init__(server)

        self.__dist_env_resolved = False

        self.__stream = None

        # log info
        self.__total_time = 0.01
        self.__engine_poll_time = 0

    def _configure(self, cfg: config.ModelWorker):
        self.config = cfg
        self.model_names = [s.id.model_name for s in cfg.shards]
        self.shard_indices = [
            cfg.model_topos[s.id.model_name].get_rank(data=s.id.dp_rank,
                                                      pipe=s.id.pp_rank,
                                                      model=s.id.mp_rank) for s in cfg.shards
        ]

        self.__experiment_name = self.config.worker_info.experiment_name
        self.__trial_name = self.config.worker_info.trial_name

        # NOTE: here worker_index is different from peer/ddp rank
        self.__worker_index = cfg.worker_info.worker_index

        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
        torch.backends.cudnn.deterministic = cfg.cudnn_deterministic

        seeding.set_random_seed(cfg.seed)

        # Reveal DDP identity of this worker to world.
        # NOTE: We include master worker in the process group, so the global rank is model_worker_index + 1
        gpu_utils.reveal_ddp_identity(self.__experiment_name, self.__trial_name, self.__worker_index + 1)
        self.__dist_env_resolved = False

        self.__clear_cache_frequency = base.timeutil.FrequencyControl(
            frequency_steps=self.config.cuda_cache_clear_freq)

        r = self.config.worker_info
        return r

    def __lazy_setup(self):
        """Setup pytorch ddp processes, and algorithms."""
        self.__stream = request_reply_stream.make_worker_stream(
            self.config.worker_info,
            sub_patterns=[s.id for s in self.config.shards],
        )

        self.__pg_info = gpu_utils.setup_ddp(
            expr_name=self.__experiment_name,
            trial_name=self.__trial_name,
            worker_index=self.__worker_index + 1,
            model_topos=self.config.model_topos,
            msid2mwid=self.config.msid2mwid,
            mw_bcast_groups=self.config.mw_bcast_groups,
        )

        base.constants.set_experiment_trial_names(self.__experiment_name, self.__trial_name)

        logger.info(f"SetUp Information - Model worker index {self.__worker_index} located at "
                    f"{socket.gethostname()} GPU {self.__pg_info.local_gpu_id}.")

        # if self.config.backend.type_ in ["ds_train", "ds_inference"]:
        deepspeed.init_distributed()
        self.logger.info("deepspeed init distributed on model worker")
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
                self.__interfaces[s.id.model_name] = api.model.make_interface(s.interface)
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

        self.__reply_storage: Dict[uuid.UUID, namedarray.NamedArray] = dict()

        # A monitoring process.
        self.__gpu_util_mp = mp.Process(target=gpu_utilization_monitor,
                                        args=(self.__pg_info.local_gpu_id, 7200))
        self.__gpu_util_mp.start()

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

    def __maybe_receive_request(self):
        recv_tik = time.perf_counter()
        try:
            request: request_reply_stream.Payload = self.__stream.poll()
        except request_reply_stream.NoMessage:
            return

        # logger.info(f"(dp, mp, pp)=({self._dp_rank}, {self._mp_rank}, {self._pp_rank}) receive request")
        self.__request_storage[request.request_id] = request
        handler = request.handler
        # ACK message to indicate ready to run dist.scatter
        if request.is_tensor:
            # logger.info(
            #     f"(dp, mp, pp)=({self._dp_rank}, {self._mp_rank}, {self._pp_rank}) receive tensor request {request.handle_name}, send ack"
            # )
            assert request.ack_reply_id is not None
            ack = request_reply_stream.Payload(
                handler="master",
                handle_name=request.handle_name,
                request_id=request.ack_reply_id,
            )
            self.__stream.post(ack)

        data = request.data
        if request.is_tensor:
            assert data is None

            data = {}
            # Maybe create or extend the size of scatter buffer.
            pg_idx = [
                dist.get_rank(group) != -1 for group in self.__pg_info.scatter_groups[handler.model_name]
            ].index(True)
            scatter_group = self.__pg_info.scatter_groups[handler.model_name][pg_idx]
            for (k, buf_shape), dtype, actual_shape in zip(request.buf_shapes.items(),
                                                           request.dtypes.values(),
                                                           request.actual_shapes.values()):
                buf = base.constants.get_global_memory_buffer().get_tensor(buf_shape, dtype, "scatter_gather")
                dist.scatter(
                    buf,
                    scatter_list=None,
                    src=0,
                    group=scatter_group,
                )
                s = tuple(slice(0, target_size) for target_size in actual_shape)
                data[k] = buf[s].clone()
            data = namedarray.from_dict(data)

        # with base.constants.model_scope(handler.model_name):
        #     if self._is_dp_head:
        #         self.logger.info(
        #             f"Model {handler.model_name} receive request {request.handle_name} time: {time.perf_counter() - recv_tik}"
        #         )

        self.__request_queue.put_nowait((request, data))

    def __model_poll_step(self) -> worker_base.PollResult:
        # interface
        try:
            request, data = self.__request_queue.get_nowait()
        except queue.Empty:
            return

        request: request_reply_stream.Payload
        tik = time.perf_counter()

        # self.logger.info(
        #     f"Model worker {self.__worker_index} #{request.handler}# "
        #     f"start handle request *{request.handle_name}*, "
        #     f"request_id {request.request_id}."
        # )
        with base.constants.model_scope(request.handler.model_name):
            try:
                if request.handle_name == "initialize":
                    self.__models[request.handler.model_name] = self._backend.initialize(self._model, data)
                    self.__engines[request.handler.model_name] = self._model.module
                    if self._is_stream_pipe:
                        assert isinstance(self._engine, StreamPipeEngine)
                    res = None
                elif request.handle_name == "save":
                    res = self._interface.save(self._model, data)  # -> None
                elif request.handle_name == "inference":
                    res = self._interface.inference(self._model, data)  # -> NamedArray
                elif request.handle_name == "train_step":
                    res = self._interface.train_step(self._model, data)  # -> Dict
                elif request.handle_name == "generate":
                    res = self._interface.generate(self._model, data)  # -> NamedArray
                elif request.handle_name == "evaluate":
                    res = self._interface.evaluate(self._model, self._eval_dataloader)  # -> Dict
                elif request.handle_name == "gather_tensor_reply":
                    res = None
                    if self._is_dp_head:
                        res = self.__gather_tensor_reply(request)
                elif request.handle_name == "offload":
                    raise NotImplementedError(f"Offload is not implemented yet.")  # FIXME: offload
                elif request.handle_name == "sync_param":
                    raise NotImplementedError(f"Sync param is not implemented yet.")  # FIXME: sync param
                else:
                    raise NotImplementedError(f"Unknown request type: {request.handle_name}.")
            except RuntimeError as e:
                # We may print some info here.
                raise e

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
                    f"Model worker #{request.handler.model_name}# issued future request *{request.handle_name}*."
                )
            else:
                if self._is_dp_head and self._dp_rank == 0:
                    blogger.info(
                        f"Model worker #{request.handler.model_name}# handle request *{request.handle_name}*"
                        f" in ${time.perf_counter() - tik:.4f}$s")
                self.__reply_queue.put_nowait((request, res))

        sample_count = data.length(0) if isinstance(data, namedarray.NamedArray) else 1
        self.__request_sample_size[request.request_id] = sample_count

    def __gather_tensor_reply(self, request: request_reply_stream.Payload):
        # This is not the ID of "gather" request, but the ID of the original RPC, e.g., generate
        # Used to index the reply storage.
        request_id = request.data
        handler = request.handler
        buf_shapes = request.buf_shapes

        ack = request_reply_stream.Payload(
            handler="master",
            handle_name="gather_tensor_reply",
            request_id=request.ack_reply_id,
        )
        self.__stream.post(ack)
        res: namedarray.NamedArray = self.__reply_storage.pop(request_id)

        pg_idx = [dist.get_rank(group) != -1
                  for group in self.__pg_info.gather_groups[handler.model_name]].index(True)
        gather_group = self.__pg_info.gather_groups[handler.model_name][pg_idx]

        # Copy data to the gather buffer.
        for k, v in res.items():
            # logger.info(f"handle_name {request.handle_name} "
            #             f"Gathering {k} with shape {v.shape}, req id = {request.request_id}")
            if v is None:
                continue
            buf_shape = buf_shapes[k]
            buf = base.constants.get_global_memory_buffer().get_tensor(buf_shape, v.dtype, "scatter_gather")
            s = tuple(slice(0, size) for size in v.shape)
            buf[s] = v
            dist.gather(
                buf,
                gather_list=None,
                dst=0,
                group=gather_group,
            )

    def __maybe_engine_poll_step(self):
        for model_name in self.model_names:
            with base.constants.model_scope(model_name):
                if self._is_stream_pipe and self._engine is not None:
                    self._engine: StreamPipeEngine
                    self._engine.poll_one_step()

    def __post_one_response(self, request: request_reply_stream.Payload, res):
        if isinstance(res, namedarray.NamedArray):
            shapes = {k: v.shape for k, v in res.items() if v is not None}
            dtypes = {k: v.dtype for k, v in res.items() if v is not None}
            reply = request_reply_stream.Payload(
                handler="master",
                request_id=request.request_id,
                handle_name=request.handle_name,
                is_tensor=True,
                actual_shapes=shapes,
                buf_shapes=dict(),
                dtypes=dtypes,
                seqlens=request.seqlens,
                buffer_indices=request.buffer_indices,
            )
        else:
            reply = request_reply_stream.Payload(
                handler="master",
                request_id=request.request_id,
                handle_name=request.handle_name,
                is_tensor=False,
                data=res,
            )

        self.__stream.post(reply)
        # logger.info(f"handle_name {request.handle_name} Posted req id = {request.request_id}")

        with base.constants.model_scope(request.handler.model_name):
            if reply.is_tensor and self._is_dp_head:
                self.__reply_storage[request.request_id] = res

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
            blogger.debug(("Model worker #{}#: MemAllocated=*{}*GB, MaxMemAllocated=${}$GB".format(
                ",".join(self.model_names),
                round(get_accelerator().memory_allocated() / 1024**3, 2),
                round(get_accelerator().max_memory_allocated() / 1024**3, 2),
            )))
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
