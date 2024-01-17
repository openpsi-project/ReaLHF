from typing import Dict
import gc
import itertools
import os
import queue
import socket
import time

from deepspeed.accelerator import get_accelerator
import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data

from base.monitor import time_mark
from base.topology import PipelineParallelGrid
from impl.model.backend.pipe_engine.stream_pipe_engine import EngineFuture, StreamPipeEngine
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


def _str_to_torch_dtype(dtype: str):
    return torch.from_numpy(np.zeros(1, dtype=np.dtype(dtype))).dtype


class ModelWorker(worker_base.Worker):

    def __init__(self, server=None):
        super().__init__(server)
        self.config = None
        self.model_name = None

        self.__ddp_env_resolved = False
        self.__ddp_rank = None

        self.__stream = None

        # log info
        self.__total_time = 0.01
        self.__engine_poll_time = 0

    def _configure(self, cfg: config.ModelWorker):
        self.config = cfg
        self.model_name = cfg.model_name

        self.__experiment_name = self.config.worker_info.experiment_name
        self.__trial_name = self.config.worker_info.trial_name

        # NOTE: here worker_index is different from peer/ddp rank
        self.__worker_index = cfg.worker_info.worker_index

        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
        torch.backends.cudnn.deterministic = cfg.cudnn_deterministic

        seeding.set_random_seed(cfg.seed)

        self.__mw_id = config.ModelWorkerID(
            self.config.model_name,
            dp_rank=self.config.dp_rank,
            mp_rank=self.config.mp_rank,
            pp_rank=self.config.pp_rank,
        )

        # Reveal DDP identity of this worker to world.
        # NOTE: We include master worker in the process group, so the global rank is model_worker_index + 1
        gpu_utils.reveal_ddp_identity(self.__experiment_name, self.__trial_name, self.__worker_index + 1)
        self.__ddp_env_resolved = False

        self.__clear_cache_frequency = base.timeutil.FrequencyControl(
            frequency_steps=self.config.cuda_cache_clear_freq)

        r = self.config.worker_info
        r.model_name = cfg.model_name
        return r

    def __lazy_setup(self):
        """Setup pytorch ddp processes, and algorithms."""
        self.__stream = request_reply_stream.make_worker_stream(
            self.config.worker_info,
            self.config.stream,
        )

        base.constants.set_model_name(self.config.model_name)
        self.__pg_info = gpu_utils.setup_ddp(
            self.__experiment_name,
            self.__trial_name,
            self.__worker_index + 1,
            mw_topos=self.config.mw_topos,
        )
        for model_name_ in self.config.mw_topos:
            base.constants.set_parallelism_group(
                model_name_,
                self.__pg_info.mw_groups[model_name_],
            )

        base.constants.set_experiment_trial_names(self.__experiment_name, self.__trial_name)

        logger.info(f"SetUp Information - Model worker index {self.__worker_index}"
                    f' type "{self.config.model_name}" located at '
                    f"{socket.gethostname()} GPU {self.__pg_info.local_gpu_id}.")

        # if self.config.backend.type_ in ["ds_train", "ds_inference"]:
        deepspeed.init_distributed()
        self.logger.info("deepspeed init distributed on model worker")
        self.__device = torch.device("cuda:0")

        offset = 1
        for model_name_, topo_ in self.config.mw_topos.items():
            grid = PipelineParallelGrid(
                topology=topo_,
                world_size=topo_.world_size(),
                process_group=self.__pg_info.mw_groups[model_name_],
                process_group_offset=offset,
            )
            base.constants.set_grid(model_name_, grid)
            offset += topo_.world_size()

        self.__model = api.model.make_model(
            self.config.model,
            name=self.model_name,
            device=self.__device,
        )
        self.__interface = api.model.make_interface(self.config.interface)
        self.__backend = api.model.make_backend(self.config.backend)
        self.__engine = None

        if self.config.eval_datasets is not None and self.config.eval_dataloader is not None:
            eval_datasets = [
                api.data.make_dataset(
                    d,
                    self.config.seed,
                    self.config.dp_rank,
                    self.config.topo.get_dim("data"),
                    self.__model.tokenizer,
                    self.config.worker_info.experiment_name,
                    self.config.worker_info.trial_name,
                    cache_root=(None
                                if not self.config.use_dataset_cache else self.config.dataset_cahce_root),
                ) for d in self.config.eval_datasets
            ]
            if len(eval_datasets) > 1:
                eval_dataset = torch.utils.data.ConcatDataset(eval_datasets)
            else:
                eval_dataset = eval_datasets[0]
            eval_dataloader = api.data.make_dataloader(self.config.eval_dataloader, eval_dataset)
        else:
            eval_dataloader = None
        self.__eval_dataloader = eval_dataloader

        self._mp_rank = base.constants.model_parallel_rank()
        self._pp_rank = base.constants.pipe_parallel_rank()
        self._dp_rank = base.constants.data_parallel_rank()
        self._pp_size = base.constants.pipe_parallel_world_size()

        # NOTE: Here "model_parallel_group" is the group of model *AND* pipeline parallel, thanks to deepspeed.
        self._bgroup = base.constants.grid().get_model_parallel_group()
        self._bsrc = dp_head_global_rank = (
            self.config.topo.get_rank(data=self._dp_rank, pipe=self._pp_size - 1, model=0) +
            base.constants.process_group_offset())
        self.logger.info(f"Get broadcast src global_rank={dp_head_global_rank} "
                         f"with dp_rank={self._dp_rank}, pp_rank={self._pp_size-1}, mp_rank=0")

        # DP head will receive data from the master, broadcast to all data parallel peers.
        # It will also return result back to master, while other workers in the data parallel group return None.
        self._is_dp_head = self._mp_rank == 0 and self._pp_rank == self._pp_size - 1
        self._is_dp_pp_head = self._mp_rank == 0

        # One for each RPC, e.g., generation and train_step use different buffers.
        self.__scatter_buffers: Dict[Dict[str, torch.Tensor]] = {}
        self.__gather_buffers: Dict[Dict[str, torch.Tensor]] = {}

        self.__request_queue = queue.Queue(maxsize=8)
        self.__reply_queue = queue.Queue(maxsize=8)
        self.__request_sample_size = dict()

        # only used by future interfaces and stream pipe engine
        self.__is_stream_pipe = self.__interface.is_future_interface

        self.__request_storage = dict()  # mapping from request id to requests
        self.__future_storage = dict()  # mapping from request id to corresponding future
        self.__post_hook_data_storage = dict()

    def __maybe_receive_request(self):
        recv_tik = time.perf_counter()
        try:
            request: request_reply_stream.Payload = self.__stream.poll()
        except request_reply_stream.NoMessage:
            return

        logger.info(f"(dp, mp, pp)=({self._dp_rank}, {self._mp_rank}, {self._pp_rank}) receive request")
        self.__request_storage[request.request_id] = request
        # ACK message to indicate ready to run dist.scatter
        if request.is_tensor and self._is_dp_pp_head:
            logger.info(
                f"(dp, mp, pp)=({self._dp_rank}, {self._mp_rank}, {self._pp_rank}) receive tensor request {request.handle_name}, send ack"
            )
            assert request.ack_reply_id is not None
            ack = request_reply_stream.Payload(handle_name=request.handle_name,
                                               request_id=request.ack_reply_id)
            self.__stream.post(ack)

        if request.handle_name not in self.__gather_buffers:
            self.__gather_buffers[request.handle_name] = {}
        if request.handle_name not in self.__scatter_buffers:
            self.__scatter_buffers[request.handle_name] = {}

        scatter_buffer = self.__scatter_buffers[request.handle_name]

        data = request.data
        if request.is_tensor:
            assert data is None

            # Maybe create or extend the size of scatter buffer.
            for (k, buf_shape), dtype in zip(request.buf_shapes.items(), request.dtypes.values()):
                if k not in scatter_buffer:
                    if self._is_dp_head:
                        logger.info(f"Create scatter buffer key {k} with shape {buf_shape}")
                    scatter_buffer[k] = torch.empty(buf_shape, dtype=dtype, device=self.__device)
                elif k in scatter_buffer and not base.numpy_utils.shape_leq(buf_shape,
                                                                            scatter_buffer[k].shape):
                    if self._is_dp_head:
                        logger.info(f"Resizing scatter buffer key {k} "
                                    f"from {scatter_buffer[k].shape} to {buf_shape}")
                    padding = tuple(
                        itertools.chain.from_iterable(
                            reversed([(0, target_size - current_size)
                                      for target_size, current_size in zip(buf_shape, scatter_buffer[k].shape)
                                      ])))
                    scatter_buffer[k] = torch.nn.functional.pad(scatter_buffer[k], padding, "constant", 0)

            # if self._is_dp_head:
            # Receive from the master worker
            for k in request.buf_shapes:
                dist.scatter(
                    scatter_buffer[k],
                    scatter_list=None,
                    src=0,
                    group=self.__pg_info.mas_pp_stage_groups[self.model_name][self._pp_rank],
                )
                logger.info(f"request {request.handle_name} scatter {k} done, {scatter_buffer[k][0]}")

            # Broadcast to the DP group / receive from the DP head
            # for k in request.buf_shapes:
            #     logger.info(f"before broadcast, scatter_buffer[k] shape {scatter_buffer[k].shape}")
            #     dist.broadcast(scatter_buffer[k], src=self._bsrc, group=self._bgroup)
            #     logger.info(f"request {request.handle_name} broadcast {k} done, {scatter_buffer[k][0]}")

            # Slice the array to get the actual data
            data = {}
            for k, target_shape in request.actual_shapes.items():
                s = tuple(slice(0, target_size) for target_size in target_shape)
                data[k] = scatter_buffer[k][s]
            data = namedarray.from_dict(data)

        if self._is_dp_head:
            self.logger.info(
                f"Model {self.model_name} receive request {request.handle_name} time: {time.perf_counter() - recv_tik}"
            )

        self.__request_queue.put_nowait((request, data))

    def __model_poll_step(self) -> worker_base.PollResult:
        # interface
        try:
            request, data = self.__request_queue.get_nowait()
        except queue.Empty:
            return

        request: request_reply_stream.Payload
        tik = time.perf_counter()

        self.logger.info(f"Model worker #{self.model_name}# start handle request *{request.handle_name}*, "
                         f"request_id {request.request_id}.")
        try:
            worker_identifier = self.__mw_id
            if request.handle_name == "initialize":
                self.__model = self.__backend.initialize(self.__model, data)
                self.__engine = self.__model.module
                if self.__is_stream_pipe:
                    assert isinstance(self.__engine, StreamPipeEngine)
                res = None
            elif request.handle_name == "save":
                res = self.__interface.save(self.__model, data)  # -> None
            elif request.handle_name == "inference":
                res = self.__interface.inference(self.__model, data)  # -> NamedArray
            elif request.handle_name == "train_step":
                res = self.__interface.train_step(self.__model, data)  # -> Dict
            elif request.handle_name == "generate":
                res = self.__interface.generate(self.__model, data)  # -> NamedArray
            elif request.handle_name == "evaluate":
                res = self.__interface.evaluate(self.__model, self.__eval_dataloader)  # -> Dict
            else:
                raise NotImplementedError(f"Unknown request type: {request.handle_name}.")
        except RuntimeError as e:
            # We may print some info here.
            raise e

        if self.__is_stream_pipe and isinstance(res, tuple):
            # When using stream pipe engine and future interface,
            # there are two kinds of APIs in the interface, one is blocking API that returns the result directly,
            # the other is non-blocking API that returns a future object.
            # When handling non-blocking API, we only store the future and leave the job to the engine
            # and check/poll the result later.
            future, cache_data = res
            assert isinstance(future, EngineFuture)
            self.__future_storage[request.request_id] = future
            self.__post_hook_data_storage[request.request_id] = cache_data
            self.logger.info(
                f"Model worker #{self.model_name}# issued future request *{request.handle_name}*.")
        else:
            if self._is_dp_head:
                blogger.info(f"Model worker #{self.model_name}# handle request *{request.handle_name}*"
                             f" in ${time.perf_counter() - tik:.4f}$s")
            self.__reply_queue.put_nowait((request, res))

        sample_count = data.length(0) if isinstance(data, namedarray.NamedArray) else 1
        self.__request_sample_size[request.request_id] = sample_count

    def __maybe_engine_poll_step(self):
        if self.__is_stream_pipe and self.__engine is not None:
            self.__engine: StreamPipeEngine
            self.__engine.poll_one_step()

    def __post_one_response(self, request: request_reply_stream.Payload, res):
        gather_buffer = self.__gather_buffers[request.handle_name]

        if not self._is_dp_pp_head:
            # Discard returned data if not DP+PP head.
            return

        if isinstance(res, namedarray.NamedArray):
            shapes = {k: v.shape for k, v in res.items()}
            dtypes = {k: v.dtype for k, v in res.items()}

            all_shapes = [None for _ in range(self.config.topo.get_dim("data"))]
            dist.all_gather_object(
                all_shapes,
                shapes,
                group=base.constants.grid().dp_head_group,
            )
            buf_shapes = {}
            for k in shapes:
                buf_shapes[k] = base.numpy_utils.shape_union(*[tuple(s[k]) for s in all_shapes])

            # Expand buffer shape if necessary.
            for (k, dtype), buf_shape in zip(dtypes.items(), buf_shapes.values()):
                if k not in gather_buffer or (k in gather_buffer and not base.numpy_utils.shape_leq(
                        buf_shape, gather_buffer[k].shape)):
                    if k in gather_buffer:
                        logger.info(
                            f"Resizing gather buffer key {k} from {gather_buffer[k].shape} to {buf_shape}")
                    else:
                        logger.info(f"Create gather buffer key {k} with shape {buf_shape}")
                    gather_buffer[k] = torch.empty(buf_shape, dtype=dtype, device=self.__device)

            reply = request_reply_stream.Payload(
                request_id=request.request_id,
                handle_name=request.handle_name,
                is_tensor=True,
                actual_shapes=shapes,
                buf_shapes=buf_shapes,
                dtypes=dtypes,
                seqlens=request.seqlens,
                buffer_indices=request.buffer_indices,
            )
        else:
            reply = request_reply_stream.Payload(
                request_id=request.request_id,
                handle_name=request.handle_name,
                is_tensor=False,
                data=res,
            )

        self.__stream.post(reply)

        if reply.is_tensor and self._is_dp_head:
            # Copy data to the gather buffer.
            for k, v in res.items():
                logger.info(f"Gathering {k} with shape {v.shape}")
                s = tuple(slice(0, size) for size in v.shape)
                gather_buffer[k][s] = v
                dist.gather(
                    gather_buffer[k],
                    gather_list=None,
                    dst=0,
                    group=self.__pg_info.mas_dp_head_groups[self.model_name],
                )

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
                res = self.__interface.execute_post_hook(request.handle_name, self.__model, data, future)
                ready_to_post.append((request, res))

        batch_size = sample_size = 0
        for request, res in ready_to_post:
            request: request_reply_stream.Payload
            self.__post_one_response(request, res)

            if self.__is_stream_pipe:
                self.__request_storage.pop(request.request_id)
                if request.request_id in self.__future_storage:
                    self.__future_storage.pop(request.request_id)
            sample_size += self.__request_sample_size.pop(request.request_id)
            batch_size += 1
        return worker_base.PollResult(sample_count=sample_size, batch_count=batch_size)

    def _poll(self):
        if not self.__ddp_env_resolved:
            self.__lazy_setup()
            self.__ddp_env_resolved = True
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
                if self._is_dp_head:
                    blogger.debug(f"Model worker {self.model_name} cleared cache in {et-st:.4f}s")

            tik = time.perf_counter()
            blogger.debug(("Model worker #{}#: MemAllocated=*{}*GB, MaxMemAllocated=${}$GB".format(
                self.model_name,
                round(get_accelerator().memory_allocated() / 1024**3, 2),
                round(get_accelerator().max_memory_allocated() / 1024**3, 2),
            )))
            blogger.debug(f"monitoring overhead {time.perf_counter()-tik}s")
            if os.environ.get("DLLM_TRACE", "0") == "1":
                self.tracer.save()
                if self.__is_stream_pipe:
                    assert isinstance(self.__engine, StreamPipeEngine)
                    blogger.debug(f"Tracer for controller saving ... ")
                    self.__engine.save_tracer()

        t = time.monotonic() - st
        self.__total_time += t
        self.__engine_poll_time += pt
        # blogger.debug(
        #     f"Model worker #{self.model_name}# poll time: {t:.4f}s, engine poll time {pt:.4f}s, percent {pt/t:.4f}"
        # )
        if r.batch_count > 0:
            blogger.debug(
                f"Total time {self.__total_time:.4f}s, engine poll time {self.__engine_poll_time:.4f}s, "
                f"percent {self.__engine_poll_time/self.__total_time:.4f}")
        return r
