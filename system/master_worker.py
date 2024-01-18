from collections import defaultdict
from typing import Dict, List, Optional, Union
import asyncio
import copy
import dataclasses
import gc
import getpass
import itertools
import os
import re
import sys
import threading
import time
import uuid

import colorama
import deepspeed
import numpy as np
import torch
import torch.distributed

from base.asyncio_utils import raise_asyncio_exception, setup_run_until_complete, teardown_run_util_complete
from base.buffer import AsyncIOSequenceBuffer
from base.cluster import spec as cluster_spec
from base.constants import MODEL_SAVE_ROOT
import api.config as config_pkg
import api.data as data_api
import api.dfg
import api.model as model_api
import base.constants
import base.dataparallel as dataparallel
import base.gpu_utils as gpu_utils
import base.logging as logging
import base.namedarray as namedarray
import base.numpy_utils
import base.timeutil
import base.topology
import base.topology as topology
import system.request_reply_stream as request_reply_stream
import system.worker_base as worker_base

logger = logging.getLogger("master worker", "system")
blogger = logging.getLogger("benchmark")


class ExperimentComplete(Exception):

    def __init__(self, message):
        disclaimer = (colorama.Fore.GREEN + "\033[1m" +
                      "<This is not an error. It is just a way to stop the experiment.> ")
        super().__init__(disclaimer + colorama.Style.RESET_ALL + colorama.Fore.YELLOW +
                         colorama.Style.BRIGHT + "\033[1m" + message + colorama.Style.RESET_ALL)


def request_all(
    streams: List[request_reply_stream.RequestReplyStream],
    handle_type: str,
    datas: List[namedarray.NamedArray],
    verbose: bool = True,
):
    """Send request of `handle_type` to multiple streams. len(streams)==len(datas)"""
    requests = [
        request_reply_stream.Payload(
            handle_name=handle_type,
            is_tensor=False,
            data=data,
        ) for data in datas
    ]
    if verbose:
        blogger.debug(f"master worker #request_all# *end* time ${time.time_ns()}$")
    tik = time.perf_counter()
    for s, r in zip(streams, requests):
        s.post(r)
    t = time.perf_counter() - tik
    if verbose:
        blogger.debug(f'Request "{handle_type}" time in total: '
                      f"{t:.4f}s, {t / len(requests):.4f}s per request")
    return [r.request_id for r in requests]


def create_exact_match_pattern(string_list: List[Union[uuid.UUID, str]]) -> re.Pattern:
    escaped_strings = [re.escape(str(s)) for s in string_list]
    pattern = f"({'|'.join(escaped_strings)})$"
    return re.compile(pattern)


async def _awaitable_response(
    stream: request_reply_stream.RequestReplyStream,
    pattern: re.Pattern | None,
) -> request_reply_stream.Payload:
    while True:
        try:
            return stream.poll(pattern=pattern, block=False)
        except request_reply_stream.NoMessage:
            await asyncio.sleep(0.01)
            continue


async def gather_all_replies(
    streams: List[request_reply_stream.RequestReplyStream],
    request_ids: List[str],
    verbose: bool = True,
) -> List:
    """Collect responses from multiple streams. Blocking method."""
    responses = await asyncio.gather(*[
        _awaitable_response(s, pattern=create_exact_match_pattern([req_id]))
        for s, req_id in zip(streams, request_ids)
    ])
    if verbose:
        blogger.debug(f"master worker #gather_all_replies# *end* time ${time.time_ns()}$")
    return responses


async def group_rpc_blocked(
    streams: List[request_reply_stream.RequestReplyStream],
    handle_type: str,
    datas: List[namedarray.NamedArray],
    verbose: bool = True,
) -> List[namedarray.NamedArray]:
    payloads = await gather_all_replies(streams, request_all(streams, handle_type, datas, verbose=verbose))
    return [p.data for p in payloads]


def split_packed_batch_into_seqs(
    sample: namedarray.NamedArray,
    input_lens: Optional[torch.Tensor] = None,
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
    return dataparallel.PackedParallelDataBroker.scatter_to(sample,
                                                            n_dp=len(input_lens),
                                                            partitions=partitions)


@dataclasses.dataclass
class RPCCorountineControl:
    stop: asyncio.Event
    # does not exceed #max_concurrent_calls
    can_do_rpc: asyncio.Semaphore
    # for synchronizing req ids between req and reply coroutines
    request_queue: asyncio.Queue
    # for counting the number of finished training steps
    # one training step corresponds to traversal of the whole DFG
    train_count: asyncio.Queue
    # for loading data
    fetch_data_queue: asyncio.Queue
    # for save and eval model
    eval_queue: asyncio.Queue
    save_queue: asyncio.Queue


async def scatter_tensor_to_mws(
    datas: List[namedarray.NamedArray],
    rpc: api.dfg.ModelRPC,
    streams: List[List[request_reply_stream.RequestReplyStream]],
    all_buffer_indices: List[List[int]],
    all_seqlens: List[List[int]],
    mas_pp_stage_groups: List[torch.distributed.ProcessGroup],
    scatter_buffer: Dict[str, List[torch.Tensor]],
    device: torch.device,
    mp_size: int,
    ctrl: RPCCorountineControl,
) -> List[uuid.UUID]:
    dtypes = {k: v.dtype for k, v in datas[0].items()}
    all_shapes = [{k: v.shape for k, v in data.items()} for data in datas]
    buf_shapes = {}
    for k in dtypes:
        buf_shapes[k] = base.numpy_utils.shape_union(*[tuple(s[k]) for s in all_shapes])

    # Expand scatter buffer if necessary
    _scatter_buffer_changed = False
    for k, buf_shape in buf_shapes.items():
        if k not in scatter_buffer or (k in scatter_buffer and
                                       not base.numpy_utils.shape_leq(buf_shape, scatter_buffer[k][0].shape)):
            # if k in scatter_buffer:
            #     logger.info(f"Resize RPC *{rpc.name}* scatter buffer on master worker for {k}"
            #                 f" from {scatter_buffer[k][0].shape} to {buf_shape}")
            # else:
            #     logger.info(
            #         f"Create RPC *{rpc.name}* scatter buffer on master worker for {k} with shape {buf_shape}")
            scatter_buffer[k] = [
                torch.empty(buf_shape, dtype=dtypes[k], device=device) for _ in range(len(datas) + 1)
            ]
            _scatter_buffer_changed = True
        elif k in scatter_buffer and not base.numpy_utils.shape_leq(buf_shape, scatter_buffer[k][0].shape):
            # logger.info(f"Resize scatter buffer on master worker for {k}"
            #             f" from {scatter_buffer[k][0].shape} to {buf_shape}")
            new_x = []
            for x in scatter_buffer[k]:
                padding = tuple(
                    itertools.chain.from_iterable(
                        reversed([(0, s2 - s1) for s1, s2 in zip(x.shape, buf_shape)])))
                new_x.append(torch.nn.functional.pad(x, pad=padding, mode="constant", value=0))
            scatter_buffer[k] = new_x
            _scatter_buffer_changed = True
    if _scatter_buffer_changed:
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    expanded_buffer = {k: [scatter_buffer[k][0]] for k in scatter_buffer}
    # Put data into scatter buffer
    for i, data in enumerate(datas):
        for k, v in data.items():
            assert len(scatter_buffer[k]) == len(datas) + 1
            s = tuple(slice(0, x) for x in v.shape)
            scatter_buffer[k][i + 1][s] = data[k]

            # expand scatter buffer to model parallel ranks
            expanded_buffer[k].extend([scatter_buffer[k][i + 1]] * mp_size)

    request_ids = []
    for pp_rank, pp_stage_streams in enumerate(streams):
        ack_ids = []
        for stream, shapes, buffer_indices, seqlens in zip(pp_stage_streams, all_shapes, all_buffer_indices,
                                                           all_seqlens):
            req = request_reply_stream.Payload(
                handle_name=rpc.interface_type.value,
                actual_shapes=shapes,
                buf_shapes=buf_shapes,
                dtypes=dtypes,
                is_tensor=True,
                buffer_indices=buffer_indices,
                seqlens=seqlens,
            )
            request_ids.append(stream.post(req))
            ack_ids.append(req.ack_reply_id)

        # logger.info(f"Waiting for ack from stage {pp_rank}")
        # Wait for the ack message from model worker
        await asyncio.gather(*[
            _awaitable_response(s, pattern=create_exact_match_pattern([ack_id]))
            for s, ack_id in zip(pp_stage_streams, ack_ids)
        ])
        # logger.info(f"Scatter data to stage {pp_rank}")

        # Scatter data to DP head model workers.
        for k in scatter_buffer:
            torch.distributed.scatter(
                expanded_buffer[k][0],
                scatter_list=expanded_buffer[k],
                src=0,
                group=mas_pp_stage_groups[pp_rank],
            )
        torch.cuda.synchronize()

    return request_ids


async def model_rpc_request_func(
    rpc: api.dfg.ModelRPC,
    buffer: AsyncIOSequenceBuffer,
    streams: List[List[request_reply_stream.RequestReplyStream]],
    mas_pp_stage_groups: List[torch.distributed.ProcessGroup],
    scatter_buffer: Dict[str, List[torch.Tensor]],
    device: torch.device,
    topo: base.topology.PipeModelDataParallelTopology,
    ctrl: RPCCorountineControl,
):
    dp_size = topo.get_dim("data")
    mp_size = topo.get_dim("model")
    pp_size = topo.get_dim("pipe")
    assert pp_size == len(streams)
    assert dp_size == len(streams[0])

    while not ctrl.stop.is_set():
        await ctrl.can_do_rpc.acquire()
        sample = await buffer.get_batch_for_rpc(rpc)
        # logger.info(f"Model rpc {rpc.name} requesting.")
        datas, n_seqs = dataparallel.get_broker(rpc.dp_broker_type).scatter_to(
            sample.data,
            dp_size,
            return_sizes=True,
        )
        all_buffer_indices = []
        all_seqlens = []
        offset = 0
        for n_seq in n_seqs:
            all_buffer_indices.append(sample.indices[offset:offset + n_seq])
            all_seqlens.append(sample.seqlens[offset:offset + n_seq])
            offset += n_seq

        # sanity check for pipeline parallel
        try:
            for d in datas:
                dataparallel.get_broker(rpc.dp_broker_type).scatter_to(d, pp_size)
        except Exception as e:
            raise RuntimeError(
                f"Data in each data parallel rank cannot be "
                f"partitioned further into {pp_size} mini-batches for pipeline parallel. Exiting.") from e

        # send partitioned data to model workers
        req_ids = await scatter_tensor_to_mws(
            datas=datas,
            rpc=rpc,
            streams=streams,
            all_buffer_indices=all_buffer_indices,
            all_seqlens=all_seqlens,
            mas_pp_stage_groups=mas_pp_stage_groups,
            scatter_buffer=scatter_buffer,
            device=device,
            mp_size=mp_size,
            ctrl=ctrl,
        )
        await ctrl.request_queue.put((req_ids, time.perf_counter()))
        logger.info(f"Model rpc {rpc.name} requested.")


async def gather_tensor_from_mws(
    streams: List[List[request_reply_stream.RequestReplyStream]],
    req_ids: List[uuid.UUID],
    rpc: api.dfg.ModelRPC,
    mas_dp_head_group: torch.distributed.ProcessGroup,
    gather_buffer: Dict[str, List[torch.Tensor]],
    device: torch.device,
    dp_size: int,
    ctrl: RPCCorountineControl,
) -> namedarray.NamedArray:
    # dp_head_streams = streams[-1]
    streams = list(itertools.chain.from_iterable(streams))
    logger.info(f"rpc {rpc.name} waiting for responses {req_ids}")
    responses = await asyncio.gather(*[
        _awaitable_response(s, pattern=create_exact_match_pattern([req_id]))
        for s, req_id in zip(streams, req_ids)
    ])
    logger.info(f"rpc {rpc.name} received responses {req_ids}")

    dp_head_responses = responses[-dp_size:]
    recv_tik = time.perf_counter()

    # logger.info([res.is_tensor for res in responses])
    if dp_head_responses[-1].is_tensor:  # responses[-1] is from dp_head
        all_buf_shapes = [response.buf_shapes for response in dp_head_responses]
        for buf_shapes in all_buf_shapes:
            for k, v in buf_shapes.items():
                assert buf_shapes[k] == all_buf_shapes[0][k]

        # Expand gather buffer if necessary.
        _gather_buffer_changed = False
        for k, buf_shape in buf_shapes.items():
            if k not in gather_buffer or (k in gather_buffer and not base.numpy_utils.shape_leq(
                    buf_shape, gather_buffer[k][0].shape)):
                # if k in gather_buffer:
                #     logger.info(
                #         f"Resize RPC *{rpc.name}* gather buffer on master worker for {k} from {gather_buffer[k][0].shape} to {buf_shape}"
                #     )
                # else:
                #     logger.info(
                #         f"Create RPC *{rpc.name}* gather buffer on master worker for {k} with shape {buf_shape}"
                #     )
                gather_buffer[k] = [
                    torch.empty(buf_shape, dtype=dp_head_responses[0].dtypes[k], device=device)
                    for _ in range(dp_size + 1)
                ]
                _gather_buffer_changed = True
            elif k in gather_buffer and not base.numpy_utils.shape_leq(buf_shape, gather_buffer[k][0].shape):
                logger.info(
                    f"Resize gather buffer on master worker for {k} from {gather_buffer[k][0].shape} to {buf_shape}"
                )
                new_x = []
                for x in gather_buffer[k]:
                    padding = tuple(
                        itertools.chain.from_iterable(
                            reversed([(0, s2 - s1) for s1, s2 in zip(x.shape, buf_shape)])))
                    new_x.append(torch.nn.functional.pad(x, pad=padding, mode="constant", value=0))
                gather_buffer[k] = new_x
                _gather_buffer_changed = True
        if _gather_buffer_changed:
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()

        # only gather from dp heads
        for k in gather_buffer:
            assert len(gather_buffer[k]) == dp_size + 1
            logger.info(f"rpc {rpc.name} Gathering {k} from dp heads")
            torch.distributed.gather(
                gather_buffer[k][0],
                gather_list=gather_buffer[k],
                dst=0,
                group=mas_dp_head_group,
            )

        all_res = []
        for i, response in enumerate(dp_head_responses):
            res_ = {}
            for k, vs in gather_buffer.items():
                assert len(vs) == len(dp_head_responses) + 1
                v = vs[i + 1]
                shape = response.actual_shapes[k]
                s = tuple(slice(0, x) for x in shape)
                res_[k] = v[s]
            all_res.append(namedarray.from_dict(res_))
        res = dataparallel.get_broker(rpc.dp_broker_type).gather_from(all_res)
    else:
        res = dataparallel.get_broker(rpc.dp_broker_type).gather_from(
            [response.data for response in dp_head_responses])
    # logger.info(f"Master worker return from model worker time: {time.perf_counter() - recv_tik:.4f}s")
    return dp_head_responses, rpc.remap_output_keys(res)


async def model_rpc_reply_func(
    rpc: api.dfg.ModelRPC,
    buffer: AsyncIOSequenceBuffer,
    streams: List[List[request_reply_stream.RequestReplyStream]],
    mas_dp_head_group: torch.distributed.ProcessGroup,
    gather_buffer: Dict[str, List[torch.Tensor]],
    device: torch.device,
    topo: base.topology.PipeModelDataParallelTopology,
    ctrl: RPCCorountineControl,
):
    dp_size = topo.get_dim("data")
    while not ctrl.stop.is_set():
        req_ids, tik = await ctrl.request_queue.get()

        responses, res = await gather_tensor_from_mws(
            streams=streams,
            req_ids=req_ids,
            rpc=rpc,
            mas_dp_head_group=mas_dp_head_group,
            gather_buffer=gather_buffer,
            device=device,
            dp_size=dp_size,
            ctrl=ctrl,
        )

        if rpc.log_return_value:
            logger.info(f"RPC name {rpc.name} returns {res}")

        ctrl.can_do_rpc.release()

        if rpc.is_dst:
            await ctrl.train_count.put(1)
        else:
            assert responses[0].is_tensor
            if "input_lens" in res:
                seqlens = res["input_lens"]
            elif "cu_seqlens" in res:
                seqlens = res["cu_seqlens"][1:] - res["cu_seqlens"][:-1]
            else:
                seqlens = torch.from_numpy(np.concatenate([r.seqlens for r in responses]))
            buffer_indices = torch.from_numpy(np.concatenate([r.buffer_indices for r in responses]))
            xs = split_packed_batch_into_seqs(res, input_lens=seqlens)
            await buffer.amend_batch(buffer_indices, xs)

        logger.info(f"Model rpc {rpc.name} finished. Run time {time.perf_counter() - tik:.4f}s.")


async def load_data_func(
    buffer: AsyncIOSequenceBuffer,
    data_streams: List[request_reply_stream.RequestReplyStream],
    fetch_ctl: asyncio.Queue,
    stop_ctl: asyncio.Event,
):
    while not stop_ctl.is_set():
        await fetch_ctl.get()
        # fetch data from dataloader to fill the sequence buffer
        blogger.info(f"Filling data into the buffer in a new epoch.")
        fetch_data_start = time.perf_counter()
        cur_epoch = latest_epoch = None
        while cur_epoch is None or cur_epoch == latest_epoch:
            data_batches: List[data_api.DataBatch] = await group_rpc_blocked(
                data_streams,
                "fetch",
                [None for _ in data_streams],
                verbose=False,
            )
            assert len(set(x.epoch for x in data_batches)) == 1
            assert len(set(x.epoch_step for x in data_batches)) == 1
            assert len(set(x.global_step for x in data_batches)) == 1

            # Update counters. All starting from 0.
            if cur_epoch is None:
                cur_epoch = latest_epoch = data_batches[0].epoch
            else:
                latest_epoch = data_batches[0].epoch

            # Merge fetched data. We assume fetched data is a flattened dict.
            datas = [x.data for x in data_batches]
            sample = dataparallel.ParallelDataBroker.gather_from([namedarray.from_dict(x) for x in datas])
            xs = split_packed_batch_into_seqs(sample)
            await buffer.put_batch(xs)

        async with buffer.lock:
            buffer.lock.notify(buffer.n_rpcs)

        blogger.info(
            f"Filling data finished. Time consumption: {time.perf_counter() - fetch_data_start:.3f}s.")


async def model_eval_thread_func(
    model_streams: Dict[str, List[request_reply_stream.RequestReplyStream]],
    eval_queue: asyncio.Queue,
    stop_ctl: asyncio.Event,
):
    while not stop_ctl.is_set():
        epoch, epoch_step = await eval_queue.get()
        all_model_streams = list(model_streams.values())
        eval_stats = dataparallel.ParallelDataBroker.gather_from(await group_rpc_blocked(
            all_model_streams, "evaluate", [None for _ in all_model_streams]))
        logger.info(f"Evaluation results at epoch {epoch + 1} step {epoch_step + 1}: {eval_stats}")


async def model_save_thread_func(
    model_streams: Dict[config_pkg.MasterStreamID, List[request_reply_stream.RequestReplyStream]],
    model_save_root: str,
    save_queue: asyncio.Queue,
    stop_ctl: asyncio.Event,
):
    while not stop_ctl.is_set():
        epoch, epoch_step = await save_queue.get()
        dp0streams = {k: v for k, v in model_streams.items() if k.dp_rank == 0}
        assert len(dp0streams) > 0
        model_save_dirs = [os.path.join(model_save_root, k.model_name) for k in dp0streams]
        await group_rpc_blocked(list(dp0streams.values()), "save", model_save_dirs)
        logger.info(f"Save models at epoch {epoch + 1} step {epoch_step + 1}.")


class MasterWorker(worker_base.Worker):
    os.makedirs(MODEL_SAVE_ROOT, exist_ok=True)

    def __init__(self, server=None):
        super().__init__(server)
        self.__initialized = False

        self._data_registry = {}

        self._epoch = -1
        self._epoch_step = self._global_step = 0

        self._train_start_time = None
        self.__exp_complete_msg = None

        # for benchmark
        self.e2e_time_history = []
        self.level_time_history = defaultdict(list)

    def _configure(self, config: config_pkg.MasterWorker):
        self.config = config

        self.__model_topos: Dict[str, topology.PipeModelDataParallelTopology] = config.model_topos

        # Build execution graph and initialize concurrency utilities.
        self.__model_rpcs, _ = api.dfg.build_graph(config.model_rpcs)
        for rpc in self.__model_rpcs:
            _dp_size = self.__model_topos[rpc.model_name].get_dim("data")
            _pp_size = self.__model_topos[rpc.model_name].get_dim("pipe")
            rpc.min_n_seqs = max(rpc.min_n_seqs, _dp_size * _pp_size)

        self.__rpc_srcs = list(filter(lambda rpc: rpc.is_src, self.__model_rpcs))
        self.__rpc_dsts = list(filter(lambda rpc: rpc.is_dst, self.__model_rpcs))
        self.__n_rpc_srcs = len(self.__rpc_srcs)
        self.__n_rpc_dsts = len(self.__rpc_dsts)

        # Save and eval control.
        self.__total_train_epochs = config.total_train_epochs
        self.__save_ctl = base.timeutil.EpochStepTimeFreqCtl(
            freq_epoch=config.save_frequency_epochs,
            freq_step=config.save_frequency_steps,
            freq_sec=config.save_frequency_seconds,
        )
        self.__eval_ctl = base.timeutil.EpochStepTimeFreqCtl(
            freq_epoch=config.eval_frequency_epochs,
            freq_step=config.eval_frequency_steps,
            freq_sec=config.eval_frequency_seconds,
        )

        self.MODEL_SAVE_ROOT = os.path.join(
            MODEL_SAVE_ROOT,
            config.worker_info.experiment_name,
            config.worker_info.trial_name,
        )
        os.makedirs(self.MODEL_SAVE_ROOT, exist_ok=True)

        gpu_utils.reveal_ddp_identity(
            expr_name=self.config.worker_info.experiment_name,
            trial_name=self.config.worker_info.trial_name,
            worker_index=0,
        )

        # Used only for benchmark
        self.__benchmark_steps = config.benchmark_steps

        return config.worker_info

    def __lazy_init(self):
        # Set up streams.
        self.__model_streams: Dict[config_pkg.MasterStreamID,
                                   List[request_reply_stream.RequestReplyStream]] = {
                                       k: request_reply_stream.make_master_stream(
                                           self.config.worker_info,
                                           v,
                                           n_subscribers=self.__model_topos[k.model_name].get_dim("model"),
                                       )
                                       for k, v in self.config.model_streams.items()
                                   }
        self.__data_stream: request_reply_stream.RequestReplyStream = request_reply_stream.make_master_stream(
            self.config.worker_info,
            self.config.data_stream,
            n_subscribers=1,
        )

        # Set up the global process group.
        self.__pg_info = gpu_utils.setup_ddp(
            expr_name=self.config.worker_info.experiment_name,
            trial_name=self.config.worker_info.trial_name,
            worker_index=0,
            mw_topos=self.config.mw_topos,
        )
        for model_name_ in self.config.mw_topos:
            base.constants.set_parallelism_group(
                model_name_,
                self.__pg_info.mw_groups[model_name_],
            )
        deepspeed.init_distributed()
        self.logger.info("deepspeed init distributed on master worker")
        self.__device = torch.device("cuda:0")

        offset = 1
        for model_name_, topo_ in self.config.mw_topos.items():
            grid = base.topology.PipelineParallelGrid(
                topology=topo_,
                process_group=self.__pg_info.mw_groups[model_name_],
                world_size=topo_.world_size(),
                process_group_offset=offset,
            )
            base.constants.set_grid(model_name_, grid)
            offset += topo_.world_size()

        # Request training specification from data workers, e.g. batch size and total train steps.
        self.__data_stream.post(request_reply_stream.Payload(handle_name="spec"))
        ft_specs: List[model_api.FinetuneSpec] = [self.__data_stream.poll(block=True).data]
        if len(set(x.steps_per_epoch for x in ft_specs)) != 1:
            raise RuntimeError(f"steps_per_epoch not equal among data workers:"
                               f" {list(x.steps_per_epoch for x in ft_specs)}. "
                               "Consider launching less data workers.")
        ft_spec = ft_specs[0]
        ft_spec.total_train_epochs = self.config.total_train_epochs
        ft_spec.total_train_steps = ft_spec.total_train_epochs * ft_spec.steps_per_epoch

        batch_size = ft_spec.batch_size_per_device
        # logger.info(
        #     "\n\n"
        #     + "=" * 40
        #     + f"\nTotal train epochs: {ft_spec.total_train_epochs}"
        #     + f"\nTotal train steps: {ft_spec.total_train_steps}"
        #     + f"\nSteps per epoch: {ft_spec.steps_per_epoch}"
        #     + f"\nEffective batch size: {batch_size}\n"
        #     + "=" * 40
        #     + "\n"
        # )
        # logger.info(f"ft_spec = {ft_spec}")

        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)

        model_ft_specs = []
        for ms_id in self.__model_streams:
            model_name = ms_id.model_name
            num_dp = self.__model_topos[model_name].get_dim("data")
            model_ft_spec = copy.deepcopy(ft_spec)
            model_ft_spec.batch_size_per_device = batch_size // num_dp
            model_ft_specs.append(model_ft_spec)

        # logger.info("before create task initialize")
        _task = event_loop.create_task(
            group_rpc_blocked(list(self.__model_streams.values()), "initialize", model_ft_specs))
        event_loop.run_until_complete(asyncio.gather(_task))
        # logger.info("initialize complete")

        self.__scatter_buffers = {rpc.name: {} for rpc in self.__model_rpcs}
        self.__gather_buffers = {rpc.name: {} for rpc in self.__model_rpcs}

        self.__fetch_data_queue = asyncio.Queue(1)
        self.__fetch_master_ctl = asyncio.Queue(1)
        self.__eval_queue = asyncio.Queue(1)
        self.__save_queue = asyncio.Queue(1)
        self.__stop_ctl = asyncio.Event()
        self.__train_count = asyncio.Queue(maxsize=len(self.__rpc_dsts))

        # NOTE: we don't set a maximum buffer size here because we want to keep all data in the buffer
        self.__seqbuffer = AsyncIOSequenceBuffer(
            self.__model_rpcs,
            max_size=int(1e6),
            fetch_ctl=self.__fetch_data_queue,
            fetch_master_ctl=self.__fetch_master_ctl,
        )

        logger.info(f"Creating asyncio coroutines...")

        coroutine_tasks = []
        for rpc in self.__model_rpcs:
            # should be a dict: pp_rank to streams
            pp_world_size = self.__model_topos[rpc.model_name].get_dim("pipe")
            stream_array = [[v for k, v in self.__model_streams.items() \
                            if k.model_name == rpc.model_name and k.pp_rank == pp_rank] for pp_rank in range(pp_world_size)]
            # dp_head_streams = stream_array[-1]

            rpc_count = asyncio.Semaphore(rpc.max_concurrent_calls)
            request_queue = asyncio.Queue(rpc.max_concurrent_calls)

            ctrl = RPCCorountineControl(
                stop=self.__stop_ctl,
                can_do_rpc=rpc_count,
                request_queue=request_queue,
                train_count=self.__train_count,
                fetch_data_queue=self.__fetch_data_queue,
                eval_queue=self.__eval_queue,
                save_queue=self.__save_queue,
            )

            request_task = event_loop.create_task(
                model_rpc_request_func(
                    rpc=rpc,
                    buffer=self.__seqbuffer,
                    streams=stream_array,
                    mas_pp_stage_groups=self.__pg_info.mas_pp_stage_groups[rpc.model_name],
                    scatter_buffer=self.__scatter_buffers[rpc.name],
                    device=self.__device,
                    topo=self.__model_topos[rpc.model_name],
                    ctrl=ctrl,
                ))
            reply_task = event_loop.create_task(
                model_rpc_reply_func(
                    rpc=rpc,
                    buffer=self.__seqbuffer,
                    streams=stream_array,
                    mas_dp_head_group=self.__pg_info.mas_dp_head_groups[rpc.model_name],
                    gather_buffer=self.__gather_buffers[rpc.name],
                    device=self.__device,
                    topo=self.__model_topos[rpc.model_name],
                    ctrl=ctrl,
                ))
            coroutine_tasks += [request_task, reply_task]

        load_data_task = event_loop.create_task(
            load_data_func(
                buffer=self.__seqbuffer,
                data_streams=[self.__data_stream],
                fetch_ctl=self.__fetch_data_queue,
                stop_ctl=self.__stop_ctl,
            ))
        eval_task = event_loop.create_task(
            model_eval_thread_func(
                model_streams=self.__model_streams,
                eval_queue=self.__eval_queue,
                stop_ctl=self.__stop_ctl,
            ))
        save_task = event_loop.create_task(
            model_save_thread_func(
                model_streams=self.__model_streams,
                model_save_root=self.MODEL_SAVE_ROOT,
                save_queue=self.__save_queue,
                stop_ctl=self.__stop_ctl,
            ))
        coroutine_tasks += [load_data_task, eval_task, save_task]

        # self.__event_loop = event_loop
        # self.__coroutine_tasks = coroutine_tasks

        # Set up a run context of EventLoop.run_util_complete, baiscally copy-paste from cpython.
        # With this context, we can call the non-block EventLoop._run_once (similar to worker._poll).
        self.__asyncio_ctx = setup_run_until_complete(event_loop, asyncio.gather(*coroutine_tasks))

        logger.info(f"asyncio coroutines created, master worker ready to run.")

        self.__initialized = True
        self._train_start_time = time.perf_counter()

    def _poll(self):
        if not self.__initialized:
            self.__lazy_init()

        try:
            self.__fetch_master_ctl.get_nowait()
            is_new_epoch = True
        except asyncio.QueueEmpty:
            is_new_epoch = False

        should_eval = self.__eval_ctl.check(epochs=int(is_new_epoch), steps=1)
        should_save = self.__save_ctl.check(epochs=int(is_new_epoch), steps=1)

        if should_eval:
            self.__eval_queue.put_nowait((self._epoch, self._epoch_step))
        if should_save:
            self.__save_queue.put_nowait((self._epoch, self._epoch_step))

        if is_new_epoch:
            self._epoch += 1
            self._epoch_step = 0
            if self._epoch >= self.__total_train_epochs:
                self.experiment_complete_exit(f"Training completes! Yeah!!!")

        # Main execution steps. The graph runs under-the-hood in RPC & stream threads.
        # Wait for the finish of the tranverse of the execution graph.
        execution_start = time.perf_counter()
        logger.info("Master worker is waiting for the finish of the execution graph.")
        _rpc_dst_cnt = 0
        while _rpc_dst_cnt < self.__n_rpc_dsts:
            try:
                self.__train_count.get_nowait()
                _rpc_dst_cnt += 1
                continue
            except asyncio.QueueEmpty:
                pass
            try:
                # Similar to worker._poll. Run multiple times until a train step is finished.
                self.__asyncio_ctx.loop._run_once()
                # NOTE: The following line will propagate errors in corountines back to the main thread.
                # It raises asyncio.exceptions.InvalidStateError if the result is not ready.
                # (In our use cases, the result will never be ready because corountines run while-loops.)
                # We just ignore this error and continue running.
                self.__asyncio_ctx.future.result()
            except asyncio.exceptions.InvalidStateError:
                # Catch the exception when future.result() is not ready.
                pass
            except:
                raise_asyncio_exception(self.__asyncio_ctx)
        logger.info("Execution finished!")

        self._epoch_step += 1
        self._global_step += 1

        total_time_consumption = time.perf_counter() - self._train_start_time
        time_per_step = total_time_consumption / (self._global_step + 1)
        e2e_time = time.perf_counter() - execution_start
        self.e2e_time_history.append(e2e_time)
        logger.info(
            f"Epoch {self._epoch + 1}/{self.config.total_train_epochs} "
            f"step {self._epoch_step + 1} "
            f"(global step {self._global_step + 1}) finishes. "
            f"#End to end# execution time: *{e2e_time:.3f}*s. "
            f"Total time consumption: {total_time_consumption:.3f}s. "
            # f"Estimated remaining time of this epoch: {self._buffer_size_tokens / buffer_size_decre_per_step * time_per_step:.3f}s."
        )

        if self.__benchmark_steps is not None and self._global_step >= self.__benchmark_steps:
            logger.info(
                f"Finished benchmark {self.__benchmark_steps}. Total time consumption {total_time_consumption:.3f}"
            )
            logger.info(f"avg #e2e# time *{np.mean(self.e2e_time_history):.3f}*")
            for i, level_time_history in self.level_time_history.items():
                logger.info(f"avg #level{i+1}# time *{np.mean(level_time_history):.3f}*")
            self.experiment_complete_exit(f"Benchmark completes! Yeah!!!")

        return worker_base.PollResult(sample_count=1, batch_count=1)

    def experiment_complete_exit(self, msg: str):
        self.__stop_ctl.set()
        self.__asyncio_ctx.loop.stop()
        try:
            teardown_run_util_complete(self.__asyncio_ctx)
        except RuntimeError as e:
            raise ExperimentComplete(msg) from e
