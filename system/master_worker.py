from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union
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
from impl.model.parallelism.model_parallel.utils import GlobalMemoryBuffer
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
    stream: request_reply_stream.RequestReplyStream,
    handlers: List[str],
    handle_type: str,
    datas: List[namedarray.NamedArray],
    verbose: bool = True,
):
    """Send request of `handle_type` to multiple streams. len(streams)==len(datas)"""
    requests = [
        request_reply_stream.Payload(
            handler=handler,
            handle_name=handle_type,
            is_tensor=False,
            data=data,
        ) for handler, data in zip(handlers, datas)
    ]
    if verbose:
        blogger.debug(f"master worker #request_all# *end* time ${time.time_ns()}$")
    tik = time.perf_counter()
    [stream.post(r) for r in requests]
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
    stream: request_reply_stream.RequestReplyStream,
    request_ids: List[str],
    verbose: bool = True,
) -> List:
    """Collect responses from multiple streams. Blocking method."""
    responses = await asyncio.gather(
        *
        [_awaitable_response(stream, pattern=create_exact_match_pattern([req_id])) for req_id in request_ids])
    if verbose:
        blogger.debug(f"master worker #gather_all_replies# *end* time ${time.time_ns()}$")
    return responses


async def group_rpc_blocked(
    stream: request_reply_stream.RequestReplyStream,
    handlers: List[Union[config_pkg.ModelShardID, str]],
    handle_type: str,
    datas: List[namedarray.NamedArray],
    verbose: bool = True,
) -> List[namedarray.NamedArray]:
    payloads = await gather_all_replies(stream,
                                        request_all(stream, handlers, handle_type, datas, verbose=verbose))
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
    ## Shared resources ##
    stop: asyncio.Event
    # for counting the number of finished training steps
    # one training step corresponds to traversal of the whole DFG
    train_count: asyncio.Queue
    # for loading data, save and eval model
    fetch_data_queue: asyncio.Queue
    eval_queue: asyncio.Queue
    save_queue: asyncio.Queue

    ## Per-coroutine resources ##
    # Used for counting the number of concurrent calls.
    can_do_rpc: Dict[str, asyncio.Semaphore]
    model_traversal: Dict[str, int]
    # for synchronizing req ids between req and reply coroutines
    request_queues: Dict[str, List[asyncio.Queue]]


async def scatter_tensor_to_mws(
    rpc: api.dfg.ModelRPC,
    stream: request_reply_stream.RequestReplyStream,
    handlers: List[config_pkg.ModelShardID],
    handler_mw_ids: List[int],
    scatter_groups: List[torch.distributed.ProcessGroup],
    data_replica_ids: List[int],
    datas: List[namedarray.NamedArray],
    all_buffer_indices: List[List[int]],
    all_seqlens: List[List[int]],
) -> List[uuid.UUID]:
    assert len(handler_mw_ids) == len(handlers) == len(datas) == len(all_buffer_indices) == len(all_seqlens)

    request_ids = []
    for scatter_group in scatter_groups:
        # Get handlers and datas belonging to this scatter group.
        assert all(torch.distributed.get_rank(scatter_group) != -1 for scatter_group in scatter_groups)
        # Since master must belong to all scatter groups, it is safe to call get_process_group_ranks here.
        scatter_mw_ids = [r - 1 for r in torch.distributed.get_process_group_ranks(scatter_group)]
        handler_indices = list(
            map(lambda x: x[0], filter(lambda ix: ix[1] in scatter_mw_ids, enumerate(handler_mw_ids))))

        this_handlers = [handlers[i] for i in handler_indices]
        this_datas = [datas[i] for i in handler_indices]
        this_all_buffer_indices = [all_buffer_indices[i] for i in handler_indices]
        this_all_seqlens = [all_seqlens[i] for i in handler_indices]
        this_replica_ids = [data_replica_ids[i] for i in handler_indices]

        dtypes = {k: v.dtype for k, v in this_datas[0].items()}
        all_shapes = [{k: v.shape for k, v in data.items()} for data in this_datas]
        buf_shapes = {}
        for k in dtypes:
            buf_shapes[k] = base.numpy_utils.shape_union(*[tuple(s[k]) for s in all_shapes])

        ack_ids = []
        for handler, shapes, buffer_indices, seqlens in zip(this_handlers, all_shapes,
                                                            this_all_buffer_indices, this_all_seqlens):
            req = request_reply_stream.Payload(
                handler=handler,
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
        [stream.poll(pattern=create_exact_match_pattern([ack_id]), block=True) for ack_id in ack_ids]
        # await asyncio.gather(
        #     *[_awaitable_response(stream, pattern=create_exact_match_pattern([ack_id])) for ack_id in ack_ids]
        # )
        # logger.info(f"Scatter data to stage {pp_rank}")

        for (k, buf_shape), dtype in zip(buf_shapes.items(), dtypes.values()):
            scatter_buffer = dict(master=base.constants.get_global_memory_buffer().get_tensor(
                buf_shape, dtype, name=f"scatter_gather_master"))
            for j in this_replica_ids:
                scatter_buffer[f"data{j}"] = base.constants.get_global_memory_buffer().get_tensor(
                    buf_shape, dtype, name=f"scatter_gather_dp{j}")
            copied_replica_ids = []
            for data, replica_id in zip(this_datas, this_replica_ids):
                if replica_id in copied_replica_ids:
                    continue
                buf = scatter_buffer[f"data{replica_id}"]
                v = data[k]
                s = tuple(slice(0, x) for x in v.shape)
                buf[s] = v
                copied_replica_ids.append(replica_id)

            scatter_list = [scatter_buffer["master"]] + [scatter_buffer[f"data{j}"] for j in this_replica_ids]

            # Scatter to dp_rank=*, pp_rank=pp_rank, mp_rank=*
            torch.distributed.scatter(
                scatter_list[0],
                scatter_list=scatter_list,
                src=0,
                group=scatter_group,
            )

    return request_ids


async def model_rpc_request_func(
    rpc: api.dfg.ModelRPC,
    stream: request_reply_stream.RequestReplyStream,
    scatter_groups: List[torch.distributed.ProcessGroup],
    msid2mwid: Dict[config_pkg.ModelShardID, int],
    buffer: AsyncIOSequenceBuffer,
    topo: base.topology.PipeModelDataParallelTopology,
    ctrl: RPCCorountineControl,
):

    handlers = [
        config_pkg.ModelShardID.from_parallelism_rank(rpc.model_name, topo, j)
        for j in range(topo.world_size())
    ]
    handler_mw_ids = [msid2mwid[h] for h in handlers]

    can_do_rpc = ctrl.can_do_rpc[rpc.name]
    request_queues = ctrl.request_queues[rpc.name]

    response_coroutine_idx = 0
    data_amount_seqs = data_amount_tokens = 0
    while not ctrl.stop.is_set():
        await can_do_rpc.acquire()

        # The following two lines are used to ensure staleness=1, but it may be unnecessary when enabling the stream engine.
        # NOTE: max-min-flow-tokens is not used here because the number of tokens can change (e.g. after generate)
        while data_amount_seqs >= (ctrl.model_traversal[rpc.model_name] + 1) * rpc.max_min_flow_seqs:
            await asyncio.sleep(0.1)

        sample = await buffer.get_batch_for_rpc(rpc)

        data_amount_seqs += len(sample.seqlens)
        data_amount_tokens += sum(sample.seqlens)

        # logger.info(f"Model rpc {rpc.name} requesting.")
        dp_size = topo.get_dim("data")
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
        pp_size = topo.get_dim("pipe")
        try:
            for d in datas:
                dataparallel.get_broker(rpc.dp_broker_type).scatter_to(d, pp_size)
        except Exception as e:
            raise RuntimeError(
                f"Data in each data parallel rank cannot be "
                f"partitioned further into {pp_size} mini-batches for pipeline parallel. Exiting.") from e

        # replicate data for each model shard
        data_replica_ids = [topo.get_coord(j).data for j in range(topo.world_size())]
        datas = [datas[topo.get_coord(j).data] for j in range(topo.world_size())]
        all_buffer_indices = [all_buffer_indices[topo.get_coord(j).data] for j in range(topo.world_size())]
        all_seqlens = [all_seqlens[topo.get_coord(j).data] for j in range(topo.world_size())]

        # send partitioned data to model workers
        req_ids = await scatter_tensor_to_mws(
            rpc=rpc,
            stream=stream,
            handlers=handlers,
            handler_mw_ids=handler_mw_ids,
            scatter_groups=scatter_groups,
            data_replica_ids=data_replica_ids,
            datas=datas,
            all_buffer_indices=all_buffer_indices,
            all_seqlens=all_seqlens,
        )
        await request_queues[response_coroutine_idx].put((req_ids, time.perf_counter()))
        response_coroutine_idx = (response_coroutine_idx + 1) % len(request_queues)
        logger.info(f"Model rpc {rpc.name} requested.")


async def gather_tensor_from_mws(
    rpc: api.dfg.ModelRPC,
    stream: request_reply_stream.RequestReplyStream,
    req_ids: List[uuid.UUID],
    dp_head_indices: List[int],
    dp_head_mw_ids: List[int],
    dp_head_handlers: List[config_pkg.ModelShardID],
    gather_groups: List[torch.distributed.ProcessGroup],
) -> Tuple[List[request_reply_stream.Payload], Union[dict, namedarray.NamedArray]]:
    # logger.info(f"rpc {rpc.name} waiting for responses {req_ids}")
    responses = await asyncio.gather(
        *[_awaitable_response(stream, pattern=create_exact_match_pattern([req_id])) for req_id in req_ids])
    # logger.info(f"rpc {rpc.name} received responses {req_ids}")

    responses = [responses[i] for i in dp_head_indices]
    recv_tik = time.perf_counter()

    if not responses[-1].is_tensor:
        res = dataparallel.get_broker(rpc.dp_broker_type).gather_from(
            [response.data for response in responses])
        # logger.info(f"Master worker return from model worker time: {time.perf_counter() - recv_tik:.4f}s")
        return responses, res

    assert all(len(r.buf_shapes) == 0 for r in responses)
    buf_shapes = {}
    for k in responses[0].dtypes.keys():
        buf_shapes[k] = base.numpy_utils.shape_union(*[tuple(s.actual_shapes[k]) for s in responses])

    all_res = [dict() for _ in range(len(responses))]
    for gather_group in gather_groups:
        assert all(torch.distributed.get_rank(gather_group) != -1 for gather_group in gather_groups)
        gather_mw_ids = [r - 1 for r in torch.distributed.get_process_group_ranks(gather_group)]

        gather_response_indices = [i for i, mw_id in enumerate(dp_head_mw_ids) if mw_id in gather_mw_ids]
        this_responses = [responses[i] for i in gather_response_indices]
        this_handlers = [dp_head_handlers[i] for i in gather_response_indices]
        this_request_indices = [dp_head_indices[i] for i in gather_response_indices]
        dtypes = this_responses[-1].dtypes

        gather_requests = [
            request_reply_stream.Payload(
                handler=h,
                handle_name="gather_tensor_reply",
                actual_shapes=None,
                buf_shapes=buf_shapes,
                dtypes=None,
                is_tensor=False,
                data=req_ids[j],
            ) for h, j in zip(this_handlers, this_request_indices)
        ]
        [stream.post(x) for x in gather_requests]
        gather_ack_ids = [r.ack_reply_id for r in gather_requests]

        # wait for the ack message from model workers
        # await asyncio.gather(
        #     *[
        #         _awaitable_response(stream, pattern=create_exact_match_pattern([ack_id]))
        #         for ack_id in gather_ack_ids
        #     ]
        # )
        [stream.poll(pattern=create_exact_match_pattern([ack_id]), block=True) for ack_id in gather_ack_ids]

        for k, buf_shape in buf_shapes.items():
            gather_buffer = [
                base.constants.get_global_memory_buffer().get_tensor(
                    buf_shape, dtypes[k], name="scatter_gather_master")
            ] + [
                base.constants.get_global_memory_buffer().get_tensor(
                    buf_shape, dtypes[k], name=f"scatter_gather_dp{i}") for i in gather_response_indices
            ]

            # Only gather from DP heads, ignoring the MP/PP dimension.
            torch.distributed.gather(
                gather_buffer[0],
                gather_list=gather_buffer,
                dst=0,
                group=gather_group,
            )

            assert len(this_responses) == len(gather_response_indices) == len(gather_buffer) - 1
            for idx, v, actual_shape in zip(
                    gather_response_indices,
                    gather_buffer[1:],
                [r.actual_shapes[k] for r in this_responses],
            ):
                s = tuple(slice(0, x) for x in actual_shape)
                all_res[idx][k] = v[s].clone()

    # logger.info(f"rpc {rpc.name} finished tensor gather {req_ids}")
    res = dataparallel.get_broker(rpc.dp_broker_type).gather_from(list(map(namedarray.from_dict, all_res)))
    # logger.info(f"Master worker return from model worker time: {time.perf_counter() - recv_tik:.4f}s")
    return responses, rpc.remap_output_keys(res)


async def model_rpc_reply_func(
    corountine_idx: int,
    rpc: api.dfg.ModelRPC,
    stream: request_reply_stream.RequestReplyStream,
    gather_groups: List[torch.distributed.ProcessGroup],
    msid2mwid: Dict[config_pkg.ModelShardID, int],
    buffer: AsyncIOSequenceBuffer,
    topo: base.topology.PipeModelDataParallelTopology,
    ctrl: RPCCorountineControl,
):
    dp_size = topo.get_dim("data")
    dp_head_indices = [topo.get_rank(data=i, pipe=topo.get_dim("pipe") - 1, model=0) for i in range(dp_size)]
    dp_head_mw_ids = [
        msid2mwid[config_pkg.ModelShardID.from_parallelism_rank(rpc.model_name, topo, i)]
        for i in dp_head_indices
    ]
    dp_head_handlers = [
        config_pkg.ModelShardID.from_parallelism_rank(rpc.model_name, topo, i) for i in dp_head_indices
    ]

    request_queue = ctrl.request_queues[rpc.name][corountine_idx]
    can_do_rpc = ctrl.can_do_rpc[rpc.name]

    while not ctrl.stop.is_set():
        req_ids, tik = await request_queue.get()

        responses, res = await gather_tensor_from_mws(
            rpc=rpc,
            stream=stream,
            req_ids=req_ids,
            dp_head_indices=dp_head_indices,
            dp_head_mw_ids=dp_head_mw_ids,
            dp_head_handlers=dp_head_handlers,
            gather_groups=gather_groups,
        )

        if rpc.log_return_value:
            logger.info(f"RPC name {rpc.name} returns {res}")

        can_do_rpc.release()

        if rpc.is_dst_of_model:
            ctrl.model_traversal[rpc.model_name] += 1

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
    stream: request_reply_stream.RequestReplyStream,
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
                stream,
                handlers=["__data__"],
                handle_type="fetch",
                datas=[None],
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
    stream: List[request_reply_stream.RequestReplyStream],
    handlers: List[config_pkg.ModelShardID],
    eval_queue: asyncio.Queue,
    stop_ctl: asyncio.Event,
):
    while not stop_ctl.is_set():
        epoch, epoch_step = await eval_queue.get()
        eval_stats = dataparallel.ParallelDataBroker.gather_from(await group_rpc_blocked(
            stream, handlers, "evaluate", [None for _ in handlers]))
        logger.info(f"Evaluation results at epoch {epoch + 1} step {epoch_step + 1}: {eval_stats}")


async def model_save_thread_func(
    stream: request_reply_stream.RequestReplyStream,
    handlers: List[config_pkg.ModelShardID],
    model_save_root: str,
    save_queue: asyncio.Queue,
    stop_ctl: asyncio.Event,
):
    while not stop_ctl.is_set():
        epoch, epoch_step = await save_queue.get()
        model_save_dirs = [os.path.join(model_save_root, s.model_name) for s in handlers]
        await group_rpc_blocked(stream, handlers, "save", model_save_dirs)
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
        self.__stream = request_reply_stream.make_master_stream(
            self.config.worker_info,
            n_subscribers=self.config.n_model_workers + 1,  # all model workers and data worker
        )
        self.__stream: request_reply_stream.RequestReplyStream

        # Set up the global process group.
        self.__pg_info = gpu_utils.setup_ddp(
            expr_name=self.config.worker_info.experiment_name,
            trial_name=self.config.worker_info.trial_name,
            worker_index=0,
            model_topos=self.config.model_topos,
            msid2mwid=self.config.msid2mwid,
            mw_bcast_groups=self.config.mw_bcast_groups,
        )
        deepspeed.init_distributed()
        self.logger.info("deepspeed init distributed on master worker")
        self.__device = torch.device("cuda:0")

        for model_name_, topo_ in self.config.model_topos.items():
            base.constants.set_parallelism_group(
                model_name_,
                self.__pg_info.model_groups[model_name_],
            )
            base.constants.set_rank_mapping(model_name_, topo_, self.config.msid2mwid)
            grid = base.topology.ParallelGrid(
                topology=topo_,
                process_group=self.__pg_info.model_groups[model_name_],
                rank_mapping=base.constants.rank_mapping_of_model(model_name_),
            )
            base.constants.set_grid(model_name_, grid)

        base.constants.set_global_memory_buffer(GlobalMemoryBuffer())

        # Request training specification from data workers, e.g. batch size and total train steps.
        self.__stream.post(request_reply_stream.Payload(
            handler="__data__",
            handle_name="spec",
        ))
        ft_specs: List[model_api.FinetuneSpec] = [self.__stream.poll(block=True).data]
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
        self.__all_model_handlers = []
        self.__dp0_model_handlers = []
        for model_name, topo in self.config.model_topos.items():
            num_dp = topo.get_dim("data")
            model_ft_spec = copy.deepcopy(ft_spec)
            model_ft_spec.batch_size_per_device = batch_size // num_dp
            model_ft_specs += [model_ft_spec] * topo.world_size()
            self.__all_model_handlers += [
                config_pkg.ModelShardID.from_parallelism_rank(model_name, topo, j)
                for j in range(topo.world_size())
            ]
            self.__dp0_model_handlers += [
                config_pkg.ModelShardID.from_parallelism_rank(model_name, topo, j)
                for j in topo.filter_match(data=0)
            ]

        # logger.info("before create task initialize")
        _task = event_loop.create_task(
            group_rpc_blocked(
                self.__stream,
                handlers=self.__all_model_handlers,
                handle_type="initialize",
                datas=model_ft_specs,
            ))
        event_loop.run_until_complete(asyncio.gather(_task))
        # logger.info("initialize complete")

        self.__rpc_ctrl = RPCCorountineControl(
            stop=asyncio.Event(),
            train_count=asyncio.Queue(maxsize=len(self.__rpc_dsts)),
            fetch_data_queue=asyncio.Queue(1),
            eval_queue=asyncio.Queue(1),
            save_queue=asyncio.Queue(1),
            model_traversal={model_name: 0
                             for model_name in set(r.model_name for r in self.__model_rpcs)},
            can_do_rpc={rpc.name: asyncio.Semaphore(rpc.max_concurrent_calls)
                        for rpc in self.__model_rpcs},
            request_queues={
                rpc.name: [asyncio.Queue(1) for _ in range(rpc.max_concurrent_calls)]
                for rpc in self.__model_rpcs
            },
        )

        self.__fetch_master_ctl = asyncio.Queue(1)

        # NOTE: we don't set a maximum buffer size here because we want to keep all data in the buffer
        self.__seqbuffer = AsyncIOSequenceBuffer(
            self.__model_rpcs,
            max_size=int(1e6),
            fetch_ctl=self.__rpc_ctrl.fetch_data_queue,
            fetch_master_ctl=self.__fetch_master_ctl,
        )

        logger.info(f"Creating asyncio coroutines...")

        coroutine_tasks = []
        for rpc in self.__model_rpcs:
            # should be a dict: pp_rank to streams

            request_task = event_loop.create_task(
                model_rpc_request_func(
                    rpc=rpc,
                    stream=self.__stream,
                    scatter_groups=self.__pg_info.scatter_groups[rpc.model_name],
                    msid2mwid=self.config.msid2mwid,
                    buffer=self.__seqbuffer,
                    topo=self.__model_topos[rpc.model_name],
                    ctrl=self.__rpc_ctrl,
                ))
            reply_tasks = []
            for j in range(rpc.max_concurrent_calls):
                _reply_task = event_loop.create_task(
                    model_rpc_reply_func(
                        corountine_idx=j,
                        rpc=rpc,
                        stream=self.__stream,
                        gather_groups=self.__pg_info.gather_groups[rpc.model_name],
                        msid2mwid=self.config.msid2mwid,
                        buffer=self.__seqbuffer,
                        topo=self.__model_topos[rpc.model_name],
                        ctrl=self.__rpc_ctrl,
                    ))
                reply_tasks.append(_reply_task)
            coroutine_tasks += [request_task] + reply_tasks

        load_data_task = event_loop.create_task(
            load_data_func(
                buffer=self.__seqbuffer,
                stream=self.__stream,
                fetch_ctl=self.__rpc_ctrl.fetch_data_queue,
                stop_ctl=self.__rpc_ctrl.stop,
            ))
        eval_task = event_loop.create_task(
            model_eval_thread_func(
                stream=self.__stream,
                handlers=self.__all_model_handlers,
                eval_queue=self.__rpc_ctrl.eval_queue,
                stop_ctl=self.__rpc_ctrl.stop,
            ))
        save_task = event_loop.create_task(
            model_save_thread_func(
                stream=self.__stream,
                handlers=self.__dp0_model_handlers,
                model_save_root=self.MODEL_SAVE_ROOT,
                save_queue=self.__rpc_ctrl.save_queue,
                stop_ctl=self.__rpc_ctrl.stop,
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
            self.__rpc_ctrl.eval_queue.put_nowait((self._epoch, self._epoch_step))
        if should_save:
            self.__rpc_ctrl.save_queue.put_nowait((self._epoch, self._epoch_step))

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
                self.__rpc_ctrl.train_count.get_nowait()
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
        self.__rpc_ctrl.stop.set()
        self.__asyncio_ctx.loop.stop()
        try:
            teardown_run_util_complete(self.__asyncio_ctx)
        except RuntimeError as e:
            raise ExperimentComplete(msg) from e
