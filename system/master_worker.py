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

from api.config.config_flash_model import FlashMQATConfig
from base.asyncio_utils import raise_asyncio_exception, setup_run_until_complete, teardown_run_util_complete
from base.buffer import AsyncIOSequenceBuffer
from base.cluster import spec as cluster_spec
from base.constants import MODEL_SAVE_ROOT
import api.config.config_system as config_pkg
import api.config.dfg
import api.data as data_api
import api.model as model_api
import base.datapack as datapack
import base.dataparallel as dataparallel
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


def _get_n_seqs_from_batch_sample(sample: namedarray.NamedArray) -> int:
    assert ("input_lens" in sample.keys() or "cu_seqlens" in sample.keys()
            or "prompt_cu_seqlens" in sample.keys() or "prompt_lens" in sample.keys()), (
                list(sample.keys()),
                sample,
            )
    if "input_lens" in sample.keys():
        return len(sample["input_lens"])
    elif "cu_seqlens" in sample.keys():
        return len(sample["cu_seqlens"]) - 1
    # NOTE: The order matters. We should first try to get the length of the generated text rather than prompts.
    elif "prompt_lens" in sample.keys():
        return len(sample["prompt_lens"])
    elif "prompt_cu_seqlens" in sample.keys():
        return len(sample["prompt_cu_seqlens"]) - 1
    else:
        raise NotImplementedError(f"Unknown seqlens keys: {list(sample.keys())}.")


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


def handle_rpc_hook(
    hook: api.config.dfg.RPCHook,
    hook_counter: int,
    model_name: str,
    stream: request_reply_stream.IpRequestClient,
    model_topos: Dict[str, base.topology.PipeModelDataParallelTopology],
    model_configs: Dict[str, None | FlashMQATConfig],
):
    logger.info(f"Dealing with RPC hook {hook}.")
    if isinstance(hook, (api.config.dfg.OffloadHook, api.config.dfg.LoadToDeviceHook)):
        topo = model_topos[model_name]
        handlers = [
            config_pkg.ModelShardID.from_parallelism_rank(model_name, topo, j)
            for j in range(topo.world_size())
        ]
        request_ids = request_all(
            stream,
            handlers,
            "offload" if isinstance(hook, api.config.dfg.OffloadHook) else "load_to_device",
            [None for _ in handlers],
        )
    elif isinstance(hook, api.config.dfg.SyncParamHook):
        print(">>>>>>>>>>>>>>>>>>", hook_counter)
        # Since the counter is increased after the hook is handled, we add one here.
        if (hook_counter + 1) % hook.interval != 0:
            return

        src_topo = model_topos[model_name]
        dst_topo = model_topos[hook.target]

        #################### Sanity check of model configs. ####################
        src_config = model_configs[model_name]
        dst_config = model_configs[hook.target]
        assert src_config is not None and dst_config is not None
        for k, v in dataclasses.asdict(src_config).items():
            if k not in [
                    "sequence_parallel",
                    "gradient_accumulation_fusion",
                    "ckpt_attn",
                    "ckpt_mlp",
            ] and v != getattr(dst_config, k):
                raise ValueError(
                    f"Can't synchronize a checkpoint with different config (key `{k}`, "
                    f"value of src model is `{v}`, value of dst model is `{getattr(dst_config, k)}`).")
        src_mp_size = src_topo.get_dim("model")
        dst_mp_size = src_topo.get_dim("model")
        if (src_config.n_kv_heads % src_mp_size == 0) != (dst_config.n_kv_heads % dst_mp_size == 0):
            raise ValueError(
                "The partition methods of KV heads of the src and dst model are not compatible. "
                "To load the checkpoint, KV heads must be able to evenly partitioned (i.e.,  #kv_heads % mp_size == 0) "
                "or unable to be partitioned (i.e., #kv_heads % mp_size != 0) for both models. "
                f"Number of kv heads={src_config.n_kv_heads}, src mp_size={src_mp_size}, dst mp_size={dst_mp_size}.",
            )
        #################### Sanity check of model configs. ####################

        src_handlers = [
            config_pkg.ModelShardID.from_parallelism_rank(model_name, src_topo, j)
            for j in range(src_topo.world_size())
        ]
        src_payloads = [
            request_reply_stream.Payload(handler=h, handle_name="send_param", data=hook.target)
            for h in src_handlers
        ]
        dst_handlers = [
            config_pkg.ModelShardID.from_parallelism_rank(hook.target, dst_topo, j)
            for j in range(dst_topo.world_size())
        ]
        dst_payloads = [
            request_reply_stream.Payload(handler=h, handle_name="recv_param", data=model_name)
            for h in dst_handlers
        ]
        request_ids = [stream.post(p) for p in src_payloads + dst_payloads]
    else:
        raise NotImplementedError()
    [stream.poll(pattern=create_exact_match_pattern([req_id]), block=True) for req_id in request_ids]
    logger.info(f"RPC hook {hook} receives responses from model worker.")


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
    rpc: api.config.dfg.ModelRPC,
    stream: request_reply_stream.RequestReplyStream,
    msid2mwid: Dict[config_pkg.ModelShardID, int],
    producer_names: Dict[str, str],
    producer_name2producer_handlers: Dict[str, List[config_pkg.ModelShardID]],
    producer_mappings: Dict[str, Dict[str, List[int]]],
    target_mapping: Dict[str, List[int]],
    buffer_indices: List[int],
    seqlens: List[int],
    handlers: List[config_pkg.ModelShardID],
) -> List[uuid.UUID]:

    dt_data = {
        "keys": rpc.input_data,
        "target": rpc.model_name,
        "producer_names": producer_names,
        "producer_mappings": producer_mappings,
        "target_mapping": target_mapping,
        "handle_name": rpc.interface_type.value,
        "input_key_remap": rpc.input_key_remap,
        "output_key_remap": rpc.output_key_remap,
        "rpc_name": rpc.name,
    }

    all_handlers = copy.deepcopy(handlers)
    all_handler_mwids = set([msid2mwid[h] for h in all_handlers])
    for producer_name in producer_names.values():
        for h in producer_name2producer_handlers[producer_name]:
            if msid2mwid[h] not in all_handler_mwids:
                all_handlers.append(h)
                all_handler_mwids.add(msid2mwid[h])

    dt_request_ids = []
    for handler in all_handlers:
        req = request_reply_stream.Payload(
            handler=handler,
            handle_name="data_transfer",
            buffer_indices=buffer_indices,
            seqlens=seqlens,
            data=dt_data,
        )
        dt_request_ids.append(stream.post(req))

    # logger.info(f"Waiting for ack from stage {pp_rank}")
    # Wait for the ack message from model worker
    await gather_all_replies(stream, dt_request_ids, verbose=False)
    # [stream.poll(pattern=create_exact_match_pattern([req_id]), block=True) for req_id in dt_request_ids]

    request_ids = []
    for handler in handlers:
        req = request_reply_stream.Payload(
            handler=handler,
            handle_name=rpc.interface_type.value,
            buffer_indices=buffer_indices,
            seqlens=seqlens,
        )
        request_ids.append(stream.post(req))

    return request_ids


async def model_rpc_request_func(
    rpc: api.config.dfg.ModelRPC,
    msid2mwid: Dict[config_pkg.ModelShardID, int],
    src_rpc_model_name: str,
    pre_hook_counters: List[int],
    stream: request_reply_stream.RequestReplyStream,
    buffer: AsyncIOSequenceBuffer,
    data_owner: Dict[Tuple[int, str], Tuple[str, int]],
    model_topos: Dict[str, base.topology.PipeModelDataParallelTopology],
    model_configs: Dict[str, None | FlashMQATConfig],
    ctrl: RPCCorountineControl,
):
    topo = model_topos[rpc.model_name]
    handlers = [
        config_pkg.ModelShardID.from_parallelism_rank(rpc.model_name, topo, j)
        for j in range(topo.world_size())
    ]

    producer_names = {}  # data key -> model name
    for k in rpc.input_data:
        if k in rpc.data_producers:
            producer_names[k] = rpc.data_producers[k]
        else:
            producer_names[k] = src_rpc_model_name
    keys_to_send = defaultdict(list)  # model name -> List[keys] to send
    for k in producer_names:
        keys_to_send[producer_names[k]].append(k)

    # convert producer model name to ModelShardID
    producer_name2producer_handlers = {}
    for producer_name in keys_to_send:
        producer_name2producer_handlers[producer_name] = [
            config_pkg.ModelShardID.from_parallelism_rank(producer_name, model_topos[producer_name], j)
            for j in range(model_topos[producer_name].world_size())
        ]

    can_do_rpc = ctrl.can_do_rpc[rpc.name]
    request_queues = ctrl.request_queues[rpc.name]

    response_coroutine_idx = 0
    data_amount_seqs = data_amount_tokens = 0
    while not ctrl.stop.is_set():
        await can_do_rpc.acquire()

        # The following two lines are used to ensure staleness=0, but it may be unnecessary when enabling the stream engine.
        # NOTE: max-min-flow-tokens is not used here because the number of tokens can change (e.g. after generate)
        while data_amount_seqs >= (ctrl.model_traversal[rpc.model_name] + 1) * rpc.max_min_flow_seqs:
            await asyncio.sleep(0.1)

        sample = await buffer.get_batch_for_rpc(rpc)

        data_amount_seqs += len(sample.seqlens)
        data_amount_tokens += sum(sample.seqlens)

        # logger.info(f"Model rpc {rpc.name} requesting.")
        dp_size = topo.get_dim("data")
        if rpc.balanced_dp:
            assert len(sample.seqlens) % dp_size == 0
            min_n_seqs_per_dp = len(sample.seqlens) // dp_size
        else:
            min_n_seqs_per_dp = 1
        partitions = datapack.min_abs_diff_partition(np.array(sample.seqlens, dtype=np.int32),
                                                     dp_size,
                                                     min_size=min_n_seqs_per_dp)
        target_mapping = {i: list(range(v[0], v[1])) for i, v in enumerate(partitions)}

        # Set data owner of produced data by this RPC, such that downstream RPCs can know
        # whether to fetch these data.
        for dp_idx, (st, ed) in enumerate(partitions):
            for i in range(st, ed):
                for k in rpc.output_data:
                    if k in rpc.output_key_remap:
                        k = rpc.output_key_remap[k]
                    data_owner[sample.indices[i], k] = (rpc.model_name, dp_idx)

        # Get the data owner of this RPC's input data.
        producer_mappings: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
        for k in rpc.input_data:
            names, dp_indices = [], []
            for buf_idx in sample.indices:
                owner_name, dp_idx = data_owner[(buf_idx, k)]
                names.append(owner_name)
                dp_indices.append(dp_idx)
            assert len(set(names)) == 1
            producer_mapping = defaultdict(list)
            for i, dp_idx in enumerate(dp_indices):
                producer_mapping[dp_idx].append(i)
            producer_mapping = {k: sorted(v) for k, v in producer_mapping.items()}
            producer_mappings[names[0], k] = producer_mapping

        for i, pre_hook in enumerate(rpc.pre_hooks):
            handle_rpc_hook(
                pre_hook,
                hook_counter=pre_hook_counters[i],
                model_name=rpc.model_name,
                stream=stream,
                model_topos=model_topos,
                model_configs=model_configs,
            )
            pre_hook_counters[i] += 1

        # send partitioned data to model workers
        req_ids = await scatter_tensor_to_mws(
            rpc=rpc,
            stream=stream,
            msid2mwid=msid2mwid,
            producer_names=producer_names,
            producer_name2producer_handlers=producer_name2producer_handlers,
            producer_mappings=producer_mappings,
            target_mapping=target_mapping,
            buffer_indices=sample.indices,
            seqlens=sample.seqlens,
            handlers=handlers,
        )
        await request_queues[response_coroutine_idx].put((req_ids, time.perf_counter()))
        response_coroutine_idx = (response_coroutine_idx + 1) % len(request_queues)
        logger.info(f"Model rpc {rpc.name} requested.")


async def model_rpc_reply_func(
    corountine_idx: int,
    rpc: api.config.dfg.ModelRPC,
    post_hook_counters: List[int],
    stream: request_reply_stream.RequestReplyStream,
    buffer: AsyncIOSequenceBuffer,
    model_topos: Dict[str, base.topology.PipeModelDataParallelTopology],
    model_configs: Dict[str, None | FlashMQATConfig],
    ctrl: RPCCorountineControl,
):
    topo = model_topos[rpc.model_name]
    dp_size = topo.get_dim("data")
    dp_head_indices = [topo.get_rank(data=i, pipe=topo.get_dim("pipe") - 1, model=0) for i in range(dp_size)]
    dp_head_handlers = [
        config_pkg.ModelShardID.from_parallelism_rank(rpc.model_name, topo, i) for i in dp_head_indices
    ]

    request_queue = ctrl.request_queues[rpc.name][corountine_idx]
    can_do_rpc = ctrl.can_do_rpc[rpc.name]

    while not ctrl.stop.is_set():
        req_ids, tik = await request_queue.get()

        responses = await asyncio.gather(
            *
            [_awaitable_response(stream, pattern=create_exact_match_pattern([req_id])) for req_id in req_ids])
        # logger.info(f"rpc {rpc.name} received responses {req_ids}")

        responses: List[request_reply_stream.Payload] = [responses[i] for i in dp_head_indices]
        recv_tik = time.perf_counter()

        if responses[-1].seqlens is None:
            assert responses[-1].buffer_indices is None
            res = dataparallel.PackedParallelDataBroker.gather_from([response.data for response in responses])
        else:
            res = []
            for k in responses[0].data:
                if k in rpc.output_key_remap:
                    res.append(rpc.output_key_remap[k])
                else:
                    res.append(k)

        if rpc.log_return_value:
            logger.info(f"RPC name {rpc.name} returns {res}")

        can_do_rpc.release()

        if rpc.is_dst_of_model:
            ctrl.model_traversal[rpc.model_name] += 1

        if rpc.is_dst:
            await ctrl.train_count.put(1)
        else:
            buffer_indices = sum([response.buffer_indices for response in responses], [])
            keys = res
            seqlens = sum([response.seqlens for response in responses], [])
            await buffer.amend_batch(buffer_indices, [(keys, seqlen) for seqlen in seqlens])

        for i, post_hook in enumerate(rpc.post_hooks):
            handle_rpc_hook(
                post_hook,
                hook_counter=post_hook_counters[i],
                model_name=rpc.model_name,
                stream=stream,
                model_topos=model_topos,
                model_configs=model_configs,
            )
            post_hook_counters[i] += 1

        logger.info(f"Model rpc {rpc.name} finished. Run time {time.perf_counter() - tik:.4f}s.")


async def load_data_func(
    src_rpc_dp_size: int,
    src_rpc_model_name: str,
    buffer: AsyncIOSequenceBuffer,
    data_owner: Dict[Tuple[int, str], Tuple[str, int]],
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
                handlers=[f"__data{i}__" for i in range(src_rpc_dp_size)],
                handle_type="fetch",
                datas=[None for _ in range(src_rpc_dp_size)],
                verbose=False,
            )

            # Update counters. All starting from 0.
            if cur_epoch is None:
                cur_epoch = latest_epoch = data_batches[0].epoch
            else:
                latest_epoch = data_batches[0].epoch

            # Merge fetched data. We assume fetched data is a flattened dict.
            datas = [x.data for x in data_batches]
            n_seqs = [_get_n_seqs_from_batch_sample(d) for d in datas]
            sample = dataparallel.ParallelDataBroker.gather_from([namedarray.from_dict(x) for x in datas])
            xs, seqlens = split_packed_batch_into_seqs(sample, return_seqlens=True)
            buffer_indices = await buffer.put_batch([(list(x.keys()), seqlen)
                                                     for x, seqlen in zip(xs, seqlens)])
            assert len(buffer_indices) == sum(n_seqs)

            for dp_i, (st, ed) in enumerate(
                    zip([0] + list(itertools.accumulate(n_seqs)), itertools.accumulate(n_seqs))):
                for buf_idx in buffer_indices[st:ed]:
                    for k in sample.keys():
                        data_owner[(buf_idx, k)] = (src_rpc_model_name, dp_i)

            await group_rpc_blocked(
                stream,
                handlers=[f"__data{i}__" for i in range(src_rpc_dp_size)],
                handle_type="store",
                datas=[
                    buffer_indices[st:ed]
                    for st, ed in zip([0] + list(itertools.accumulate(n_seqs)), itertools.accumulate(n_seqs))
                ],
                verbose=False,
            )

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

    def _configure(self, config: config_pkg.MasterWorker):
        self.config = config

        self.__model_topos: Dict[str, topology.PipeModelDataParallelTopology] = config.model_topos

        # Build execution graph and initialize concurrency utilities.
        self.__model_rpcs, _ = api.config.dfg.build_graph(config.model_rpcs)
        for rpc in self.__model_rpcs:
            _dp_size = self.__model_topos[rpc.model_name].get_dim("data")
            _pp_size = self.__model_topos[rpc.model_name].get_dim("pipe")
            if rpc.min_n_seqs < _dp_size * _pp_size:
                logger.warning(f"The batch size of RPC `{rpc.name}` in terms of #seqs is smaller than "
                               f"dp_size * pp_size ({_dp_size}*{_pp_size}). Forcely enlarge the batch size "
                               f"to {_dp_size * _pp_size} (dp_size * pp_size). (original: {rpc.min_n_seqs})")
                rpc.min_n_seqs_per_dp = 1
                rpc.min_n_seqs = _dp_size * _pp_size

        self.__rpc_srcs = list(filter(lambda rpc: rpc.is_src, self.__model_rpcs))
        self.__rpc_dsts = list(filter(lambda rpc: rpc.is_dst, self.__model_rpcs))
        self.__n_rpc_srcs = len(self.__rpc_srcs)
        self.__n_rpc_dsts = len(self.__rpc_dsts)

        # Save and eval control.
        self.__total_train_epochs = config.exp_ctrl.total_train_epochs
        self.__save_ctl = base.timeutil.EpochStepTimeFreqCtl(
            freq_epoch=config.exp_ctrl.save_frequency_epochs,
            freq_step=config.exp_ctrl.save_frequency_steps,
            freq_sec=config.exp_ctrl.save_frequency_seconds,
        )
        self.__eval_ctl = base.timeutil.EpochStepTimeFreqCtl(
            freq_epoch=config.exp_ctrl.eval_frequency_epochs,
            freq_step=config.exp_ctrl.eval_frequency_steps,
            freq_sec=config.exp_ctrl.eval_frequency_seconds,
        )

        self.MODEL_SAVE_ROOT = os.path.join(
            MODEL_SAVE_ROOT,
            config.worker_info.experiment_name,
            config.worker_info.trial_name,
        )
        os.makedirs(self.MODEL_SAVE_ROOT, exist_ok=True)

        self.__initialized = False
        self._epoch = -1
        self._epoch_step = self._global_step = 0

        # for benchmark
        self.e2e_time_history = []
        self.level_time_history = defaultdict(list)
        self.__benchmark_steps = config.exp_ctrl.benchmark_steps

        return config.worker_info

    def __lazy_init(self):
        # Set up streams.
        self.__stream = request_reply_stream.make_master_stream(
            self.config.worker_info,
            n_subscribers=self.config.n_model_workers,
        )
        self.__stream: request_reply_stream.RequestReplyStream

        # Request training specification from data workers, e.g. batch size and total train steps.
        self.__stream.post(request_reply_stream.Payload(
            handler="__data0__",
            handle_name="spec",
        ))
        ft_spec: model_api.FinetuneSpec = self.__stream.poll(block=True).data
        ft_spec.total_train_epochs = self.config.exp_ctrl.total_train_epochs
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
        self.__all_model_handlers: List[config_pkg.ModelShardID] = []
        self.__dp0_model_handlers: List[config_pkg.ModelShardID] = []
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
        init_res = event_loop.run_until_complete(asyncio.gather(_task))[0]
        self.__model_configs: Dict[str, None | FlashMQATConfig] = {}
        assert len(init_res) == len(self.__all_model_handlers), (
            len(init_res),
            len(self.__all_model_handlers),
        )
        for h, r in zip(self.__all_model_handlers, init_res):
            if h.model_name not in self.__model_configs:
                self.__model_configs[h.model_name] = r
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

        self.__data_owner: Dict[Tuple[int, str], Tuple[str, int]] = {}

        logger.info(f"Creating asyncio coroutines...")

        self.__pre_hook_counters = [[0 for _ in rpc.pre_hooks] for rpc in self.__model_rpcs]
        self.__post_hook_counters = [[0 for _ in rpc.post_hooks] for rpc in self.__model_rpcs]

        src_rpc = [rpc for rpc in self.config.model_rpcs][0]
        src_rpc_model_name = src_rpc.model_name
        src_rpc_dp_size = self.config.model_topos[src_rpc.model_name].get_dim("data")

        coroutine_tasks = []
        for rpc, pre_hook_counters, post_hook_counters in zip(self.__model_rpcs, self.__pre_hook_counters,
                                                              self.__post_hook_counters):
            # should be a dict: pp_rank to streams

            request_task = event_loop.create_task(
                model_rpc_request_func(
                    rpc=rpc,
                    msid2mwid=self.config.msid2mwid,
                    src_rpc_model_name=src_rpc_model_name,
                    pre_hook_counters=pre_hook_counters,
                    data_owner=self.__data_owner,
                    stream=self.__stream,
                    buffer=self.__seqbuffer,
                    model_topos=self.__model_topos,
                    model_configs=self.__model_configs,
                    ctrl=self.__rpc_ctrl,
                ))
            reply_tasks = []
            for j in range(rpc.max_concurrent_calls):
                _reply_task = event_loop.create_task(
                    model_rpc_reply_func(
                        corountine_idx=j,
                        rpc=rpc,
                        post_hook_counters=post_hook_counters,
                        stream=self.__stream,
                        buffer=self.__seqbuffer,
                        model_topos=self.__model_topos,
                        model_configs=self.__model_configs,
                        ctrl=self.__rpc_ctrl,
                    ))
                reply_tasks.append(_reply_task)
            coroutine_tasks += [request_task] + reply_tasks

        load_data_task = event_loop.create_task(
            load_data_func(
                src_rpc_dp_size=src_rpc_dp_size,
                src_rpc_model_name=src_rpc_model_name,
                data_owner=self.__data_owner,
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
            f"Epoch {self._epoch + 1}/{self.config.exp_ctrl.total_train_epochs} "
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
