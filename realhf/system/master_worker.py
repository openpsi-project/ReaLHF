import asyncio
import collections
import copy
import dataclasses
import getpass
import itertools
import os
import pprint
import re
import time
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

import colorama
import deepspeed
import numpy as np
import torch
import torch.distributed

import realhf.api.core.config as config_api
import realhf.api.core.data_api as data_api
import realhf.api.core.dfg as dfg
import realhf.api.core.model_api as model_api
import realhf.api.core.system_api as config_pkg
import realhf.base.recover as recover
import realhf.system.request_reply_stream as request_reply_stream
import realhf.system.worker_base as worker_base
from realhf.api.core.config import ModelName
from realhf.api.core.model_api import ReaLModelConfig
from realhf.base import (
    constants,
    datapack,
    logging,
    name_resolve,
    names,
    timeutil,
    topology,
)
from realhf.base.asyncio_utils import (
    raise_asyncio_exception,
    setup_run_until_complete,
    teardown_run_util_complete,
)
from realhf.base.monitor import (
    caculuate_llama_forward_flops,
    calculate_llama_gen_flops,
    calculate_llama_train_flops,
)
from realhf.system.buffer import AsyncIOSequenceBuffer

logger = logging.getLogger("master worker", "system")
blogger = logging.getLogger("benchmark")


def request_all(
    stream: request_reply_stream.NameResolvingRequestClient,
    handlers: List[str],
    handle_type: str,
    datas: List,
    verbose: bool = True,
):
    requests = [
        request_reply_stream.Payload(
            handler=handler,
            handle_name=handle_type,
            data=data,
        )
        for handler, data in zip(handlers, datas)
    ]
    if verbose:
        blogger.debug(f"master worker #request_all# *end* time ${time.time_ns()}$")
    tik = time.perf_counter()

    # A protocol to ensure that any model worker execute jobs in the same order.
    [stream.post(r) for r in requests]
    [
        stream.poll(block=True, pattern=create_exact_match_pattern([p.syn_reply_id]))
        for p in requests
    ]
    [
        stream.post(
            request_reply_stream.Payload(
                handler=r.handler, handle_name="ack", request_id=r.ack_reply_id
            )
        )
        for r in requests
    ]
    t = time.perf_counter() - tik

    if verbose:
        blogger.debug(
            f'Request "{handle_type}" time in total: '
            f"{t:.4f}s, {t / len(requests):.4f}s per request"
        )
    return [r.request_id for r in requests]


def create_exact_match_pattern(string_list: List[Union[uuid.UUID, str]]) -> re.Pattern:
    escaped_strings = [re.escape(str(s)) for s in string_list]
    pattern = f"({'|'.join(escaped_strings)})$"
    return re.compile(pattern)


async def _awaitable_response(
    stream: request_reply_stream.NameResolvingRequestClient,
    pattern: re.Pattern | None,
) -> request_reply_stream.Payload:
    while True:
        try:
            return stream.poll(pattern=pattern, block=False)
        except request_reply_stream.NoMessage:
            await asyncio.sleep(0.01)
            continue


async def async_gather_replies(
    stream: request_reply_stream.NameResolvingRequestClient,
    request_ids: List[str],
    verbose: bool = True,
) -> List:
    """Collect responses from multiple streams.

    Blocking method.
    """
    responses = await asyncio.gather(
        *[
            _awaitable_response(stream, pattern=create_exact_match_pattern([req_id]))
            for req_id in request_ids
        ]
    )
    if verbose:
        blogger.debug(
            f"master worker #async_gather_replies# *end* time ${time.time_ns()}$"
        )
    return responses


async def async_group_rpc(
    stream: request_reply_stream.NameResolvingRequestClient,
    handlers: List[Union[config_pkg.ModelShardID, str]],
    handle_type: str,
    datas: List,
    verbose: bool = True,
) -> List:
    payloads = await async_gather_replies(
        stream,
        request_all(stream, handlers, handle_type, datas, verbose=verbose),
    )
    return [p.data for p in payloads]


def group_rpc_blocked(
    stream: request_reply_stream.NameResolvingRequestClient,
    handlers: List[Union[config_pkg.ModelShardID, str]],
    handle_type: str,
    datas: List,
    verbose: bool = True,
):
    req_ids = request_all(stream, handlers, handle_type, datas, verbose=verbose)
    payloads = [
        stream.poll(pattern=create_exact_match_pattern([req_id]), block=True)
        for req_id in req_ids
    ]
    return [p.data for p in payloads]


def _request_parameter_sync(
    stream: request_reply_stream.NameResolvingRequestClient,
    msid2mwid: Dict[config_pkg.ModelShardID, int],
    from_model_name: ModelName,
    to_model_name: ModelName,
    from_topo: topology.PipeModelDataParallelTopology,
    to_topo: topology.PipeModelDataParallelTopology,
    to_model_config: ReaLModelConfig,
):

    model_name = from_model_name
    target = to_model_name
    # Prioritize handlers of `from_model`, then handlers of `to_model`.
    # As a result, if both `from_model` and `to_model` reside in a model worker,
    # the handler in the received request will be `from_model`. Layers will also built in `from_model`.
    # After that, we assign layers of the `from_model` to `to_model`.
    handlers = [
        config_pkg.ModelShardID.from_parallelism_rank(model_name, from_topo, j)
        for j in range(from_topo.world_size())
    ]
    all_handler_mwids = set([msid2mwid[h] for h in handlers])
    dst_handlers = [
        config_pkg.ModelShardID.from_parallelism_rank(target, to_topo, j)
        for j in range(to_topo.world_size())
    ]
    for h in dst_handlers:
        if msid2mwid[h] not in all_handler_mwids:
            handlers.append(h)
            all_handler_mwids.add(msid2mwid[h])

    ps_data = {
        "from_model_name": model_name,
        "to_model_name": target,
        "from_topo": from_topo,
        "to_topo": to_topo,
        "to_model_config": to_model_config,
        "eta": 1.0,
    }
    payloads = [
        request_reply_stream.Payload(
            handler=h,
            handle_name="empty",
            pre_hooks=["param_realloc"],
            pre_hook_data=[ps_data],
        )
        for h in handlers
    ]
    request_ids = [stream.post(p) for p in payloads]
    [
        stream.poll(pattern=create_exact_match_pattern([p.syn_reply_id]), block=True)
        for p in payloads
    ]
    [
        stream.post(
            request_reply_stream.Payload(
                handler=p.handler, handle_name="ack", request_id=p.ack_reply_id
            )
        )
        for p in payloads
    ]
    [
        stream.poll(pattern=create_exact_match_pattern([req_id]), block=True)
        for req_id in request_ids
    ]


@dataclasses.dataclass
class InterfaceDataAmount:
    train_configs: List[ReaLModelConfig] = dataclasses.field(default_factory=list)
    train_bs: List[int] = dataclasses.field(default_factory=list)
    train_seqlens: List[List[int]] = dataclasses.field(default_factory=list)

    inf_configs: List[ReaLModelConfig] = dataclasses.field(default_factory=list)
    inf_bs: List[int] = dataclasses.field(default_factory=list)
    inf_seqlens: List[List[int]] = dataclasses.field(default_factory=list)

    gen_configs: List[ReaLModelConfig] = dataclasses.field(default_factory=list)
    gen_bs: List[int] = dataclasses.field(default_factory=list)
    prompt_lens: List[List[int]] = dataclasses.field(default_factory=list)
    gen_len: List[int] = dataclasses.field(default_factory=list)

    def clear(self):
        self.train_bs.clear()
        self.train_seqlens.clear()

        self.inf_bs.clear()
        self.inf_seqlens.clear()

        self.gen_bs.clear()
        self.prompt_lens.clear()
        self.gen_len.clear()

        self.train_configs.clear()
        self.inf_configs.clear()
        self.gen_configs.clear()


@dataclasses.dataclass
class RPCCorountineControl:
    ## Shared resources ##
    stop: asyncio.Event
    # for counting the number of finished training steps
    # one training step corresponds to traversal of the whole DFG
    train_count: asyncio.Queue
    # for loading data, save and eval model
    fetch_data_queue: asyncio.Queue
    data_spec_queue: asyncio.Queue
    eval_queue: asyncio.Queue
    save_queue: asyncio.Queue

    ## Per-coroutine resources ##
    # Used for counting the number of concurrent calls.
    can_do_rpc: Dict[str, asyncio.Semaphore]
    rpc_traversal: Dict[str, int]
    # for synchronizing req ids between req and reply coroutines
    request_queues: Dict[str, asyncio.Queue]

    # for training data management and data cleaning after each step
    ids_to_clear: Set[int] = dataclasses.field(default_factory=set)
    data_amount: InterfaceDataAmount = dataclasses.field(
        default_factory=InterfaceDataAmount
    )

    # recover information
    used_hash_vals_this_epoch: Set[int] = dataclasses.field(default_factory=set)
    is_recover_epoch: bool = False
    hash_vals_to_ignore_in_recover: Set[int] = dataclasses.field(default_factory=set)


def _attach_payloads_with_hooks(
    rpc: dfg.MFCDef,
    payloads: Dict[config_api.ModelShardID, request_reply_stream.Payload],
    mwids: List[int],
    msid2mwid: Dict[config_pkg.ModelShardID, int],
    model_configs: Dict[str, None | ReaLModelConfig],
    model_topos: Dict[str, topology.PipeModelDataParallelTopology],
    main_handlers: List[config_pkg.ModelShardID],
    hook_type: str,
) -> Tuple[Dict[config_api.ModelShardID, request_reply_stream.Payload], List[int]]:
    assert hook_type in ["pre", "post"], hook_type

    main_mwids = set([msid2mwid[h] for h in main_handlers])
    for hook in getattr(rpc, f"_{hook_type}_hooks"):
        if isinstance(hook, dfg.ParamReallocHook):
            assert (hook.source is None) != (hook.target is None), hook
            if hook.source is None:
                src_topo = model_topos[rpc.model_name]
                dst_topo = model_topos[hook.target]
                dst_config = model_configs[hook.target]
                src_model_name, dst_model_name = rpc.model_name, hook.target
                other_model_name = hook.target
                other_topo = dst_topo
            else:
                src_topo = model_topos[hook.source]
                dst_topo = model_topos[rpc.model_name]
                dst_config = model_configs[rpc.model_name]
                src_model_name, dst_model_name = hook.source, rpc.model_name
                other_model_name = hook.source
                other_topo = src_topo

            ps_data = {
                "from_model_name": src_model_name,
                "to_model_name": dst_model_name,
                "from_topo": src_topo,
                "to_topo": dst_topo,
                "to_model_config": dst_config,
                "eta": hook.eta,
            }
            for h in main_handlers:
                getattr(payloads[h], f"{hook_type}_hooks").append("param_realloc")
                getattr(payloads[h], f"{hook_type}_hook_data").append(ps_data)
            other_handlers = [
                config_api.ModelShardID.from_parallelism_rank(
                    other_model_name, other_topo, j
                )
                for j in range(other_topo.world_size())
            ]
            for h in other_handlers:
                if msid2mwid[h] not in mwids:
                    payloads[h] = request_reply_stream.Payload(
                        handler=h,
                        handle_name="empty",
                    )
                    setattr(payloads[h], f"{hook_type}_hooks", ["param_realloc"])
                    setattr(payloads[h], f"{hook_type}_hook_data", [ps_data])
                    mwids.append(msid2mwid[h])
                elif msid2mwid[h] not in main_mwids:
                    hh = next(hh for hh in payloads if msid2mwid[hh] == msid2mwid[h])
                    getattr(payloads[hh], f"{hook_type}_hooks").append("param_realloc")
                    getattr(payloads[hh], f"{hook_type}_hook_data").append(ps_data)

        elif isinstance(hook, dfg.OffloadHook):
            for h in main_handlers:
                getattr(payloads[h], f"{hook_type}_hooks").append("offload")
                getattr(payloads[h], f"{hook_type}_hook_data").append(
                    dict(model_name=h.model_name)
                )
        else:
            raise NotImplementedError(f"Unknown hook type: {hook}")
    return payloads, mwids


def _request_model_function_call(
    rpc: dfg.MFCDef,
    stream: request_reply_stream.NameResolvingRequestClient,
    msid2mwid: Dict[config_pkg.ModelShardID, int],
    model_topos: Dict[str, topology.PipeModelDataParallelTopology],
    model_configs: Dict[str, None | ReaLModelConfig],
    producer_names: Dict[str, str],
    producer_name2producer_handlers: Dict[str, List[config_pkg.ModelShardID]],
    producer_mappings: Dict[str, Dict[str, List[int]]],
    target_mapping: Dict[str, List[int]],
    meta_sample: data_api.SequenceSample,
    handlers: List[config_pkg.ModelShardID],
) -> List[uuid.UUID]:

    dt_data = {
        "keys": rpc.input_keys,
        "target": rpc.model_name,
        "producer_names": producer_names,
        "producer_mappings": producer_mappings,
        "target_mapping": target_mapping,
        "handle_name": rpc.interface_type.value,
        "rpc_name": rpc.name,
        "meta_sample": meta_sample,
    }

    payloads = {
        handler: request_reply_stream.Payload(
            handler=handler,
            handle_name=rpc.interface_type.value,
            pre_hooks=["data_transfer"],
            pre_hook_data=[dt_data],
            data=rpc.name,
        )
        for handler in handlers
    }
    mwids = [msid2mwid[h] for h in handlers]
    assert len(mwids) == len(set(mwids))

    for producer_name in producer_names.values():
        for h in producer_name2producer_handlers[producer_name]:
            if msid2mwid[h] not in mwids:
                payloads[h] = request_reply_stream.Payload(
                    handler=h,
                    handle_name="empty",
                    pre_hooks=["data_transfer"],
                    pre_hook_data=[dt_data],
                )
                mwids.append(msid2mwid[h])

    payloads, mwids = _attach_payloads_with_hooks(
        rpc,
        payloads,
        mwids,
        msid2mwid=msid2mwid,
        model_configs=model_configs,
        model_topos=model_topos,
        main_handlers=handlers,
        hook_type="pre",
    )
    payloads, mwids = _attach_payloads_with_hooks(
        rpc,
        payloads,
        mwids,
        msid2mwid=msid2mwid,
        model_configs=model_configs,
        model_topos=model_topos,
        main_handlers=handlers,
        hook_type="post",
    )
    req_ids = [stream.post(p) for h, p in payloads.items() if h in handlers]
    other_req_ids = [stream.post(p) for h, p in payloads.items() if h not in handlers]
    [
        stream.poll(block=True, pattern=create_exact_match_pattern([p.syn_reply_id]))
        for p in payloads.values()
    ]
    [
        stream.post(
            request_reply_stream.Payload(
                handler=p.handler, handle_name="ack", request_id=p.ack_reply_id
            )
        )
        for p in payloads.values()
    ]
    return req_ids, other_req_ids


async def model_rpc_request_func(
    rpc: dfg.MFCDef,
    msid2mwid: Dict[config_pkg.ModelShardID, int],
    src_rpc_model_name: ModelName,
    stream: request_reply_stream.NameResolvingRequestClient,
    buffer: AsyncIOSequenceBuffer,
    data_owner: Dict[Tuple[int, str], Tuple[ModelName, int]],
    model_topos: Dict[str, topology.PipeModelDataParallelTopology],
    model_configs: Dict[str, None | ReaLModelConfig],
    ctrl: RPCCorountineControl,
):
    """The corountine for sending requests to model workers."""

    topo = model_topos[rpc.model_name]
    handlers = [
        config_pkg.ModelShardID.from_parallelism_rank(rpc.model_name, topo, j)
        for j in range(topo.world_size())
    ]

    producer_names = {}  # data key -> model name
    for k in rpc.input_keys:
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
            config_pkg.ModelShardID.from_parallelism_rank(
                producer_name, model_topos[producer_name], j
            )
            for j in range(model_topos[producer_name].world_size())
        ]

    request_queue = ctrl.request_queues[rpc.name]
    can_do_rpc = ctrl.can_do_rpc[rpc.name]

    this_rpc_consumed_seqs = 0
    while not ctrl.stop.is_set():

        await can_do_rpc.acquire()

        # Ensure that parent RPCs will not be over-consumed.
        while any(
            this_rpc_consumed_seqs >= (ctrl.rpc_traversal[c.name] + 1) * c.n_seqs
            for c in rpc.all_successors()
            if c.interface_type == dfg.ModelInterfaceType.TRAIN_STEP
            and c.model_name.role == rpc.model_name.role
        ):
            await asyncio.sleep(0.1)

        buf_indices, sample = await buffer.get_batch_for_rpc(rpc)

        if rpc.is_src:
            ctrl.used_hash_vals_this_epoch = ctrl.used_hash_vals_this_epoch.union(
                sample.ids
            )

        # Record the data amount for each interface to compute FLOPs.
        # Since the user may arbitrarily specify input/output keys,
        # we can only try to find the most probable key name for computing FLOPs.
        # If such keys do not exist, we will use the key with the longest
        # sequence length in this model function call.
        acc_seqlens = {
            k: sum(sum(x) for x in slens) for k, slens in sample.seqlens.items()
        }
        seqlen_key = max(sample.seqlens, key=acc_seqlens.get)
        flops_seqlens = [sum(x) for x in sample.seqlens[seqlen_key]]
        if rpc.interface_type == dfg.ModelInterfaceType.GENERATE:
            ctrl.data_amount.gen_configs.append(model_configs[rpc.model_name])
            ctrl.data_amount.gen_bs.append(sample.bs)
            ctrl.data_amount.gen_len.append(
                rpc.interface_impl.args["generation_config"]["min_new_tokens"]
            )
            ctrl.data_amount.prompt_lens.append(flops_seqlens)
        elif rpc.interface_type == dfg.ModelInterfaceType.TRAIN_STEP:
            ctrl.data_amount.train_configs.append(model_configs[rpc.model_name])
            ctrl.data_amount.train_bs.append(sample.bs)
            ctrl.data_amount.train_seqlens.append(flops_seqlens)
        elif rpc.interface_type == dfg.ModelInterfaceType.INFERENCE:
            ctrl.data_amount.inf_configs.append(model_configs[rpc.model_name])
            ctrl.data_amount.inf_bs.append(sample.bs)
            ctrl.data_amount.inf_seqlens.append(flops_seqlens)

        this_rpc_consumed_seqs += sample.bs

        # logger.info(f"Model rpc {rpc.name} requesting.")

        # Dispatch data to different data parallel ranks.
        dp_size = topo.get_dim("data")
        if rpc.balanced_dp:
            assert sample.bs % dp_size == 0
            min_n_seqs_per_dp = sample.bs // dp_size
        else:
            min_n_seqs_per_dp = rpc.n_mbs if rpc.n_mbs is not None else 1
        split_spec = sample.get_split_spec(dp_size, min_size=min_n_seqs_per_dp)
        partitions = split_spec.partitions
        target_mapping = {i: list(range(v[0], v[1])) for i, v in enumerate(partitions)}

        # Set data owner of produced data by this RPC, such that downstream RPCs can know
        # where to fetch these data.
        for dp_idx, (st, ed) in enumerate(partitions):
            for i in range(st, ed):
                for k in rpc.output_keys:
                    data_owner[sample.ids[i], k] = (rpc.model_name, dp_idx)

        # Get the data owner of this RPC's input data.
        # We use it to determine the source of data transfer.
        producer_mappings = {}
        for k in rpc.input_keys:
            names, dp_indices = [], []
            for sample_id in sample.ids:
                owner_name, dp_idx = data_owner[(sample_id, k)]
                names.append(owner_name)
                dp_indices.append(dp_idx)
            assert len(set(names)) == 1
            producer_mapping = defaultdict(list)
            for i, dp_idx in enumerate(dp_indices):
                producer_mapping[dp_idx].append(i)
            producer_mapping = {k: sorted(v) for k, v in producer_mapping.items()}
            producer_mappings[names[0], k] = producer_mapping

        # send partitioned data to model workers
        req_ids, other_req_ids = _request_model_function_call(
            rpc=rpc,
            stream=stream,
            msid2mwid=msid2mwid,
            model_topos=model_topos,
            model_configs=model_configs,
            producer_names=producer_names,
            producer_name2producer_handlers=producer_name2producer_handlers,
            producer_mappings=producer_mappings,
            target_mapping=target_mapping,
            meta_sample=sample,
            handlers=handlers,
        )
        await request_queue.put(
            (buf_indices, sample.ids, req_ids, other_req_ids, time.perf_counter())
        )
        logger.info(f"Model rpc {rpc.name} requested.")


async def model_rpc_reply_func(
    rpc: dfg.MFCDef,
    stream: request_reply_stream.NameResolvingRequestClient,
    buffer: AsyncIOSequenceBuffer,
    model_topos: Dict[str, topology.PipeModelDataParallelTopology],
    ctrl: RPCCorountineControl,
):
    topo = model_topos[rpc.model_name]
    dp_size = topo.get_dim("data")
    dp_head_indices = [
        topo.get_rank(data=i, pipe=topo.get_dim("pipe") - 1, model=0)
        for i in range(dp_size)
    ]

    request_queue = ctrl.request_queues[rpc.name]
    can_do_rpc = ctrl.can_do_rpc[rpc.name]

    while not ctrl.stop.is_set():
        # Wait for master worker's request.
        buf_indices, ids, req_ids, other_req_ids, tik = await request_queue.get()

        # Then, wait for all main requests to finish.
        responses = await asyncio.gather(
            *[
                _awaitable_response(
                    stream, pattern=create_exact_match_pattern([req_id])
                )
                for req_id in req_ids
            ]
        )
        # logger.info(f"rpc {rpc.name} received responses {req_ids}")

        # Filter out responses other than DP heads.
        # Other repsonses are duplicated or None.
        responses: List[request_reply_stream.Payload] = [
            responses[i] for i in dp_head_indices
        ]

        # If the returned data is a SequenceSample, it is the data returned by
        # model function calls. The data shoulbe be amended into buffer.
        # Otherwise, it's the train statistics and should be reduced and logged.
        if isinstance(responses[-1].data, data_api.SequenceSample):
            res = data_api.SequenceSample.gather([r.data for r in responses])
        else:
            res = _gather_stat([response.data for response in responses])

        if rpc.log_return_value:
            logger.info(f"RPC name {rpc.name} returns {res}")

        logger.info(
            f"Model rpc {rpc.name} finished. Run time {time.perf_counter() - tik:.4f}s."
        )

        # Release the semaphore to let the request corountine continue running.
        can_do_rpc.release()
        ctrl.rpc_traversal[rpc.name] += 1

        # If this RPC is the final node in the dataflow graph,
        # update the train counter.
        # Otherwise, amend data in the buffer.
        if rpc.is_dst:
            ctrl.ids_to_clear = ctrl.ids_to_clear.union(ids)
            await ctrl.train_count.put(1)
        else:
            logger.info(f"Amending RPC {rpc.name} output keys: {res.keys}")
            await buffer.amend_batch(buf_indices, res.unpack())

        # Wait for all side-effect requests to finish.
        # Side-effect or empty requests are required for data transfer
        # and parameter synchronization.
        # Wait them after the main request to log the oorrect MFC time.
        await asyncio.gather(
            *[
                _awaitable_response(
                    stream, pattern=create_exact_match_pattern([req_id])
                )
                for req_id in other_req_ids
            ]
        )


async def load_data_func(
    src_rpc: dfg.MFCDef,
    src_rpc_dp_size: int,
    src_rpc_model_name: str,
    buffer: AsyncIOSequenceBuffer,
    data_owner: Dict[Tuple[int, str], Tuple[ModelName, int]],
    stream: request_reply_stream.NameResolvingRequestClient,
    ctrl: RPCCorountineControl,
):
    while not ctrl.stop.is_set():
        await ctrl.fetch_data_queue.get()
        # fetch data from dataloader to fill the sequence buffer
        blogger.info(f"Filling data into the buffer in a new epoch.")
        fetch_data_start = time.perf_counter()

        # NOTE: PyTorch dataloader will shuffle data for us.
        all_data: List[data_api.SequenceSample] = []
        received_ids = set()

        # NOTE: Currently we send dataloading requests until iterating
        # over the entire dataset. This may lead to a huge memory waste
        # with super-large datasets. Empirically, it's fine.
        is_final_batch = False
        while not is_final_batch:
            # Send request to model workers to get the specification of data.
            # Data itself is not transferred to the master worker.
            data_batches: List[data_api.DataBatchMeta] = await async_group_rpc(
                stream,
                handlers=[f"__data{i}__" for i in range(src_rpc_dp_size)],
                handle_type="fetch",
                datas=[None for _ in range(src_rpc_dp_size)],
                verbose=False,
            )
            cur_epoch = data_batches[0].epoch

            # Unpack batched sequences into individual sequences.
            for x in data_batches:
                if x.meta_sample is None:
                    continue
                for xx in x.meta_sample.unpack():
                    all_data.append(xx)
                    if xx.ids[0] in received_ids:
                        raise ValueError(
                            f"Duplicate data id {xx.ids[0]}. Is the final batch? {is_final_batch}."
                        )
                    received_ids.add(xx.ids[0])

            # Store the owner information of the data.
            # RPCs corountines will use this information to
            # determine the src and dst of data transfer.
            for dp_rank, db_meta in enumerate(data_batches):
                if db_meta.meta_sample is None:
                    continue
                for s in db_meta.meta_sample.unpack():
                    for k in s.keys:
                        data_owner[(s.ids[0], k)] = (src_rpc_model_name, dp_rank)

            is_final_batch = data_batches[0].is_final_batch
            assert all(is_final_batch == x.is_final_batch for x in data_batches)

        steps_per_epoch = len(all_data) // src_rpc.n_seqs
        # Since different keys may have different sequence lengths, we cannot
        # count tokens accurately. Here we just assume that the key with the
        # longest sequence length is the number of tokens.
        seqlens = [max(sum(v[0]) for v in x.seqlens.values()) for x in all_data]
        avg_tokens_per_batch = sum(seqlens) / steps_per_epoch
        logger.info(
            f"Training epoch {cur_epoch + 1} approximately has {steps_per_epoch} steps. "
            f"Each batch has {avg_tokens_per_batch:.2f} tokens in average."
        )
        await ctrl.data_spec_queue.put((steps_per_epoch, avg_tokens_per_batch))

        # Reorder loaded (meta-)data and store them into the buffer.
        # NOTE: The reordered indices prioritize longer sequences for detecting OOM errors early.
        reorder_indices, _ = datapack.reorder_to_balanced_batches(
            np.array(seqlens), src_rpc.n_seqs
        )
        all_data: List[data_api.SequenceSample] = [all_data[i] for i in reorder_indices]

        if ctrl.is_recover_epoch:
            all_data = list(
                filter(
                    lambda x: x.ids[0] not in ctrl.hash_vals_to_ignore_in_recover,
                    all_data,
                )
            )

        # Store into buffer!
        buffer_indices = await buffer.put_batch(all_data)
        assert len(buffer_indices) == len(all_data)

        # Awake other model RPC corountines to start running the dataflow graph.
        async with buffer.lock:
            buffer.lock.notify(buffer.n_rpcs)

        blogger.info(
            f"Filling data finished. Time consumption: "
            f"{time.perf_counter() - fetch_data_start:.3f}s."
        )


def _gather_stat(src: List[Dict]) -> Dict:
    cnt, stats = {}, {}
    for reply in src:
        for k, v in reply.items():
            cnt[k] = cnt.get(k, 0) + 1
            stats[k] = stats.get(k, 0) + v
    res = {k: v / cnt for k, v, cnt in zip(stats.keys(), stats.values(), cnt.values())}
    for k, c in cnt.items():
        if c != len(src):
            logger.warning(f"Gathered `{k}` is not present in every returned stats.")
    for k, v in res.items():
        if any(abs(v - x.get(k, None)) > 1e-4 for x in src):
            logger.warning(
                f"Gathered `{k}` is not all-reduced "
                f"before returning: ({[x.get(k, None) for x in src]}, {v})."
            )
    return res


async def model_eval_thread_func(
    stream: request_reply_stream.NameResolvingRequestClient,
    handlers: List[config_pkg.ModelShardID],
    eval_queue: asyncio.Queue,
    stop_ctl: asyncio.Event,
):
    while not stop_ctl.is_set():
        epoch, epoch_step = await eval_queue.get()
        eval_stats = await async_group_rpc(
            stream, handlers, "evaluate", [None for _ in handlers]
        )
        eval_stats = _gather_stat(list(filter(lambda x: bool(x), eval_stats)))
        logger.info(
            f"Evaluation results at epoch {epoch} step {epoch_step}: {eval_stats}"
        )


async def model_save_thread_func(
    stream: request_reply_stream.NameResolvingRequestClient,
    handlers: List[config_pkg.ModelShardID],
    model_save_root: str,
    save_queue: asyncio.Queue,
    stop_ctl: asyncio.Event,
):
    while not stop_ctl.is_set():
        epoch, epoch_step, global_step = await save_queue.get()
        model_save_dirs = [
            os.path.join(
                model_save_root,
                s.model_name.role,
                f"epoch{epoch}epochstep{epoch_step}globalstep{global_step}",
            )
            for s in handlers
        ]
        await async_group_rpc(stream, handlers, "save", model_save_dirs)
        logger.info(f"Save models at epoch {epoch} step {epoch_step}.")


class MasterWorker(worker_base.Worker):
    os.makedirs(constants.MODEL_SAVE_ROOT, exist_ok=True)
    global_exp_tik = time.perf_counter()

    def _configure(self, config: config_pkg.MasterWorker):
        self.config = config

        self.__model_topos: Dict[ModelName, topology.PipeModelDataParallelTopology] = (
            config.model_topos
        )

        # Build execution graph and initialize concurrency utilities.
        self.__model_rpcs = config.model_rpcs
        for rpc in self.__model_rpcs:
            _dp_size = self.__model_topos[rpc.model_name].get_dim("data")
            _pp_size = self.__model_topos[rpc.model_name].get_dim("pipe")
            if rpc.n_seqs < _dp_size * _pp_size:
                logger.warning(
                    f"The batch size of RPC `{rpc.name}` in terms of #seqs is smaller than "
                    f"dp_size * pp_size ({_dp_size}*{_pp_size}). Forcely enlarge the batch size "
                    f"to {_dp_size * _pp_size} (dp_size * pp_size). (original: {rpc.n_seqs})"
                )
                rpc.n_seqs = _dp_size * _pp_size

        self.__mwid2msids = defaultdict(list)
        for msid, mwid in self.config.msid2mwid.items():
            self.__mwid2msids[mwid].append(msid)

        self.__rpc_srcs = list(filter(lambda rpc: rpc.is_src, self.__model_rpcs))
        self.__rpc_dsts = list(filter(lambda rpc: rpc.is_dst, self.__model_rpcs))
        self.__n_rpc_srcs = len(self.__rpc_srcs)
        self.__n_rpc_dsts = len(self.__rpc_dsts)

        # Save and eval control.
        self.__total_train_epochs = config.exp_ctrl.total_train_epochs
        self.__save_ctl = timeutil.EpochStepTimeFreqCtl(
            freq_epoch=config.exp_ctrl.save_freq_epochs,
            freq_step=config.exp_ctrl.save_freq_steps,
            freq_sec=config.exp_ctrl.save_freq_secs,
        )
        self.__eval_ctl = timeutil.EpochStepTimeFreqCtl(
            freq_epoch=config.exp_ctrl.eval_freq_epochs,
            freq_step=config.exp_ctrl.eval_freq_steps,
            freq_sec=config.exp_ctrl.eval_freq_secs,
        )

        self.MODEL_SAVE_ROOT = os.path.join(
            constants.MODEL_SAVE_ROOT,
            config.worker_info.experiment_name,
            config.worker_info.trial_name,
        )
        os.makedirs(self.MODEL_SAVE_ROOT, exist_ok=True)

        self.__initialized = False
        self.__recover_run = os.environ.get("REAL_RECOVER_RUN", "0") == "1"
        self.__recover_info = (
            recover.load_recover_info() if self.__recover_run else None
        )
        self.__recover_first_epoch_done = False

        self._epoch = 0
        self._epoch_step = self._global_step = 0
        if self.__recover_run:
            self._epoch = self.__recover_info.recover_start.epoch
            self._epoch_step = self.__recover_info.recover_start.epoch_step
            self._start_global_step = self._global_step = (
                self.__recover_info.recover_start.global_step
            )
            logger.info(
                f"Recovering from previous run: "
                f"{(self._epoch, self._epoch_step, self._global_step)}"
            )

        self.__this_step_info = recover.StepInfo(
            epoch=0,
            epoch_step=0,
            global_step=0,
        )
        self.__last_step_info = None

        # for benchmark
        self.e2e_time_history = []
        self.__benchmark_steps = config.exp_ctrl.benchmark_steps

        return config.worker_info

    def __lazy_init(self):
        # Set up streams.
        handler_routing = copy.deepcopy(self.config.msid2mwid)
        src_rpc = self.__rpc_srcs[0]
        src_rpc_topo = self.config.model_topos[src_rpc.model_name]
        src_rpc_dp_size = src_rpc_topo.get_dim("data")
        src_rpc_pp_size = src_rpc_topo.get_dim("pipe")
        for i in range(src_rpc_dp_size):
            rank = src_rpc_topo.get_rank(data=i, pipe=src_rpc_pp_size - 1, model=0)
            handler_routing[f"__data{i}__"] = self.config.msid2mwid[
                config_pkg.ModelShardID.from_parallelism_rank(
                    model_name=src_rpc.model_name,
                    topo=src_rpc_topo,
                    parallelism_rank=rank,
                )
            ]
        self.__stream = request_reply_stream.make_master_stream(
            self.config.worker_info,
            n_subscribers=self.config.n_model_workers,
            handler_routing=handler_routing,
        )
        self.__stream: request_reply_stream.NameResolvingRequestClient

        src_rpc = [rpc for rpc in self.config.model_rpcs if rpc.is_src][0]
        src_rpc_model_name = src_rpc.model_name
        src_rpc_dp_size = self.config.model_topos[src_rpc.model_name].get_dim("data")

        # Request training specification from data workers.
        total_n_seqs = 0
        for i in range(src_rpc_dp_size):
            p = request_reply_stream.Payload(
                handler=f"__data{i}__",
                handle_name="spec",
            )
            self.__stream.post(p)
            self.__stream.poll(
                block=True, pattern=create_exact_match_pattern([p.syn_reply_id])
            )
            self.__stream.post(
                request_reply_stream.Payload(
                    handler=f"__data{i}__",
                    handle_name="ack",
                    request_id=p.ack_reply_id,
                )
            )
            n_seqs_per_dataset_shard: int = self.__stream.poll(
                block=True, pattern=create_exact_match_pattern([p.request_id])
            ).data
            total_n_seqs += n_seqs_per_dataset_shard

        event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(event_loop)

        # Build some data required for subsequent model function calls.
        self.__all_model_handlers: List[config_pkg.ModelShardID] = []
        self.__all_mw_handlers: List[config_pkg.ModelShardID] = []
        _covered_mws = set()
        self.__dp0_model_handlers: List[config_pkg.ModelShardID] = []
        self.__trainable_model_handlers: List[config_pkg.ModelShardID] = []
        for model_name, topo in self.config.model_topos.items():
            for j in range(topo.world_size()):
                h = config_pkg.ModelShardID.from_parallelism_rank(model_name, topo, j)
                _mw_id = self.config.msid2mwid[h]
                if _mw_id not in _covered_mws:
                    _covered_mws.add(_mw_id)
                    self.__all_mw_handlers.append(h)
            num_dp = topo.get_dim("data")
            self.__all_model_handlers += [
                config_pkg.ModelShardID.from_parallelism_rank(model_name, topo, j)
                for j in range(topo.world_size())
            ]

            if any(
                rpc.model_name == model_name
                and rpc.interface_type == dfg.ModelInterfaceType.TRAIN_STEP
                for rpc in self.__model_rpcs
            ):
                self.__trainable_model_handlers += [
                    config_pkg.ModelShardID.from_parallelism_rank(model_name, topo, j)
                    for j in range(topo.world_size())
                ]
            self.__dp0_model_handlers += [
                config_pkg.ModelShardID.from_parallelism_rank(model_name, topo, j)
                for j in topo.filter_match(data=0)
            ]

        # Request model configs from model workers.
        # Return None if the model is not a ReaLModel.
        self.__model_configs: Dict[ModelName, None | ReaLModelConfig] = {}
        for model_name in self.config.model_topos:
            p = request_reply_stream.Payload(
                handler=config_pkg.ModelShardID.from_parallelism_rank(
                    model_name, topo, 0
                ),
                handle_name="model_config",
            )
            self.__stream.post(p)
            self.__stream.poll(
                block=True, pattern=create_exact_match_pattern([p.syn_reply_id])
            )
            self.__stream.post(
                request_reply_stream.Payload(
                    handler=p.handler,
                    handle_name="ack",
                    request_id=p.ack_reply_id,
                )
            )
            self.__model_configs[model_name] = self.__stream.poll(
                pattern=create_exact_match_pattern([p.request_id]), block=True
            ).data

        # Initialize model backends.
        # For models with the same role, they share the same model parameters.
        # Therefore, we must call reallocate parameters from A to B
        # before we send requests to initialize B.
        _param_senders = [v[0] for v in self.config.sync_param_pairs]
        _param_recevers = [v[1] for v in self.config.sync_param_pairs]

        # The parameters are by default held by the trainable model.
        # If all replicas are not trainable, the parameters are held in replica 0.
        _model_is_trainable = collections.defaultdict(list)
        for rpc in self.__model_rpcs:
            _model_is_trainable[rpc.model_name].append(
                rpc.interface_type == dfg.ModelInterfaceType.TRAIN_STEP
            )

        _model_is_trainable = {
            model_name: any(values)
            for model_name, values in _model_is_trainable.items()
        }

        _roles = set([rpc.model_name.role for rpc in self.__model_rpcs])
        _role_cnt = {
            role: len(
                set(
                    [
                        rpc.model_name
                        for rpc in self.__model_rpcs
                        if rpc.model_name.role == role
                    ]
                )
            )
            for role in _roles
        }
        _reordered_model_names = []
        for role in _roles:
            if _role_cnt[role] == 1:
                _reordered_model_names.append(ModelName(role, 0))
                continue
            _indices = list(range(_role_cnt[role]))
            _trainable_this_role = [
                _model_is_trainable[ModelName(role, i)] for i in range(_role_cnt[role])
            ]
            if any(_trainable_this_role):
                assert sum(_trainable_this_role) == 1
                _trainable_idx = _trainable_this_role.index(True)
                _reordered_model_names.append(ModelName(role, _trainable_idx))
                _indices.remove(_trainable_idx)
            for i in _indices:
                _reordered_model_names.append(ModelName(role, i))

        # Send initialization requests.
        self.logger.info(
            f"Initialize model backends with order: {_reordered_model_names}."
        )
        _initialized_roles = []
        for model_name in _reordered_model_names:
            topo = self.config.model_topos[model_name]
            # Build FinetuneSpec, which is required to initialize backends.
            _handlers = [
                config_pkg.ModelShardID.from_parallelism_rank(model_name, topo, j)
                for j in range(topo.world_size())
            ]
            train_rpcs = list(
                filter(
                    lambda rpc: rpc.model_name == model_name
                    and rpc.interface_type == dfg.ModelInterfaceType.TRAIN_STEP,
                    self.__model_rpcs,
                )
            )
            if len(train_rpcs) > 0:
                assert len(train_rpcs) == 1
                steps_per_epoch = (
                    total_n_seqs * self.config.exp_ctrl.total_train_epochs
                    + train_rpcs[0].n_seqs
                    - 1
                ) // train_rpcs[0].n_seqs
                ft_spec = model_api.FinetuneSpec(
                    total_train_epochs=self.config.exp_ctrl.total_train_epochs,
                    total_train_steps=steps_per_epoch
                    * self.config.exp_ctrl.total_train_epochs,
                    steps_per_epoch=steps_per_epoch,
                )
            else:
                ft_spec = model_api.FinetuneSpec(
                    total_train_epochs=self.config.exp_ctrl.total_train_epochs,
                    total_train_steps=-1,
                    steps_per_epoch=-1,
                )
            model_ft_specs = [ft_spec] * topo.world_size()

            # Reallocate parameters if necessary.
            if model_name.role in _initialized_roles and model_name in _param_recevers:
                _param_realloc_src = _param_senders[_param_recevers.index(model_name)]
                _request_parameter_sync(
                    stream=self.__stream,
                    msid2mwid=self.config.msid2mwid,
                    from_model_name=_param_realloc_src,
                    to_model_name=model_name,
                    from_topo=self.config.model_topos[_param_realloc_src],
                    to_topo=self.config.model_topos[model_name],
                    to_model_config=self.__model_configs[model_name],
                )

            group_rpc_blocked(
                self.__stream,
                handlers=_handlers,
                handle_type="initialize",
                datas=model_ft_specs,
            )

            # Reallocate parameters back.
            if model_name.role in _initialized_roles and model_name in _param_recevers:
                # Reversely sync parameters
                _request_parameter_sync(
                    stream=self.__stream,
                    msid2mwid=self.config.msid2mwid,
                    from_model_name=model_name,
                    to_model_name=_param_realloc_src,
                    to_topo=self.config.model_topos[_param_realloc_src],
                    from_topo=self.config.model_topos[model_name],
                    to_model_config=self.__model_configs[_param_realloc_src],
                )

            _initialized_roles.append(model_name.role)

        logger.info("Initializations of models and backends complete.")

        # Create corountine control objects for running the dataflow graph.
        self.__rpc_ctrl = RPCCorountineControl(
            stop=asyncio.Event(),
            train_count=asyncio.Queue(maxsize=len(self.__rpc_dsts)),
            fetch_data_queue=asyncio.Queue(1),
            data_spec_queue=asyncio.Queue(1),
            eval_queue=asyncio.Queue(1),
            save_queue=asyncio.Queue(1),
            rpc_traversal={rpc.name: 0 for rpc in self.__model_rpcs},
            request_queues={rpc.name: asyncio.Queue(1) for rpc in self.__model_rpcs},
            can_do_rpc={rpc.name: asyncio.Semaphore(1) for rpc in self.__model_rpcs},
            is_recover_epoch=self.__recover_run,
            used_hash_vals_this_epoch=(
                self.__recover_info.hash_vals_to_ignore if self.__recover_run else set()
            ),
            hash_vals_to_ignore_in_recover=(
                self.__recover_info.hash_vals_to_ignore if self.__recover_run else set()
            ),
        )

        self.__fetch_master_ctl = asyncio.Queue(1)

        # NOTE: We don't set a configurable maximum buffer size here because we want to keep all data in the buffer.
        # We assume that the dataset size is at most 1M. We have warnings if the buffer is 95% full.
        self.__seqbuffer = AsyncIOSequenceBuffer(
            self.__model_rpcs,
            max_size=int(1e6),
            fetch_ctl=self.__rpc_ctrl.fetch_data_queue,
            fetch_master_ctl=self.__fetch_master_ctl,
        )

        self.__data_owner: Dict[Tuple[int, str], Tuple[str, int]] = {}

        logger.info(f"Creating asyncio coroutines...")

        # Create coroutines for model RPCs.
        coroutine_tasks = []
        for rpc in self.__model_rpcs:
            request_task = event_loop.create_task(
                model_rpc_request_func(
                    rpc=rpc,
                    msid2mwid=self.config.msid2mwid,
                    src_rpc_model_name=src_rpc_model_name,
                    data_owner=self.__data_owner,
                    stream=self.__stream,
                    buffer=self.__seqbuffer,
                    model_topos=self.__model_topos,
                    model_configs=self.__model_configs,
                    ctrl=self.__rpc_ctrl,
                )
            )
            reply_task = event_loop.create_task(
                model_rpc_reply_func(
                    rpc=rpc,
                    stream=self.__stream,
                    buffer=self.__seqbuffer,
                    model_topos=self.__model_topos,
                    ctrl=self.__rpc_ctrl,
                )
            )
            coroutine_tasks += [request_task, reply_task]

        # Append some utilization coroutines to handle data loading, saving, and evaluation.
        load_data_task = event_loop.create_task(
            load_data_func(
                src_rpc=src_rpc,
                src_rpc_dp_size=src_rpc_dp_size,
                src_rpc_model_name=src_rpc_model_name,
                data_owner=self.__data_owner,
                buffer=self.__seqbuffer,
                stream=self.__stream,
                ctrl=self.__rpc_ctrl,
            )
        )
        eval_task = event_loop.create_task(
            model_eval_thread_func(
                stream=self.__stream,
                handlers=self.__all_model_handlers,
                eval_queue=self.__rpc_ctrl.eval_queue,
                stop_ctl=self.__rpc_ctrl.stop,
            )
        )
        save_task = event_loop.create_task(
            model_save_thread_func(
                stream=self.__stream,
                handlers=self.__trainable_model_handlers,
                model_save_root=self.MODEL_SAVE_ROOT,
                save_queue=self.__rpc_ctrl.save_queue,
                stop_ctl=self.__rpc_ctrl.stop,
            )
        )
        coroutine_tasks += [load_data_task, eval_task, save_task]

        # Set up a run context of EventLoop.run_util_complete, baiscally copy-paste from cpython.
        # With this context, we can call the non-block EventLoop._run_once (similar to worker._poll).
        self.__asyncio_tasks: List[asyncio.Task] = coroutine_tasks
        self.__asyncio_ctx = setup_run_until_complete(
            event_loop, asyncio.gather(*coroutine_tasks)
        )

        logger.info(f"Coroutines created. The master worker is ready to run.")

        self.__initialized = True
        self._train_start_time = time.perf_counter()

        self.__cur_steps_per_epoch = None
        self.__cur_avg_tokens_per_batch = None

    def _poll(self):
        if not self.__initialized:
            self.__lazy_init()

        # Main execution steps. The graph runs under-the-hood in RPC & stream threads.
        # Wait for the finish of the traversal of the execution graph.
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
            except KeyboardInterrupt as e:
                raise_asyncio_exception(self.__asyncio_ctx, raise_error=False)
                raise e
            except:
                raise_asyncio_exception(self.__asyncio_ctx)
        logger.info("Execution finished!")

        # Check whether we have entered a new epoch.
        try:
            self.__fetch_master_ctl.get_nowait()
            is_new_epoch = True
        except asyncio.QueueEmpty:
            is_new_epoch = False

        try:
            (self.__cur_steps_per_epoch, self.__cur_avg_tokens_per_batch) = (
                self.__rpc_ctrl.data_spec_queue.get_nowait()
            )
        except asyncio.QueueEmpty:
            pass

        # Check whether we should evaluate or save models.
        should_eval = self.__eval_ctl.check(epochs=int(is_new_epoch), steps=1)
        should_save = self.__save_ctl.check(epochs=int(is_new_epoch), steps=1)

        if is_new_epoch:
            # do not update epoch counter when recover for the first step
            if self.__recover_run and not self.__recover_first_epoch_done:
                self.__recover_first_epoch_done = True
            else:
                self._epoch += 1
                self._epoch_step = 0
                self.__rpc_ctrl.used_hash_vals_this_epoch = set()

        # Updata counters.
        self._epoch_step += 1
        self._global_step += 1

        if should_eval:
            self.__rpc_ctrl.eval_queue.put_nowait((self._epoch, self._epoch_step))
        if should_save:
            self.__rpc_ctrl.save_queue.put_nowait(
                (self._epoch, self._epoch_step, self._global_step)
            )

        # update step info
        self.__last_step_info = self.__this_step_info
        self.__this_step_info = recover.StepInfo(
            epoch=self._epoch,
            epoch_step=self._epoch_step,
            global_step=self._global_step,
        )

        time_since_configure = time.perf_counter() - self._train_start_time
        time_per_step = time_since_configure / (self._global_step + 1)
        e2e_time = time.perf_counter() - execution_start
        self.e2e_time_history.append(e2e_time)

        self._log_training_stats(e2e_time, time_since_configure)

        # Pause the worker if experiment or system-wise benchmark completes.
        if (
            self.__benchmark_steps is not None
            and self._global_step >= self.__benchmark_steps
        ) or (is_new_epoch and self._epoch > self.__total_train_epochs):
            if should_eval:
                eval_stats = group_rpc_blocked(
                    self.__stream,
                    self.__all_model_handlers,
                    "evaluate",
                    [None for _ in self.__all_model_handlers],
                )
                eval_stats = _gather_stat(list(filter(lambda x: bool(x), eval_stats)))
                logger.info(
                    f"Evaluation results at epoch {self._epoch} step {self._epoch_step}: {eval_stats}"
                )
            if should_save:
                model_save_dirs = [
                    os.path.join(
                        self.MODEL_SAVE_ROOT,
                        s.model_name.role,
                        f"epoch{self._epoch}epochstep{self._epoch_step}globalstep{self._global_step}",
                    )
                    for s in self.__trainable_model_handlers
                ]
                group_rpc_blocked(
                    self.__stream,
                    self.__trainable_model_handlers,
                    "save",
                    model_save_dirs,
                )
                logger.info(
                    f"Save models at epoch {self._epoch} step {self._epoch_step}."
                )
            if self.__benchmark_steps is not None:
                logger.info(
                    f"Finished benchmark {self.__benchmark_steps}. "
                    f"Time consumption of this setup: {time_since_configure:.3f}"
                )
                logger.info(f"avg #e2e# time *{np.mean(self.e2e_time_history):.3f}*")
            return self.experiment_complete_exit()

        # Send clear cache requests to model workers.
        self._clear_gpu_cache()

        return worker_base.PollResult(sample_count=1, batch_count=1)

    def _log_training_stats(self, e2e_time: float, time_since_configure: float):
        # calculate flops
        #########################################
        if not all(
            isinstance(v, ReaLModelConfig) for v in self.__model_configs.values()
        ):
            logger.warning(
                f"Not all models are ReaLModels. Unable to calculate FLOP/s."
            )
            flops = None
        else:
            flops = 0
            for train_bs, train_seqlens, real_config in zip(
                self.__rpc_ctrl.data_amount.train_bs,
                self.__rpc_ctrl.data_amount.train_seqlens,
                self.__rpc_ctrl.data_amount.train_configs,
            ):
                flops += calculate_llama_train_flops(
                    checkpoint_activations_factor=4,
                    batch_size=train_bs,
                    seqlens=train_seqlens,
                    num_layers=real_config.n_layers,
                    hidden_size=real_config.hidden_dim,
                    intermediate_size=real_config.intermediate_dim,
                    vocab_size=real_config.vocab_size,
                )
            for inf_bs, inf_seqlens, real_config in zip(
                self.__rpc_ctrl.data_amount.inf_bs,
                self.__rpc_ctrl.data_amount.inf_seqlens,
                self.__rpc_ctrl.data_amount.inf_configs,
            ):
                flops += caculuate_llama_forward_flops(
                    batch_size=inf_bs,
                    seqlens=inf_seqlens,
                    num_layers=real_config.n_layers,
                    hidden_size=real_config.hidden_dim,
                    intermediate_size=real_config.intermediate_dim,
                    vocab_size=real_config.vocab_size,
                )
            for gen_bs, prompt_lens, gen_len, real_config in zip(
                self.__rpc_ctrl.data_amount.gen_bs,
                self.__rpc_ctrl.data_amount.prompt_lens,
                self.__rpc_ctrl.data_amount.gen_len,
                self.__rpc_ctrl.data_amount.gen_configs,
            ):
                flops += calculate_llama_gen_flops(
                    batch_size=gen_bs,
                    prompt_lens=prompt_lens,
                    gen_len=gen_len,
                    num_layers=real_config.n_layers,
                    hidden_size=real_config.hidden_dim,
                    intermediate_size=real_config.intermediate_dim,
                    vocab_size=real_config.vocab_size,
                )
            tflops = flops / (e2e_time * (10**12))
            tflops_per_gpu = flops / (e2e_time * self.config.n_model_workers * (10**12))
        self.__rpc_ctrl.data_amount.clear()
        #########################################

        s = f"Epoch {self._epoch}/{self.config.exp_ctrl.total_train_epochs} "
        if self.__cur_steps_per_epoch is not None:
            s += f"step {self._epoch_step}/{self.__cur_steps_per_epoch} "
        else:
            s += f"step {self._epoch_step} "
        s += f"(global step {self._global_step}) finishes. "
        if self.__cur_avg_tokens_per_batch is not None:
            s += f"Average #tokens per batch is {self.__cur_avg_tokens_per_batch:.0f}. "
        s += f"#End to end# execution time: *{e2e_time:.3f}*s. "
        s += f"Total time consumption: {time_since_configure:.3f}s. "
        if self.__cur_steps_per_epoch is not None and len(self.e2e_time_history) > 2:
            remaining_steps = self.__cur_steps_per_epoch - self._epoch_step
            remaining_epochs = self.__total_train_epochs - self._epoch
            avg_t = np.mean(self.e2e_time_history[2:])
            remain_t = avg_t * remaining_steps
            remain_t += avg_t * self.__cur_steps_per_epoch * remaining_epochs
            s += f"Estimated remaining time: {remain_t:.3f}s. "
        if flops is not None:
            s += f"TFLOP/s per GPU: {tflops_per_gpu:.2f}, total TFLOP/s: {tflops:.2f}."
        logger.info(s)
        logger.info(
            f"Time taken so far across all configurations: {time.perf_counter() - self.global_exp_tik:.2f}s"
        )

    def _clear_gpu_cache(self):
        request_all(
            self.__stream,
            self.__all_mw_handlers,
            "clear_data_cache",
            [self.__rpc_ctrl.ids_to_clear for _ in self.__all_mw_handlers],
        )
        self.__rpc_ctrl.ids_to_clear.clear()

    def experiment_complete_exit(self):
        self.__rpc_ctrl.stop.set()
        for task in self.__asyncio_tasks:
            task.cancel()
        self.__asyncio_ctx.future.set_result(None)
        # NOTE: stopping the loop immediately after cancelling tasks may
        # raise warnings sometimes, but it doesn't matter.
        self.__asyncio_ctx.loop.stop()
        teardown_run_util_complete(self.__asyncio_ctx)
        logger.info(
            colorama.Style.RESET_ALL
            + colorama.Fore.YELLOW
            + colorama.Style.BRIGHT
            + "\033[1m"
            + "Experiment Completes! Yeah!!!!!!!!"
            + colorama.Style.RESET_ALL
        )

        # Send requests to pause model workers.
        # Model workers will not respond to this message.
        request_all(
            self.__stream,
            handlers=self.__all_mw_handlers,
            handle_type="reset",
            datas=[None for _ in self.__all_mw_handlers],
        )
        self.__stream.close()
        constants.reset_run()
        # Reset names used for distributed training.
        # The next round of training will set up a new distributed environment.
        name_resolve.clear_subtree(
            names.distributed_root(constants.experiment_name(), constants.trial_name())
        )
        name_resolve.clear_subtree(
            names.request_reply_stream_root(
                constants.experiment_name(), constants.trial_name()
            )
        )
        self.__initialized = False
        self.pause()
        return worker_base.PollResult(0, 0)

    def __recover_save(self):
        # save step info for recover
        recover_info = recover.RecoverInfo(
            # recover_start=self.__last_step_info,
            recover_start=self.__this_step_info,
            last_step_info=self.__last_step_info,
            hash_vals_to_ignore=self.__rpc_ctrl.used_hash_vals_this_epoch,
        )

        logger.info("dumped recover info to file")
        pprint.pprint(recover_info.recover_start)
        pprint.pprint(recover_info.last_step_info)
        pprint.pprint(len(recover_info.hash_vals_to_ignore))
        recover.dump_recover_info(recover_info)

    def _exit_hook(self, exit_status: worker_base.WorkerServerStatus):
        logger.info(f"Master worker exits with {exit_status}.")
        if os.environ["REAL_SAVE_RECOVER_STATES"] == "0":
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
