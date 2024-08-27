import dataclasses
import itertools
from typing import *

import torch
import torch.distributed as dist

from realhf.api.core.config import ModelName, ModelShardID
from realhf.api.core.data_api import SequenceSample
from realhf.base import topology
from realhf.impl.model.comm.global_comm import filter_match_mwids
from realhf.impl.model.comm.param_realloc import pipeline_repartition_strategy


@dataclasses.dataclass(unsafe_hash=True)
class DataTransferPair:
    src: ModelName
    src_dp_rank: int
    dst: ModelName
    dst_dp_rank: int


@dataclasses.dataclass
class DataTransferInfo:
    # Groups for data transfer among model workers.
    data_transfer_groups: Dict[DataTransferPair, dist.ProcessGroup]
    data_transfer_src_ranks: Dict[DataTransferPair, int]
    data_transfer_dst_ranks: Dict[DataTransferPair, List[int]]


def setup_data_transfer(
    model_topos: Optional[Dict[str, topology.PipeModelDataParallelTopology]] = None,
    msid2mwid: Optional[Dict[ModelShardID, int]] = None,
    data_transfer_pairs: Optional[List[Tuple[ModelName, ModelName]]] = None,
) -> DataTransferInfo:

    # Stores the ranks given a (model_name, dp_rank) pair.
    # These workers correspond to a complete set of model parameters sharded by TP+PP.
    mw_dp_ranks: Dict[Tuple[ModelName, int], List[int]] = {}

    # Stores the dp_head (i.e., mp_rank=0, pp_rank=-1) ranks given a model_name.
    mw_dp_head_ranks: Dict[ModelName, List[int]] = {}

    if model_topos is not None:
        assert msid2mwid is not None
        for model_name, topo in model_topos.items():
            mw_dp_head_ranks[model_name] = filter_match_mwids(
                model_name,
                topo,
                msid2mwid,
                pipe=topo.get_dim("pipe") - 1,
                model=0,
            )
            dp_size = topo.get_dim("data")
            for dp_i in range(dp_size):
                mw_dp_ranks[model_name, dp_i] = filter_match_mwids(
                    model_name,
                    topo,
                    msid2mwid,
                    data=dp_i,
                )

    data_transfer_groups, data_transfer_src_ranks = {}, {}
    data_transfer_dst_ranks = {}
    if data_transfer_pairs is not None:
        for src, dst in data_transfer_pairs:
            src_topo = model_topos[src]
            dst_topo = model_topos[dst]

            # Construct all src-dst pairs, from any src dp rank to any dst dp rank.
            # Note that a dp rank corresponds to multiple parameter shards (TP+PP),
            # so each pair is a group-to-group communication.
            # Since the models in the source group have duplicate data (TP+PP),
            # we just use its "head" as the broadcast source,
            # and broadcast to all the ranks in the destination group.
            for src_dp, dst_dp in itertools.product(
                range(src_topo.get_dim("data")), range(dst_topo.get_dim("data"))
            ):
                key = DataTransferPair(
                    src=src, src_dp_rank=src_dp, dst=dst, dst_dp_rank=dst_dp
                )
                src_mw_rank = mw_dp_head_ranks[src][src_dp]
                dst_mw_ranks = mw_dp_ranks[dst, dst_dp]
                data_transfer_dst_ranks[key] = dst_mw_ranks
                # The src and dst groups can be disjoint or overlapped.
                # If they are disjoint, we need to include the src_mw_rank in the group.
                # Otherwise, we only need to include the dst_mw_ranks.
                if src_mw_rank not in dst_mw_ranks:
                    _ranks = [src_mw_rank] + dst_mw_ranks
                else:
                    _ranks = dst_mw_ranks
                data_transfer_groups[key] = topology.new_or_get_group(
                    _ranks, backend="nccl"
                )
                data_transfer_src_ranks[key] = src_mw_rank

    return DataTransferInfo(
        data_transfer_groups=data_transfer_groups,
        data_transfer_src_ranks=data_transfer_src_ranks,
        data_transfer_dst_ranks=data_transfer_dst_ranks,
    )


@dataclasses.dataclass
class DataTransferSenderStep:
    rank: int
    dst_ranks: List[int]
    group: dist.ProcessGroup
    key: str
    ids: List[int]


@dataclasses.dataclass
class DataTransferReceiverStep:
    rank: int
    src: int
    dst_ranks: List[int]
    group: dist.ProcessGroup
    key: str
    ids: List[int]


def derive_data_transfer_plan(
    keys: List[str],
    global_ids: List[int],
    consumer_name: ModelName,
    consumer_mapping: Dict[int, List[int]],
    producer_names: Dict[str, ModelName],
    producer_mappings: Dict[Tuple[ModelName, str], Dict[int, List[int]]],
    data_transfer_info: DataTransferInfo,
) -> List[DataTransferReceiverStep | DataTransferSenderStep]:
    comm_plan = []

    for k in keys:
        producer_name = producer_names[k]
        producer_mapping = producer_mappings[(producer_name, k)]

        # partition mapping starts from zero, which is different from buffer indices
        repart_strat = pipeline_repartition_strategy(producer_mapping, consumer_mapping)

        for (dp_i, dp_j), comm_slots in repart_strat.items():
            if len(comm_slots) == 0:
                continue

            group_key = DataTransferPair(
                src=producer_name,
                src_dp_rank=dp_i,
                dst=consumer_name,
                dst_dp_rank=dp_j,
            )
            bcast_src = data_transfer_info.data_transfer_src_ranks[group_key]
            group = data_transfer_info.data_transfer_groups[group_key]
            dst_ranks = data_transfer_info.data_transfer_dst_ranks[group_key]

            ids = [global_ids[_i] for _i in comm_slots]

            for dst_rank in dst_ranks:
                comm_plan.append(
                    DataTransferReceiverStep(
                        rank=dst_rank,
                        src=bcast_src,
                        dst_ranks=dst_ranks,
                        group=group,
                        key=k,
                        ids=ids,
                    )
                )

            comm_plan.append(
                DataTransferSenderStep(
                    rank=bcast_src,
                    dst_ranks=dst_ranks,
                    group=group,
                    key=k,
                    ids=ids,
                )
            )

    return comm_plan


def run_data_transfer(
    comm_plan: List[DataTransferReceiverStep | DataTransferSenderStep],
    meta_samples: Dict[int, SequenceSample],
    storage: Dict[int, SequenceSample],
    sent_worker_idx_table: Dict[int, Dict[str, Set[int]]],
    received_worker_idx_table: Dict[int, Dict[str, Set[int]]],
) -> Tuple[Set[int], Set[str]]:
    for step in comm_plan:

        if isinstance(step, DataTransferReceiverStep) and step.rank == dist.get_rank():
            ids = step.ids
            if step.src == dist.get_rank():
                # The receiver is also a sender.
                # We can directly use the data without comm.
                for _id in step.ids:
                    if storage[_id].data[step.key] is not None:
                        storage[_id].data[step.key] = storage[_id].data[step.key].cuda()
            else:
                # If we have to receive remote data, we first check whether
                # the data has been sent here in previous function calls.
                # If so, just fetch it from the cache.
                cached = all(
                    [
                        set(step.dst_ranks).issubset(
                            set(received_worker_idx_table[_id][step.key])
                        )
                        for _id in ids
                    ]
                )

                metadata_cached = all(
                    [
                        set(step.dst_ranks).issubset(
                            set(received_worker_idx_table[_id]["__metadata__"])
                        )
                        for _id in ids
                    ]
                )
                if cached:
                    pass
                else:
                    dtype = meta_samples[ids[0]].dtypes[step.key]
                    total_len = sum(
                        sum(meta_samples[_id].seqlens[step.key][0]) for _id in ids
                    )
                    trailing_shape = meta_samples[ids[0]].trailing_shapes[step.key]

                    # Receive data if it is not None.
                    if trailing_shape is not None:
                        buf = torch.zeros(
                            (total_len, *trailing_shape),
                            dtype=dtype,
                            device=torch.cuda.current_device(),
                        )
                        dist.broadcast(buf, src=step.src, group=step.group)
                    else:
                        buf = None

                    # Receive metadata if not cached.
                    if not metadata_cached:
                        metadatas = [{} for _ in step.ids]
                        dist.broadcast_object_list(
                            metadatas, src=step.src, group=step.group
                        )

                    # Mark that the data has been received.
                    for _id in ids:
                        received_worker_idx_table[_id][step.key].union(step.dst_ranks)
                        received_worker_idx_table[_id]["__metadata__"].union(
                            step.dst_ranks
                        )

                    # Split the received data and put it into the storage.
                    offset = 0
                    for _id, metadata in zip(ids, metadatas):
                        seqlens = meta_samples[_id].seqlens[step.key]
                        assert len(seqlens) == 1
                        seqlen = sum(seqlens[0])
                        if buf is not None:
                            vs = buf[offset : offset + seqlen]
                        else:
                            vs = None
                        offset = offset + seqlen
                        with SequenceSample.disable_validation():
                            s = SequenceSample(
                                keys=[step.key],
                                dtypes={step.key: vs.dtype if vs is not None else None},
                                trailing_shapes={
                                    step.key: vs.shape[1:] if vs is not None else None
                                },
                                ids=[_id],
                                seqlens={step.key: seqlens},
                                data={step.key: vs},
                                metadata=metadata,
                            )
                        if _id in storage:
                            storage[_id].update_(s)
                        else:
                            storage[_id] = s

        if isinstance(step, DataTransferSenderStep) and step.rank == dist.get_rank():
            # Similar to the receiver, we first check whether the data has been sent to all destinations.
            cached = all(
                [
                    set(step.dst_ranks).issubset(
                        set(sent_worker_idx_table[_id][step.key])
                    )
                    for _id in step.ids
                ]
            )
            metadata_cached = all(
                [
                    set(step.dst_ranks).issubset(
                        set(sent_worker_idx_table[_id]["__metadata__"])
                    )
                    for _id in step.ids
                ]
            )
            if cached:
                pass
            else:
                # If not cached, we fetch the data from the storage and send it to all destinations.
                for _id in step.ids:
                    if storage[_id].data[step.key] is not None:
                        storage[_id].data[step.key] = storage[_id].data[step.key].cuda()
                if all([storage[_id].data[step.key] is not None for _id in step.ids]):
                    vs = torch.cat(
                        [storage[_id].data[step.key] for _id in step.ids],
                        dim=0,
                    )
                    dist.broadcast(vs, src=step.rank, group=step.group)

                if not metadata_cached:
                    dist.broadcast_object_list(
                        [storage[_id].metadata for _id in step.ids],
                        src=step.rank,
                        group=step.group,
                    )

                for _id in step.ids:
                    sent_worker_idx_table[_id][step.key].union(step.dst_ranks)
                    sent_worker_idx_table[_id]["__metadata__"].union(step.dst_ranks)
