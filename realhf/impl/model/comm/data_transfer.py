from collections import defaultdict
from typing import *
import dataclasses
import itertools

import torch.distributed

from realhf.api.core import system_api
from realhf.api.core.config import ModelName
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
    data_transfer_groups: Dict[DataTransferPair, torch.distributed.ProcessGroup]
    data_transfer_src_ranks: Dict[DataTransferPair, int]
    data_transfer_dst_ranks: Dict[DataTransferPair, List[int]]


def setup_data_transfer(
    model_topos: Optional[
        Dict[str, topology.PipeModelDataParallelTopology]
    ] = None,
    msid2mwid: Optional[Dict[system_api.ModelShardID, int]] = None,
    data_transfer_pairs: Optional[List[Tuple[ModelName, ModelName]]] = None,
) -> DataTransferInfo:

    mw_dp_ranks: Dict[Tuple[ModelName, int], List[int]] = {}
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
            for src_dp, dst_dp in itertools.product(
                range(src_topo.get_dim("data")), range(dst_topo.get_dim("data"))
            ):
                key = DataTransferPair(
                    src=src, src_dp_rank=src_dp, dst=dst, dst_dp_rank=dst_dp
                )
                src_mw_rank = mw_dp_head_ranks[src][src_dp]
                dst_mw_ranks = mw_dp_ranks[dst, dst_dp]
                data_transfer_dst_ranks[key] = dst_mw_ranks
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
    group: torch.distributed.ProcessGroup
    key: str
    buf_indices: List[int]
    seqlens: List[int]


@dataclasses.dataclass
class DataTransferReceiverStep:
    rank: int
    src: int
    dst_ranks: List[int]
    group: torch.distributed.ProcessGroup
    key: str
    buf_indices: List[int]
    seqlens: List[int]


def derive_data_transfer_plan(
    keys: List[str],
    global_buffer_indices: List[int],
    global_seqlens: List[int],
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
        repart_strat = pipeline_repartition_strategy(
            producer_mapping, consumer_mapping
        )

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

            buf_indices = [global_buffer_indices[_i] for _i in comm_slots]
            seqlens = [global_seqlens[_i] for _i in comm_slots]

            for dst_rank in dst_ranks:
                comm_plan.append(
                    DataTransferReceiverStep(
                        rank=dst_rank,
                        src=bcast_src,
                        dst_ranks=dst_ranks,
                        group=group,
                        key=k,
                        buf_indices=buf_indices,
                        seqlens=seqlens,
                    )
                )

            comm_plan.append(
                DataTransferSenderStep(
                    rank=bcast_src,
                    dst_ranks=dst_ranks,
                    group=group,
                    key=k,
                    buf_indices=buf_indices,
                    seqlens=seqlens,
                )
            )

    return comm_plan
