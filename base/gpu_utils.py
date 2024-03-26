from collections import defaultdict
from typing import *
import dataclasses
import itertools
import os
import platform
import socket
import time

import torch
import torch.distributed

from api.config.config_base import ModelName
import api.config.config_system
import base.logging as logging
import base.name_resolve as name_resolve
import base.names as names
import base.network as network
import base.topology as topology

logger = logging.getLogger("System-GPU", "system")

GPU_DEVICES_ISOLATED = False
GLOBAL_PROCESS_GROUP_NAME = "master"


def gpu_count():
    """Returns the number of gpus on a node. Ad-hoc to frl cluster."""
    if platform.system() == "Darwin":
        return 0
    elif platform.system() == "Windows":
        try:
            import torch

            return torch.cuda.device_count()
        except ImportError:
            return 0
    else:
        dev_directories = list(os.listdir("/dev/"))
        for cnt in itertools.count():
            if "nvidia" + str(cnt) in dev_directories:
                continue
            else:
                break
        return cnt


def set_cuda_device(device):
    """Set the default cuda-device. Useful on multi-gpu nodes. Should be called in every gpu-thread."""
    # logger.info(f"Setting device to {device}.")
    if device != "cpu":
        import torch

        torch.cuda.set_device(device)


def reveal_ddp_identity(expr_name, trial_name, worker_index):
    master_group_name = names.trainer_ddp_peer(expr_name, trial_name, GLOBAL_PROCESS_GROUP_NAME)
    name_resolve.add_subentry(master_group_name, str(worker_index), keepalive_ttl=30)
    # local_peer_name = names.trainer_ddp_local_peer(expr_name, trial_name, socket.gethostname(), peer_name)
    # name_resolve.add_subentry(local_peer_name, peer_index, keepalive_ttl=30)


@dataclasses.dataclass(unsafe_hash=True)
class DataTransferPair:
    src: ModelName
    src_dp_rank: int
    dst: ModelName
    dst_dp_rank: int


@dataclasses.dataclass(unsafe_hash=True)
class ParamSyncPair:
    src: ModelName
    src_dp_rank: int
    src_mp_rank: int
    src_pp_rank: int
    dst: ModelName
    dst_mp_rank: int
    dst_pp_rank: int


@dataclasses.dataclass
class NCCLProcessGroupInfo:
    world_size: int
    global_rank: int
    local_gpu_id: int
    # 3D parallelism groups of each model.
    model_groups: Dict[str, torch.distributed.ProcessGroup]
    # Groups for data transfer among model workers.
    data_transfer_groups: Dict[DataTransferPair, torch.distributed.ProcessGroup]
    data_transfer_src_ranks: Dict[DataTransferPair, int]
    # Groups for parameter synchronization.
    param_sync_groups: Dict[ParamSyncPair, torch.distributed.ProcessGroup]
    param_sync_src_ranks: Dict[ParamSyncPair, int]
    param_sync_dst_ranks: Dict[ParamSyncPair, List[int]]


def _filter_match_mwids(
    model_name: ModelName,
    topo: topology.PipeModelDataParallelTopology,
    msid2mwid: Dict[api.config.config_system.ModelShardID, int],
    **conditions,
) -> List[int]:
    if len(conditions) == 0:
        mwids_this_model = [
            msid2mwid[api.config.config_system.ModelShardID.from_parallelism_rank(model_name, topo, j)]
            for j in range(topo.world_size())
        ]
    else:
        mwids_this_model = [
            msid2mwid[api.config.config_system.ModelShardID.from_parallelism_rank(model_name, topo, j)]
            for j in topo.filter_match(**conditions)
        ]
    mwids_this_model = sorted(mwids_this_model)
    assert len(set(mwids_this_model)) == len(mwids_this_model)
    return list(mwids_this_model)


# def _partition_into_sub_bcast_groups(group_ranks: List[int],
#                                      bcast_groups: List[List[int]]) -> List[List[int]]:
#     bcast_sub_group_ranks = defaultdict(list)
#     # -1 here because the rank in the model worker broadcast group starts from 0.
#     group_ids = [next(filter(lambda jg: r - 1 in jg[1], enumerate(bcast_groups)))[0] for r in group_ranks]
#     for r, gid in zip(group_ranks, group_ids):
#         bcast_sub_group_ranks[gid].append(r)
#     return list(map(sorted, bcast_sub_group_ranks.values()))


def _max_match(_src_ranks: List[int], _grouped_dst_ranks: List[List[int]]):
    from scipy.optimize import linear_sum_assignment

    cost_matrix = []
    for source in _src_ranks:
        costs = []
        for destinations in _grouped_dst_ranks:
            cost = 0 if source in destinations else 1
            costs.append(cost)
        cost_matrix.append(costs)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind


def _create_param_sync_groups(
    from_topo: topology.PipeModelDataParallelTopology,
    to_topo: topology.PipeModelDataParallelTopology,
    src: ModelName,
    dst: ModelName,
    msid2mwid: Dict[api.config.config_system.ModelShardID, int],
    param_sync_groups: Dict[ParamSyncPair, torch.distributed.ProcessGroup],
    param_sync_src_ranks: Dict[ParamSyncPair, int],
    param_sync_dst_ranks: Dict[ParamSyncPair, List[int]],
):
    mwid2msid: Dict[int, Dict[ModelName, api.config.config_system.ModelShardID]] = defaultdict(dict)
    for k, v in msid2mwid.items():
        mwid2msid[v][k.model_name] = k
    for pp_i, pp_j in itertools.product(range(from_topo.get_dim("pipe")), range(to_topo.get_dim("pipe"))):
        # create tensor reshard groups
        src_mp_size = from_topo.get_dim("model")
        dst_mp_size = to_topo.get_dim("model")

        for mp_j in range(dst_mp_size):
            _all_dst_ranks = _filter_match_mwids(dst, to_topo, msid2mwid, pipe=pp_j, model=mp_j)
            if src_mp_size > dst_mp_size:
                factor = src_mp_size // dst_mp_size
                mp_is = list(range(factor * mp_j, factor * (mp_j + 1)))
                _all_src_ranks = [
                    _filter_match_mwids(src, from_topo, msid2mwid, model=mp_i, pipe=pp_i) for mp_i in mp_is
                ]
            else:
                factor = dst_mp_size // src_mp_size
                _all_src_ranks = [
                    _filter_match_mwids(src, from_topo, msid2mwid, model=mp_j // factor, pipe=pp_i)
                ]
            for _src_ranks in _all_src_ranks:
                # All GPUs in _src_ranks have the data required by (pp_j, mp_j)
                # We split _dst_ranks evenly among the GPUs in _src_ranks, such that they can broadcast in parallel.
                subgroup_size = (len(_all_dst_ranks) + len(_src_ranks) - 1) // len(_src_ranks)
                _grouped_dst_ranks = [[] for _ in range(len(_src_ranks))]
                counter = 0
                for _src_rank in _src_ranks:
                    if _src_rank in _all_dst_ranks:
                        _grouped_dst_ranks[counter].append(_src_rank)
                        counter += 1
                for _dst_rank in _all_dst_ranks:
                    if _dst_rank in _src_ranks:
                        continue
                    for _this_group in _grouped_dst_ranks:
                        if len(_this_group) >= subgroup_size:
                            continue
                        _this_group.append(_dst_rank)
                        break
                assert all(len(_x) <= subgroup_size for _x in _grouped_dst_ranks), (
                    _grouped_dst_ranks,
                    _all_dst_ranks,
                    _src_ranks,
                )
                # prioritize the _src_rank that is also in _dst_ranks
                src_rank_ids, group_dst_rank_ids = _max_match(_src_ranks, _grouped_dst_ranks)
                for _src_rank_id, _dst_ranks_group_id in zip(src_rank_ids, group_dst_rank_ids):
                    _src_rank = _src_ranks[_src_rank_id]
                    _dst_ranks = _grouped_dst_ranks[_dst_ranks_group_id]

                    dp_i, mp_i = (
                        from_topo.get_coord(mwid2msid[_src_rank][src].parallelism_rank).data,
                        from_topo.get_coord(mwid2msid[_src_rank][src].parallelism_rank).model,
                    )
                    key = ParamSyncPair(
                        src=src,
                        src_dp_rank=dp_i,
                        src_mp_rank=mp_i,
                        src_pp_rank=pp_i,
                        dst=dst,
                        dst_mp_rank=mp_j,
                        dst_pp_rank=pp_j,
                    )
                    param_sync_dst_ranks[key] = _dst_ranks
                    if _src_rank not in _dst_ranks:
                        _dst_ranks = [_src_rank] + _dst_ranks
                    assert len(set(_dst_ranks)) == len(_dst_ranks)
                    if len(_dst_ranks) > 1:
                        param_sync_groups[key] = torch.distributed.new_group(_dst_ranks)
                    else:
                        param_sync_groups[key] = None
                    param_sync_src_ranks[key] = _src_rank


def setup_ddp(
    expr_name: str,
    trial_name: str,
    worker_index: int,
    model_topos: Optional[Dict[str, topology.PipeModelDataParallelTopology]] = None,
    msid2mwid: Optional[Dict[api.config.config_system.ModelShardID, int]] = None,
    param_sync_pairs: Optional[List[Tuple[ModelName, ModelName]]] = None,
    data_transfer_pairs: Optional[List[Tuple[ModelName, ModelName]]] = None,
) -> NCCLProcessGroupInfo:
    peers: List[int] = list(
        sorted(
            map(
                int,
                name_resolve.get_subtree(
                    names.trainer_ddp_peer(expr_name, trial_name, GLOBAL_PROCESS_GROUP_NAME)),
            )))
    assert len(peers) == len(set(peers)), f"Duplicated trainer worker index. {peers}"
    world_size = len(peers)
    global_rank = peers.index(worker_index)

    mw_ranks = {}
    mw_dp_ranks: Dict[Tuple[ModelName, int], List[int]] = {}
    mw_dp_head_ranks: Dict[ModelName, List[int]] = {}
    if model_topos is not None:
        assert msid2mwid is not None
        for model_name, topo in model_topos.items():
            mw_ranks[model_name] = _filter_match_mwids(model_name, topo, msid2mwid)
            mw_dp_head_ranks[model_name] = _filter_match_mwids(model_name,
                                                               topo,
                                                               msid2mwid,
                                                               pipe=topo.get_dim("pipe") - 1,
                                                               model=0)
            dp_size = topo.get_dim("data")
            for dp_i in range(dp_size):
                mw_dp_ranks[model_name, dp_i] = _filter_match_mwids(
                    model_name,
                    topo,
                    msid2mwid,
                    data=dp_i,
                )

    if "GPU_DEVICES_ISOLATED" not in os.environ and "RAY" not in os.environ["DLLM_MODE"]:
        raise RuntimeError("GPU devices not isolated in slurm or local mode. This should not happen.")

    assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1, os.environ["CUDA_VISIBLE_DEVICES"]
    local_gpu_id = int(os.environ["CUDA_VISIBLE_DEVICES"])

    ddp_master_name = names.trainer_ddp_master(expr_name, trial_name, GLOBAL_PROCESS_GROUP_NAME)

    if worker_index == 0:
        host_ip = socket.gethostbyname(socket.gethostname())
        port = network.find_free_port()
        ddp_init_address = f"tcp://{host_ip}:{port}"
        name_resolve.add(ddp_master_name, ddp_init_address, keepalive_ttl=300)
    else:
        try:
            ddp_init_address = name_resolve.wait(ddp_master_name, timeout=300)
        except TimeoutError:
            raise TimeoutError(
                f"global_rank={global_rank} worker_index={worker_index} wait for ddp_init_method timeout.")

    torch_dist_kwargs = dict(
        world_size=world_size,
        rank=global_rank,
        init_method=ddp_init_address,
        backend="nccl",
    )
    torch.cuda.set_device(0)  # initialize CUDA here with only a single visible device
    # This environment variable is used by DeepSpeed.
    os.environ["LOCAL_RANK"] = "0"

    torch.distributed.init_process_group(**torch_dist_kwargs, group_name=GLOBAL_PROCESS_GROUP_NAME)

    model_groups = {}
    # scatter_groups = defaultdict(list)
    for model_name, ranks in mw_ranks.items():
        model_groups[model_name] = torch.distributed.new_group(ranks, backend="nccl")
        # all_group_ranks = _partition_into_sub_bcast_groups(ranks, mw_bcast_groups)
        # for group_ranks in all_group_ranks:
        #     scatter_groups[model_name].append(torch.distributed.new_group([0] + group_ranks, backend="nccl"))
        # logger.info("Created process group for model %s with ranks %s", model_name, ranks)

    # gather_groups = defaultdict(list)
    # for model_name, ranks in mw_head_ranks.items():
    #     all_group_ranks = _partition_into_sub_bcast_groups(ranks, mw_bcast_groups)
    #     for group_ranks in all_group_ranks:
    #         gather_groups[model_name].append(torch.distributed.new_group([0] + group_ranks, backend="nccl"))
    # logger.info("Created master-DP head group for model %s with ranks %s", model_name, ranks)

    data_transfer_groups, data_transfer_src_ranks = {}, {}
    if data_transfer_pairs is not None:
        for src, dst in data_transfer_pairs:
            src_topo = model_topos[src]
            dst_topo = model_topos[dst]
            for src_dp, dst_dp in itertools.product(range(src_topo.get_dim("data")),
                                                    range(dst_topo.get_dim("data"))):
                key = DataTransferPair(src=src, src_dp_rank=src_dp, dst=dst, dst_dp_rank=dst_dp)
                src_mw_rank = mw_dp_head_ranks[src][src_dp]
                dst_mw_ranks = mw_dp_ranks[dst, dst_dp]
                if src_mw_rank not in dst_mw_ranks:
                    _ranks = [src_mw_rank] + dst_mw_ranks
                else:
                    _ranks = dst_mw_ranks
                data_transfer_groups[key] = torch.distributed.new_group(_ranks, backend="nccl")
                data_transfer_src_ranks[key] = src_mw_rank

    param_sync_groups = {}
    param_sync_src_ranks = {}
    param_sync_dst_ranks = {}
    if param_sync_pairs is not None:
        for src, dst in param_sync_pairs:
            _create_param_sync_groups(
                model_topos[src],
                model_topos[dst],
                src,
                dst,
                msid2mwid,
                param_sync_groups,
                param_sync_src_ranks,
                param_sync_dst_ranks,
            )

    # logger.info(f"Setup process group finishes for worker_index={worker_index}")

    return NCCLProcessGroupInfo(
        world_size=world_size,
        global_rank=global_rank,
        local_gpu_id=local_gpu_id,
        model_groups=model_groups,
        data_transfer_groups=data_transfer_groups,
        data_transfer_src_ranks=data_transfer_src_ranks,
        param_sync_groups=param_sync_groups,
        param_sync_src_ranks=param_sync_src_ranks,
        param_sync_dst_ranks=param_sync_dst_ranks,
    )


def isolate_cuda_device(worker_type: str, rank: int, world_size: int, experiment_name: str, trial_name: str):
    """Isolate CUDA_VISIBLE_DEVICES for each Slurm jobstep.

    To distinguish the concept of job/jobstep/worker/task, check scheduler/slurm/utils.py.
    A slurm job with multiple jobsteps will not set CUDA_VISIBLE_DEVICES properly.
    For example, if a job has 2 jobsteps, each with 1 GPU, and is allocated onto GPU 0 and 1,
    then CUDA_VISIBLE_DEVICES of these jobsteps will be 0,1, instead of 0 and 1.
    We use this function in `apps.remote` to isolate CUDA_VISIBLE_DEVICES for each jobstep.

    Note that this function is completely independent of `setup_ddp`.

    Args:
        worker_type (str): .
        rank (int): Rank of the **jobstep**.
        world_size (int): Size of the **jobsteps**, aka SLURM_NPROCS. However, we may call this function
            in other schedulers (e.g. local scheduler), so we don't use SLURM_NPROCS directly.
        experiment_name (str): .
        trial_name (str): .
    """
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        return

    name_resolve_identifier = f"__type_{worker_type}"
    name_resolve.add_subentry(
        names.trainer_ddp_local_peer(experiment_name, trial_name, socket.gethostname(),
                                     name_resolve_identifier),
        rank,
        keepalive_ttl=60,
    )
    name_resolve.add_subentry(
        names.trainer_ddp_peer(experiment_name, trial_name, name_resolve_identifier),
        rank,
        keepalive_ttl=30,
    )
    logger.info(f"Worker type {worker_type} rank {rank} waiting for peers, world size {world_size}...")
    while (len(
            name_resolve.get_subtree(
                names.trainer_ddp_peer(experiment_name, trial_name, name_resolve_identifier))) < world_size):
        time.sleep(0.1)
    # logger.info(f"Rank {rank} discovers all peers, resolving local rank...")
    local_peer_name = names.trainer_ddp_local_peer(
        experiment_name,
        trial_name,
        socket.gethostname(),
        name_resolve_identifier,
    )
    local_peers = list([str(x) for x in sorted([int(x) for x in name_resolve.get_subtree(local_peer_name)])])
    # logger.info(f"Rank {rank} discovers local peers with global ranks {local_peers}")

    local_peer_index = local_peers.index(str(rank))
    if len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == len(local_peers):
        local_gpu_id = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))[local_peer_index]
    elif len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1:
        local_gpu_id = int(os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        if not os.environ.get("DLLM_MODE") == "LOCAL":
            raise RuntimeError(
                f"Unresolvable CUDA_VISIBLE_DEVICES {os.environ['CUDA_VISIBLE_DEVICES']} on host {network.gethostname()}, "
                f"local peers (global ranks) {local_peers}, local peer index {local_peer_index}.")
        devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        local_gpu_id = int(devices[local_peer_index % len(devices)])

    # logger.info(
    #     f"Worker type {worker_type} rank {rank} running on host {socket.gethostname()}, "
    #     f"local peer index: {local_peer_index}, local gpu id {local_gpu_id}."
    # )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_gpu_id)
    os.environ["GPU_DEVICES_ISOLATED"] = "1"
