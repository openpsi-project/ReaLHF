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

import api.config
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


@dataclasses.dataclass
class ParamSyncPair:
    src: str
    src_mp_rank: int
    src_pp_rank: int
    dst: str
    dst_mp_rank: int
    dst_pp_rank: int

    def __hash__(self):
        return (
            self.src,
            self.src_mp_rank,
            self.src_pp_rank,
            self.dst,
            self.dst_mp_rank,
            self.dst_pp_rank,
        ).__hash__()

    def __eq__(self, other: "ParamSyncPair"):
        return (
            self.src,
            self.src_mp_rank,
            self.src_pp_rank,
            self.dst,
            self.dst_mp_rank,
            self.dst_pp_rank,
        ) == (
            other.src,
            other.src_mp_rank,
            other.src_pp_rank,
            other.dst,
            other.dst_mp_rank,
            other.dst_pp_rank,
        )


@dataclasses.dataclass
class NCCLProcessGroupInfo:
    world_size: int
    global_rank: int
    local_gpu_id: int
    # 3D parallelism groups of each model.
    model_groups: Dict[str, torch.distributed.ProcessGroup]
    # Groups to broadcast data to model workers, which contains master and all model workers of the same name.
    # Model workers are further partitioned into several groups to avoid scattering to different
    # pipeline parallelism ranks simultaneously.
    scatter_groups: Dict[str, List[torch.distributed.ProcessGroup]]
    # Groups to gather returned data from model workers, which contains master and DP heads of model workers
    # with the same name. DP head => dp_rank=*, pp_rank=pp_size-1, mp_rank=0.
    # Similar to scatter groups, gather groups are also partitioned to avoid synchronization across pp ranks.
    gather_groups: Dict[str, List[torch.distributed.ProcessGroup]]
    # Groups for parameter synchronization.
    param_sync_groups: Dict[ParamSyncPair, torch.distributed.ProcessGroup]
    param_sync_src_ranks: Dict[ParamSyncPair, int]


def _filter_match_mwids(
    model_name: str,
    topo: topology.PipeModelDataParallelTopology,
    msid2mwid: Dict[api.config.ModelShardID, int],
    **conditions,
) -> List[int]:
    if len(conditions) == 0:
        mwids_this_model = [
            msid2mwid[api.config.ModelShardID.from_parallelism_rank(model_name, topo, j)] + 1
            for j in range(topo.world_size())
        ]
    else:
        mwids_this_model = [
            msid2mwid[api.config.ModelShardID.from_parallelism_rank(model_name, topo, j)] + 1
            for j in topo.filter_match(**conditions)
        ]
    mwids_this_model = sorted(mwids_this_model)
    assert len(set(mwids_this_model)) == len(mwids_this_model)
    return list(mwids_this_model)


def _partition_into_sub_bcast_groups(
    group_ranks: List[int], bcast_groups: List[List[int]]
) -> List[List[int]]:
    bcast_sub_group_ranks = defaultdict(list)
    # -1 here because the rank in the model worker broadcast group starts from 0.
    group_ids = [next(filter(lambda jg: r - 1 in jg[1], enumerate(bcast_groups)))[0] for r in group_ranks]
    for r, gid in zip(group_ranks, group_ids):
        bcast_sub_group_ranks[gid].append(r)
    return list(map(sorted, bcast_sub_group_ranks.values()))


def setup_ddp(
    expr_name: str,
    trial_name: str,
    worker_index: int,
    model_topos: Optional[Dict[str, topology.PipeModelDataParallelTopology]] = None,
    msid2mwid: Optional[Dict[api.config.ModelShardID, int]] = None,
    mw_bcast_groups: Optional[List[List[int]]] = None,
    param_sync_pairs: Optional[List[Tuple[str, str]]] = None,
) -> NCCLProcessGroupInfo:
    peers: List[int] = list(
        sorted(
            map(
                int,
                name_resolve.get_subtree(
                    names.trainer_ddp_peer(expr_name, trial_name, GLOBAL_PROCESS_GROUP_NAME)
                ),
            )
        )
    )
    assert len(peers) == len(set(peers)), f"Duplicated trainer worker index. {peers}"
    world_size = len(peers)
    global_rank = peers.index(worker_index)

    mw_ranks = {}
    mw_head_ranks = {}
    mw_pp_mp_ranks: Dict[Tuple[str, int, int], List[int]] = {}
    mw_pp_mp_head_rank: Dict[Tuple[str, int, int], int] = {}
    if model_topos is not None:
        assert msid2mwid is not None
        for model_name, topo in model_topos.items():
            mw_ranks[model_name] = _filter_match_mwids(model_name, topo, msid2mwid)
            mw_head_ranks[model_name] = _filter_match_mwids(
                model_name, topo, msid2mwid, pipe=topo.get_dim("pipe") - 1, model=0
            )
            pp_size = topo.get_dim("pipe")
            mp_size = topo.get_dim("model")
            for pp_i, mp_i in itertools.product(range(pp_size), range(mp_size)):
                mw_pp_mp_ranks[model_name, pp_i, mp_i] = _filter_match_mwids(
                    model_name,
                    topo,
                    msid2mwid,
                    pipe=pp_i,
                    model=mp_i,
                )
                mw_pp_mp_head_rank[model_name, pp_i, mp_i] = _filter_match_mwids(
                    model_name,
                    topo,
                    msid2mwid,
                    pipe=pp_i,
                    model=mp_i,
                    data=0,
                )[0]

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
                f"global_rank={global_rank} worker_index={worker_index} wait for ddp_init_method timeout."
            )

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
    scatter_groups = defaultdict(list)
    for model_name, ranks in mw_ranks.items():
        model_groups[model_name] = torch.distributed.new_group(ranks, backend="nccl")
        all_group_ranks = _partition_into_sub_bcast_groups(ranks, mw_bcast_groups)
        for group_ranks in all_group_ranks:
            scatter_groups[model_name].append(torch.distributed.new_group([0] + group_ranks, backend="nccl"))
        # logger.info("Created process group for model %s with ranks %s", model_name, ranks)

    gather_groups = defaultdict(list)
    for model_name, ranks in mw_head_ranks.items():
        all_group_ranks = _partition_into_sub_bcast_groups(ranks, mw_bcast_groups)
        for group_ranks in all_group_ranks:
            gather_groups[model_name].append(torch.distributed.new_group([0] + group_ranks, backend="nccl"))
        # logger.info("Created master-DP head group for model %s with ranks %s", model_name, ranks)

    param_sync_groups = {}
    param_sync_src_ranks = {}
    if param_sync_pairs is not None:
        for src, dst in param_sync_pairs:
            src_topo = model_topos[src]
            dst_topo = model_topos[dst]
            for src_pp_i, src_mp_i in itertools.product(
                range(src_topo.get_dim("pipe")), range(src_topo.get_dim("model"))
            ):
                for dst_pp_i, dst_mp_i in itertools.product(
                    range(dst_topo.get_dim("pipe")), range(dst_topo.get_dim("model"))
                ):
                    key = ParamSyncPair(
                        src,
                        src_mp_rank=src_mp_i,
                        src_pp_rank=src_pp_i,
                        dst=dst,
                        dst_mp_rank=dst_mp_i,
                        dst_pp_rank=dst_pp_i,
                    )
                    src_rank = mw_pp_mp_head_rank[src, src_pp_i, src_mp_i]
                    dst_ranks = mw_pp_mp_ranks[dst, dst_pp_i, dst_mp_i]
                    _ranks = [src_rank] + dst_ranks
                    if len(set(_ranks)) != len(_ranks):
                        raise RuntimeError(
                            f"Trying to synchronize parameters from {src} to {dst}, "
                            f"but their GPUs are overlapped. src topo={src_topo}, "
                            f"occupying process group/model worker ranks {mw_ranks[src]}. "
                            f"dst topo={dst_topo}, occupying process group/model worker"
                            f" ranks {mw_ranks[dst]}."
                        )
                    param_sync_groups[key] = torch.distributed.new_group(_ranks, backend="nccl")
                    param_sync_src_ranks[key] = src_rank
    # logger.info(f"Setup process group finishes for worker_index={worker_index}")

    return NCCLProcessGroupInfo(
        world_size=world_size,
        global_rank=global_rank,
        local_gpu_id=local_gpu_id,
        model_groups=model_groups,
        scatter_groups=scatter_groups,
        gather_groups=gather_groups,
        param_sync_groups=param_sync_groups,
        param_sync_src_ranks=param_sync_src_ranks,
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
        names.trainer_ddp_local_peer(
            experiment_name, trial_name, socket.gethostname(), name_resolve_identifier
        ),
        rank,
        keepalive_ttl=60,
    )
    name_resolve.add_subentry(
        names.trainer_ddp_peer(experiment_name, trial_name, name_resolve_identifier),
        rank,
        keepalive_ttl=30,
    )
    logger.info(f"Worker type {worker_type} rank {rank} waiting for peers, world size {world_size}...")
    while (
        len(
            name_resolve.get_subtree(
                names.trainer_ddp_peer(experiment_name, trial_name, name_resolve_identifier)
            )
        )
        < world_size
    ):
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
                f"local peers (global ranks) {local_peers}, local peer index {local_peer_index}."
            )
        devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        local_gpu_id = int(devices[local_peer_index % len(devices)])

    # logger.info(
    #     f"Worker type {worker_type} rank {rank} running on host {socket.gethostname()}, "
    #     f"local peer index: {local_peer_index}, local gpu id {local_gpu_id}."
    # )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_gpu_id)
    os.environ["GPU_DEVICES_ISOLATED"] = "1"


def reveal_ddp_identity_single_model(expr_name, trial_name, model_name, worker_index):
    """Legacy method. Reveal DDP identity for a single model, for testing scripts only"""
    master_group_name = names.trainer_ddp_peer(expr_name, trial_name, model_name)
    name_resolve.add_subentry(master_group_name, str(worker_index), keepalive_ttl=30)
    # local_peer_name = names.trainer_ddp_local_peer(expr_name, trial_name, socket.gethostname(), peer_name)
    # name_resolve.add_subentry(local_peer_name, peer_index, keepalive_ttl=30)


def setup_ddp_single_model(expr_name: str, trial_name: str, model_name: str, worker_index: int):
    """Legacy method. Setup DDP for a single model, for testing scripts only"""
    # logger.info(f"Setup DDP {worker_index} for model {model_name}")

    global_peers = list(
        sorted(map(int, name_resolve.get_subtree(names.trainer_ddp_peer(expr_name, trial_name, model_name))))
    )
    assert len(global_peers) == len(set(global_peers)), f"Duplicated trainer worker index. {global_peers}"
    world_size = len(global_peers)
    ddp_rank = global_peers.index(worker_index)

    if "GPU_DEVICES_ISOLATED" not in os.environ and "RAY" not in os.environ["DLLM_MODE"]:
        raise RuntimeError("GPU devices not isolated in slurm or local mode. This should not happen.")

    assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1, os.environ["CUDA_VISIBLE_DEVICES"]
    local_gpu_id = int(os.environ["CUDA_VISIBLE_DEVICES"])

    ddp_master_name = names.trainer_ddp_master(expr_name, trial_name, model_name)

    if ddp_rank == 0:
        host_ip = socket.gethostbyname(socket.gethostname())
        port = network.find_free_port()
        ddp_init_address = f"tcp://{host_ip}:{port}"
        name_resolve.add(ddp_master_name, ddp_init_address, keepalive_ttl=60)
    else:
        try:
            ddp_init_address = name_resolve.wait(ddp_master_name, timeout=60)
        except TimeoutError:
            raise TimeoutError(
                f"DDP trainer(index:{worker_index}), rank {ddp_rank} for model "
                f"{model_name} wait for ddp_init_method timeout."
            )

    torch_dist_kwargs = dict(
        world_size=world_size, rank=ddp_rank, init_method=ddp_init_address, backend="nccl"
    )
    torch.cuda.set_device(0)  # initialize CUDA here with only a single visible device
    # This environment variable is used by DeepSpeed.
    os.environ["LOCAL_RANK"] = "0"

    torch.distributed.init_process_group(**torch_dist_kwargs, group_name=model_name)

    return world_size, ddp_rank, local_gpu_id
