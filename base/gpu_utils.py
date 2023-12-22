from typing import *
import itertools
import os
import platform
import socket
import time

import torch
import torch.distributed

import base.logging as logging
import base.name_resolve as name_resolve
import base.names as names
import base.network as network

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
    logger.info(f"Setting device to {device}.")
    if device != "cpu":
        import torch

        torch.cuda.set_device(device)


def reveal_ddp_identity(expr_name, trial_name, worker_index):
    master_group_name = names.trainer_ddp_peer(expr_name, trial_name, GLOBAL_PROCESS_GROUP_NAME)
    name_resolve.add_subentry(master_group_name, str(worker_index), keepalive_ttl=30)
    # local_peer_name = names.trainer_ddp_local_peer(expr_name, trial_name, socket.gethostname(), peer_name)
    # name_resolve.add_subentry(local_peer_name, peer_index, keepalive_ttl=30)


def setup_ddp(
    expr_name: str,
    trial_name: str,
    worker_index: int,
    peer_indices: Optional[List[int]] = None,
) -> Tuple[int, int, int, Optional[torch.distributed.ProcessGroup]]:
    logger.info(f"Setup process group for worker_index={worker_index}")

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

    if peer_indices is not None:
        peer_ranks = list(sorted([peers.index(x) for x in peer_indices]))

    if "GPU_DEVICES_ISOLATED" not in os.environ and "RAY" not in os.environ["DLLM_MODE"]:
        raise RuntimeError("GPU devices not isolated in slurm or local mode. This should not happen.")

    assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1, os.environ["CUDA_VISIBLE_DEVICES"]
    local_gpu_id = int(os.environ["CUDA_VISIBLE_DEVICES"])

    ddp_master_name = names.trainer_ddp_master(expr_name, trial_name, GLOBAL_PROCESS_GROUP_NAME)

    if worker_index == 0:
        host_ip = socket.gethostbyname(socket.gethostname())
        port = network.find_free_port()
        ddp_init_address = f"tcp://{host_ip}:{port}"
        name_resolve.add(ddp_master_name, ddp_init_address, keepalive_ttl=60)
    else:
        try:
            ddp_init_address = name_resolve.wait(ddp_master_name, timeout=60)
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

    peer_group = None
    if peer_indices is not None:
        logger.info(f"peer ranks {peer_ranks}")
        peer_group = torch.distributed.new_group(peer_ranks, backend="nccl")
    
    torch.distributed.barrier()

    # FIXME: 
    import time
    time.sleep(10)
    return world_size, global_rank, local_gpu_id, peer_group


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
    logger.info(f"Rank {rank} discovers all peers, resolving local rank...")
    local_peer_name = names.trainer_ddp_local_peer(
        experiment_name,
        trial_name,
        socket.gethostname(),
        name_resolve_identifier,
    )
    local_peers = list([str(x) for x in sorted([int(x) for x in name_resolve.get_subtree(local_peer_name)])])
    logger.info(f"Rank {rank} discovers local peers with global ranks {local_peers}")

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

    logger.info(
        f"Worker type {worker_type} rank {rank} running on host {socket.gethostname()}, "
        f"local peer index: {local_peer_index}, local gpu id {local_gpu_id}."
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_gpu_id)
    os.environ["GPU_DEVICES_ISOLATED"] = "1"
