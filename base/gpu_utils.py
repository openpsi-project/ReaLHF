from typing import List
import itertools
import logging
import os
import platform
import socket

import torch

import base.name_resolve as name_resolve
import base.names as names

import base.network as network

logger = logging.getLogger("System-GPU")


def gpu_count():
    """Returns the number of gpus on a node. Ad-hoc to frl cluster.
    """
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


def resolve_cuda_environment():
    """Pytorch DDP does not work if more than one processes (with different environment variable CUDA_VISIBLE_DEVICES)
     are inited on the same node(w/ multiple GPUS). This function works around the issue by setting another variable.
     Currently all devices should use `base.gpu_utils.get_gpu_device()` to get the proper gpu device.
    """
    if "MARL_CUDA_DEVICES" in os.environ.keys():
        return

    cuda_devices = [str(i) for i in range(gpu_count())]
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        if len(cuda_devices) > 0:
            os.environ["MARL_CUDA_DEVICES"] = "0"
        else:
            os.environ["MARL_CUDA_DEVICES"] = "cpu"
    else:
        if os.environ["CUDA_VISIBLE_DEVICES"] != "":
            for s in os.environ["CUDA_VISIBLE_DEVICES"].split(","):
                assert s.isdigit() and s in cuda_devices, f"Cuda device {s} cannot be resolved."
            os.environ["MARL_CUDA_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]  # Store assigned device.
        else:
            os.environ["MARL_CUDA_DEVICES"] = "cpu"  # Use CPU if no cuda device available.
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(cuda_devices)  # Make all devices visible.


def get_gpu_device() -> List[str]:
    """
    Returns:
        List of assigned devices.
    """
    if "MARL_CUDA_DEVICES" not in os.environ:
        resolve_cuda_environment()

    if os.environ["MARL_CUDA_DEVICES"] == "cpu":
        return ["cpu"]
    else:
        return [f"cuda:{device}" for device in os.environ["MARL_CUDA_DEVICES"].split(",")]


def set_cuda_device(device):
    """Set the default cuda-device. Useful on multi-gpu nodes. Should be called in every gpu-thread.
    """
    logger.info(f"Setting device to {device}.")
    if device != "cpu":
        import torch
        torch.cuda.set_device(device)


def reveal_ddp_identity(expr_name, trial_name, model_name, worker_index):
    global_peer_name = names.trainer_ddp_peer(expr_name, trial_name, model_name)
    name_resolve.add_subentry(global_peer_name, worker_index, keepalive_ttl=30)
    local_peer_name = names.trainer_ddp_local_peer(expr_name, trial_name, socket.gethostname(), model_name)
    name_resolve.add_subentry(local_peer_name, worker_index, keepalive_ttl=30)


def setup_ddp(expr_name, trial_name, model_name, worker_index, use_gpu: bool = True):
    if use_gpu:
        import deepspeed
    logger.info(f"Setup DDP {worker_index} for model {model_name}")

    global_peers = list(
        sorted(name_resolve.get_subtree(names.trainer_ddp_peer(expr_name, trial_name, model_name))))

    assert len(global_peers) == len(set(global_peers)), f"Duplicated trainer worker index. {global_peers}"
    world_size = len(global_peers)
    ddp_rank = global_peers.index(str(worker_index))

    local_peer_name = names.trainer_ddp_local_peer(
        expr_name,
        trial_name,
        socket.gethostname(),
        model_name,
    )
    local_peers = list([str(x) for x in sorted([int(x) for x in name_resolve.get_subtree(local_peer_name)])])
    local_peer_index = local_peers.index(str(worker_index))

    if use_gpu:
        if len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == len(local_peers):
            local_gpu_id = list(map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(',')))[local_peer_index]
        elif len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 1:
            local_gpu_id = int(os.environ['CUDA_VISIBLE_DEVICES'])
        else:
            raise RuntimeError(
                f"Unresolvable CUDA_VISIBLE_DEVICES {os.environ['CUDA_VISIBLE_DEVICES']}, "
                f"local peers (global ranks) {local_peers}, local peer index {local_peer_index}.")

        logger.info(f"DDP rank {ddp_rank} running on host {socket.gethostname()}, "
                    f"local peer index: {local_peer_index}, local gpu id {local_gpu_id}.")
    else:
        local_gpu_id = None
        logger.info(f"DDP rank {ddp_rank} running on host {socket.gethostname()}, "
                    f"local peer index: {local_peer_index}.")

    ddp_master_name = names.trainer_ddp_master(expr_name, trial_name, model_name)
    if ddp_rank == 0:
        host_ip = socket.gethostbyname(socket.gethostname())
        port = network.find_free_port()
        ddp_init_address = f"tcp://{host_ip}:{port}"
        name_resolve.add(ddp_master_name, ddp_init_address, keepalive_ttl=15)
    else:
        try:
            ddp_init_address = name_resolve.wait(ddp_master_name, timeout=15)
        except TimeoutError:
            raise TimeoutError(f"DDP trainer(index:{worker_index}), rank {ddp_rank} for model "
                               f"{model_name} wait for ddp_init_method timeout.")

    torch_dist_kwargs = dict(world_size=world_size,
                             rank=ddp_rank,
                             init_method=ddp_init_address,
                             backend='nccl')
    if use_gpu:
        # Set local rank and make all devices visible. These variables are used by DeepSpeed.
        os.environ['CUDA_VISIBLE_DEVICES'] = str(local_gpu_id)
        torch.cuda.set_device(0)

        os.environ['LOCAL_RANK'] = str(local_gpu_id)
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(i) for i in range(gpu_count())])

        dist_kwrags = dict(world_size=world_size,
                           rank=ddp_rank,
                           init_method=ddp_init_address,
                           dist_backend='nccl',
                           auto_mpi_discovery=False)
        deepspeed.init_distributed(**dist_kwrags)
    else:
        torch_dist_kwargs['backend'] = 'gloo'  # this key word differs from deepspeed.
        torch.distributed.init_process_group(**torch_dist_kwargs)

    return world_size, ddp_rank, local_gpu_id
