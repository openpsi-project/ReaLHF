from collections import defaultdict
from typing import *
import dataclasses
import itertools
import os
import socket

import torch.distributed

from realhf.api.core import system_api
from realhf.api.core.config import ModelName
from realhf.base import (
    constants,
    gpu_utils,
    name_resolve,
    names,
    network,
    topology,
)


@dataclasses.dataclass
class NCCLProcessGroupInfo:
    world_size: int
    global_rank: int
    local_gpu_id: int
    # 3D parallelism groups of each model.
    model_groups: Dict[str, torch.distributed.ProcessGroup]


def filter_match_mwids(
    model_name: ModelName,
    topo: topology.PipeModelDataParallelTopology,
    msid2mwid: Dict[system_api.ModelShardID, int],
    **conditions,
) -> List[int]:
    if len(conditions) == 0:
        mwids_this_model = [
            msid2mwid[
                system_api.ModelShardID.from_parallelism_rank(
                    model_name, topo, j
                )
            ]
            for j in range(topo.world_size())
        ]
    else:
        mwids_this_model = [
            msid2mwid[
                system_api.ModelShardID.from_parallelism_rank(
                    model_name, topo, j
                )
            ]
            for j in topo.filter_match(**conditions)
        ]
    mwids_this_model = sorted(mwids_this_model)
    assert len(set(mwids_this_model)) == len(mwids_this_model)
    return list(mwids_this_model)


def setup_global_comm(
    expr_name: str,
    trial_name: str,
    worker_index: int,
    model_topos: Optional[
        Dict[str, topology.PipeModelDataParallelTopology]
    ] = None,
    msid2mwid: Optional[Dict[system_api.ModelShardID, int]] = None,
    world_size: Optional[int] = None,  # for testing only
    global_rank: Optional[int] = None,  # for testing only
) -> NCCLProcessGroupInfo:
    assert (world_size is None) == (global_rank is None)
    if world_size is None:
        peers: List[int] = list(
            sorted(
                map(
                    int,
                    name_resolve.get_subtree(
                        names.trainer_ddp_peer(
                            expr_name,
                            trial_name,
                            gpu_utils.GLOBAL_PROCESS_GROUP_NAME,
                        )
                    ),
                )
            )
        )
        assert len(peers) == len(
            set(peers)
        ), f"Duplicated trainer worker index. {peers}"
        world_size = len(peers)
        global_rank = peers.index(worker_index)

    mw_ranks = {}
    if model_topos is not None:
        assert msid2mwid is not None
        for model_name, topo in model_topos.items():
            mw_ranks[model_name] = filter_match_mwids(
                model_name, topo, msid2mwid
            )

    if (
        "GPU_DEVICES_ISOLATED" not in os.environ
        and "RAY" not in os.environ["REAL_MODE"]
    ):
        raise RuntimeError(
            "GPU devices not isolated in slurm or local mode. This should not happen."
        )

    assert len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) == 1, os.environ[
        "CUDA_VISIBLE_DEVICES"
    ]
    local_gpu_id = int(os.environ["CUDA_VISIBLE_DEVICES"])

    ddp_master_name = names.trainer_ddp_master(
        expr_name, trial_name, gpu_utils.GLOBAL_PROCESS_GROUP_NAME
    )

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
    torch.cuda.set_device(
        0
    )  # initialize CUDA here with only a single visible device
    # This environment variable is used by DeepSpeed.
    os.environ["LOCAL_RANK"] = "0"

    torch.distributed.init_process_group(
        **torch_dist_kwargs, group_name=gpu_utils.GLOBAL_PROCESS_GROUP_NAME
    )

    model_groups = {}
    for model_name, ranks in mw_ranks.items():
        model_groups[model_name] = topology.new_or_get_group(
            ranks, backend="nccl"
        )
        constants.set_parallelism_group(
            model_name, model_groups[model_name], ranks
        )

    self_group = None
    for i in range(world_size):
        group = topology.new_or_get_group([i], backend="nccl")
        if i == global_rank:
            self_group = group
            constants.set_self_group(self_group)

    # logger.info(f"Setup process group finishes for worker_index={worker_index}")

    return NCCLProcessGroupInfo(
        world_size=world_size,
        global_rank=global_rank,
        local_gpu_id=local_gpu_id,
        model_groups=model_groups,
    )
