from typing import *
import torch.distributed as dist
import torch
import random
import multiprocessing as mp
import dataclasses
import os
import time

import torch.distributed
from api.config.config_system import ModelShardID
from base.topology import PipeModelDataParallelTopology
import itertools


param_size = 8
n_layers = 16


def repartition_strategy(
    layer_mapping1: Dict[int, List[int]],
    layer_mapping2: Dict[int, List[int]],
):
    assert set(sum(layer_mapping1.values(), [])) == set(sum(layer_mapping2.values(), []))

    layer_map: Dict[Tuple[int, int], List[int]] = {}
    for pp_rank2, layer_indices2 in layer_mapping2.items():
        for pp_rank1, layer_indices1 in layer_mapping1.items():
            layer_map[(pp_rank1, pp_rank2)] = sorted(
                list(set(layer_indices1).intersection(set(layer_indices2)))
            )

    return layer_map


def create_global_param(param_size, n_layers):
    params = {i: torch.randn(param_size, device="cuda") for i in range(n_layers)}
    for v in params.values():
        torch.distributed.all_reduce(v)
    return params


def get_layer_mapping(pp_size: int):
    layer_mapping = {}
    for i in range(pp_size):
        layer_mapping[i] = list(range(i * n_layers // pp_size, (i + 1) * n_layers // pp_size))
    return layer_mapping


def get_param_shape(layer_idx, mp_size):
    return (param_size // mp_size,)


def get_param_dtype(layer_idx):
    return torch.float32


def setup_comm_groups(from_shard_id: ModelShardID, to_shard_id: ModelShardID):
    comm_groups = {}
    comm_src = {}
    comm_dst_ranks = {}
    for src_pp_rank, dst_pp_rank in itertools.product(
        range(from_shard_id.topo.get_dim("pipe")), range(to_shard_id.topo.get_dim("pipe"))
    ):
        # create tensor reshard groups
        src_mp_size = from_shard_id.topo.get_dim("model")
        dst_mp_size = to_shard_id.topo.get_dim("model")

        for mp_j in range(dst_mp_size):
            _all_dst_ranks = to_shard_id.topo.filter_match(pipe=dst_pp_rank, model=mp_j)
            if src_mp_size > dst_mp_size:
                factor = src_mp_size // dst_mp_size
                mp_is = list(range(factor * mp_j, factor * (mp_j + 1)))
                _all_src_ranks = [
                    from_shard_id.topo.filter_match(model=mp_i, pipe=src_pp_rank) for mp_i in mp_is
                ]
            else:
                factor = dst_mp_size // src_mp_size
                _all_src_ranks = [from_shard_id.topo.filter_match(model=mp_j // factor, pipe=src_pp_rank)]
            for receiver_portion_id, _src_ranks in enumerate(_all_src_ranks):
                # All GPUs in _src_ranks have the data required by (pp_j, mp_j)
                # We split _dst_ranks evenly among the GPUs in _src_ranks, such that they can broadcast in parallel.
                subgroup_size = (len(_all_dst_ranks) + len(_src_ranks) - 1) // len(_src_ranks)
                for subgroup_id, _src_rank in enumerate(_src_ranks):
                    _dst_ranks = _all_dst_ranks[
                        subgroup_id * subgroup_size : (subgroup_id + 1) * subgroup_size
                    ]
                    dp_i, mp_i = (
                        from_shard_id.topo.get_coord(_src_rank).data,
                        from_shard_id.topo.get_coord(_src_rank).model,
                    )
                    comm_dst_ranks[(dp_i, mp_i, src_pp_rank, mp_j, dst_pp_rank)] = _dst_ranks
                    if _src_rank not in _dst_ranks:
                        _dst_ranks = [_src_rank] + _dst_ranks
                    if len(_dst_ranks) > 1:
                        comm_groups[(dp_i, mp_i, src_pp_rank, mp_j, dst_pp_rank)] = dist.new_group(_dst_ranks)
                    else:
                        comm_groups[(dp_i, mp_i, src_pp_rank, mp_j, dst_pp_rank)] = None
                    comm_src[(dp_i, mp_i, src_pp_rank, mp_j, dst_pp_rank)] = _src_rank

    return comm_groups, comm_src, comm_dst_ranks


def worker(idx: int, from_topo: PipeModelDataParallelTopology, to_topo: PipeModelDataParallelTopology):
    torch.cuda.set_device(idx)
    dist.init_process_group(backend="nccl", init_method="tcp://localhost:7777", rank=idx, world_size=8)
    torch.cuda.set_device(idx)

    global_state_dict = create_global_param(param_size, n_layers)

    from_shard_id = ModelShardID.from_parallelism_rank("default", from_topo, idx)
    to_shard_id = ModelShardID.from_parallelism_rank("default", to_topo, idx)

    dst_mp_size = to_shard_id.topo.get_dim("model")
    src_mp_size = from_shard_id.topo.get_dim("model")

    from_layer_idx_start = n_layers // from_topo.get_dim("pipe") * from_shard_id.pp_rank
    from_layer_idx_end = n_layers // from_topo.get_dim("pipe") * (from_shard_id.pp_rank + 1)

    _param_size_per_mp = param_size // from_topo.get_dim("model")
    state_dict = {
        i: x[from_shard_id.mp_rank * _param_size_per_mp : (from_shard_id.mp_rank + 1) * _param_size_per_mp]
        for i, x in global_state_dict.items()
        if i in range(from_layer_idx_start, from_layer_idx_end)
    }

    # initialize groups for communication
    comm_groups, comm_src, comm_dst_ranks = setup_comm_groups(from_shard_id, to_shard_id)

    from_layer_mapping = get_layer_mapping(from_shard_id.topo.get_dim("pipe"))
    to_layer_mapping = get_layer_mapping(to_shard_id.topo.get_dim("pipe"))
    repart_strat = repartition_strategy(from_layer_mapping, to_layer_mapping)
    print(repart_strat)

    @dataclasses.dataclass
    class SenderStep:
        rank: int
        sender_mp_portion_id: int
        receiver_mp_portion_id: int
        param_key: str | int
        group: torch.distributed.ProcessGroup
        dst_ranks: List[int]
        remove: bool = False

    @dataclasses.dataclass
    class ReceverStep:
        rank: int
        sender_mp_portion_id: int
        receiver_mp_portion_id: int
        param_key: str | int
        param_shape: torch.Size
        param_dtype: torch.dtype
        src: int
        group: torch.distributed.ProcessGroup

    comm_plan = []

    src_dp_size = from_shard_id.topo.get_dim("data")
    dst_dp_size = to_shard_id.topo.get_dim("data")

    # derive a global NCCL communication plan
    for (pp_i, pp_j), layer_indices in repart_strat.items():
        sub_sd = {
            i: (get_param_shape(i, from_shard_id.topo.get_dim("model")), get_param_dtype(i))
            for i in layer_indices
        }
        for k, (shape, dtype) in sub_sd.items():
            for mp_i in range(src_mp_size):
                if dst_mp_size > src_mp_size:
                    factor = dst_mp_size // src_mp_size
                    mp_js = [i + factor * mp_i for i in range(factor)]
                    dtypes = [dtype for _ in range(factor)]
                    shapes = [(shape[0] // factor,) for _ in range(factor)]
                    receiver_mp_portion_id = 0
                else:
                    factor = src_mp_size // dst_mp_size
                    mp_js = [mp_i // factor]
                    dtypes = [dtype]
                    shapes = [shape]
                    receiver_mp_portion_id = mp_i % factor
                for sender_mp_portion_id, (mp_j, _dtype, _shape) in enumerate(zip(mp_js, dtypes, shapes)):
                    for dp_i in range(src_dp_size):
                        key = (
                            dp_i,
                            mp_i,
                            pp_i,
                            mp_j,
                            pp_j,
                        )
                        src = comm_src[key]
                        group = comm_groups[key]
                        dst_ranks = comm_dst_ranks[key]

                        for dst_rank in dst_ranks:
                            comm_plan.append(
                                ReceverStep(
                                    rank=dst_rank,
                                    sender_mp_portion_id=sender_mp_portion_id,
                                    receiver_mp_portion_id=receiver_mp_portion_id,
                                    param_key=k,
                                    param_shape=_shape,
                                    param_dtype=_dtype,
                                    src=src,
                                    group=group,
                                )
                            )
                        comm_plan.append(
                            SenderStep(
                                rank=src,
                                sender_mp_portion_id=sender_mp_portion_id,
                                receiver_mp_portion_id=receiver_mp_portion_id,
                                param_key=k,
                                group=group,
                                dst_ranks=dst_ranks,
                            )
                        )

    for i, step in enumerate(comm_plan):
        if isinstance(step, ReceverStep):
            continue
        step: SenderStep
        required_by_nex_steps = False
        for nex_step in comm_plan[i + 1 :]:
            if (
                isinstance(nex_step, SenderStep)
                and nex_step.rank == step.rank
                and nex_step.param_key == step.param_key
            ):
                required_by_nex_steps = True
                break
        step.remove = not required_by_nex_steps

    if idx == 0:
        print(comm_plan)

    # input()

    new_state_dict = {}
    for step in comm_plan:
        portion_size = min(param_size // dst_mp_size, param_size // src_mp_size)
        if isinstance(step, ReceverStep) and step.rank == idx:
            if step.param_key not in new_state_dict:
                new_state_dict[step.param_key] = torch.zeros(
                    param_size // to_shard_id.topo.get_dim("model"), dtype=step.param_dtype, device="cuda"
                )

            if step.rank == step.src:
                buf = state_dict[step.param_key][
                    step.sender_mp_portion_id * portion_size : (step.sender_mp_portion_id + 1) * portion_size
                ]
            else:
                buf = torch.zeros(step.param_shape, dtype=step.param_dtype, device="cuda")
                torch.distributed.broadcast(buf, src=step.src, group=step.group)

            new_state_dict[step.param_key][
                step.receiver_mp_portion_id * portion_size : (step.receiver_mp_portion_id + 1) * portion_size
            ] = buf
        if isinstance(step, SenderStep) and step.rank == idx:
            if step.group is not None:
                buf = state_dict[step.param_key][
                    step.sender_mp_portion_id * portion_size : (step.sender_mp_portion_id + 1) * portion_size
                ]
                torch.distributed.broadcast(buf, src=step.rank, group=step.group)
            if step.remove:
                state_dict.pop(step.param_key)

    assert len(state_dict) == 0, (idx, state_dict.keys(), new_state_dict.keys())
    assert len(new_state_dict) == n_layers // to_shard_id.topo.get_dim("pipe"), len(new_state_dict)

    to_layer_idx_start = n_layers // to_topo.get_dim("pipe") * to_shard_id.pp_rank
    to_layer_idx_end = n_layers // to_topo.get_dim("pipe") * (to_shard_id.pp_rank + 1)
    for i in range(to_layer_idx_start, to_layer_idx_end):
        to_mp_rank = to_shard_id.mp_rank
        a = global_state_dict[i][
            param_size
            // to_shard_id.topo.get_dim("model")
            * to_mp_rank : param_size
            // to_shard_id.topo.get_dim("model")
            * (to_mp_rank + 1)
        ]
        assert torch.allclose(new_state_dict[i], a), (new_state_dict[i], a, i)

    torch.distributed.barrier()
    if idx == 0:
        print("success!")


def decompose_to_three_factors(n: int):
    factors = []
    for i in range(1, int(n ** (1 / 2)) + 1):
        if n % i == 0:
            for j in range(i, int((n // i) ** (1 / 2)) + 1):
                if (n // i) % j == 0:
                    k = (n // i) // j
                    factors += list(set(itertools.permutations([i, j, k])))
    return factors


if __name__ == "__main__":
    decompositions = decompose_to_three_factors(8)
    for x1, x2 in itertools.product(decompositions, decompositions):
        print(f"testing from {x1} to {x2}")
        from_topo = PipeModelDataParallelTopology(num_pp=x1[0], num_mp=x1[1], num_dp=x1[2])
        to_topo = PipeModelDataParallelTopology(num_pp=x2[0], num_mp=x2[1], num_dp=x2[2])
        procs = []
        for i in range(8):
            proc = mp.Process(target=worker, args=(i, from_topo, to_topo))
            procs.append(proc)
            proc.start()
        for proc in procs:
            proc.join()
