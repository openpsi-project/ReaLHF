from typing import *
import dataclasses
import itertools
import multiprocessing as mp
import os
import queue
import random
import time

import torch
import torch.distributed
import torch.distributed as dist

from api.config.config_system import ModelName, ModelShardID
from base.topology import PipeModelDataParallelTopology

param_size = 24
n_layers = 24


def repartition_strategy(
    layer_mapping1: Dict[int, List[int]],
    layer_mapping2: Dict[int, List[int]],
):
    assert set(sum(layer_mapping1.values(), [])) == set(sum(layer_mapping2.values(), []))

    layer_map: Dict[Tuple[int, int], List[int]] = {}
    for pp_rank2, layer_indices2 in layer_mapping2.items():
        for pp_rank1, layer_indices1 in layer_mapping1.items():
            layer_map[(pp_rank1,
                       pp_rank2)] = sorted(list(set(layer_indices1).intersection(set(layer_indices2))))

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


def setup_comm_groups(from_topo: PipeModelDataParallelTopology, to_topo: PipeModelDataParallelTopology):
    comm_groups = {}
    comm_src = {}
    comm_dst_ranks = {}
    for pp_i, pp_j in itertools.product(range(from_topo.get_dim("pipe")), range(to_topo.get_dim("pipe"))):
        # create tensor reshard groups
        src_mp_size = from_topo.get_dim("model")
        dst_mp_size = to_topo.get_dim("model")

        for mp_j in range(dst_mp_size):
            _all_dst_ranks = [
                _x + 8 - to_topo.world_size() for _x in to_topo.filter_match(pipe=pp_j, model=mp_j)
            ]
            if src_mp_size > dst_mp_size:
                factor = src_mp_size // dst_mp_size
                mp_is = list(range(factor * mp_j, factor * (mp_j + 1)))
                _all_src_ranks = [from_topo.filter_match(model=mp_i, pipe=pp_i) for mp_i in mp_is]
            else:
                factor = dst_mp_size // src_mp_size
                _all_src_ranks = [from_topo.filter_match(model=mp_j // factor, pipe=pp_i)]
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
                        from_topo.get_coord(_src_rank).data,
                        from_topo.get_coord(_src_rank).model,
                    )
                    comm_dst_ranks[(dp_i, mp_i, pp_i, mp_j, pp_j)] = _dst_ranks
                    if _src_rank not in _dst_ranks:
                        _dst_ranks = [_src_rank] + _dst_ranks
                    assert len(set(_dst_ranks)) == len(_dst_ranks)
                    if len(_dst_ranks) > 1:
                        comm_groups[(dp_i, mp_i, pp_i, mp_j, pp_j)] = dist.new_group(_dst_ranks)
                    else:
                        comm_groups[(dp_i, mp_i, pp_i, mp_j, pp_j)] = None
                    comm_src[(dp_i, mp_i, pp_i, mp_j, pp_j)] = _src_rank

    return comm_groups, comm_src, comm_dst_ranks


def worker(
    idx: int,
    from_topo: PipeModelDataParallelTopology,
    to_topo: PipeModelDataParallelTopology,
    err_queue: mp.Queue,
):
    try:
        _worker(idx, from_topo, to_topo)
    except Exception as e:
        err_queue.put_nowait(1)
        raise


def _worker(
    idx: int,
    from_topo: PipeModelDataParallelTopology,
    to_topo: PipeModelDataParallelTopology,
):
    # The first from_topo.world_size() processes own the first model, which will send its parameters
    # the the last to_topo.world_size() processes that host the second model.
    torch.cuda.set_device(idx)
    dist.init_process_group(backend="nccl", init_method="tcp://localhost:7777", rank=idx, world_size=8)
    torch.cuda.set_device(idx)

    global_state_dict = create_global_param(param_size, n_layers)

    if idx < from_topo.world_size():
        from_shard_id = ModelShardID.from_parallelism_rank(ModelName("default", 0), from_topo, idx)
    if idx >= 8 - to_topo.world_size():
        to_shard_id = ModelShardID.from_parallelism_rank(ModelName("default", 0), to_topo,
                                                         idx - (8 - to_topo.world_size()))

    dst_mp_size = to_topo.get_dim("model")
    src_mp_size = from_topo.get_dim("model")

    if idx < from_topo.world_size():
        from_layer_idx_start = n_layers // from_topo.get_dim("pipe") * from_shard_id.pp_rank
        from_layer_idx_end = n_layers // from_topo.get_dim("pipe") * (from_shard_id.pp_rank + 1)

        _param_size_per_mp = param_size // from_topo.get_dim("model")
        state_dict = {
            i: x[from_shard_id.mp_rank * _param_size_per_mp:(from_shard_id.mp_rank + 1) * _param_size_per_mp]
            for i, x in global_state_dict.items() if i in range(from_layer_idx_start, from_layer_idx_end)
        }
    else:
        state_dict = {}

    # initialize groups for communication
    comm_groups, comm_src, comm_dst_ranks = setup_comm_groups(from_topo, to_topo)

    from_layer_mapping = get_layer_mapping(from_topo.get_dim("pipe"))
    to_layer_mapping = get_layer_mapping(to_topo.get_dim("pipe"))
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

    src_dp_size = from_topo.get_dim("data")
    dst_dp_size = to_topo.get_dim("data")

    # derive a global NCCL communication plan
    for (pp_i, pp_j), layer_indices in repart_strat.items():
        sub_sd = {
            i: (get_param_shape(i, from_topo.get_dim("model")), get_param_dtype(i))
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
                                ))
                        comm_plan.append(
                            SenderStep(
                                rank=src,
                                sender_mp_portion_id=sender_mp_portion_id,
                                receiver_mp_portion_id=receiver_mp_portion_id,
                                param_key=k,
                                group=group,
                                dst_ranks=dst_ranks,
                            ))

    for i, step in enumerate(comm_plan):
        if isinstance(step, ReceverStep):
            continue
        step: SenderStep
        required_by_nex_steps = False
        for nex_step in comm_plan[i + 1:]:
            if (isinstance(nex_step, SenderStep) and nex_step.rank == step.rank
                    and nex_step.param_key == step.param_key):
                required_by_nex_steps = True
                break
        step.remove = not required_by_nex_steps

    # if idx == 0:
    #     print(comm_plan)

    # input()

    comm_volume = 0
    new_state_dict = {}
    for step in comm_plan:
        portion_size = min(param_size // dst_mp_size, param_size // src_mp_size)
        if isinstance(step, ReceverStep) and step.rank == idx:
            if step.param_key not in new_state_dict:
                new_state_dict[step.param_key] = torch.zeros(param_size // to_topo.get_dim("model"),
                                                             dtype=step.param_dtype,
                                                             device="cuda")

            if step.rank == step.src:
                buf = state_dict[step.param_key][step.sender_mp_portion_id *
                                                 portion_size:(step.sender_mp_portion_id + 1) * portion_size]
            else:
                buf = torch.zeros(step.param_shape, dtype=step.param_dtype, device="cuda")
                comm_volume += buf.numel()
                torch.distributed.broadcast(buf, src=step.src, group=step.group)

            new_state_dict[step.param_key][step.receiver_mp_portion_id *
                                           portion_size:(step.receiver_mp_portion_id + 1) *
                                           portion_size] = buf
        if isinstance(step, SenderStep) and step.rank == idx:
            if step.group is not None:
                buf = state_dict[step.param_key][step.sender_mp_portion_id *
                                                 portion_size:(step.sender_mp_portion_id + 1) * portion_size]
                # comm_volume += buf.numel() * len(step.dst_ranks)
                torch.distributed.broadcast(buf, src=step.rank, group=step.group)
            if step.remove:
                state_dict.pop(step.param_key)

    assert len(state_dict) == 0, (idx, state_dict.keys(), new_state_dict.keys())

    if idx >= 8 - to_topo.world_size():
        assert len(new_state_dict) == n_layers // to_topo.get_dim("pipe"), len(new_state_dict)
        to_layer_idx_start = n_layers // to_topo.get_dim("pipe") * to_shard_id.pp_rank
        to_layer_idx_end = n_layers // to_topo.get_dim("pipe") * (to_shard_id.pp_rank + 1)
        for i in range(to_layer_idx_start, to_layer_idx_end):
            to_mp_rank = to_shard_id.mp_rank
            a = global_state_dict[i][param_size // to_topo.get_dim("model") * to_mp_rank:param_size //
                                     to_topo.get_dim("model") * (to_mp_rank + 1)]
            assert torch.allclose(new_state_dict[i], a), (new_state_dict[i], a, i)

    layer_indices_before = {}
    param_portion_before = {}
    for i in range(8):
        if i < from_topo.world_size():
            coord = from_topo.get_coord(i)
            layer_indices_before[i] = list(
                range(
                    coord.pipe * n_layers // from_topo.get_dim("pipe"),
                    (coord.pipe + 1) * n_layers // from_topo.get_dim("pipe"),
                ))
            param_portion_before[i] = (
                coord.model * param_size // from_topo.get_dim("model"),
                (coord.model + 1) * param_size // from_topo.get_dim("model"),
            )
        else:
            layer_indices_before[i] = []
            param_portion_before[i] = (0, 0)

    layer_indices_after = {}
    param_portion_after = {}
    for i in range(8):
        if i >= 8 - to_topo.world_size():
            coord = to_topo.get_coord(i - (8 - to_topo.world_size()))
            layer_indices_after[i] = list(
                range(
                    coord.pipe * n_layers // to_topo.get_dim("pipe"),
                    (coord.pipe + 1) * n_layers // to_topo.get_dim("pipe"),
                ))
            param_portion_after[i] = (
                coord.model * param_size // to_topo.get_dim("model"),
                (coord.model + 1) * param_size // to_topo.get_dim("model"),
            )
        else:
            layer_indices_after[i] = []
            param_portion_after[i] = (0, 0)

    est_comm_volume = 0
    for i in range(8):
        for layer_idx in layer_indices_after[i]:
            if layer_idx in layer_indices_before[i]:
                est_comm_volume += len(
                    set(range(param_portion_after[i][0], param_portion_after[i][1])).difference(
                        range(param_portion_before[i][0], param_portion_before[i][1])))
            else:
                est_comm_volume += param_portion_after[i][1] - param_portion_after[i][0]

    comm_volume = torch.tensor(comm_volume, dtype=torch.int32, device="cuda")
    torch.distributed.all_reduce(comm_volume)
    assert comm_volume == est_comm_volume, (comm_volume, est_comm_volume, from_topo, to_topo)
    torch.distributed.barrier()
    if idx == 0:
        print("success!")


def decompose_to_three_factors(n: int):
    factors = []
    for i in range(1, int(n**(1 / 2)) + 1):
        if n % i == 0:
            for j in range(i, int((n // i)**(1 / 2)) + 1):
                if (n // i) % j == 0:
                    k = (n // i) // j
                    factors += list(set(itertools.permutations([i, j, k])))
    return factors


if __name__ == "__main__":
    err_queue = mp.Queue(8)
    for a, b in [(6, 6), (4, 4)]:
        for x1, x2 in itertools.product(decompose_to_three_factors(a), decompose_to_three_factors(b)):
            # for x1, x2 in itertools.product([(1,3,2)], [(2,1,3)]):
            if not (x1[1] % x2[1] == 0 or x2[1] % x1[1] == 0):
                continue
            print(f"testing from {x1} to {x2}")
            from_topo = PipeModelDataParallelTopology(num_pp=x1[0], num_mp=x1[1], num_dp=x1[2])
            to_topo = PipeModelDataParallelTopology(num_pp=x2[0], num_mp=x2[1], num_dp=x2[2])
            procs = []
            for i in range(8):
                proc = mp.Process(target=worker, args=(i, from_topo, to_topo, err_queue))
                procs.append(proc)
                proc.start()
            for proc in procs:
                proc.join()
            for i in range(8):
                try:
                    err_code = err_queue.get_nowait()
                    print("error!!!!!!!!!!!!")
                    exit(0)
                except queue.Empty:
                    pass
