from collections import defaultdict
from typing import *
import dataclasses
import itertools
import json
import time

import torch
import torch.distributed

from reallm.api.core import model_api, system_api
from reallm.api.core.config import ModelName
from reallm.base.testing import get_llama7b_real_config
from reallm.impl.model.nn.real_llm_api import (_keys_from_layer_indices, _param_size_from_keys,
                                               ReparallelizeReceiverStep, ReparallelizeSenderStep)
from reallm.impl.model.nn.real_llm_base import (real_model_embed_param_count, real_model_head_param_count,
                                                real_model_tblock_param_count, ReaLModelConfig)
from reallm.impl.model.nn.real_llm_parallel import (get_real_model_param_shape, partition_pipeline_layers,
                                                    pipeline_repartition_strategy)
import reallm.api.core.system_api
import reallm.base.gpu_utils as gpu_utils
import reallm.base.topology
import reallm.base.topology as topology


def _filter_match_mwids(
    model_name: ModelName,
    topo: topology.PipeModelDataParallelTopology,
    msid2mwid: Dict[system_api.ModelShardID, int],
    **conditions,
) -> List[int]:
    if len(conditions) == 0:
        mwids_this_model = [
            msid2mwid[system_api.ModelShardID.from_parallelism_rank(model_name, topo, j)]
            for j in range(topo.world_size())
        ]
    else:
        mwids_this_model = [
            msid2mwid[system_api.ModelShardID.from_parallelism_rank(model_name, topo, j)]
            for j in topo.filter_match(**conditions)
        ]
    mwids_this_model = sorted(mwids_this_model)
    assert len(set(mwids_this_model)) == len(mwids_this_model)
    return list(mwids_this_model)


def _group_mwids_by_node(ranks: List[int]) -> Dict[int, List[int]]:
    node2ranks = defaultdict(list)
    for r in ranks:
        node2ranks[r // 8].append(r)
    return {k: node2ranks[k] for k in sorted(node2ranks.keys())}


def _squeeze_mwids_by_node(ranks: List[int]) -> List[int]:
    node2ranks = _group_mwids_by_node(ranks)
    return [ranks[0] for ranks in node2ranks.values()]


def _assign_src_to_dsts(node2srcs: Dict[int, List[int]], node2dsts: Dict[int,
                                                                         List[int]]) -> Dict[int, List[int]]:
    """Assign nodes with a greedy algorithm.

    All ranks in the values of node2srcs have the data required by all dst ranks.
    Source ranks can be assigned to zero or multiple destination ranks.
    All destination ranks must be assigned to exactly one src rank.

    Args:
        node2srcs (Dict[int, List[int]]): Node index -> source ranks.
        node2dsts (Dict[int, List[int]]): Node index -> destination ranks.

    Returns:
        Dict[int, List[int]]: src rank -> dst ranks.
    """
    # First, assign all destination ranks to source nodes.
    # If a destination node is also a source node, assign it to itself.
    # Otherwise, find the node with the minimum workload for load balancing.
    dst_src_nodes = {}
    for dst_node in node2dsts.keys():
        if dst_node in node2srcs:
            # dst node is also src node
            dst_src_nodes[dst_node] = dst_node
    src_node_workloads = {k: int(k in dst_src_nodes) for k in node2srcs}
    assert sum(src_node_workloads.values()) == len(dst_src_nodes)
    for dst_node in node2dsts.keys():
        if dst_node not in node2srcs:
            # find a source node with the minimum workload
            src_node = min(src_node_workloads, key=src_node_workloads.get)
            dst_src_nodes[dst_node] = src_node
    assert all(dst_node in dst_src_nodes for dst_node in node2dsts)

    # Revert the key-value of the dict.
    src_dst_nodes = defaultdict(list)
    for dst_node, src_node in dst_src_nodes.items():
        src_dst_nodes[src_node].append(dst_node)

    # Next, find an appropriate source rank on each source node.
    # If the source rank is also the destination rank, assign it to the destination rank.
    # Otherwise, assign the first one to the destination ranks.
    assignment = {}
    for src_node, dst_nodes in src_dst_nodes.items():
        assigned = False
        src_ranks = node2srcs[src_node]
        dst_ranks = sum([node2dsts[dst_node] for dst_node in dst_nodes], start=[])
        for s in src_ranks:
            if s in dst_ranks:
                assignment[s] = dst_ranks
                assigned = True
                break
        if not assigned:
            assignment[src_ranks[0]] = dst_ranks
    assert len(set(assignment.keys())) == len(assignment.keys())
    assert len(set(sum(assignment.values(), []))) == len(sum(assignment.values(), []))

    return assignment


def _create_param_realloc_groups(
    from_topo: reallm.base.topology.PipeModelDataParallelTopology,
    to_topo: reallm.base.topology.PipeModelDataParallelTopology,
    src: ModelName,
    dst: ModelName,
    msid2mwid: Dict[system_api.ModelShardID, int],
    param_realloc_groups: Dict[gpu_utils.ParamReallocPair, torch.distributed.ProcessGroup],
    param_realloc_src_ranks: Dict[gpu_utils.ParamReallocPair, int],
    param_realloc_dst_ranks: Dict[gpu_utils.ParamReallocPair, List[int]],
):
    mwid2msid: Dict[int, Dict[ModelName, system_api.ModelShardID]] = defaultdict(dict)
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
            # All GPUs in _src_ranks have the data required by (pp_j, mp_j)
            for _src_ranks in _all_src_ranks:
                # NOTE: inter-node communication cost is significantly larger than intra-node communication cost.
                # We only select one sender per host/node to prevent multiple senders occupying the same network bandwidth.
                # This is not the optimal solution for intra-node communication
                # because there may exist a source rank that is also dst rank,
                # but we forcely select the first source rank on each node here.
                assignment = _assign_src_to_dsts(
                    _group_mwids_by_node(_src_ranks),
                    _group_mwids_by_node(_all_dst_ranks),
                )
                _idle_src_ranks = [r for r in _src_ranks if r not in assignment]
                for _src_rank in _idle_src_ranks:
                    dp_i, mp_i = (
                        from_topo.get_coord(mwid2msid[_src_rank][src].parallelism_rank).data,
                        from_topo.get_coord(mwid2msid[_src_rank][src].parallelism_rank).model,
                    )
                    key = gpu_utils.ParamReallocPair(
                        src=src,
                        src_dp_rank=dp_i,
                        src_mp_rank=mp_i,
                        src_pp_rank=pp_i,
                        dst=dst,
                        dst_mp_rank=mp_j,
                        dst_pp_rank=pp_j,
                    )
                    param_realloc_dst_ranks[key] = []
                    param_realloc_groups[key] = None
                    param_realloc_src_ranks[key] = _src_rank
                for _src_rank, _dst_ranks in assignment.items():
                    dp_i, mp_i = (
                        from_topo.get_coord(mwid2msid[_src_rank][src].parallelism_rank).data,
                        from_topo.get_coord(mwid2msid[_src_rank][src].parallelism_rank).model,
                    )
                    key = gpu_utils.ParamReallocPair(
                        src=src,
                        src_dp_rank=dp_i,
                        src_mp_rank=mp_i,
                        src_pp_rank=pp_i,
                        dst=dst,
                        dst_mp_rank=mp_j,
                        dst_pp_rank=pp_j,
                    )
                    param_realloc_dst_ranks[key] = _dst_ranks
                    if _src_rank not in _dst_ranks:
                        _dst_ranks = [_src_rank] + _dst_ranks
                    assert len(set(_dst_ranks)) == len(_dst_ranks)
                    if len(_dst_ranks) > 1:
                        param_realloc_groups[key] = 1
                    else:
                        param_realloc_groups[key] = None
                    param_realloc_src_ranks[key] = _src_rank


def _derive_reparallelize_comm_plan(
    from_model_name: ModelName,
    to_model_name: ModelName,
    from_topo: reallm.base.topology.PipeModelDataParallelTopology,
    to_topo: reallm.base.topology.PipeModelDataParallelTopology,
    from_model_config: ReaLModelConfig,
    to_model_config: ReaLModelConfig,
    pg_info: gpu_utils.NCCLProcessGroupInfo,
    dtype: Optional[torch.dtype] = torch.float16,
) -> List[ReparallelizeReceiverStep | ReparallelizeSenderStep]:
    src_mp_size = from_topo.get_dim("model")
    dst_mp_size = to_topo.get_dim("model")
    assert src_mp_size % dst_mp_size == 0 or dst_mp_size % src_mp_size == 0
    for k, v in dataclasses.asdict(to_model_config).items():
        if k not in [
                "is_critic",
                "sequence_parallel",
                "gradient_accumulation_fusion",
        ] and v != getattr(from_model_config, k):
            raise ValueError(
                f"Can't load a checkpoint with different config (key `{k}`, "
                f"value in checkpoint is `{v}`, current value is `{getattr(from_model_config, k)}`).")
    if (from_model_config.n_kv_heads % src_mp_size == 0) != (from_model_config.n_kv_heads % dst_mp_size == 0):
        raise ValueError("Whether to partition kv heads should remain the same.")

    from_layer_mapping = partition_pipeline_layers(
        from_model_config,
        from_topo.get_dim("pipe"),
        real_model_embed_param_count,
        real_model_tblock_param_count,
        real_model_head_param_count,
    )
    from_layer_mapping = {k: list(range(v[0], v[1])) for k, v in from_layer_mapping.items()}
    to_layer_mapping = partition_pipeline_layers(
        to_model_config,
        to_topo.get_dim("pipe"),
        real_model_embed_param_count,
        real_model_tblock_param_count,
        real_model_head_param_count,
    )
    to_layer_mapping = {k: list(range(v[0], v[1])) for k, v in to_layer_mapping.items()}
    repart_strat = pipeline_repartition_strategy(from_layer_mapping, to_layer_mapping)

    comm_plan = []

    src_dp_size = from_topo.get_dim("data")

    # derive a global NCCL communication plan
    for (pp_i, pp_j), layer_indices in repart_strat.items():
        if len(layer_indices) == 0:
            continue

        for mp_i in range(src_mp_size):
            if dst_mp_size > src_mp_size:
                factor = dst_mp_size // src_mp_size
                mp_js = [i + factor * mp_i for i in range(factor)]
                receiver_mp_portion_id = 0
            else:
                factor = src_mp_size // dst_mp_size
                mp_js = [mp_i // factor]
                receiver_mp_portion_id = mp_i % factor
            for sender_mp_portion_id, mp_j in enumerate(mp_js):

                for dp_i in range(src_dp_size):
                    key = gpu_utils.ParamReallocPair(
                        src=from_model_name,
                        src_dp_rank=dp_i,
                        src_mp_rank=mp_i,
                        src_pp_rank=pp_i,
                        dst=to_model_name,
                        dst_mp_rank=mp_j,
                        dst_pp_rank=pp_j,
                    )
                    src = pg_info.param_realloc_src_ranks[key]
                    group = pg_info.param_realloc_groups[key]
                    dst_ranks = pg_info.param_realloc_dst_ranks[key]

                    param_intervals = param_keys = receiver_param_intervals = None
                    max_param_interval_size = max_receiver_param_interval_size = -1
                    param_intervals_cpu = receiver_param_intervals_cpu = None
                    param_size = -1
                    param_keys = _keys_from_layer_indices(from_model_config, layer_indices)
                    param_size = _param_size_from_keys(
                        config=from_model_config,
                        src_mp_size=src_mp_size,
                        sd_keys=param_keys,
                        src2dst_tp_size=max(dst_mp_size // src_mp_size, 1),
                        src2dst_tp_rank=sender_mp_portion_id,
                    )

                    for dst_rank in dst_ranks:
                        comm_plan.append(
                            ReparallelizeReceiverStep(
                                rank=dst_rank,
                                sender_mp_portion_id=sender_mp_portion_id,
                                receiver_mp_portion_id=receiver_mp_portion_id,
                                param_keys=param_keys,
                                sender_param_intervals_cpu=param_intervals_cpu,
                                sender_param_intervals=param_intervals,
                                sender_max_interval_size=max_param_interval_size,
                                receiver_param_intervals_cpu=receiver_param_intervals_cpu,
                                receiver_param_intervals=receiver_param_intervals,
                                receiver_max_interval_size=max_receiver_param_interval_size,
                                param_size=param_size,
                                param_dtype=dtype,
                                src=src,
                                dst_ranks=dst_ranks,
                                group=group,
                            ))
                    comm_plan.append(
                        ReparallelizeSenderStep(
                            rank=src,
                            sender_mp_portion_id=sender_mp_portion_id,
                            receiver_mp_portion_id=receiver_mp_portion_id,
                            param_keys=param_keys,
                            param_intervals=param_intervals,
                            param_intervals_cpu=param_intervals_cpu,
                            max_interval_size=max_param_interval_size,
                            param_size=param_size,
                            group=group,
                            dst_ranks=dst_ranks,
                        ))

    return comm_plan


def bcast_cost(param_size: float, bw: float, src: int, dsts: List[int], n_nodes_per_gpu=8):
    src_node = src // n_nodes_per_gpu
    dst_nodes = [dst // n_nodes_per_gpu for dst in dsts]
    if src_node == dst_nodes[0] and all(dst_node == dst_nodes[0] for dst_node in dst_nodes):
        return param_size * 2 * 8 / (1800 * 1024**3)
        # return 0.0
    else:
        # param size is in float16, bw is in Gbps
        return param_size * 2 * 8 / (bw * 1024**3) * len(set(dst_nodes))


def compute_cost(
    world_size: int,
    from_model_name: ModelName,
    to_model_name: ModelName,
    from_topo: reallm.base.topology.PipeModelDataParallelTopology,
    to_topo: reallm.base.topology.PipeModelDataParallelTopology,
    model_config: ReaLModelConfig,
    bw: float,  # Gbps
    set_interval_cost: float,
) -> int:

    param_realloc_groups = {}
    param_realloc_src_ranks = {}
    param_realloc_dst_ranks = {}
    msid2mwid = {}
    for i in range(from_topo.world_size()):
        msid2mwid[system_api.ModelShardID.from_parallelism_rank(from_model_name, from_topo, i)] = i
    for i in range(to_topo.world_size()):
        msid2mwid[system_api.ModelShardID.from_parallelism_rank(to_model_name, to_topo,
                                                                i)] = (i + world_size - to_topo.world_size())
    _create_param_realloc_groups(
        from_topo,
        to_topo,
        from_model_name,
        to_model_name,
        msid2mwid,
        param_realloc_groups,
        param_realloc_src_ranks,
        param_realloc_dst_ranks,
    )
    pg_info = gpu_utils.NCCLProcessGroupInfo(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        param_realloc_groups,
        param_realloc_src_ranks,
        param_realloc_dst_ranks,
    )
    comm_plan = _derive_reparallelize_comm_plan(
        from_model_name,
        to_model_name,
        from_topo,
        to_topo,
        model_config,
        model_config,
        pg_info,
    )

    # Run boradcast!
    max_cost = max_comm_volume = max_bcast_cnt = 0
    total_local_comm_volume = 0
    total_remote_comm_volume = 0
    assert world_size % 8 == 0
    node_send_v = {(i, 0): 0 for i in range(world_size // 8)}
    node_send_v.update({(i, 1): 0 for i in range(world_size // 8)})
    node_recv_v = {(i, 0): 0 for i in range(world_size // 8)}
    node_recv_v.update({(i, 1): 0 for i in range(world_size // 8)})
    for _rank in range(world_size):
        cost = comm_volume = bcast_cnt = 0
        for step in comm_plan:
            if isinstance(step, ReparallelizeReceiverStep) and step.rank == _rank:
                if step.rank != step.src:
                    cost += bcast_cost(step.param_size, bw, step.src, step.dst_ranks)
                    comm_volume += step.param_size
                    bcast_cnt += 1
                    src_node = step.rank // 8
                    self_node = step.src // 8
                    cur_node_receivers = [rank for rank in step.dst_ranks if rank // 8 == src_node]
                    if src_node != self_node and step.rank == cur_node_receivers[0]:
                        node_recv_v[self_node] += step.param_size
                cost += set_interval_cost
            if isinstance(step, ReparallelizeSenderStep) and step.rank == _rank:
                if step.group is not None:
                    cost += bcast_cost(step.param_size, bw, step.rank, step.dst_ranks)
                    src_node = step.rank // 8
                    dst_nodes = list(sorted(set([dst // 8 for dst in step.dst_ranks])))
                    if len(dst_nodes) == 1 and dst_nodes[0] == src_node:
                        total_local_comm_volume += step.param_size
                    else:
                        for n in [src_node] + dst_nodes[:-1]:
                            node_send_v[n] += step.param_size
                        total_remote_comm_volume += step.param_size * sum(
                            [src_node != dst_node for dst_node in dst_nodes])
                    # bcast_cnt += 1
        max_cost = max(max_cost, cost)
        max_comm_volume = max(max_comm_volume, comm_volume)
        max_bcast_cnt = max(max_bcast_cnt, bcast_cnt)

    return (
        max_cost,
        total_local_comm_volume,
        total_remote_comm_volume,
        max_bcast_cnt,
        max_comm_volume,
        node_send_v,
        node_recv_v,
    )


def main():
    from_model_name = ModelName("actor", 0)
    to_model_name = ModelName("actor", 1)
    with open("memshift_cost.jsonl", "r") as f:
        inaccurate_cnt = total_cnt = 0
        for ff in f.readlines():
            data = json.loads(ff)
            from_pp_mp_dp = (
                data["from_pp_size"],
                data["from_mp_size"],
                data["from_dp_size"],
            )
            to_pp_mp_dp = (data["to_pp_size"], data["to_mp_size"], data["to_dp_size"])
            world_size = data["world_size"]
            profile_res = data["mem_shift_time_ns"] / 1e9

            from_topo = reallm.base.topology.PipeModelDataParallelTopology(*from_pp_mp_dp)
            to_topo = reallm.base.topology.PipeModelDataParallelTopology(*to_pp_mp_dp)
            assert world_size >= from_topo.world_size()
            assert world_size >= to_topo.world_size()

            mconfig = get_llama7b_real_config()

            tik = time.perf_counter()
            cost = compute_cost(
                world_size,
                from_model_name,
                to_model_name,
                from_topo,
                to_topo,
                mconfig,
                bw=200.0,
                set_interval_cost=0.03,
            )
            total_cnt += 1
            if abs(cost - profile_res) > 0.1:
                print(
                    f"direction: {from_pp_mp_dp} -> {to_pp_mp_dp}, cost {cost:.4f}, "
                    f"profiled {profile_res:.4f}, abs diff {(cost-profile_res):.4f}, "
                    f"rel diff {((cost-profile_res) / profile_res):.4f} ",
                    time.perf_counter() - tik,
                )
                inaccurate_cnt += 1
    print(
        "Inaccurate count:",
        inaccurate_cnt,
        "Total count:",
        total_cnt,
        "inaccurate ratio",
        inaccurate_cnt / total_cnt,
    )


def decompose_to_three_factors(n: int):
    factors = []
    for i in range(1, int(n**(1 / 2)) + 1):
        if n % i == 0:
            for j in range(i, int((n // i)**(1 / 2)) + 1):
                if (n // i) % j == 0:
                    k = (n // i) // j
                    factors += list(set(itertools.permutations([i, j, k])))
    return factors


def get_table():
    from_model_name = ModelName("actor", 0)
    to_model_name = ModelName("actor", 1)
    for a, b in [(64, 56), (56, 64)]:
        all_configs = list(itertools.product(decompose_to_three_factors(a), decompose_to_three_factors(b)))
        all_configs = list(filter(lambda x: x[0][1] <= 8 and x[1][1] <= 8, all_configs))
        all_configs = list(filter(lambda x: x[0][2] <= 8 and x[1][2] <= 8, all_configs))
        all_configs = list(filter(
            lambda x: x[0][1] in [1, 2, 4, 8] and x[1][1] in [1, 2, 4, 8],
            all_configs,
        ))
        all_configs = list(filter(lambda x: x[0][0] <= 16 and x[1][0] <= 16, all_configs))
        all_configs = list(filter(lambda x: x[0][1] % x[1][1] == 0 or x[1][1] % x[0][1] == 0, all_configs))
        for config_id, (from_pp_mp_dp, to_pp_mp_dp) in enumerate(all_configs):
            world_size = max(a, b)

            from_topo = reallm.base.topology.PipeModelDataParallelTopology(*from_pp_mp_dp)
            to_topo = reallm.base.topology.PipeModelDataParallelTopology(*to_pp_mp_dp)
            assert world_size >= from_topo.world_size()
            assert world_size >= to_topo.world_size()

            mconfig = get_llama7b_real_config()

            tik = time.perf_counter()
            cost = compute_cost(
                world_size,
                from_model_name,
                to_model_name,
                from_topo,
                to_topo,
                mconfig,
                bw=200.0,
                set_interval_cost=0.03,
            )
            print(
                f"direction: {from_pp_mp_dp} -> {to_pp_mp_dp}, cost {cost:.4f}",
                time.perf_counter() - tik,
            )
            cost = compute_cost(
                world_size,
                to_model_name,
                from_model_name,
                to_topo,
                from_topo,
                mconfig,
                bw=200.0,
                set_interval_cost=0.03,
            )
            print(
                f"direction: {to_pp_mp_dp} -> {from_pp_mp_dp}, cost {cost:.4f}",
                time.perf_counter() - tik,
            )


if __name__ == "__main__":
    # main()
    get_table()
