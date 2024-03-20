from typing import *
import torch.distributed as dist
import torch
import random
import multiprocessing as mp
import dataclasses
import os
import time


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


def random_partition(n, parts):
    partition = []
    for _ in range(parts - 1):
        rand_part = random.randint(0, n - sum(partition) - (parts - len(partition) - 1))
        partition.append(rand_part)
    partition.append(n - sum(partition))
    random.shuffle(partition)
    return partition


tensor_size = 10
sender_ranks = [[0], [1], [2], [3]]
sender_partition = random_partition(tensor_size, len(sender_ranks))
receiver_ranks = [[2, 3], [0, 1]]
receiver_partition = random_partition(tensor_size, len(receiver_ranks))


def worker(idx: int):
    torch.cuda.set_device(idx)
    dist.init_process_group(backend="nccl", init_method="tcp://localhost:7777", rank=idx, world_size=8)
    torch.cuda.set_device(idx)
    groups = {}
    group_sources = {}
    for dp_i, s_ranks in enumerate(sender_ranks):
        for dp_j, r_ranks in enumerate(receiver_ranks):
            src = s_ranks[0]
            if src not in r_ranks:
                _ranks = [src] + r_ranks
            else:
                _ranks = r_ranks
            groups[(dp_i, dp_j)] = dist.new_group(_ranks)
            group_sources[(dp_i, dp_j)] = s_ranks[0]

    tensor_storage = torch.randn(tensor_size).cuda()
    recv_buf = torch.zeros_like(tensor_storage)
    
    sender_mapping= {}
    offset = 0
    for i, rank in enumerate(sender_ranks):
        sender_mapping[i] = list(range(offset, offset + sender_partition[i]))
        offset += sender_partition[i]
    receiver_mapping = {}
    offset = 0
    for i, rank in enumerate(receiver_ranks):
        receiver_mapping[i] = list(range(offset, offset + receiver_partition[i]))
        offset += receiver_partition[i]
    repart_strat = repartition_strategy(sender_mapping, receiver_mapping)
    if idx == 0:
        print(sender_partition, receiver_partition, repart_strat)
    
    dist.barrier()
    for (dp_i, dp_j), comm_slots in repart_strat.items():
        if len(comm_slots) > 0:
            source = sender_ranks[dp_i][0]
            assert source == group_sources[(dp_i, dp_j)]
            r_ranks = receiver_ranks[dp_j]
            if idx == source:
                # print(f"calling broadcast send from {idx} to {r_ranks} with slots {comm_slots}")
                dist.broadcast(tensor_storage[comm_slots], src=idx, group=groups[(dp_i, dp_j)])
            elif idx in r_ranks:
                # print(f">>> calling broadcast recv from {source} to {r_ranks} with slots {comm_slots}")
                if idx == source:
                    recv_buf[comm_slots] = tensor_storage[comm_slots]
                else:
                    dist.broadcast(recv_buf[comm_slots], src=group_sources[(dp_i, dp_j)], group=groups[(dp_i, dp_j)])
    dist.barrier()
    
    dist.barrier()
    tik = time.time()
    for (dp_i, dp_j), comm_slots in repart_strat.items():
        if len(comm_slots) > 0:
            source = sender_ranks[dp_i][0]
            assert source == group_sources[(dp_i, dp_j)]
            r_ranks = receiver_ranks[dp_j]
            if idx == source:
                # print(f"calling broadcast send from {idx} to {r_ranks} with slots {comm_slots}")
                dist.broadcast(tensor_storage[comm_slots], src=idx, group=groups[(dp_i, dp_j)])
            elif idx in r_ranks:
                # print(f">>> calling broadcast recv from {source} to {r_ranks} with slots {comm_slots}")
                if idx == source:
                    recv_buf[comm_slots] = tensor_storage[comm_slots]
                else:
                    dist.broadcast(recv_buf[comm_slots], src=group_sources[(dp_i, dp_j)], group=groups[(dp_i, dp_j)])

    print(f"rank {idx} took {time.time() - tik} seconds")
    for dp_i, s_ranks in enumerate(sender_ranks):
        if s_ranks[0] == idx:
            print(f"sender partition {dp_i} has tensor {tensor_storage[sender_mapping[dp_i]]}")
    for dp_j, r_ranks in enumerate(receiver_ranks):
        if idx in r_ranks:
            print(f"receiver partition {dp_j} has tensor {tensor_storage[receiver_mapping[dp_j]]}")
    time.sleep(2)
    print("success!")

if __name__ == "__main__":
    procs = []
    for i in range(8):
        proc = mp.Process(target=worker, args=(i,))
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()
