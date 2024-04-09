import torch
import torch.distributed as dist
import multiprocessing as mp
import time
import argparse


def main(rank, world_size, bcast_src, dst, n_iterations=10):
    torch.cuda.set_device(rank % 8)
    print("initialize process group...")
    dist.init_process_group("nccl", init_method="tcp://10.119.12.145:7777", rank=rank, world_size=world_size)
    print(f"initialize process group...rank {rank} finish")
    torch.cuda.set_device(rank % 8)

    assert bcast_src not in dst
    g = dist.new_group([bcast_src] + dst, backend="nccl")

    volume = 3.5e9
    tensor = torch.zeros(int(volume), dtype=torch.float16, device="cuda")
    if rank == bcast_src:
        tensor[:] = 1
    for _ in range(n_iterations):
        if rank == bcast_src or rank in dst:

            torch.cuda.synchronize()
            tik = time.perf_counter_ns()
            dist.broadcast(tensor, src=bcast_src, group=g)
            torch.cuda.synchronize()
            print(f"rank {rank} broadcast time: {(time.perf_counter_ns() - tik) / 1e6} ms, "
                  f"estimated bandwidth: {volume * 2 * 8/ (time.perf_counter_ns() - tik) * 1e9 / 1024**3} Gb/s")

    dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--node_idx", "-i", type=int, default=0)
    parser.add_argument("--n_nodes", "-n", type=int, default=1)
    args = parser.parse_args()

    procs = []
    for i in range(8):
        p = mp.Process(target=main, args=(args.node_idx * 8 + i, args.n_nodes * 8, 0, [8,10,12,14]))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
