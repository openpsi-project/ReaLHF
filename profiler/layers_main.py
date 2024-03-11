import argparse
import time

import torch
import torch.distributed
import torch.multiprocessing as mp

from tests.utils import clear_name_resolve, init_global_constants, setup_barrier, setup_gpu


def main(rank, world_size, args):
    device = setup_gpu(rank, world_size)
    init_global_constants(1, world_size, 1)

    from profiler.layers import make_profile_layers
    profile_layers = make_profile_layers(device, args.model_path, args.model_name)

    st = time.monotonic_ns()
    for i in range(args.warm_up_rounds + args.profile_rounds):
        profile_layers.fwd_gen(args.batch_size, args.seq_len)
        profile_layers.fwd_bwd_opt(args.batch_size, args.seq_len)

        if i < args.warm_up_rounds:
            profile_layers.reset_stats()
    profile_layers.print_stats()
    profile_layers.sync_stats()
    profile_layers.dump_stats(world_size, args.batch_size, args.seq_len)
    t = (time.monotonic_ns() - st) / int(1e9)
    print(f"rank {rank} cost {t:4f} seconds")


if __name__ == "__main__":
    st = time.monotonic_ns()
    parser = argparse.ArgumentParser(prog="profile_compute")
    parser.add_argument("--model_path",
                        type=str,
                        default="/lustre/public/pretrained_model_weights/sharded/Llama-2-7b-hf_4pp_3s")
    parser.add_argument("--model_name", type=str, default="Llama-2-7b")
    parser.add_argument("--warm_up_rounds", type=int, default=2)
    parser.add_argument("--profile_rounds", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=128)
    args = parser.parse_args()
    mp_ranks = [1, 2, 4, 8]

    for mp_rank in mp_ranks:
        clear_name_resolve()
        setup_barrier(mp_rank)

        processes = [mp.Process(target=main, args=(rank, mp_rank, args)) for rank in range(mp_rank)]
        for p in processes:
            p.start()

        for p in processes:
            p.join()
    t = (time.monotonic_ns() - st) / int(1e9)
    print("total cost {:4f} seconds".format(t))
