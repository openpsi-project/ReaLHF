import argparse
import time

import torch
import torch.distributed
import torch.multiprocessing as mp

from tests.utils import clear_name_resolve, init_global_constants, MODEL_NAME, setup_barrier, setup_gpu


def main(rank, world_size, args):
    import base.constants
    with base.constants.model_scope(MODEL_NAME):
        device = setup_gpu(rank, world_size)
        init_global_constants(1, world_size, 1)

        # device = "cpu"

        from profiler.layers import make_profile_layers
        profile_layers = make_profile_layers(device, args.model_path, args.model_name,
                                             args.use_sequence_parallel and world_size > 1,
                                             args.use_gradient_checkpointing)

        st = time.monotonic_ns()
        for i in range(args.warm_up_rounds + args.profile_rounds):
            for bs, seq_len in zip(args.batch_size_list, args.seq_len_list):
                profile_layers.fwd_gen(bs, seq_len)
                profile_layers.fwd_bwd_opt(bs, seq_len)

            if i < args.warm_up_rounds:
                profile_layers.reset_stats()
        profile_layers.print_stats()
        profile_layers.sync_stats()
        profile_layers.dump_stats(world_size)
        t = (time.monotonic_ns() - st) / int(1e9)
        print(f"rank {rank} cost {t:4f} seconds")


if __name__ == "__main__":
    st = time.monotonic_ns()
    parser = argparse.ArgumentParser(prog="profile_compute")
    parser.add_argument("--model_path",
                        type=str,
                        default="/lustre/public/pretrained_model_weights/sharded/Llama-2-70b-hf_8pp_3s")
    parser.add_argument("--model_name", type=str, default="Llama-2-70b")
    parser.add_argument("--warm_up_rounds", type=int, default=2)
    parser.add_argument("--profile_rounds", type=int, default=3)
    parser.add_argument("--batch_size_list",
                        type=str,
                        default="16,32,64",
                        help="batch size list divided by comma, example: 32,64,128")
    parser.add_argument("--seq_len_list",
                        type=str,
                        default="128,128,256",
                        help="sequence length list divided by comma, "
                        "length should be the same as batch size list"
                        "each entry corresponds to the sequence length for each batch size")
    parser.add_argument("--use_sequence_parallel", action="store_true")
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    args = parser.parse_args()

    args.batch_size_list = [int(x) for x in args.batch_size_list.split(",")]
    args.seq_len_list = [int(x) for x in args.seq_len_list.split(",")]
    mp_ranks = [1, 2, 4, 8]
    # mp_ranks = [1]

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
