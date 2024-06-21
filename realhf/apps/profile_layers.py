import argparse
import itertools
import time

import realhf.base.testing as testing

BATCH_SIZE_RANGE = [1, 2, 4, 8, 16, 32, 64, 128]
SEQ_LEN_RANGE = [128, 256, 512]


def profile_layer_func(
    world_size,
    model_path,
    model_name,
    warm_up_rounds,
    profile_rounds,
    batch_size_range,
    seq_len_range,
    use_sequence_parallel=False,
    use_gradient_checkpointing=False,
):
    # FIXME: use_sequence_parallel=True and use_gradient_checkpointing=True will cause bugs
    import torch

    import realhf.base.constants as constants

    testing.init_global_constants(
        1, world_size, 1, sequence_parallel=False, gradient_checkpointing=False
    )
    device = torch.device("cuda")
    with constants.model_scope(testing.MODEL_NAME):
        from realhf.search_engine.layers import make_profile_layers

        profile_layers = make_profile_layers(device, model_path, model_name)

        st = time.monotonic_ns()
        for i in range(warm_up_rounds + profile_rounds):
            for bs, seq_len in itertools.product(batch_size_range, seq_len_range):
                profile_layers.fwd_gen(bs, seq_len)
                profile_layers.fwd_bwd_opt(bs, seq_len)

            if i < warm_up_rounds:
                profile_layers.reset_stats()
        profile_layers.make_dataframe_and_print()
        profile_layers.dump_stats(world_size)
        t = (time.monotonic_ns() - st) / int(1e9)
        print(f"profile world size {world_size} cost {t:4f} seconds")


if __name__ == "__main__":
    st = time.monotonic_ns()
    parser = argparse.ArgumentParser(prog="profile_layers")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
    )
    parser.add_argument("--expr_name", type=str, default="profile")
    parser.add_argument("--trial_name", type=str, default="profile")
    parser.add_argument("--model_name", type=str, default="Llama-2-70b")
    parser.add_argument("--warm_up_rounds", type=int, default=1)
    parser.add_argument("--profile_rounds", type=int, default=3)
    # parser.add_argument("--use_sequence_parallel", action="store_true")
    # parser.add_argument("--use_gradient_checkpointing", action="store_true")
    args = parser.parse_args()

    world_sizes = [1, 2, 4, 8]

    for world_size in world_sizes:
        testing.clear_name_resolve(args.expr_name, args.trial_name)
        mp = testing.LocalMultiProcessTest(
            world_size,
            profile_layer_func,
            world_size,
            args.model_path,
            args.model_name,
            args.warm_up_rounds,
            args.profile_rounds,
            BATCH_SIZE_RANGE,
            SEQ_LEN_RANGE,
            expr_name=args.expr_name,
            trial_name=args.trial_name,
        )
        mp.launch()

    t = (time.monotonic_ns() - st) / int(1e9)
    print(f"profile model {args.model_name} time cost {t:4f} seconds")
