from typing import *
import subprocess
import os
import argparse


def get_partitions(model_size: int) -> List[Tuple[int, int]]:
    # mp * pp
    if model_size == 7:
        return [(1, 2), (2, 1), (2, 2), (1, 3), (1, 4)]
    elif model_size == 13:
        return [(1, 2), (2, 1), (2, 2), (1, 4), (4, 1), (2, 4)]
    elif model_size == 34:
        return [(2, 2), (2, 4), (4, 2), (2, 3), (2, 1)]
    elif model_size == 70:
        return [(2, 2), (2, 4), (4, 2), (2, 3), (4, 3), (4, 1)]


def main(args):
    for model_size in args.model_size:
        partitions = get_partitions(model_size)
        for mp_size, pp_size in partitions:
            assert pp_size > 1 or mp_size > 1
            if model_size in [7, 13, 70]:
                model_dir_prefix = f"Llama-2-{model_size}b-hf"
                model_type = "llama"
            else:
                model_dir_prefix = "CodeLlama-34b-hf"
                model_type = "codellama"
            ckpt_base_dir = "/lustre/public/pretrained_model_weights/"
            model_path = os.path.join(ckpt_base_dir, model_dir_prefix)
            partitioned_ckpt_base_dir = "/lustre/public/pretrained_model_weights/sharded_new/"
            output_path = os.path.join(
                partitioned_ckpt_base_dir, f"{model_dir_prefix}_{pp_size}pp_{mp_size}mp"
            )
            if os.path.exists(output_path):
                print(f"{model_size}B model partition pp{pp_size}*mp{mp_size} already exists. Skipping.")
                continue
            cmd = [
                "python3",
                "-m",
                "scripts.transform_to_pipe_ckpt",
                "--model_dir",
                model_path,
                "--output_dir",
                output_path,
                "--model_type",
                model_type,
                "--num_pp",
                pp_size,
                "--num_mp",
                mp_size,
            ]
            cmd = [str(x) for x in cmd]
            os.system(" ".join(cmd))
            # print(" ".join(cmd))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", "-x", type=int, nargs="+", choices=[7, 13, 34, 70], required=True)
    args = parser.parse_args()

    main(args)
