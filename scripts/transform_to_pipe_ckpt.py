import argparse
import os
import shutil

from deepspeed.runtime import utils as ds_utils
import torch
import torch.nn as nn

from base.datapack import partition_balanced as true_partition_balanced
from impl.model.nn.flash_mqat.flash_mqat_base import *
from impl.model.nn.flash_mqat.flash_mqat_parallel import (make_causal_flash_mqat_pipe_module,
                                                          mp_partition_flash_mqat_state_dict)
from impl.model.parallelism.pipeline_parallel.pipeline_module import LayerSpec
from impl.model.utils.save_load import save_to_disk
import base.constants

MODEL_CONFIG_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
]


def count_layer_params(num_layers: int, state_dict_list: List[Dict[str, torch.Tensor]]) -> List[int]:
    param_counts = []
    for i in range(num_layers):
        cnt = 0
        for sd in state_dict_list:
            for k, v in sd.items():
                if k.startswith(f"{str(i)}.") and v is not None:
                    cnt += v.numel()
        param_counts.append(cnt)
    print(f"Count layer paramters: {param_counts}")
    return param_counts


def partition_layers(layer_specs, state_dict_list, num_stages, method="uniform"):
    # Each stage gets a simple uniform number of layers.
    parts = None
    num_layers = len(layer_specs)
    if method == "uniform":
        parts = ds_utils.partition_uniform(num_items=num_layers, num_parts=num_stages)
    elif method == "parameters":
        param_counts = count_layer_params(num_layers, state_dict_list)
        parts = ds_utils.partition_balanced(weights=param_counts, num_parts=num_stages)
    elif method == "parameters_balanced":
        param_counts = count_layer_params(num_layers, state_dict_list)
        import numpy as np

        param_counts = np.array(param_counts)
        parts = true_partition_balanced(nums=param_counts, k=num_stages)
    else:
        raise NotImplementedError(f"Partitioning method {method} not implemented.")

    stage_to_layer_idx = {}
    for stage in range(num_stages):
        start = parts[stage]
        stop = parts[stage + 1]
        print(f"stage={stage} layers={stop - start}")
        for idx, layer in enumerate(layer_specs[start:stop]):
            name = str(layer)
            if isinstance(layer, LayerSpec):
                name = layer.typename.__name__
            if isinstance(layer, nn.Module):
                name = layer.__class__.__name__
            else:
                try:
                    name = layer.__name__
                except AttributeError:
                    pass
            print(f"    {idx+start:2d}: {name}")
        stage_to_layer_idx[stage] = (start, stop)
    return stage_to_layer_idx


def split_state_dict_by_stage(state_dict, stage_to_layer_idx):
    stage_to_state_dict = {}
    for stage, (start, stop) in stage_to_layer_idx.items():
        stage_state_dict = {}
        for k, v in state_dict.items():
            for i in range(start, stop):
                if k.startswith(f"{i}."):
                    stage_state_dict[k] = v
                    # print(f"stage {stage} k={k}")
                    break
        stage_to_state_dict[stage] = stage_state_dict
    return stage_to_state_dict


def save_state_dict(state_dict, stage_index, mp_rank, shard_index, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    from impl.model import USE_TE_BACKEND
    suffix = "safetensors" if not USE_TE_BACKEND else "bin"
    save_type = "st" if not USE_TE_BACKEND else "pt"
    output_fn = f"model-pp-{stage_index:02d}-mp-{mp_rank:02d}-s-{shard_index:02d}.{suffix}"
    save_to_disk(state_dict,
                 model_dir,
                 output_fn=output_fn,
                 save_type=save_type,
                 n_shards=1,
                 no_shard_suffix=True)
    print(
        f"saved {len(state_dict.keys())} keys to {model_dir}/model-pp-{stage_index:02d}-mp-00-s-{shard_index:02d}.{suffix}"
    )
    print(f"saved {state_dict.keys()} to "
          f"{model_dir}/pytorch_model-pp-{stage_index:02d}-mp-{mp_rank:02d}-s-{shard_index:02d}.bin")


def fit_state_dict_to_critic(num_layers, state_dict):
    # modify last layer shape
    for k, v in state_dict.items():
        if k.startswith(f"{num_layers-1}."):
            print(f"last layer key {k} tensor shape {v.shape}")
            state_dict[k] = v[0].unsqueeze(0)
            print(f"critic head shape {state_dict[k].shape}")
    return state_dict


def copy_configs(src_model_dir, dst_model_dir):
    for file in MODEL_CONFIG_FILES:
        try:
            shutil.copy(os.path.join(src_model_dir, file), os.path.join(dst_model_dir, file))
            print(f"copied {file} from {src_model_dir} to {dst_model_dir}")
        except FileNotFoundError:
            print(f"{file} not exist in {src_model_dir} skipping.")


def split_state_dict_into_shards(state_dict, n_shards):
    if n_shards == 1:
        return [state_dict]

    keys = list(state_dict.keys())
    if len(keys) < n_shards:
        raise ValueError(f"state_dict has {len(keys)} keys, but n_shards={n_shards}")

    shard_size = len(keys) // n_shards
    extra = len(keys) % n_shards
    shard_size_list = [shard_size for _ in range(n_shards)]
    shard_size_list[-1] = shard_size + extra
    start, shards = 0, []
    for i, size in enumerate(shard_size_list):
        shard = {}
        for j in range(start, start + size):
            shard[keys[j]] = state_dict[keys[j]]
            # print(f"shard {i} key {keys[j]}")
        start += size
        shards.append(shard)
    return shards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",
                        type=str,
                        default="/lustre/public/pretrained_model_weights/Llama-2-7b-hf")
    parser.add_argument("--model_type", type=str, default="llama")
    parser.add_argument("--num_pp", type=int, default=1)
    parser.add_argument("--num_mp", type=int, default=4)
    parser.add_argument("--num_shards", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--partition_method", type=str, default="parameters_balanced")
    parser.add_argument(
        "--to_critic",
        action="store_true",
        help="transform actor model to critic model by changing the last layer, only for test purposes.",
    )
    args = parser.parse_args()
    if args.partition_method == "parameters":
        logger.warning(
            "method 'parameters' does not partition parameters to each stage evenly, "
            "preserving the option as default due to checkpoints already partitioned in this way."
            "Update to use 'parameters_balanced' option instead!"
            "Change the option both in model partition script and pipe/pipe+model model wrapper configs!")

    assert args.num_mp > 1 or args.num_pp > 1
    if args.output_dir is None:
        model_name = args.model_dir.rstrip("/").split("/")[-1]
        if args.num_mp == 1:
            output_dir = f"{model_name}_{args.num_pp}pp_{args.num_shards}s"
        elif args.num_pp == 1:
            output_dir = f"{model_name}_{args.num_mp}mp_{args.num_shards}s"
        else:
            output_dir = f"{model_name}_{args.num_pp}pp_{args.num_mp}mp_{args.num_shards}s"
        if args.to_critic:
            output_dir += "_critic"
        default_save_root = "/lustre/public/pretrained_model_weights/sharded"
        output_dir = os.path.join(default_save_root, output_dir)
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # TODO: load and process full statedict by shard for large model that can not fit into memory
    cfg = None
    base.constants.set_fake_mp_world_size(args.num_mp)
    base.constants.set_fake_mp_rank(0)
    cfg, state_dict = getattr(FlashMQATModel,
                              f"config_and_param_from_{args.model_type}")(model_path=args.model_dir)
    if args.num_mp > 1:
        state_dict_list = mp_partition_flash_mqat_state_dict(state_dict, cfg, args.num_mp)
    else:
        state_dict_list = [state_dict]

    if args.num_pp > 1:
        layer_specs = make_causal_flash_mqat_pipe_module(cfg,
                                                         partition_method=args.partition_method,
                                                         output_layer_specs_only=True)
        if args.to_critic:
            state_dict_list = [fit_state_dict_to_critic(len(layer_specs), sd) for sd in state_dict_list]
        print("loaded full state_dict")
        stage_to_layer_idx = partition_layers(layer_specs,
                                              state_dict_list,
                                              num_stages=args.num_pp,
                                              method=args.partition_method)
        stage_to_state_dict_list = [
            split_state_dict_by_stage(sd, stage_to_layer_idx) for sd in state_dict_list
        ]
        for mp_rank, stage_to_state_dict in enumerate(stage_to_state_dict_list):
            for stage, state_dict in stage_to_state_dict.items():
                shards = split_state_dict_into_shards(state_dict, args.num_shards)
                # print(f"stage {stage} state_dict keys: {state_dict.keys()}")
                for shard_index, shard in enumerate(shards):
                    save_state_dict(shard, stage, mp_rank, shard_index, output_dir)
    elif args.num_pp == 1:
        for mp_rank, state_dict in enumerate(state_dict_list):
            shards = split_state_dict_into_shards(state_dict, args.num_shards)
            # print(f"state_dict keys: {state_dict.keys()}")
            for shard_index, shard in enumerate(shards):
                save_state_dict(shard, 0, mp_rank, shard_index, output_dir)

    copy_configs(args.model_dir, output_dir)
    with open(os.path.join(output_dir, "flash_mqat_config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=4)


if __name__ == "__main__":
    main()
