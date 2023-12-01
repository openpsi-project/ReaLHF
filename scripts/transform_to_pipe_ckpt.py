import argparse
import os
import shutil

from deepspeed.runtime import utils as ds_utils
import torch
import torch.nn as nn

from impl.model.nn.flash_mqat.flash_mqat_base import *
from impl.model.utils.pipeline_module import LayerSpec

MODEL_CONFIG_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
]


def get_layer_specs(config: FlashMQATConfig):
    layer_specs = []
    # vocab pos embedding
    embedding_layer = LayerSpec(VocabPositionEmbedding, config, dtype=None, device=None)

    layer_specs.append(embedding_layer)

    for i in range(config.n_layers):
        flash_mqat_block = LayerSpec(
            FlashMQATBlock,
            config,
            layer_index=i,
            output_layernorm=(i == config.n_layers - 1),
            ckpt_attn=(i > 0 and config.ckpt_attn),
            ckpt_mlp=(i > 0 and config.ckpt_mlp),
            dtype=None,
            device=None,
        )
        layer_specs.append(flash_mqat_block)

    lm_head = LayerSpec(
        LanguageModelHead,
        config.hidden_dim,
        config.vocab_size,
        bias=False,
        device=None,
        dtype=None,
    )
    layer_specs.append(lm_head)

    return layer_specs


def count_layer_params(layer_specs):
    param_counts = [0] * len(layer_specs)
    for idx, layer in enumerate(layer_specs):
        if isinstance(layer, LayerSpec):
            l = layer.build()
            params = filter(lambda p: p.requires_grad, l.parameters())
            param_counts[idx] = sum(p.numel() for p in params)
        elif isinstance(layer, nn.Module):
            params = filter(lambda p: p.requires_grad, layer.parameters())
            param_counts[idx] = sum(p.numel() for p in params)
        print(f"count_layer_params build layer {layer.typename.__name__}")
    return param_counts


def partition_layers(layer_specs, num_stages, method="uniform"):
    # Each stage gets a simple uniform number of layers.
    parts = None
    if method == "uniform":
        num_layers = len(layer_specs)
        parts = ds_utils.partition_uniform(num_items=num_layers, num_parts=num_stages)
    elif method == "parameters":
        param_counts = count_layer_params(layer_specs)
        parts = ds_utils.partition_balanced(weights=param_counts, num_parts=num_stages)
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
                print(k)
                if k.startswith(f"{i}."):
                    stage_state_dict[k] = v
                    print(f"stage {stage} k={k}")
                    break
        stage_to_state_dict[stage] = stage_state_dict
    return stage_to_state_dict


def save_state_dict(state_dict, stage_index, shard_index, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    torch.save(
        state_dict,
        os.path.join(model_dir, f"pytorch_model-pp-{stage_index:02d}-mp-00-s-{shard_index:02d}.bin"),
    )
    print(
        f"saved {state_dict.keys()} to {model_dir}/pytorch_model-pp-{stage_index:02d}-mp-00-s-{shard_index:02d}.bin"
    )


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
            print(f"shard {i} key {keys[j]}")
        start += size
        shards.append(shard)
    return shards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/lustre/public/pretrained_model_weights/testOnly/llama-2-4l",
    )
    parser.add_argument("--model_type", type=str, default="llama")
    parser.add_argument("--num_stages", type=int, default=4)
    parser.add_argument("--num_shards", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        output_dir = f"{args.model_dir}_{args.num_stages}pp_{args.num_shards}s"
    else:
        output_dir = args.output_dir

    # TODO: load and process full statedict by shard for large model that can not fit into memory
    cfg, state_dict = getattr(FlashMQATForCausalLM,
                              f"config_and_param_from_{args.model_type}")(model_path=args.model_dir)
    layer_specs = get_layer_specs(cfg)
    state_dict = FlashMQATForCausalLM.map_to_pipe_state_dict(cfg, state_dict)
    print("loaded full state_dict")
    stage_to_layer_idx = partition_layers(layer_specs, num_stages=args.num_stages, method="parameters")
    stage_to_state_dict = split_state_dict_by_stage(state_dict, stage_to_layer_idx)
    for stage, state_dict in stage_to_state_dict.items():
        shards = split_state_dict_into_shards(state_dict, args.num_shards)
        print(f"stage {stage} state_dict keys: {state_dict.keys()}")
        for shard_index, shard in enumerate(shards):
            save_state_dict(shard, stage, shard_index, output_dir)
    copy_configs(args.model_dir, output_dir)


if __name__ == "__main__":
    main()
