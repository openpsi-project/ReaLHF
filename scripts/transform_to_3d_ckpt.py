import os
import sys

sys.path.append("../")
from typing import List, Optional, Tuple
import dataclasses
import shutil

from deepspeed.runtime import utils as ds_utils
import torch
import torch.nn as nn
import transformers

from base.monitor import process_memory_mb
from impl.model.nn.flash_mqat import *
from impl.model.utils.pipeline_module import LayerSpec

FULL_MODEL_DIR = "/lustre/meizy/models/starcoder_4l"
NUM_PIPE_STAGES = 4
NUM_TP = 2
NUM_SHARDS = 1
PIPE_MODEL_DIR = f"/lustre/meizy/models/3d/starcoder_4l_{NUM_PIPE_STAGES}pp_{NUM_TP}tp_{NUM_SHARDS}s"
TEST_EXPR_NAME = "test"
TEST_TRIAL_NAME = "test"
TEST_MODEL_NAME = "default"
MODEL_CONFIG_FILES = [
    "config.json", "generation_config.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "tokenizer.json"
]


@dataclasses.dataclass
class TransformKeyMapping:
    before_key: str
    after_keys: List[str]
    before_shape: Tuple
    after_shapes: List[Tuple]


def flash_mqat_config(model_path: str):
    starcoder_config = transformers.AutoConfig.from_pretrained(os.path.join(model_path, "config.json"))
    config = FlashMQATConfig(
        n_layers=starcoder_config.n_layer,
        n_kv_heads=1,
        attn_pdrop=starcoder_config.attn_pdrop,
        embd_pdrop=starcoder_config.embd_pdrop,
        layer_norm_epsilon=starcoder_config.layer_norm_epsilon,
        hidden_dim=starcoder_config.n_embd,
        head_dim=starcoder_config.n_embd // starcoder_config.n_head,
        intermediate_dim=starcoder_config.n_inner,
        n_positions=starcoder_config.n_positions,
        resid_pdrop=starcoder_config.resid_pdrop,
        vocab_size=starcoder_config.vocab_size,
    )
    return config


def layer_specs_and_transform_mappings(config: FlashMQATConfig):
    layer_specs = []
    # vocab pos embedding
    embedding_layer = LayerSpec(VocabPositionEmbedding,
                                config.vocab_size,
                                config.n_positions,
                                config.hidden_dim,
                                config.embd_pdrop,
                                config.fixed_abs_position_ids,
                                dtype=None,
                                device=None)

    layer_specs.append(embedding_layer)

    for i in range(config.n_layers):
        flash_mqat_block = LayerSpec(FlashMQATBlock,
                                     config,
                                     layer_index=i,
                                     output_layernorm=(i == config.n_layers - 1),
                                     ckpt_attn=(i > 0 and config.ckpt_attn),
                                     ckpt_mlp=(i > 0 and config.ckpt_mlp),
                                     dtype=None,
                                     device=None)
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

    mappings = []

    mappings.append(
        TransformKeyMapping(
            before_key="transformer.wte.weight",
            after_keys=["0.wte.weight"],
            before_shape=(config.vocab_size, config.hidden_dim),
            after_shapes=[(config.vocab_size // NUM_TP, config.hidden_dim)],
        ))
    mappings.append(
        TransformKeyMapping(
            before_key="transformer.wpe.weight",
            after_keys=["0.wpe.weight"],
            before_shape=(config.n_positions, config.hidden_dim),
            after_shapes=[(config.n_positions // NUM_TP, config.hidden_dim)],
        ))

    n_heads = config.hidden_dim // config.head_dim
    for i in range(config.n_layers):
        mappings.append(
            TransformKeyMapping(before_key=f"transformer.h.{i}.attn.c_attn.weight",
                                after_keys=[f"{i+1}.attn.q_attn.weight", f"{i+1}.attn.kv_attn.weight"],
                                before_shape=(config.head_dim * (n_heads + 2 * config.n_kv_heads),
                                              config.hidden_dim),
                                after_shapes=[
                                    (config.head_dim * n_heads // NUM_TP, config.hidden_dim),
                                    (2 * config.head_dim * config.n_kv_heads, config.hidden_dim),
                                ]))
        mappings.append(
            TransformKeyMapping(before_key=f"transformer.h.{i}.attn.c_attn.bias",
                                after_keys=[f"{i+1}.attn.q_attn.bias", f"{i+1}.attn.kv_attn.bias"],
                                before_shape=(config.head_dim * (n_heads + 2 * config.n_kv_heads),),
                                after_shapes=[
                                    (config.head_dim * n_heads // NUM_TP,),
                                    (2 * config.head_dim * config.n_kv_heads,),
                                ]))
        mappings.append(
            TransformKeyMapping(
                before_key=f"transformer.h.{i}.attn.c_proj.weight",
                after_keys=[f"{i+1}.attn.c_proj.weight"],
                before_shape=(config.hidden_dim, config.hidden_dim),
                after_shapes=[(config.hidden_dim, config.hidden_dim // NUM_TP)],
            ))
        mappings.append(
            TransformKeyMapping(
                before_key=f"transformer.h.{i}.attn.c_proj.bias",
                after_keys=[f"{i+1}.attn.c_proj.bias"],
                before_shape=(config.hidden_dim,),
                after_shapes=[(config.hidden_dim,)],
            ))
        mappings.append(
            TransformKeyMapping(
                before_key=f"transformer.h.{i}.ln_1.",
                after_keys=[f"{i+1}.attn.ln."],
                before_shape=None,
                after_shapes=[None],
            ))
        mappings.append(
            TransformKeyMapping(
                before_key=f"transformer.h.{i}.mlp.c_fc.weight",
                after_keys=[f"{i+1}.mlp.c_fc.weight"],
                before_shape=(config.intermediate_dim, config.hidden_dim),
                after_shapes=[(config.intermediate_dim // NUM_TP, config.hidden_dim)],
            ))
        mappings.append(
            TransformKeyMapping(
                before_key=f"transformer.h.{i}.mlp.c_fc.bias",
                after_keys=[f"{i+1}.mlp.c_fc.bias"],
                before_shape=(config.intermediate_dim,),
                after_shapes=[(config.intermediate_dim // NUM_TP,)],
            ))
        mappings.append(
            TransformKeyMapping(
                before_key=f"transformer.h.{i}.mlp.c_proj.weight",
                after_keys=[f"{i+1}.mlp.c_proj.weight"],
                before_shape=(config.hidden_dim, config.intermediate_dim),
                after_shapes=[(config.hidden_dim, config.intermediate_dim // NUM_TP)],
            ))
        mappings.append(
            TransformKeyMapping(
                before_key=f"transformer.h.{i}.mlp.c_proj.bias",
                after_keys=[f"{i+1}.mlp.c_proj.bias"],
                before_shape=(config.hidden_dim,),
                after_shapes=[(config.hidden_dim,)],
            ))
        mappings.append(
            TransformKeyMapping(
                before_key=f"transformer.h.{i}.ln_2.",
                after_keys=[f"{i+1}.mlp.ln."],
                before_shape=None,
                after_shapes=[None],
            ))
        if i == config.n_layers - 1:
            mappings.append(
                TransformKeyMapping(
                    before_key=f"transformer.ln_f.",
                    after_keys=[f"{i+1}.ln_f."],
                    before_shape=None,
                    after_shapes=[None],
                ))
    mappings.append(
        TransformKeyMapping(
            before_key="lm_head.",
            after_keys=[f"{config.n_layers+1}."],
            before_shape=None,
            after_shapes=[None],
        ))

    return layer_specs, mappings


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
    if method == 'uniform':
        num_layers = len(layer_specs)
        parts = ds_utils.partition_uniform(num_items=num_layers, num_parts=num_stages)
    elif method == 'parameters':
        param_counts = count_layer_params(layer_specs)
        parts = ds_utils.partition_balanced(weights=param_counts, num_parts=num_stages)
    else:
        raise NotImplementedError(f'Partitioning method {method} not implemented.')

    stage_to_layer_idx = {}
    for stage in range(num_stages):
        start = parts[stage]
        stop = parts[stage + 1]
        print(f'stage={stage} layers={stop - start}')
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
            print(f'    {idx+start:2d}: {name}')
        stage_to_layer_idx[stage] = (start, stop)
    return stage_to_layer_idx


def partition_tensor(tensor: torch.Tensor, before_shape: Tuple[int], after_shape: Tuple[int]):
    assert tensor.shape == before_shape
    # partition a tensor of before_shape into a list of tensors of after_shape
    before_dim = len(before_shape)
    after_dim = len(after_shape)
    assert before_dim == after_dim
    assert all([before_shape[i] % after_shape[i] == 0 for i in range(before_dim)])
    partition_size = [before_shape[i] // after_shape[i] for i in range(before_dim)]
    # only one dimension is to be partitioned
    num_partitions, partition_dim = None, None
    for k, p in enumerate(partition_size):
        if p > 1:
            num_partitions = p
            partition_dim = k
            break
    assert num_partitions is not None
    partitions = torch.split(tensor, num_partitions, dim=partition_dim)
    assert num_partitions == NUM_TP
    return partitions


def update_state_dict(state_dict, mappings: List[TransformKeyMapping]):
    new_state_dict = {}
    for k, v in state_dict.items():
        k: str
        v: torch.Tensor
        for m in mappings:
            if k.startswith(m.before_key) and m.before_shape is None:
                assert len(m.after_keys) == 1
                new_state_dict[m.after_keys[0]] = v
                break
            elif k == m.before_key:
                for after_key, after_shape in zip(m.after_keys, m.after_shapes):
                    new_state_dict[after_key] = partition_tensor(v, m.before_shape, after_shape)
                break
            else:
                new_state_dict[k] = v
    return new_state_dict


def split_state_dict_by_stage(state_dict, stage_to_layer_idx):
    stage_to_state_dict = {}
    for stage, (start, stop) in stage_to_layer_idx.items():
        stage_state_dict = {}
        for k, v in state_dict.items():
            for i in range(start, stop):
                if k.startswith(f"{i}."):
                    stage_state_dict[k] = v
                    print(f"stage {stage} k={k}")
                    break
        stage_to_state_dict[stage] = stage_state_dict
    return stage_to_state_dict


def save_state_dict(state_dict, stage_index, shard_index, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    for tp_rank in range(NUM_TP):
        state_dict_of_rank = {k: v[tp_rank] if isinstance(v, list) else v for k, v in state_dict.items()}
        torch.save(
            state_dict_of_rank,
            os.path.join(model_dir,
                         f"pytorch_model-pp-{stage_index:02d}-tp-{tp_rank}-s-{shard_index:02d}.bin"))
        print(f"saved {state_dict.keys()} to "
              f"{model_dir}/pytorch_model-pp-{stage_index:02d}-tp-{tp_rank}-s-{shard_index:02d}.bin")


def copy_configs(src_model_dir, dst_model_dir):
    for file in MODEL_CONFIG_FILES:
        shutil.copy(os.path.join(src_model_dir, file), os.path.join(dst_model_dir, file))
        print(f"copied {file} from {src_model_dir} to {dst_model_dir}")


def load_full_ckpt(path):
    single_file_path = os.path.join(path, "pytorch_model.bin")
    state_dict = None
    n_shards = 0
    if os.path.exists(single_file_path):
        state_dict = torch.load(single_file_path)
        n_shards += 1
    else:
        state_dict = {}
        for file in os.listdir(path):
            if file.endswith(".bin"):
                state_dict.update(torch.load(os.path.join(path, file)))
                n_shards += 1
    return state_dict, n_shards


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
    cfg = flash_mqat_config(FULL_MODEL_DIR)
    layer_specs, mappings = layer_specs_and_transform_mappings(cfg)
    # TODO: load and process full statedict by shard for large model that can not fit into memory
    state_dict, _ = load_full_ckpt(FULL_MODEL_DIR)
    print("loaded full state_dict")
    state_dict = update_state_dict(state_dict, mappings)
    stage_to_layer_idx = partition_layers(layer_specs, num_stages=NUM_PIPE_STAGES, method="parameters")
    stage_to_state_dict = split_state_dict_by_stage(state_dict, stage_to_layer_idx)
    for stage, state_dict in stage_to_state_dict.items():
        shards = split_state_dict_into_shards(state_dict, NUM_SHARDS)
        print(f"stage {stage} state_dict keys: {state_dict.keys()}")
        for shard_index, shard in enumerate(shards):
            save_state_dict(shard, stage, shard_index, PIPE_MODEL_DIR)
    copy_configs(FULL_MODEL_DIR, PIPE_MODEL_DIR)


if __name__ == "__main__":
    main()
