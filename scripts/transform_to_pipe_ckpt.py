from typing import Dict, List, Optional, Tuple, Union
import logging
import os
import sys

sys.path.append("../")
import shutil

from deepspeed.runtime import utils as ds_utils
import torch
import torch.nn as nn
import transformers

from base.monitor import process_memory_mb
from impl.model.backend.ds_pipe_engine import (LayerSpec, PipeDataParallelTopology, PipelineModule,
                                               ProcessTopology)
from impl.model.nn.flash_mqat import *
from impl.model.utils.data import tensor_data_list_to_tuple, tuple_to_tensor_data_list

FULL_MODEL_DIR = "/lustre/meizy/backup_zy/model_saves/four_layers_starcoder"
PIPE_MODEL_DIR = "/lustre/meizy/backup_zy/model_saves/pipe_4l_starcoder"
NUM_PIPE_STAGES = 4
TEST_EXPR_NAME = "test"
TEST_TRIAL_NAME = "test"
TEST_MODEL_NAME = "default"
MODEL_CONFIG_FILES = [
    "config.json", "generation_config.json", "tokenizer_config.json", "vocab.json", "merges.txt",
    "special_tokens_map.json", "tokenizer.json"
]


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


def layer_specs_and_key_mappings(config: FlashMQATConfig):
    layer_specs = []
    # vocab pos embedding
    embedding_layer = LayerSpec(VocabPositionEmbedding,
                                config.vocab_size,
                                config.n_positions,
                                config.hidden_dim,
                                config.embd_pdrop,
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

    layer_key_mappings = {
        "transformer.wte.": "0.wte.",
        "transformer.wpe.": "0.wpe.",
    }
    for i in range(config.n_layers):
        layer_key_mappings[f"transformer.h.{i}.attn.c_proj."] = f"{i+1}.attn.c_proj."
        layer_key_mappings[f"transformer.h.{i}.mlp.c_proj."] = f"{i+1}.mlp.c_proj."
        layer_key_mappings[f"transformer.h.{i}.mlp.c_fc."] = f"{i+1}.mlp.c_fc."
        layer_key_mappings[f"transformer.h.{i}.ln_1."] = f"{i+1}.attn.c_attn.ln."
        layer_key_mappings[f"transformer.h.{i}.ln_2."] = f"{i+1}.mlp.ln."
        layer_key_mappings[f"transformer.h.{i}.attn.c_attn."] = f"{i+1}.attn.c_attn.linear."
        if i == config.n_layers - 1:
            layer_key_mappings[f"transformer.ln_f."] = f"{i+1}.ln_f."
    layer_key_mappings["lm_head."] = f"{config.n_layers+1}."
    return layer_specs, layer_key_mappings


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


def update_state_dict_keys(state_dict, key_mappings):
    new_state_dict = {}
    for k, v in state_dict.items():
        for old_key, new_key in key_mappings.items():
            if k.startswith(old_key):
                new_key = k.replace(old_key, new_key)
                new_state_dict[new_key] = v
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


def save_stage_state_dict(stage_to_state_dict, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    for stage, state_dict in stage_to_state_dict.items():
        torch.save(state_dict, os.path.join(model_dir, f"pytorch_model-stage-{stage}.bin"))
        print(f"saved {state_dict.keys()} to {model_dir}/pytorch_model-stage-{stage}.bin")


def copy_configs(src_model_dir, dst_model_dir):
    for file in MODEL_CONFIG_FILES:
        shutil.copy(os.path.join(src_model_dir, file), os.path.join(dst_model_dir, file))
        print(f"copied {file} from {src_model_dir} to {dst_model_dir}")


def main():
    cfg = flash_mqat_config(FULL_MODEL_DIR)
    layer_specs, key_mappings = layer_specs_and_key_mappings(cfg)
    state_dict = torch.load(os.path.join(FULL_MODEL_DIR, "pytorch_model.bin"))
    print("loaded full state_dict")
    state_dict = update_state_dict_keys(state_dict, key_mappings)
    stage_to_layer_idx = partition_layers(layer_specs, num_stages=NUM_PIPE_STAGES, method="parameters")
    stage_to_state_dict = split_state_dict_by_stage(state_dict, stage_to_layer_idx)
    save_stage_state_dict(stage_to_state_dict, PIPE_MODEL_DIR)
    copy_configs(FULL_MODEL_DIR, PIPE_MODEL_DIR)


if __name__ == "__main__":
    main()
