import os
import sys

sys.path.append("../")
from typing import Dict
import shutil

from deepspeed.runtime import utils as ds_utils
import torch
import torch.nn as nn
import transformers

from base.monitor import process_memory_mb
from impl.model.nn.flash_mqat import *
from impl.model.utils.pipeline_module import LayerSpec

MODEL_TYPE = "llama"
FULL_MODEL_DIR = "/home/meizy/models/Llama-2-4l"
NUM_PIPE_STAGES = 4
NUM_SHARDS = 3
PIPE_MODEL_DIR = f"/home/meizy/models/llama-2-4l_{NUM_PIPE_STAGES}pp_{NUM_SHARDS}s"
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


def llama_config(model_path: str):
    hf_config = transformers.AutoConfig.from_pretrained(os.path.join(model_path, "config.json"))
    config = FlashMQATConfig(
        n_layers=hf_config.num_hidden_layers,
        n_kv_heads=hf_config.num_key_value_heads,
        hidden_dim=hf_config.hidden_size,
        head_dim=hf_config.hidden_size // hf_config.num_attention_heads,
        intermediate_dim=hf_config.intermediate_size,
        vocab_size=hf_config.vocab_size,
        n_positions=hf_config.max_position_embeddings,
        embd_pdrop=0.0,
        attn_pdrop=hf_config.attention_dropout if hasattr(hf_config, "attention_dropout") else 0.1,
        layer_norm_epsilon=hf_config.rms_norm_eps,
        activation_function=hf_config.hidden_act,
        use_attention_bias=hf_config.attention_bias,
        scale_attn_by_inverse_layer_idx=False,
        layer_norm_type="rms",
        mlp_type="llama",
        apply_rotary=True,
        rotary_base=hf_config.rope_theta,
        rotary_interleaved=False,
        rotary_scaling=None if hf_config.rope_scaling is None else hf_config.rope_scaling["factor"],
        rotary_scaling_type=None if hf_config.rope_scaling is None else hf_config.rope_scaling["type"],
    )
    return config


def layer_specs_and_key_mappings(config: FlashMQATConfig, model_type: str):
    layer_specs = []
    # vocab pos embedding
    embedding_layer = LayerSpec(VocabPositionEmbedding, config, dtype=None, device=None)

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

    if model_type == "starcoder":
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
    elif model_type == "llama":
        layer_key_mappings = {
            "model.embed_tokens.": "0.wte.",
        }
        for i in range(config.n_layers):
            layer_key_mappings[f"model.layers.{i}."] = f"{i+1}."
            if i == config.n_layers - 1:
                layer_key_mappings[f"model.norm."] = f"{i+1}.ln_f."

        layer_key_mappings[".self_attn."] = ".attn."
        layer_key_mappings[".post_attention_layernorm."] = ".mlp.ln."
        layer_key_mappings[".input_layernorm."] = ".attn.c_attn.ln."
        layer_key_mappings["attn.o_proj."] = "attn.c_proj."
        layer_key_mappings["lm_head."] = f"{config.n_layers+1}."
    else:
        raise NotImplementedError("currently only support llama and starcoder")
    return layer_specs, layer_key_mappings


def llama_state_dict_transfrom(config: FlashMQATConfig, state_dict: Dict[str, torch.Tensor]):
    # merge k_proj, o_proj, q_proj into a single layer
    for i in range(config.n_layers):
        q_proj_w = state_dict[f"model.layers.{i}.self_attn.q_proj.weight"]
        k_proj_w = state_dict[f"model.layers.{i}.self_attn.k_proj.weight"]
        v_proj_w = state_dict[f"model.layers.{i}.self_attn.v_proj.weight"]
        w = torch.cat([q_proj_w, k_proj_w, v_proj_w], dim=0)
        state_dict[f"model.layers.{i}.self_attn.c_attn.linear.weight"] = w
        state_dict.pop(f"model.layers.{i}.self_attn.q_proj.weight")
        state_dict.pop(f"model.layers.{i}.self_attn.k_proj.weight")
        state_dict.pop(f"model.layers.{i}.self_attn.v_proj.weight")
    return state_dict


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


def update_state_dict_keys(state_dict, key_mappings, config, model_type):
    if model_type == "llama":
        state_dict = llama_state_dict_transfrom(config, state_dict)
    for old_key, new_key in key_mappings.items():
        new_state_dict = {}
        for k, v in state_dict.items():
            if old_key in k:
                k = k.replace(old_key, new_key)
            new_state_dict[k] = v
        state_dict = new_state_dict
    print(f"state dict keys = {list(state_dict.keys())}")
    return state_dict


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
    torch.save(state_dict,
               os.path.join(model_dir, f"pytorch_model-pp-{stage_index:02d}-mp-00-s-{shard_index:02d}.bin"))
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
    if MODEL_TYPE == "llama":
        cfg = llama_config(FULL_MODEL_DIR)
    elif MODEL_TYPE == "starcoder":
        cfg = flash_mqat_config(FULL_MODEL_DIR)
    else:
        raise NotImplementedError("currently only support llama and starcoder")
    layer_specs, key_mappings = layer_specs_and_key_mappings(cfg, MODEL_TYPE)
    # TODO: load and process full statedict by shard for large model that can not fit into memory
    state_dict, _ = load_full_ckpt(FULL_MODEL_DIR)
    print("loaded full state_dict")
    state_dict = update_state_dict_keys(state_dict, key_mappings, cfg, MODEL_TYPE)
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
