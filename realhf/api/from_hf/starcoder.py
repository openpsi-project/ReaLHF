from typing import *

import torch
import transformers

from realhf.api.core.model_api import (
    ReaLModelConfig,
    register_hf_family,
    register_hf_path,
)
from realhf.base.constants import use_te_impl


def convert_config_starcoder(
    starcoder_config: transformers.GPTBigCodeConfig,
) -> ReaLModelConfig:
    return ReaLModelConfig(
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


def to_starcoder_config(
    config: ReaLModelConfig,
) -> transformers.GPTBigCodeConfig:
    return transformers.GPTBigCodeConfig(
        vocab_size=config.vocab_size,
        n_positions=config.n_positions,
        n_embd=config.hidden_dim,
        n_layer=config.n_layers,
        n_head=config.hidden_dim // config.head_dim,
        n_inner=config.intermediate_dim,
        resid_pdrop=config.resid_pdrop,
        embd_pdrop=config.embd_pdrop,
        attn_pdrop=config.attn_pdrop,
        layer_norm_epsilon=config.layer_norm_epsilon,
        multi_query=(config.n_kv_heads == 1),
        activation_function="gelu",
    )


def _starcoder_key_mapping_fn(config: ReaLModelConfig) -> Dict[str, str]:
    assert not use_te_impl(), "starcoder does not support TE backend now"
    key_mappings = {
        "transformer.wte.weight": "0.wte.weight",
        "transformer.wpe.weight": "0.wpe.weight",
        "transformer.ln_f.weight": f"{config.n_layers}.ln_f.weight",
        "transformer.ln_f.bias": f"{config.n_layers}.ln_f.bias",
        "lm_head.weight": f"{config.n_layers+1}.weight",
    }
    for i in range(config.n_layers):
        key_mappings[f"transformer.h.{i}.ln_1.weight"] = (
            f"{i + 1}.attn.c_attn.ln.weight"
        )
        key_mappings[f"transformer.h.{i}.ln_1.bias"] = (
            f"{i + 1}.attn.c_attn.ln.bias"
        )
        key_mappings[f"transformer.h.{i}.attn.c_attn.weight"] = (
            f"{i + 1}.attn.c_attn.linear.weight"
        )
        key_mappings[f"transformer.h.{i}.attn.c_attn.bias"] = (
            f"{i + 1}.attn.c_attn.linear.bias"
        )
        key_mappings[f"transformer.h.{i}.attn.c_proj.weight"] = (
            f"{i + 1}.attn.c_proj.weight"
        )
        key_mappings[f"transformer.h.{i}.attn.c_proj.bias"] = (
            f"{i + 1}.attn.c_proj.bias"
        )
        key_mappings[f"transformer.h.{i}.ln_2.weight"] = (
            f"{i + 1}.mlp.ln.weight"
        )
        key_mappings[f"transformer.h.{i}.ln_2.bias"] = f"{i + 1}.mlp.ln.bias"
        key_mappings[f"transformer.h.{i}.mlp.c_fc.weight"] = (
            f"{i + 1}.mlp.c_fc.weight"
        )
        key_mappings[f"transformer.h.{i}.mlp.c_fc.bias"] = (
            f"{i + 1}.mlp.c_fc.bias"
        )
        key_mappings[f"transformer.h.{i}.mlp.c_proj.weight"] = (
            f"{i + 1}.mlp.c_proj.weight"
        )
        key_mappings[f"transformer.h.{i}.mlp.c_proj.bias"] = (
            f"{i + 1}.mlp.c_proj.bias"
        )
    return key_mappings


def state_dict_from_starcoder(state_dict, config):
    key_mappings = _starcoder_key_mapping_fn(config)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[key_mappings[k]] = v

    # split qkv weight and bias
    qdim = config.hidden_dim
    kvdim = config.head_dim * config.n_kv_heads
    for k in ["weight", "bias"]:
        for i in range(config.n_layers):
            if f"{i + 1}.attn.c_attn.linear.{k}" not in new_state_dict:
                continue
            w = new_state_dict[f"{i + 1}.attn.c_attn.linear.{k}"]
            assert w.shape[0] == qdim + kvdim * 2
            new_state_dict[f"{i + 1}.attn.c_attn.q_attn.{k}"] = w[:qdim]
            new_state_dict[f"{i + 1}.attn.c_attn.k_attn.{k}"] = w[
                qdim : qdim + kvdim
            ]
            new_state_dict[f"{i + 1}.attn.c_attn.v_attn.{k}"] = w[
                qdim + kvdim :
            ]
            new_state_dict.pop(f"{i + 1}.attn.c_attn.linear.{k}")
    return new_state_dict


def state_dict_to_starcoder(state_dict, config):
    # merge qkv weight and bias
    qdim = config.hidden_dim
    kvdim = config.head_dim * config.n_kv_heads
    layer_indices = list(set(int(k.split(".")[0]) for k in state_dict.keys()))
    for k in ["weight", "bias"]:
        for i in layer_indices:
            if i == 0 or i == config.n_layers + 1:
                continue
            qw = state_dict[f"{i}.attn.c_attn.q_attn.{k}"]
            kw = state_dict[f"{i}.attn.c_attn.k_attn.{k}"]
            vw = state_dict[f"{i}.attn.c_attn.v_attn.{k}"]
            w = torch.cat([qw, kw, vw], dim=0)
            assert w.shape[0] == qdim + kvdim * 2
            state_dict[f"{i}.attn.c_attn.linear.{k}"] = w
            state_dict.pop(f"{i}.attn.c_attn.q_attn.{k}")
            state_dict.pop(f"{i}.attn.c_attn.k_attn.{k}")
            state_dict.pop(f"{i}.attn.c_attn.v_attn.{k}")

    key_mappings = {v: k for k, v in _starcoder_key_mapping_fn(config).items()}
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[key_mappings[k]] = v
    return new_state_dict


def starcoder_embedding_layer_names(config: ReaLModelConfig) -> List[str]:
    return ["transformer.wte.weight", "transformer.wpe.weight"]


def starcoder_tblock_param_name(
    config: ReaLModelConfig, layer_idx: int
) -> List[str]:
    names = [
        f"transformer.h.{layer_idx}.ln_1.weight",
        f"transformer.h.{layer_idx}.ln_1.bias",
        f"transformer.h.{layer_idx}.attn.c_attn.weight",
        f"transformer.h.{layer_idx}.attn.c_attn.bias",
        f"transformer.h.{layer_idx}.attn.c_proj.weight",
        f"transformer.h.{layer_idx}.attn.c_proj.bias",
        f"transformer.h.{layer_idx}.ln_2.weight",
        f"transformer.h.{layer_idx}.ln_2.bias",
        f"transformer.h.{layer_idx}.mlp.c_fc.weight",
        f"transformer.h.{layer_idx}.mlp.c_fc.bias",
        f"transformer.h.{layer_idx}.mlp.c_proj.weight",
        f"transformer.h.{layer_idx}.mlp.c_proj.bias",
    ]
    if layer_idx == config.n_layers - 1:
        names += ["transformer.ln_f.weight", "transformer.ln_f.bias"]
    return names


def starcoder_output_head_names(config: ReaLModelConfig) -> List[str]:
    return ["lm_head.weight"]


register_hf_family(
    name="starcoder",
    hf_cls_name="GPTBigCodeForCausalLM",
    config_from_hf_converter=convert_config_starcoder,
    config_to_hf_converter=to_starcoder_config,
    sd_from_hf_converter=state_dict_from_starcoder,
    sd_to_hf_converter=state_dict_to_starcoder,
    embedding_param_names=starcoder_embedding_layer_names,
    tblock_param_names=starcoder_tblock_param_name,
    head_param_names=starcoder_output_head_names,
)
