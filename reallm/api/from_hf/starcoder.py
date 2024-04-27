from typing import *

import transformers

from reallm.api.core.model_api import ReaLModelConfig
from reallm.base.constants import use_te_impl


def convert_config_starcoder(starcoder_config: transformers.GPTBigCodeConfig) -> ReaLModelConfig:
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


def _starcoder_key_mapping_fn(config: ReaLModelConfig) -> Dict[str, str]:

    assert not use_te_impl(), "starcoder does not support TE backend now"
    key_mappings = {
        "transformer.wte.weight": "transformer.embedding_layer.wte.weight",
        "transformer.wpe.weight": "transformer.embedding_layer.wpe.weight",
        "transformer.ln_f.weight": f"transformer.h.{config.n_layers-1}.ln_f.weight",
        "transformer.ln_f.bias": f"transformer.h.{config.n_layers-1}.ln_f.bias",
        "lm_head.weight": "head.weight",
    }
    for i in range(config.n_layers):
        key_mappings[f"transformer.h.{i}.ln_1.weight"] = f"transformer.h.{i}.attn.c_attn.ln.weight"
        key_mappings[f"transformer.h.{i}.ln_1.bias"] = f"transformer.h.{i}.attn.c_attn.ln.bias"
        key_mappings[f"transformer.h.{i}.attn.c_attn.weight"] = f"transformer.h.{i}.attn.c_attn.linear.weight"
        key_mappings[f"transformer.h.{i}.attn.c_attn.bias"] = f"transformer.h.{i}.attn.c_attn.linear.bias"
        key_mappings[f"transformer.h.{i}.attn.c_proj.weight"] = f"transformer.h.{i}.attn.c_proj.weight"
        key_mappings[f"transformer.h.{i}.attn.c_proj.bias"] = f"transformer.h.{i}.attn.c_proj.bias"
        key_mappings[f"transformer.h.{i}.ln_2.weight"] = f"transformer.h.{i}.mlp.ln.weight"
        key_mappings[f"transformer.h.{i}.ln_2.bias"] = f"transformer.h.{i}.mlp.ln.bias"
        key_mappings[f"transformer.h.{i}.mlp.c_fc.weight"] = f"transformer.h.{i}.mlp.c_fc.weight"
        key_mappings[f"transformer.h.{i}.mlp.c_fc.bias"] = f"transformer.h.{i}.mlp.c_fc.bias"
        key_mappings[f"transformer.h.{i}.mlp.c_proj.weight"] = f"transformer.h.{i}.mlp.c_proj.weight"
        key_mappings[f"transformer.h.{i}.mlp.c_proj.bias"] = f"transformer.h.{i}.mlp.c_proj.bias"
    return key_mappings


def state_dict_from_starcoder(state_dict, config):
    key_mappings = _starcoder_key_mapping_fn(config)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[key_mappings[k]] = v
    return new_state_dict


def state_dict_to_starcoder(state_dict, config):
    key_mappings = {v: k for k, v in _starcoder_key_mapping_fn(config).items()}
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[key_mappings[k]] = v
    return new_state_dict


"GPTBigCodeForCausalLM"
