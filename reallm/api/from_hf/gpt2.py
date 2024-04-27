from typing import *

import transformers

from reallm.api.core.model_api import ReaLModelConfig
from reallm.base.constants import use_te_impl


def gpt2_config_converter(gpt2config: transformers.GPT2Config) -> ReaLModelConfig:
    return ReaLModelConfig(
        n_layers=gpt2config.n_layer,
        n_kv_heads=gpt2config.n_head,
        attn_pdrop=gpt2config.attn_pdrop,
        embd_pdrop=gpt2config.embd_pdrop,
        layer_norm_epsilon=gpt2config.layer_norm_epsilon,
        hidden_dim=gpt2config.n_embd,
        head_dim=gpt2config.n_embd // gpt2config.n_head,
        intermediate_dim=gpt2config.n_inner if gpt2config.n_inner is not None else 4 * gpt2config.n_embd,
        n_positions=gpt2config.n_positions,
        resid_pdrop=gpt2config.resid_pdrop,
        vocab_size=gpt2config.vocab_size,
        activation_function=gpt2config.activation_function,
        scale_attn_by_inverse_layer_idx=False,
        fixed_abs_position_ids=True,
    )


def gpt2_state_dict_converter(state_dict: Dict, config: ReaLModelConfig) -> Dict:
    assert not use_te_impl(), "gpt does not support TE backend now"
    new_state_dict = {}
    replace_from = [
        "wte.weight",
        "wpe.weight",
        ".ln_1.",
        ".ln_2.",
        ".c_attn.weight",
        ".c_attn.bias",
        "ln_f.weight",
        "ln_f.bias",
        "lm_head",
    ]
    replace_to = [
        "embedding_layer.wte.weight",
        "embedding_layer.wpe.weight",
        ".attn.c_attn.ln.",
        ".mlp.ln.",
        ".c_attn.linear.weight",
        ".c_attn.linear.bias",
        f"h.{config.n_layers - 1}.ln_f.weight",
        f"h.{config.n_layers - 1}.ln_f.bias",
        "head",
    ]
    for k, v in state_dict.items():
        for rf, rt in zip(replace_from, replace_to):
            if rf in k:
                k = k.replace(rf, rt)
        if k.endswith(".attn.bias"):
            continue
        if k.endswith(".linear.weight") or k.endswith("proj.weight") or k.endswith("fc.weight"):
            v = v.transpose(0, 1)
        new_state_dict[k] = v
    new_state_dict["head.weight"] = state_dict["transformer.wte.weight"]
    return new_state_dict


def state_dict_to_gpt2(state_dict, config):
    replace_to = [
        "wte.weight",
        "wpe.weight",
        ".ln_1.",
        ".ln_2.",
        ".c_attn.weight",
        ".c_attn.bias",
        "ln_f.weight",
        "ln_f.bias",
        "lm_head",
    ]
    replace_from = [
        "embedding_layer.wte.weight",
        "embedding_layer.wpe.weight",
        ".attn.c_attn.ln.",
        ".mlp.ln.",
        ".c_attn.linear.weight",
        ".c_attn.linear.bias",
        f"h.{config.n_layers - 1}.ln_f.weight",
        f"h.{config.n_layers - 1}.ln_f.bias",
        "head",
    ]
    new_state_dict = {}
    for k, v in state_dict.items():
        for rf, rt in zip(replace_from, replace_to):
            if rf in k:
                k = k.replace(rf, rt)
        if (k.endswith(".linear.weight") or k.endswith("proj.weight") or k.endswith("fc.weight")
                or k.endswith("c_attn.weight")):
            v = v.transpose(0, 1)
        new_state_dict[k] = v
    return new_state_dict


"GPT2LMHeadModel"
