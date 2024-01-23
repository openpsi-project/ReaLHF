from typing import Dict, Optional, Tuple
import os

import torch
import transformers

from impl.model.nn.flash_mqat.flash_mqat_api import FlashMQATModel
from impl.model.nn.flash_mqat.flash_mqat_base import FlashMQATConfig
import base.constants

try:
    import transformer_engine.pytorch as te

    TE_ENABLED = True
except ImportError:
    TE_ENABLED = False
USE_TE_BACKEND = TE_ENABLED and os.getenv("FLASH_MQAT_USE_TE") == "1"

################################ StarCoder Begin ################################


def convert_config_starcoder(starcoder_config: transformers.GPTBigCodeConfig) -> FlashMQATConfig:
    return FlashMQATConfig(
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


def _starcoder_key_mapping_fn(config: FlashMQATConfig) -> Dict[str, str]:
    assert not USE_TE_BACKEND, "starcoder does not support TE backend now"
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


FlashMQATModel.register_hf_model(
    "starcoder", convert_config_starcoder, state_dict_from_starcoder, state_dict_to_starcoder
)

################################ StarCoder End ################################

################################ GPT2 Begin ################################


def gpt2_config_converter(gpt2config: transformers.GPT2Config) -> FlashMQATConfig:
    return FlashMQATConfig(
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


def gpt2_state_dict_converter(state_dict: Dict, config: FlashMQATConfig) -> Dict:
    assert not USE_TE_BACKEND, "gpt does not support TE backend now"
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
        if (
            k.endswith(".linear.weight")
            or k.endswith("proj.weight")
            or k.endswith("fc.weight")
            or k.endswith("c_attn.weight")
        ):
            v = v.transpose(0, 1)
        new_state_dict[k] = v
    return new_state_dict


FlashMQATModel.register_hf_model(
    "gpt2",
    gpt2_config_converter,
    gpt2_state_dict_converter,
    state_dict_to_gpt2,
    force_load_from_hf_pretrained=True,
)

################################ GPT2 End ################################

################################ LLaMa Begin ################################


def convert_config_llama(hf_config: transformers.LlamaConfig) -> FlashMQATConfig:
    return FlashMQATConfig(
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


def convert_state_dict_llama(state_dict: Dict, config: FlashMQATConfig) -> Dict:
    # merge k_proj, o_proj, q_proj into a single layer
    for i in range(config.n_layers):
        q_proj_w = state_dict[f"model.layers.{i}.self_attn.q_proj.weight"]
        k_proj_w = state_dict[f"model.layers.{i}.self_attn.k_proj.weight"]
        v_proj_w = state_dict[f"model.layers.{i}.self_attn.v_proj.weight"]
        w = torch.cat([q_proj_w, k_proj_w, v_proj_w], dim=0)
        state_dict[f"model.layers.{i}.attn.c_attn.linear.weight"] = w
        state_dict.pop(f"model.layers.{i}.self_attn.q_proj.weight")
        state_dict.pop(f"model.layers.{i}.self_attn.k_proj.weight")
        state_dict.pop(f"model.layers.{i}.self_attn.v_proj.weight")

    replace_pairs = [
        ("model.", "transformer."),
        (".embed_tokens.", ".embedding_layer.wte."),
        (".layers.", ".h."),
        (".self_attn.", ".attn."),
        (".post_attention_layernorm.", ".mlp.ln."),
        (".input_layernorm.", ".attn.c_attn.ln."),
        ("attn.o_proj.", "attn.c_proj."),
        (f".norm.", f".h.{config.n_layers - 1}.ln_f."),
        ("lm_head", "head"),
    ]
    for k1, k2 in replace_pairs:
        new_state_dict = {}
        for k, v in state_dict.items():
            if k1 in k:
                k = k.replace(k1, k2)
            new_state_dict[k] = v
        state_dict = new_state_dict

    keys_to_pop = [k for k in state_dict.keys() if "rotary_emb.inv_freq" in k]
    for k in keys_to_pop:
        state_dict.pop(k)

    if USE_TE_BACKEND:
        state_dict = new_state_dict
        new_state_dict = {}
        te_replace_pairs = [
            (".mlp.ln.weight", ".mlp.layer_norm_weight"),
            (".mlp.down_proj.weight", ".mlp.fc2_weight"),
        ]
        for k, v in state_dict.items():
            for k1, k2 in te_replace_pairs:
                if k1 in k:
                    k = k.replace(k1, k2)
            new_state_dict[k] = v

        # fuse gate && up weight
        for i in range(config.n_layers):
            gate_w = new_state_dict[f"transformer.h.{i}.mlp.gate_proj.weight"]
            upproj_w = new_state_dict[f"transformer.h.{i}.mlp.up_proj.weight"]
            w = torch.cat([gate_w, upproj_w], dim=0)
            new_state_dict[f"transformer.h.{i}.mlp.fc1_weight"] = w
            new_state_dict[f"transformer.h.{i}.mlp._extra_state"] = None
            new_state_dict.pop(f"transformer.h.{i}.mlp.gate_proj.weight")
            new_state_dict.pop(f"transformer.h.{i}.mlp.up_proj.weight")
    return new_state_dict


def to_llama_state_dict(state_dict: Dict[str, torch.Tensor], config: FlashMQATConfig) -> Dict:
    if USE_TE_BACKEND:
        # remove all extra states
        keys = list(state_dict.keys())
        for k in keys:
            if k.endswith("_extra_state"):
                state_dict.pop(k)

        # split gate && up weight
        for i in range(config.n_layers):
            w = state_dict[f"transformer.h.{i}.mlp.fc1_weight"]
            gate_w, upproj_w = w.split((w.shape[0] // 2, w.shape[0] // 2), dim=0)
            state_dict[f"transformer.h.{i}.mlp.gate_proj.weight"] = gate_w.contiguous()
            state_dict[f"transformer.h.{i}.mlp.up_proj.weight"] = upproj_w.contiguous()
            state_dict.pop(f"transformer.h.{i}.mlp.fc1_weight")

        # rename
        new_state_dict = {}
        te_replace_pairs = [
            (".mlp.layer_norm_weight", ".mlp.ln.weight"),
            (".mlp.fc2_weight", ".mlp.down_proj.weight"),
        ]
        for k, v in state_dict.items():
            for k1, k2 in te_replace_pairs:
                if k1 in k:
                    k = k.replace(k1, k2)
            new_state_dict[k] = v
        state_dict = new_state_dict

    replace_pairs = [
        ("model.", "transformer."),
        (".embed_tokens.", ".embedding_layer.wte."),
        (".layers.", ".h."),
        (".self_attn.", ".attn."),
        (".post_attention_layernorm.", ".mlp.ln."),
        (".input_layernorm.", ".attn.c_attn.ln."),
        ("attn.o_proj.", "attn.c_proj."),
        (f".norm.", f".h.{config.n_layers - 1}.ln_f."),
        ("lm_head", "head"),
        (".input_layernorm.", ".self_attn.c_attn.ln."),
        ("model.norm.weight", f"model.layers.{config.n_layers - 1}.ln_f.weight"),
    ]
    for k2, k1 in replace_pairs:
        new_state_dict = {}
        for k, v in state_dict.items():
            if k1 in k:
                k = k.replace(k1, k2)
            new_state_dict[k] = v
        state_dict = new_state_dict
    for i in range(config.n_layers):
        w = state_dict[f"model.layers.{i}.self_attn.c_attn.linear.weight"]
        nq = config.hidden_dim // config.head_dim
        q_proj_w = w[: nq * config.head_dim]
        k_proj_w = w[nq * config.head_dim : (nq + config.n_kv_heads) * config.head_dim]
        v_proj_w = w[(nq + config.n_kv_heads) * config.head_dim :]
        w = torch.cat([q_proj_w, k_proj_w, v_proj_w], dim=0)
        state_dict[f"model.layers.{i}.self_attn.q_proj.weight"] = q_proj_w
        state_dict[f"model.layers.{i}.self_attn.k_proj.weight"] = k_proj_w
        state_dict[f"model.layers.{i}.self_attn.v_proj.weight"] = v_proj_w
        state_dict.pop(f"model.layers.{i}.self_attn.c_attn.linear.weight")
    return state_dict


for name in ["llama", "deepseek", "codellama"]:
    FlashMQATModel.register_hf_model(
        name, convert_config_llama, convert_state_dict_llama, to_llama_state_dict
    )
################################ LLaMa End ################################
