from typing import Dict

import torch
import transformers

from impl.model.nn.flash_mqat.flash_mqat_base import FlashMQATConfig
from impl.model.nn.flash_mqat.flash_mqat_interface import FlashMQATForCausalLM

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


def convert_state_dict_starcoder(state_dict: Dict, config: FlashMQATConfig) -> Dict:
    new_state_dict = {}
    replace_from = [
        ".wte",
        ".wpe",
        ".ln_1.",
        ".ln_2.",
        ".c_attn.weight",
        ".c_attn.bias",
        "transformer.ln_f.",
    ]
    replace_to = [
        ".embedding_layer.wte",
        ".embedding_layer.wpe",
        ".attn.c_attn.ln.",
        ".mlp.ln.",
        ".c_attn.linear.weight",
        ".c_attn.linear.bias",
        f"transformer.h.{config.n_layers - 1}.ln_f.",
    ]
    for k, v in state_dict.items():
        for rf, rt in zip(replace_from, replace_to):
            if rf in k:
                k = k.replace(rf, rt)
        new_state_dict[k] = v
    return new_state_dict


FlashMQATForCausalLM.register_hf_model("starcoder", convert_config_starcoder, convert_state_dict_starcoder)

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
    return new_state_dict


FlashMQATForCausalLM.register_hf_model("gpt2", gpt2_config_converter, gpt2_state_dict_converter)

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
    ]
    for k1, k2 in replace_pairs:
        new_state_dict = {}
        for k, v in state_dict.items():
            if k1 in k:
                k = k.replace(k1, k2)
            new_state_dict[k] = v
        state_dict = new_state_dict
    return new_state_dict


################################ LLaMa End ################################
