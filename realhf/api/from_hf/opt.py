from typing import *

import torch
import transformers

from realhf.api.core.model_api import ReaLModelConfig, register_hf_family
from realhf.base import constants


def sd_from_opt(state_dict: Dict, config: ReaLModelConfig) -> Dict:
    new_sd = {}

    if "model.decoder.embed_tokens.weight" in state_dict:
        if constants.is_first_pipe_stage():
            new_sd["0.wte.weight"] = state_dict["model.decoder.embed_tokens.weight"]
            new_sd["0.wpe.weight"] = state_dict["model.decoder.embed_positions.weight"]
        if (
            constants.is_last_pipe_stage()
            and config.tied_embedding
            and not config.is_critic
        ):
            new_sd[f"{config.n_layers + 1}.weight"] = state_dict[
                "model.decoder.embed_tokens.weight"
            ]
    if "lm_head.weight" in state_dict:
        new_sd[f"{config.n_layers + 1}.weight"] = state_dict["lm_head.weight"]
    if "model.decoder.final_layer_norm.weight" in state_dict:
        new_sd[f"{config.n_layers}.ln_f.weight"] = state_dict[
            "model.decoder.final_layer_norm.weight"
        ]
        new_sd[f"{config.n_layers}.ln_f.bias"] = state_dict[
            "model.decoder.final_layer_norm.bias"
        ]

    for i in range(config.n_layers):
        if not any(k.startswith(f"model.decoder.layers.{i}.") for k in state_dict):
            continue
        for k in ["weight", "bias"]:
            new_sd[f"{i+1}.attn.c_attn.q_attn.{k}"] = state_dict[
                f"model.decoder.layers.{i}.self_attn.q_proj.{k}"
            ]
            new_sd[f"{i+1}.attn.c_attn.k_attn.{k}"] = state_dict[
                f"model.decoder.layers.{i}.self_attn.k_proj.{k}"
            ]
            new_sd[f"{i+1}.attn.c_attn.v_attn.{k}"] = state_dict[
                f"model.decoder.layers.{i}.self_attn.v_proj.{k}"
            ]
            new_sd[f"{i+1}.attn.c_proj.{k}"] = state_dict[
                f"model.decoder.layers.{i}.self_attn.out_proj.{k}"
            ]
            new_sd[f"{i+1}.attn.c_attn.ln.{k}"] = state_dict[
                f"model.decoder.layers.{i}.self_attn_layer_norm.{k}"
            ]
            new_sd[f"{i+1}.mlp.ln.{k}"] = state_dict[
                f"model.decoder.layers.{i}.final_layer_norm.{k}"
            ]
            new_sd[f"{i+1}.mlp.c_fc.{k}"] = state_dict[
                f"model.decoder.layers.{i}.fc1.{k}"
            ]
            new_sd[f"{i+1}.mlp.c_proj.{k}"] = state_dict[
                f"model.decoder.layers.{i}.fc2.{k}"
            ]
    return new_sd


def sd_to_opt(state_dict: Dict, config: ReaLModelConfig) -> Dict:
    new_sd = {}
    if constants.is_first_pipe_stage():
        new_sd["model.decoder.embed_tokens.weight"] = state_dict["0.wte.weight"]
        new_sd["model.decoder.embed_positions.weight"] = state_dict["0.wpe.weight"]
    if config.is_critic or not config.tied_embedding:
        if constants.is_last_pipe_stage():
            new_sd["lm_head.weight"] = state_dict[f"{config.n_layers + 1}.weight"]

    if f"{config.n_layers}.ln_f.weight" in state_dict:
        new_sd["model.decoder.final_layer_norm.weight"] = state_dict[
            f"{config.n_layers}.ln_f.weight"
        ]
        new_sd["model.decoder.final_layer_norm.bias"] = state_dict[
            f"{config.n_layers}.ln_f.bias"
        ]

    for i in range(config.n_layers):
        if not any(k.startswith(f"{i+1}.") for k in state_dict):
            continue
        for k in ["weight", "bias"]:
            new_sd[f"model.decoder.layers.{i}.self_attn.q_proj.{k}"] = state_dict[
                f"{i+1}.attn.c_attn.q_attn.{k}"
            ]
            new_sd[f"model.decoder.layers.{i}.self_attn.k_proj.{k}"] = state_dict[
                f"{i+1}.attn.c_attn.k_attn.{k}"
            ]
            new_sd[f"model.decoder.layers.{i}.self_attn.v_proj.{k}"] = state_dict[
                f"{i+1}.attn.c_attn.v_attn.{k}"
            ]
            new_sd[f"model.decoder.layers.{i}.self_attn.out_proj.{k}"] = state_dict[
                f"{i+1}.attn.c_proj.{k}"
            ]
            new_sd[f"model.decoder.layers.{i}.self_attn_layer_norm.{k}"] = state_dict[
                f"{i+1}.attn.c_attn.ln.{k}"
            ]
            new_sd[f"model.decoder.layers.{i}.final_layer_norm.{k}"] = state_dict[
                f"{i+1}.mlp.ln.{k}"
            ]
            new_sd[f"model.decoder.layers.{i}.fc1.{k}"] = state_dict[
                f"{i+1}.mlp.c_fc.{k}"
            ]
            new_sd[f"model.decoder.layers.{i}.fc2.{k}"] = state_dict[
                f"{i+1}.mlp.c_proj.{k}"
            ]
    return new_sd


# param name is used to load directly from huggingface checkpoints
def opt_embedding_layer_names(config: ReaLModelConfig) -> List[str]:
    return [
        "model.decoder.embed_tokens.weight",
        "model.decoder.embed_positions.weight",
    ]


def opt_transformer_block_param_name(config: ReaLModelConfig, idx: int) -> List[str]:
    names = [
        f"model.decoder.layers.{idx}.self_attn.q_proj.weight",
        f"model.decoder.layers.{idx}.self_attn.q_proj.bias",
        f"model.decoder.layers.{idx}.self_attn.k_proj.weight",
        f"model.decoder.layers.{idx}.self_attn.k_proj.bias",
        f"model.decoder.layers.{idx}.self_attn.v_proj.weight",
        f"model.decoder.layers.{idx}.self_attn.v_proj.bias",
        f"model.decoder.layers.{idx}.self_attn.out_proj.weight",
        f"model.decoder.layers.{idx}.self_attn.out_proj.bias",
        f"model.decoder.layers.{idx}.self_attn_layer_norm.weight",
        f"model.decoder.layers.{idx}.self_attn_layer_norm.bias",
        f"model.decoder.layers.{idx}.fc1.weight",
        f"model.decoder.layers.{idx}.fc1.bias",
        f"model.decoder.layers.{idx}.fc2.weight",
        f"model.decoder.layers.{idx}.fc2.bias",
        f"model.decoder.layers.{idx}.final_layer_norm.weight",
        f"model.decoder.layers.{idx}.final_layer_norm.bias",
    ]
    if idx == config.n_layers - 1:
        names += [
            "model.decoder.final_layer_norm.weight",
            "model.decoder.final_layer_norm.bias",
        ]
    return names


def opt_output_head_param_name(config: ReaLModelConfig) -> List[str]:
    if config.tied_embedding and not config.is_critic:
        return ["model.decoder.embed_tokens.weight"]
    else:
        return ["lm_head.weight"]


def convert_config_opt(
    hf_config: transformers.OPTConfig,
) -> ReaLModelConfig:
    if hf_config.word_embed_proj_dim != hf_config.hidden_size:
        raise ValueError("OPT word_embed_proj_dim must be equal to hidden_size.")
    return ReaLModelConfig(
        n_layers=hf_config.num_hidden_layers,
        n_kv_heads=hf_config.num_attention_heads,
        hidden_dim=hf_config.hidden_size,
        head_dim=hf_config.hidden_size // hf_config.num_attention_heads,
        n_q_heads=hf_config.num_attention_heads,
        intermediate_dim=hf_config.ffn_dim,
        vocab_size=hf_config.vocab_size,
        n_positions=hf_config.max_position_embeddings,
        embd_pdrop=hf_config.dropout,
        attn_pdrop=hf_config.attention_dropout,
        resid_pdrop=hf_config.dropout,
        layer_norm_epsilon=1e-5,
        activation_function=hf_config.activation_function,
        use_attention_bias=hf_config.enable_bias,
        use_attn_proj_bias=hf_config.enable_bias,
        tied_embedding=hf_config.tie_word_embeddings,
        scale_attn_by_inverse_layer_idx=False,
        scale_attn_weights=True,
        abs_position_embedding_offset=2,
    )


def to_opt_config(config: ReaLModelConfig) -> transformers.OPTConfig:
    assert config.abs_position_embedding_offset == 2
    assert config.n_q_heads == config.n_kv_heads
    return transformers.OPTConfig(
        num_hidden_layers=config.n_layers,
        num_attention_heads=config.n_q_heads,
        hidden_size=config.hidden_dim,
        ffn_dim=config.intermediate_dim,
        vocab_size=config.vocab_size,
        max_position_embeddings=config.n_positions,
        dropout=config.embd_pdrop,
        attention_dropout=config.attn_pdrop,
        activation_function=config.activation_function,
        enable_bias=True,
        word_embed_proj_dim=config.hidden_dim,
    )


register_hf_family(
    name="opt",
    hf_cls_name="OPTForCausalLM",
    config_from_hf_converter=convert_config_opt,
    config_to_hf_converter=to_opt_config,
    sd_from_hf_converter=sd_from_opt,
    sd_to_hf_converter=sd_to_opt,
    embedding_param_names=opt_embedding_layer_names,
    tblock_param_names=opt_transformer_block_param_name,
    head_param_names=opt_output_head_param_name,
)
