from typing import Dict, List, Optional, Tuple, Union
import functools
import os

import torch
import torch.nn as nn
import transformers

from base.monitor import process_memory_mb
from base.topology import PipeDataParallelTopology
from impl.model.nn.flash_mqat import *
from impl.model.utils.pipeline_module import LayerSpec, PipelineModule
import api.huggingface
import api.model
import base.logging as logging

logger = logging.getLogger("pipe_nn")


def make_causal_flash_mqat_pipe_module(
    config: FlashMQATConfig,
    topology: PipeDataParallelTopology,
    from_type: str = 'starcoder',
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
):
    layer_specs = []
    # vocab pos embedding
    embedding_layer = LayerSpec(VocabPositionEmbedding, config, dtype=dtype, device=device)

    layer_specs.append(embedding_layer)

    for i in range(config.n_layers):
        flash_mqat_block = LayerSpec(FlashMQATBlock,
                                     config,
                                     layer_index=i,
                                     output_layernorm=(i == config.n_layers - 1),
                                     ckpt_attn=(i > 0 and config.ckpt_attn),
                                     ckpt_mlp=(i > 0 and config.ckpt_mlp),
                                     dtype=dtype,
                                     device=device)
        layer_specs.append(flash_mqat_block)

    lm_head = LayerSpec(
        LanguageModelHead,
        config.hidden_dim,
        config.vocab_size,
        bias=False,
        device=device,
        dtype=dtype,
    )
    layer_specs.append(lm_head)

    if from_type == 'starcoder':
        layer_key_mappings = starcoder_layer_key_mappings(config)
    elif from_type == 'llama':
        layer_key_mappings = {}

    def compute_loss(output, label):
        return output.loss

    return PipelineModule(layers=layer_specs, loss_fn=compute_loss, topology=topology,
                          config=config), layer_key_mappings


def starcoder_layer_key_mappings(config: FlashMQATConfig):
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
    return layer_key_mappings


def make_starcoder_flash_mqat_pipe_module(
    model_path: str,
    topology: PipeDataParallelTopology,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
):
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
    return make_causal_flash_mqat_pipe_module(config,
                                              topology,
                                              from_type="starcoder",
                                              dtype=dtype,
                                              device=device)


def make_llama_flash_mqat_pipe_module(
    model_path: str,
    topology: PipeDataParallelTopology,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
):
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
    return make_causal_flash_mqat_pipe_module(config, topology, from_type="llama", dtype=dtype, device=device)


def load_starcoder_flash_mqat_pipe(module: PipelineModule,
                                   layer_key_mappings: Dict[str, str],
                                   load_from_full_ckpt: Optional[bool] = False,
                                   model_path: Optional[str] = None):
    if load_from_full_ckpt:
        process_memory_mb("before_init_state_dict")
        try:
            state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        except FileNotFoundError:
            state_dict = transformers.AutoModelForCausalLM.from_pretrained(model_path).state_dict()

        process_memory_mb("after_init_state_dict")

        new_state_dict = {}
        for k, v in state_dict.items():
            for replace_from, replace_to in layer_key_mappings.items():
                if replace_from in k:
                    k = k.replace(replace_from, replace_to)
            new_state_dict[k] = v
        module.load_state_dict(new_state_dict, strict=False)

        process_memory_mb("after_load_state_dict")
    else:
        process_memory_mb("before_load")
        module.load(model_path)
        # process_memory_mb("after_load")
    return module


def load_llama_flash_mqat_pipe(module: PipelineModule, model_path: str):
    module.load(model_path)
    return module


def make_flash_mqat_pipe_model(
    name: str,
    device: torch.device,
    model_path: str,
    num_pp: int,
    num_dp: int,
    dtype: torch.dtype = torch.float16,
    from_type: str = 'starcoder',
    tokenizer_path: Optional[str] = None,
    load_from_full_ckpt: Optional[bool] = False,
    ckpt_path: Optional[str] = None,
    # used to load from ckpt stored in a different path from original model with huggingface config and tokenizers
):
    if tokenizer_path is None:
        tokenizer_path = model_path
    if from_type == 'starcoder':
        tokenizer = api.huggingface.load_hf_tokenizer(model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        # logger.info("tokenizer initialized")
        topology = PipeDataParallelTopology(num_pp=num_pp, num_dp=num_dp)
        module, layer_key_mappings = make_starcoder_flash_mqat_pipe_module(model_path, topology, dtype,
                                                                           device)
        process_memory_mb("after_make_pipe_module")
        # logger.info("module initialized")
        if ckpt_path:
            model_path = ckpt_path
        module = load_starcoder_flash_mqat_pipe(module,
                                                layer_key_mappings,
                                                load_from_full_ckpt,
                                                model_path=model_path)
        # logger.info("model loaded")
    elif from_type == 'llama':
        tokenizer = api.huggingface.load_hf_tokenizer(tokenizer_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        topology = PipeDataParallelTopology(num_pp=num_pp, num_dp=num_dp)
        module, layer_key_mappings = make_llama_flash_mqat_pipe_module(model_path, topology, dtype, device)
        module = load_llama_flash_mqat_pipe(module, model_path)
    else:
        raise NotImplementedError()
    return api.model.Model(name, module, tokenizer, device)


api.model.register_model("starcoder_flash_mqat_pipe", make_flash_mqat_pipe_model)
api.model.register_model("llama_flash_mqat_pipe",
                         functools.partial(make_flash_mqat_pipe_model, from_type="llama"))
