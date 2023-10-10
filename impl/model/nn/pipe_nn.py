from typing import Dict, List, Optional, Tuple, Union
import os

import torch
import torch.nn as nn
import transformers

from impl.model.backend.pipe_engine import (LayerSpec, PipeDataParallelTopology, PipelineModule,
                                            ProcessTopology)
from impl.model.nn.flash_mqat import *
from impl.model.utils.data import tensor_data_list_to_tuple, tuple_to_tensor_data_list
import api.model

# in pipeline transformer model, the entire module is divided into stages with a minimum unit of a wrapped module
# the wrapped module should take a tuple of tensors as input and output a tuple of tensors
# the wrapped module should be a subclass of nn.Module


class WrappedPipeLayer(nn.Module):

    def __init__(self, inner_module: nn.Module):
        super().__init__()
        self.inner_module = inner_module

    def forward(self, forward_input: Tuple[torch.Tensor, ...],
                other_input: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        args = tuple_to_tensor_data_list([forward_input, other_input])
        out = self.inner_module(*args)
        return tensor_data_list_to_tuple(out)


class WrappedPipeModule(PipelineModule):

    def __init__(self, modules: List[WrappedPipeLayer], topology: PipeDataParallelTopology):
        self.layer_specs = []
        for module in modules:
            self.layer_specs.append(LayerSpec(WrappedPipeLayer, module))

        def compute_loss(output, label):
            return output.loss

        super().__init__(layers=self.layer_specs, loss_fn=compute_loss, topology=topology)


def make_causal_flash_mqat_pipe_module(
    config: FlashMQATConfig,
    topology: PipeDataParallelTopology,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
):
    layers = []
    # vocab pos embedding
    embedding_layer = VocabPositionEmbedding(config.vocab_size,
                                             config.n_positions,
                                             config.hidden_dim,
                                             config.embd_pdrop,
                                             dtype=dtype,
                                             device=device)
    layers.append(embedding_layer)

    for i in range(config.n_layers):
        flash_mqat_block = FlashMQATBlock(config,
                                          layer_index=i,
                                          output_layernorm=(i == config.n_layers - 1),
                                          ckpt_attn=(i > 0 and config.ckpt_attn),
                                          ckpt_mlp=(i > 0 and config.ckpt_mlp),
                                          dtype=dtype,
                                          device=device)
        layers.append(flash_mqat_block)

    lm_head = LanguageModelHead(
        config.hidden_dim,
        config.vocab_size,
        bias=False,
        device=device,
        dtype=dtype,
    )
    layers.append(lm_head)

    layer_key_mappings = {
        "transformer.wte.": "0.inner_module.wte.",
        "transformer.wpe.": "0.inner_module.wpe.",
    }
    for i in range(config.n_layers):
        layer_key_mappings[f"transformer.h.{i}.attn.c_proj."] = f"{i+1}.inner_module.attn.c_proj."
        layer_key_mappings[f"transformer.h.{i}.mlp.c_proj."] = f"{i+1}.inner_module.mlp.c_proj."
        layer_key_mappings[f"transformer.h.{i}.mlp.c_fc."] = f"{i+1}.inner_module.mlp.c_fc."
        layer_key_mappings[f"transformer.h.{i}.ln_1."] = f"{i+1}.inner_module.attn.c_attn.ln."
        layer_key_mappings[f"transformer.h.{i}.ln_2."] = f"{i+1}.inner_module.mlp.ln."
        layer_key_mappings[f"transformer.h.{i}.attn.c_attn."] = f"{i+1}.inner_module.attn.c_attn.linear."
        if i == config.n_layers - 1:
            layer_key_mappings[f"transformer.ln_f."] = f"{i+1}.inner_module.ln_f."
    layer_key_mappings["lm_head."] = f"{config.n_layers+1}.inner_module."

    return WrappedPipeModule(layers, topology), layer_key_mappings


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
    return make_causal_flash_mqat_pipe_module(config, topology, dtype, device)


def load_starcoder_flash_mqat_pipe(module: WrappedPipeModule,
                                   layer_key_mappings: Dict[str, str],
                                   model_path: Optional[str] = None):
    try:
        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
    except FileNotFoundError:
        state_dict = transformers.AutoModelForCausalLM.from_pretrained(model_path).state_dict()

    new_state_dict = {}
    for k, v in state_dict.items():
        for replace_from, replace_to in layer_key_mappings.items():
            if replace_from in k:
                k = k.replace(replace_from, replace_to)
        new_state_dict[k] = v
    module.load_state_dict(new_state_dict, strict=False)
    return module


def make_flash_mqat_pipe_model(
    name: str,
    device: torch.device,
    model_path: str,
    dtype: torch.dtype,
    topology: PipeDataParallelTopology,
    from_type: str = 'starcoder',
    tokenizer_path: Optional[str] = None,
):
    if tokenizer_path is None:
        tokenizer_path = model_path
    if from_type == 'starcoder':
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        module, layer_key_mappings = make_starcoder_flash_mqat_pipe_module(model_path, topology, dtype,
                                                                           device)
        module = load_starcoder_flash_mqat_pipe(module, layer_key_mappings, model_path=model_path)
    else:
        raise NotImplementedError()
    return api.model.Model(name, module, tokenizer, device)


api.model.register_model("starcoder_flash_mqat_pipe", make_flash_mqat_pipe_model)
