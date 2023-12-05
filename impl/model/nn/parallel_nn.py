from typing import Dict, List, Optional, Tuple, Union
import os

import torch
import torch.nn as nn
import transformers

from base.monitor import process_memory_mb
from base.topology import PipeModelDataParallelTopology
from impl.model.nn.mp_flash_mqat import *
from impl.model.utils.pipeline_module import LayerSpec, PipelineModule
import api.huggingface
import api.model
import base.logging as logging

logger = logging.getLogger("3d_parallel_nn")


def make_causal_flash_mqat_3d_module(
    config: FlashMQATConfig,
    topology: PipeModelDataParallelTopology,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
):
    layer_specs = []
    # vocab pos embedding
    embedding_layer = LayerSpec(VocabPositionEmbedding,
                                config.vocab_size,
                                config.n_positions,
                                config.hidden_dim,
                                config.embd_pdrop,
                                config.fixed_abs_position_ids,
                                dtype=dtype,
                                device=device)

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

    head = LayerSpec(
        OutputHead,
        config.hidden_dim,
        config.vocab_size,
        bias=False,
        device=device,
        dtype=dtype,
    )
    layer_specs.append(head)

    def compute_loss(output, label):
        return output.loss

    return PipelineModule(layers=layer_specs, config=config, loss_fn=compute_loss, topology=topology), {}


def make_starcoder_flash_mqat_3d_module(
    model_path: str,
    topology: PipeModelDataParallelTopology,
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
    return make_causal_flash_mqat_3d_module(config, topology, dtype, device)


def load_starcoder_flash_mqat_3d(module: PipelineModule, model_path: str):
    module.load(model_path)
    return module


def make_flash_mqat_3d_model(
    name: str,
    device: torch.device,
    model_path: str,
    num_pp: int,
    num_dp: int,
    num_tp: int,
    dtype: torch.dtype = torch.float16,
    from_type: str = 'starcoder',
    tokenizer_path: Optional[str] = None,
    ckpt_path: Optional[str] = None,
    # used to load from ckpt stored in a different path from original model with huggingface config and tokenizers
):
    if tokenizer_path is None:
        tokenizer_path = model_path
    if from_type == 'starcoder':
        tokenizer = api.huggingface.load_hf_tokenizer(model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        topology = PipeModelDataParallelTopology(num_pp=num_pp, num_dp=num_dp, num_tp=num_tp)
        module, _ = make_starcoder_flash_mqat_3d_module(model_path, topology, dtype, device)
        if ckpt_path:
            model_path = ckpt_path
        module = load_starcoder_flash_mqat_3d(module, model_path)
        # logger.info("model loaded")
    else:
        raise NotImplementedError()
    return api.model.Model(name, module, tokenizer, device)


api.model.register_model("starcoder_flash_mqat_3d", make_flash_mqat_3d_model)
