from typing import Tuple
import dataclasses
import json
import os

import torch
import transformers

import api.model
import base.logging as logging

logger = logging.getLogger("Model Save")


def save_hf_format(model: transformers.PreTrainedModel,
                   tokenizer: transformers.PreTrainedTokenizerFast,
                   output_dir: str,
                   sub_folder="",
                   exclude_lora: bool = True):
    model_to_save = model.module if hasattr(model, 'module') else model
    output_dir = os.path.join(output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
    output_config_file = os.path.join(output_dir, "config.json")
    save_dict = model_to_save.state_dict()
    if exclude_lora:
        for key in list(save_dict.keys()):
            if "lora" in key:
                save_dict.pop(key)
    torch.save(save_dict, output_model_file)
    if isinstance(model_to_save.config, transformers.PretrainedConfig):
        model_to_save.config.to_json_file(output_config_file)
    else:
        config = dataclasses.asdict(model_to_save.config)
        with open(output_config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
    tokenizer.save_vocabulary(output_dir)


def save_hf_or_lora_model(model: api.model.Model, output_dir: str):
    from impl.model.nn.lora import get_lora_state_dict, is_lora_model
    module = model.module
    tokenizer = model.tokenizer
    logger.info(f'saving the model for epoch {model.version.epoch} step {model.version.epoch_step}...')
    os.makedirs(os.path.abspath(
        os.path.join(
            output_dir,
            f"epoch{model.version.epoch}step{model.version.epoch_step}",
        )),
                exist_ok=True)
    if not is_lora_model(module):
        save_hf_format(
            module,
            tokenizer,
            output_dir,
            sub_folder=f"epoch{model.version.epoch}step{model.version.epoch_step}",
        )
        return
    lora_sd = get_lora_state_dict(module)
    torch.save(
        lora_sd,
        os.path.join(
            output_dir,
            f"epoch{model.version.epoch}step{model.version.epoch_step}",
            "lora.bin",
        ),
    )