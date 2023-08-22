from typing import Tuple
import logging
import os

import torch

import api.model
import api.utils

logger = logging.getLogger("Model Save")


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
        api.utils.save_hf_format(
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