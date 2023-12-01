from typing import Dict, Optional, Tuple
import dataclasses
import json
import os
import re

from safetensors import safe_open
from safetensors.torch import save_file as save_safetensors_file
import torch
import transformers

from base.monitor import process_memory_mb
import api.model
import base.logging as logging

logger = logging.getLogger("Model Save")


def save_hf_format(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizerFast,
    output_dir: str,
    sub_folder="",  # this allows for saving models of different epochs/steps in the same folder
    exclude_lora: bool = True,
):
    model_to_save = model.module if hasattr(model, "module") else model
    output_dir = os.path.join(output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_config_file = os.path.join(output_dir, "config.json")
    save_dict = model_to_save.state_dict()
    if exclude_lora:
        for key in list(save_dict.keys()):
            if "lora" in key:
                save_dict.pop(key)
    save_to_disk(save_dict, output_dir)
    if isinstance(model_to_save.config, transformers.PretrainedConfig):
        model_to_save.config.to_json_file(output_config_file)
    else:
        config = dataclasses.asdict(model_to_save.config)
        with open(output_config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
    try:
        tokenizer.save_vocabulary(output_dir)
    except ValueError:
        logger.warning("Cannot save fast tokenizer for llama.")


def save_hf_or_lora_model(model: api.model.Model, output_dir: str):
    from impl.model.nn.lora import get_lora_state_dict, is_lora_model

    module = model.module
    tokenizer = model.tokenizer
    logger.info(f"saving the model for epoch {model.version.epoch} step {model.version.epoch_step}...")
    os.makedirs(
        os.path.abspath(os.path.join(
            output_dir,
            f"epoch{model.version.epoch}step{model.version.epoch_step}",
        )),
        exist_ok=True,
    )
    if not is_lora_model(module):
        save_hf_format(
            module,
            tokenizer,
            output_dir,
            sub_folder=f"epoch{model.version.epoch}step{model.version.epoch_step}",
        )
        return
    lora_sd = get_lora_state_dict(module)
    save_to_disk(lora_sd, os.path.join(output_dir,
                                       f"epoch{model.version.epoch}step{model.version.epoch_step}"))


def save_pipeline_model(model: api.model.Model, output_dir: str):
    module = model.module
    sub_folder = f"epoch{model.version.epoch}step{model.version.epoch_step}"
    output_dir = os.path.join(output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    module.save(output_dir)
    output_config_file = os.path.join(output_dir, "config.json")
    config = dataclasses.asdict(module.config)
    with open(output_config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


def split_state_dict_into_shards(state_dict: Dict, n_shards: int) -> Dict:
    if n_shards == 1:
        return [state_dict]

    keys = list(state_dict.keys())
    if len(keys) < n_shards:
        raise ValueError(f"state_dict has {len(keys)} keys, but n_shards={n_shards}")

    shard_size = len(keys) // n_shards
    extra = len(keys) % n_shards
    shard_size_list = [shard_size for _ in range(n_shards)]
    shard_size_list[-1] = shard_size + extra
    start, shards = 0, []
    for i, size in enumerate(shard_size_list):
        shard = {}
        for j in range(start, start + size):
            shard[keys[j]] = state_dict[keys[j]]
            print(f"shard {i} key {keys[j]}")
        start += size
        shards.append(shard)
    return shards


def save_to_disk(
        state_dict: Dict[str, torch.Tensor],
        output_dir: str,
        output_fn: Optional[str] = None,
        save_type: str = "pt",
        n_shards: Optional[int] = None,
        no_shard_suffix: bool = False,
        max_shard_size_byte: int = int(1e10),
):
    if n_shards is None:
        param_size = sum([value.numel() * value.element_size() for value in state_dict.values()])
        n_shards = (param_size + max_shard_size_byte - 1) // max_shard_size_byte

    if no_shard_suffix:
        if n_shards != 1:
            raise RuntimeError("no_shard_suffix is True, but n_shards != 1")

        if output_fn is None:
            if save_type == "pt":
                output_fn = "pytorch_model.bin"
            elif save_type == "st":
                output_fn = "model.safetensors"
            else:
                raise NotImplementedError(f"save_type {save_type} is not supported")

        if save_type == "pt":
            assert output_fn.endswith("bin")
            torch.save(state_dict, os.path.join(output_dir, output_fn))
        elif save_type == "st":
            assert output_fn.endswith("safetensors")
            save_safetensors_file(state_dict, os.path.join(output_dir, output_fn))
        else:
            raise NotImplementedError(f"save_type {save_type} is not supported")
        return

    if output_fn is None:
        if save_type == "pt":
            output_fn = "pytorch_model" + "-{shard:05d}" + f"-of-{n_shards:05d}.bin"
        elif save_type == "st":
            output_fn = "model" + "-{shard:05d}" + f"-of-{n_shards:05d}.safetensors"
        else:
            raise NotImplementedError(f"save_type {save_type} is not supported")

    shards = split_state_dict_into_shards(state_dict, n_shards)
    if save_type == "pt":
        assert output_fn.endswith("bin")
        for i, shard in enumerate(shards):
            torch.save(shard, os.path.join(output_dir, output_fn.format(shard=i + 1)))
    elif save_type == "st":
        assert output_fn.endswith("safetensors")
        for i, shard in enumerate(shards):
            save_safetensors_file(shard, os.path.join(output_dir, output_fn.format(shard=i + 1)))
    else:
        raise NotImplementedError(f"save_type {save_type} is not supported")


def load_from_safetensors(model_dir: str,
                          ext: str = ".safetensors",
                          pattern: Optional[str] = None) -> Tuple[Dict, int]:
    state_dict = {}
    cnt = 0
    for fn in os.listdir(model_dir):
        if not fn.endswith(ext):
            continue
        if pattern is not None and not re.match(pattern, fn):
            continue
        with safe_open(os.path.join(model_dir, fn), framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
            cnt += 1
        process_memory_mb(f"after_load_shard_{cnt}")
    return state_dict, cnt


def load_from_pytorch(model_dir: str, ext: str = ".bin", pattern: Optional[str] = None) -> Tuple[Dict, int]:
    state_dict = {}
    cnt = 0
    for fn in os.listdir(model_dir):
        if not fn.endswith(ext):
            continue
        if pattern is not None and not re.match(pattern, fn):
            continue
        state_dict.update(torch.load(os.path.join(model_dir, fn), map_location="cpu"))
        process_memory_mb(f"after_load_shard_{cnt}")
        cnt += 1
    return state_dict, cnt


def load_from_disk(model_dir: str, fn_pattern: Optional[str] = None, return_n_shards: bool = False) -> Dict:
    # Load safetensors whenever possible, which is extremely fast.
    fns = list(os.listdir(model_dir))
    if any(fn.endswith(".DLLMsafetensors") for fn in fns):
        state_dict, n_shards = load_from_safetensors(model_dir, ext=".DLLMsafetensors", pattern=fn_pattern)
    elif any(fn.endswith(".DLLMbin") for fn in fns):
        state_dict, n_shards = load_from_pytorch(model_dir, ext=".DLLMbin", pattern=fn_pattern)
    elif any(fn.endswith(".safetensors") for fn in fns):
        state_dict, n_shards = load_from_safetensors(model_dir, pattern=fn_pattern)
    elif any(fn.endswith(".bin") for fn in fns):
        state_dict, n_shards = load_from_pytorch(model_dir, pattern=fn_pattern)
    else:
        logger.error(f"Cannot find any model file ending with `.bin` or `.safetensors` in {model_dir}.")
    if return_n_shards:
        return state_dict, n_shards
    return state_dict
