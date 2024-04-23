from typing import Dict, Optional, Tuple
import dataclasses
import json
import os
import re
import shutil

from safetensors import safe_open
from safetensors.torch import save_file as save_safetensors_file
import torch
import tqdm
import transformers

from base.monitor import process_memory_mb
import api.model
import reallm.base.constants
import reallm.base.logging as logging

logger = logging.getLogger("Model Save")


################################ these functions are currently not used ################################
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


################################ these functions are currently not used ################################


def split_state_dict_into_shards(state_dict: Dict, n_shards: int, verbose: bool = False) -> Dict:
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
    for i, size in enumerate(
            tqdm.tqdm(shard_size_list, desc=f"Splitting state dict into {len(shard_size_list)} shards...")):
        shard = {}
        for j in range(start, start + size):
            shard[keys[j]] = state_dict[keys[j]]
            # print(f"shard {i} key {keys[j]}")
        start += size
        shards.append(shard)
    return shards


HF_MODEL_CONFIG_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
]


def copy_hf_configs(src_model_dir, dst_model_dir):
    for file in HF_MODEL_CONFIG_FILES:
        try:
            shutil.copy(os.path.join(src_model_dir, file), os.path.join(dst_model_dir, file))
            print(f"copied {file} from {src_model_dir} to {dst_model_dir}")
        except FileNotFoundError:
            print(f"{file} not exist in {src_model_dir} skipping.")


def save_to_disk(
    state_dict: Dict[str, torch.Tensor],
    output_dir: str,
    output_fn: Optional[str] = None,
    save_type: str = "pt",
    n_shards: Optional[int] = None,
    no_shard_suffix: bool = False,
    max_shard_size_byte: int = int(1e10),
    with_hf_format: bool = False,
    hf_base_model_path: Optional[str] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    param_size = sum([value.numel() * value.element_size() for value in state_dict.values()])
    if n_shards is None:
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
        bin_index = {}
        bin_index["metadata"] = dict(total_size=param_size)
        bin_index['weight_map'] = {}
        for i, shard in enumerate(shards):
            torch.save(shard, os.path.join(output_dir, output_fn.format(shard=i + 1)))
            for k in shard:
                bin_index['weight_map'][k] = output_fn.format(shard=i + 1)
        # NOTE: we may require this to call `from_pretrained` like huggingface models
        if with_hf_format:
            with open(os.path.join(output_dir, "pytorch_model.bin.index.json"), "w") as f:
                json.dump(bin_index, f, indent=4)
    elif save_type == "st":
        # NOTE: calling `from_pretrained` like huggingface models is not supported for safetensors now
        assert output_fn.endswith("safetensors")
        for i, shard in enumerate(tqdm.tqdm(shards, desc="Dumping safetensors to disk...")):
            save_safetensors_file(shard, os.path.join(output_dir, output_fn.format(shard=i + 1)))
    else:
        raise NotImplementedError(f"save_type {save_type} is not supported")

    if with_hf_format and hf_base_model_path is not None:
        copy_hf_configs(hf_base_model_path, output_dir)


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


@dataclasses.dataclass
class CheckpointSpec:
    mp_size: int
    pp_size: int
    n_shard: int


def get_ckpt_spec(model_dir: str):
    fns = list(os.listdir(model_dir))
    max_mp_rank = 0
    max_pp_rank = 0
    max_n_shard = 0
    for fn in fns:
        if not re.match(r".*pp-(\d{2})-mp-(\d{2})-s-(\d{2}).*", fn):
            continue
        mp_rank = int(fn.split("-")[4])
        if mp_rank > max_mp_rank:
            max_mp_rank = mp_rank
        pp_rank = int(fn.split("-")[2])
        if pp_rank > max_pp_rank:
            max_pp_rank = pp_rank
        n_shard = int(fn.split("-")[6].split('.')[0])
        if n_shard > max_n_shard:
            max_n_shard = n_shard
    return CheckpointSpec(max_mp_rank + 1, max_pp_rank + 1, max_n_shard + 1)


def load_from_disk(
    model_dir: str,
    fn_pattern: Optional[str] = None,
    return_n_shards: bool = False,
) -> Dict:
    fns = list(os.listdir(model_dir))

    def load_model_dir_fn_pattern(model_dir, fn_pattern):
        if any(fn.endswith(".DLLMbin") for fn in fns):
            state_dict, n_shards = load_from_pytorch(model_dir, ext=".DLLMbin", pattern=fn_pattern)
        elif any(fn.endswith(".DLLMsafetensors") for fn in fns):
            state_dict, n_shards = load_from_safetensors(model_dir,
                                                         ext=".DLLMsafetensors",
                                                         pattern=fn_pattern)
        elif any(fn.endswith(".bin") for fn in fns):
            state_dict, n_shards = load_from_pytorch(model_dir, pattern=fn_pattern)
        # Load safetensors whenever possible, which is extremely fast.
        elif any(fn.endswith(".safetensors") for fn in fns):
            state_dict, n_shards = load_from_safetensors(model_dir, pattern=fn_pattern)
        else:
            logger.error(f"Cannot find any model file ending with `.bin` or `.safetensors` in {model_dir}.")
        return state_dict, n_shards

    state_dict, n_shards = load_model_dir_fn_pattern(model_dir, fn_pattern)
    if return_n_shards:
        return state_dict, n_shards
    return state_dict
