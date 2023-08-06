from typing import Any, Dict, Optional, Type
import logging
import os
import time

import torch
import transformers

logger = logging.getLogger("HuggingFace Model")


def load_hf_tokenizer(model_name_or_path: str, fast_tokenizer=True) -> transformers.PreTrainedTokenizerFast:
    if os.path.exists(model_name_or_path):
        # Locally tokenizer loading has some issue, so we need to force download
        model_json = os.path.join(model_name_or_path, "config.json")
        if "codet5" in model_name_or_path:
            tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name_or_path,
                                                                      fast_tokenizer=fast_tokenizer)
        if os.path.exists(model_json):
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path,
                                                                   fast_tokenizer=fast_tokenizer)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path,
                                                               fast_tokenizer=fast_tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def create_hf_nn(
    model_class: Type,
    model_name_or_path: str,
    init_from_scratch: bool = False,
    dtype: torch.dtype = torch.float16,
    generation_kwargs: Optional[Dict[str, Any]] = None,
) -> transformers.PreTrainedModel:
    tokenizer = load_hf_tokenizer(model_name_or_path)
    model_config = transformers.AutoConfig.from_pretrained(model_name_or_path)
    # FIXME: there may be an error when using ZeRO-3
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration

    logger.info(f"Loading from {model_name_or_path}, dtype {dtype}")
    st = time.monotonic()
    model = None
    if init_from_scratch:
        try:
            model = model_class.from_config(model_config)
        except:
            pass
    if model is None:
        model = model_class.from_pretrained(model_name_or_path,
                                            from_tf=False,
                                            config=model_config,
                                            torch_dtype=dtype)  # torch_dtype = torch.float16
    logger.info(f"Loaded from {model_name_or_path}, dtype {dtype}, time cost {time.monotonic() - st}")

    if "pad_token_id" not in model.config.to_dict():
        model.config.pad_token_id = tokenizer.pad_token_id
    raw_generation_config = transformers.GenerationConfig.from_model_config(model.config)
    assert tokenizer.pad_token_id is not None
    raw_generation_config.pad_token_id = tokenizer.pad_token_id
    assert tokenizer.eos_token_id is not None
    raw_generation_config.eos_token_id = tokenizer.eos_token_id
    if generation_kwargs is None:
        model.generation_config = raw_generation_config
    else:
        model.generation_config = transformers.GenerationConfig.from_dict({
            **raw_generation_config.to_dict(),
            **generation_kwargs
        })
    logger.debug("Hugginface model generation config: ", model.generation_config)
    return model.to(dtype=dtype)


def save_hf_format(model, tokenizer, output_dir, sub_folder=""):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = os.path.join(output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    for key in list(save_dict.keys()):
        if "lora" in key:
            save_dict.pop(key)
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor
