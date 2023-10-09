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


def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    r"""
    Taken from peft: https://github.com/huggingface/peft/blob/main/src/peft/utils/other.py#L81

    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_kbit = getattr(model, "is_loaded_bnb_8bit", False) or getattr(model, "is_loaded_in_4bit", False)

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    # cast all non INT8 parameters to fp32
    for param in model.parameters():
        if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
            param.data = param.data.to(torch.float32)

    if loaded_in_kbit and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    return model


def create_hf_nn(
    model_class: Type,
    model_name_or_path: str,
    init_from_scratch: bool = False,
    from_pretrained_kwargs: Optional[Dict[str, Any]] = None,
    generation_kwargs: Optional[Dict[str, Any]] = None,
    quantization_kwargs: Optional[Dict[str, Any]] = None,
) -> transformers.PreTrainedModel:

    # load model
    model_config = transformers.AutoConfig.from_pretrained(model_name_or_path)
    logger.info(f"Loading from {model_name_or_path}...")
    st = time.monotonic()
    if init_from_scratch:
        model: transformers.PreTrainedModel = model_class.from_config(model_config)
        if quantization_kwargs is not None:
            from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
            qconfig = BnbQuantizationConfig(**quantization_kwargs)
            model = load_and_quantize_model(model, qconfig)
    else:
        if quantization_kwargs is not None:
            qconfig = transformers.BitsAndBytesConfig(**quantization_kwargs)
            from_pretrained_kwargs['quantization_config'] = qconfig
        model: transformers.PreTrainedModel = model_class.from_pretrained(
            model_name_or_path,
            **from_pretrained_kwargs,
        )
    if quantization_kwargs is not None:
        # TODO: fix gradient checkpointing here
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    logger.info(f"Loaded from {model_name_or_path}, time cost {time.monotonic() - st}")

    # overwrite and fix generation config
    tokenizer = load_hf_tokenizer(model_name_or_path)
    if "pad_token_id" not in model.config.to_dict():
        model.config.pad_token_id = tokenizer.pad_token_id
    raw_generation_config = transformers.GenerationConfig.from_model_config(model.config)
    assert tokenizer.pad_token_id is not None
    raw_generation_config.pad_token_id = tokenizer.pad_token_id
    assert tokenizer.eos_token_id is not None
    raw_generation_config.eos_token_id = tokenizer.eos_token_id
    raw_generation_config_dict = raw_generation_config.to_dict()
    if generation_kwargs is not None:
        raw_generation_config_dict.update(generation_kwargs)
    model.generation_config.update(**raw_generation_config_dict)
    logger.debug("Hugginface model generation config: ", model.generation_config)

    return model


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor
