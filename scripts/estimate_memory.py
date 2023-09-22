import deepspeed
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer)
from transformers.deepspeed import HfDeepSpeedConfig
import math

import torch
import os
import gc

EXAMPLE_COMPUTING_RESOURCE = [
    (1, 1),
    (1, 8),
    (2, 8)
] # (num_nodes, num_gpus)
ADDITIONAL_BUFFER_FACTOR=1.2
MODEL_DIR = "/data/meizy/models/cfgonly"

def get_tokenizer(model_name_or_path, fast_tokenizer=True):
    if "llama" in model_name_or_path:
        from transformers.models.llama import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, fast_tokenizer=fast_tokenizer)
        if tokenizer.pad_token is None:
            # assert tokenizer.eos_token is not None
            # tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.padding_side = 'right'
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, fast_tokenizer=fast_tokenizer)
        tokenizer.pad_token = tokenizer.eos_token
        # make sure tokenizer is right pad in our logic
        tokenizer.padding_side = 'right'
    return tokenizer


def load_hf_tokenizer(model_name_or_path, fast_tokenizer=True):
    assert os.path.exists(model_name_or_path), model_name_or_path
    # Locally tokenizer loading has some issue, so we need to force download
    model_json = os.path.join(model_name_or_path, "config.json")
    assert os.path.exists(model_json), model_json
    # model_json_file = json.load(open(model_json))
    # model_name = model_json_file["_name_or_path"]
    tokenizer = get_tokenizer(model_name_or_path, fast_tokenizer=fast_tokenizer)
    # else:
    #     tokenizer = get_tokenizer(model_name_or_path,
    #                               fast_tokenizer=fast_tokenizer)
    return tokenizer


def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    rlhf_training=False,
                    init_from_scratch=False,
                    disable_dropout=False):
    init_from_scratch = init_from_scratch or rlhf_training
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    if disable_dropout:
        model_config.dropout = 0.0
    if init_from_scratch:
        # the weight loading is handled by create critic model
        model = model_class.from_config(model_config)
    else:
        model = model_class.from_pretrained(model_name_or_path,
                                            from_tf=bool(".ckpt" in model_name_or_path),
                                            config=model_config)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(8 *
                                      math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    try:
        model._num_params = sum(
            [p.ds_numel if hasattr(p, "ds_tensor") else p.numel() for p in model.parameters()])
        print(f"Created model {type(model)}, num params {model._num_params/(1e9):.4f} B")
    except Exception as e:
        print(f"Calculate num params failed, exception: {e}")

    return model

def estimate_from_path(model_path):
    print(f"Estimating {model_path}")
    tokenizer = load_hf_tokenizer(model_path, fast_tokenizer=True)
    model = create_hf_model(AutoModelForCausalLM, model_path, tokenizer, init_from_scratch=True)
    for num_nodes, num_gpus in EXAMPLE_COMPUTING_RESOURCE:
        deepspeed.runtime.zero.stage_1_and_2.estimate_zero2_model_states_mem_needs_all_live(
            model, num_gpus_per_node=num_gpus, num_nodes=num_nodes, additional_buffer_factor=ADDITIONAL_BUFFER_FACTOR)
        deepspeed.runtime.zero.stage3.estimate_zero3_model_states_mem_needs_all_live(
            model, num_gpus_per_node=num_gpus, num_nodes=num_nodes, additional_buffer_factor=ADDITIONAL_BUFFER_FACTOR)
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    
def estimate_all():
    for d in os.listdir(MODEL_DIR):
        model_path = os.path.join(MODEL_DIR, d)
        if os.path.isdir(model_path):
            estimate_from_path(model_path)

if __name__ == "__main__":
    # estimate_all()
    configs = [
        (3072, 24),
        (4096, 36),
        (6144, 40)
    ]
    show_paths = [os.path.join(MODEL_DIR, f"opt-{hs}-{nlayers}") for hs, nlayers in configs]
    
    tokenizer = load_hf_tokenizer(show_paths[0], fast_tokenizer=True)
    for path in show_paths:
        model = create_hf_model(AutoModelForCausalLM, path, tokenizer, init_from_scratch=True)
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()