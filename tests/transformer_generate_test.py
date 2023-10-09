# model level generation test
# dp_size = pp_size = 2
import os
import sys

sys.path.append("../")
import argparse
import multiprocessing as mp
import random
import time

import deepspeed
import torch

mp.set_start_method('spawn', force=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

from base.namedarray import NamedArray
from impl.model.nn.mqa_transformer import generate, TransformerConfig
import api.config as config_package
import api.model
import base.gpu_utils
import base.name_resolve as name_resolve
import base.names as names

MODEL_PATH = "/lustre/meizy/backup_zy/model_saves/four_layers_starcoder/"
IF_LOG = True


def setup_gpu():
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    torch.cuda.set_device(0)
    torch_dist_kwargs = dict(world_size=1, rank=0, init_method="tcp://localhost:12345", backend='nccl')
    torch.distributed.init_process_group(**torch_dist_kwargs, group_name="generate")
    return torch.device("cuda", 0)


def get_model(model_path, device):
    config_file = os.path.join(model_path, "config.json")
    # model_file_path = os.path.join(model_path, "pytorch_model.bin")
    config = TransformerConfig.from_huggingface_config(config_file)
    model_config = config_package.Model(type_="mqa_transformer",
                                        args=dict(
                                            model_name_or_path=model_path,
                                            config=config,
                                        ))
    model = api.model.make_model(model_config, name="generate", device=device)
    return model


def get_huggingface_model(model_path, device):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model


def get_interface():
    return api.model.make_interface(config_package.ModelInterface(type_="simple", args=dict()))


def get_backend():
    return api.model.make_backend(config_package.ModelBackend(type_='ds_inference', args=dict()))


def get_input(tokenizer, device, s):
    prompts = tokenizer(s, return_tensors="pt", padding=True)

    input_ids, attention_mask = prompts["input_ids"], prompts["attention_mask"]
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    return input_ids, attention_mask


def random_sentence(min_len=1, max_len=20):
    words = ["the", "quick", "brown", "fox", "jumped", "over", "the", "lazy", "dog"]
    sentence_length = random.randint(min_len, max_len)
    return " ".join(random.choices(words, k=sentence_length))


def get_example_batch(tokenizer, device, batch_size, dp_rank, dp_worldsize, seed=1):
    random.seed(seed)
    whole_batch = [random_sentence() for _ in range(batch_size)]
    dp_batch = whole_batch[batch_size // dp_worldsize * dp_rank:batch_size // dp_worldsize * (dp_rank + 1)]
    # print(f"global rank {torch.distributed.get_rank()}, dp rank {dp_rank} batch: {dp_batch}")
    return get_input(tokenizer, device, dp_batch)


def get_batch(tokenizer, device, batch_size, seed=1):
    random.seed(seed)
    whole_batch = [random_sentence() for _ in range(batch_size)]
    return get_input(tokenizer, device, whole_batch)


def log(s):
    if IF_LOG:
        print(f"Generate test: {s}")


def main():
    device = setup_gpu()
    # worker_index, device = setup_gpu_deepspeed_cli()
    model = get_model(MODEL_PATH, device)
    interface = get_interface()
    backend = get_backend()

    model = backend.initialize(model, None)
    # test generate
    log(f"model initialized")
    input_ids, attention_mask = get_batch(model.tokenizer, device, 4)
    data = NamedArray(
        prompts=input_ids,
        prompt_att_mask=attention_mask,
    )
    log(f"generate inputs: ")
    log(f"prompts: {data.prompts}")
    log(f"prompt_att_mask: {data.prompt_att_mask}")

    log("begin generate")
    outputs = interface.generate(model, data)
    log("end generate")
    log(f"generate outputs: {outputs}")


def huggingface_generate():
    device = setup_gpu()
    model = get_huggingface_model(MODEL_PATH, device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side="left", fast_tokenizer=True)
    tokenizer.pad_token = tokenizer.eos_token
    log(f"model initialized")
    input_ids, attention_mask = get_batch(tokenizer, device, 4)
    log(f"generate inputs: ")
    log(f"prompts: {input_ids}")
    log(f"prompt_att_mask: {attention_mask}")

    log("begin generate")
    outputs = model.generate(input_ids,
                             attention_mask=attention_mask,
                             eos_token_id=tokenizer.eos_token_id,
                             pad_token_id=tokenizer.pad_token_id,
                             min_new_tokens=10,
                             max_new_tokens=50)
    log("end generate")
    log(f"generate outputs: {outputs}")


if __name__ == "__main__":
    main()
    # huggingface_generate()
