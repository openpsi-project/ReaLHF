import os
import sys

sys.path.append("../")

import json
import logging

from transformers import AutoModelForCausalLM
import torch

from scripts.transform_to_pipe_ckpt import copy_configs

from api.huggingface import create_hf_nn
from impl.model.utils.save_load import load_from_disk, save_to_disk

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
DATE_FORMAT = "%Y%m%d-%H:%M:%S"
logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT, level="INFO")

LOAD_PATH = "/lustre/public/pretrained_model_weights/Llama-2-13b-hf"
SAVE_PATH = "/lustre/public/pretrained_model_weights/testOnly/llama-2-4l"
NUM_SHRINKED_LAYERS = 4

if __name__ == "__main__":
    state_dict = load_from_disk(LOAD_PATH)

    new_state_dict = {}
    for k in state_dict.keys():
        if k.startswith("transformer.h.") or k.startswith("model.layers."):
            layer_idx = int(k.split(".")[2])
            if layer_idx < NUM_SHRINKED_LAYERS:
                new_state_dict[k] = state_dict[k]
        elif k.startswith("h."):
            layer_idx = int(k.split(".")[1])
            if layer_idx < NUM_SHRINKED_LAYERS:
                new_state_dict[k] = state_dict[k]
        else:
            new_state_dict[k] = state_dict[k]

    for k in new_state_dict.keys():
        print(k)

    os.makedirs(SAVE_PATH, exist_ok=True)
    copy_configs(LOAD_PATH, SAVE_PATH)
    save_to_disk(new_state_dict, SAVE_PATH, save_type='pt', n_shards=1, no_shard_suffix=True)

    config = json.load(open(os.path.join(LOAD_PATH, "config.json"), "r"))
    # for starcoder
    if "n_layer" in config:
        config["n_layer"] = NUM_SHRINKED_LAYERS
    elif "num_hidden_layers" in config:
        config["num_hidden_layers"] = NUM_SHRINKED_LAYERS
    json.dump(config, open(os.path.join(SAVE_PATH, "config.json"), "w"))
    os.system("chmod -R 775 " + SAVE_PATH)
