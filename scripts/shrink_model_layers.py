import os
import sys

sys.path.append("../")

import json
import logging

from transformers import AutoModelForCausalLM
import torch

from scripts.transform_to_pipe_ckpt import copy_configs

from api.huggingface import create_hf_nn

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
DATE_FORMAT = "%Y%m%d-%H:%M:%S"
logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT, level="INFO")

LOAD_PATH = "/lustre/meizy/models/starcoder_scratch"
SAVE_PATH = "/lustre/meizy/models/starcoder_4l"
NUM_SHRINKED_LAYERS = 4

if __name__ == "__main__":
    state_dict = {}
    for fn in os.listdir(LOAD_PATH):
        if fn.endswith(".bin"):
            state_dict.update(torch.load(os.path.join(LOAD_PATH, fn)))
            print(f"loaded {fn}")

    new_state_dict = {}
    for k in state_dict.keys():
        if k.startswith("transformer.h."):
            layer_idx = int(k.split(".")[2])
            if layer_idx < NUM_SHRINKED_LAYERS:
                new_state_dict[k] = state_dict[k]
        else:
            new_state_dict[k] = state_dict[k]

    for k in new_state_dict.keys():
        print(k)

    os.makedirs(SAVE_PATH, exist_ok=True)
    copy_configs(LOAD_PATH, SAVE_PATH)
    abs_save_path = os.path.join(SAVE_PATH, "pytorch_model.bin")
    torch.save(new_state_dict, abs_save_path)
    print(f"saved to {abs_save_path}")

    config = json.load(open(os.path.join(LOAD_PATH, "config.json"), "r"))
    # for starcoder
    config["n_layer"] = NUM_SHRINKED_LAYERS
    json.dump(config, open(os.path.join(SAVE_PATH, "config.json"), "w"))
