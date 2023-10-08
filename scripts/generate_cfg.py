import json
import os
import shutil

from scripts.cfg_templates.model_info import *

def generate_cfg(name):
    example_config_dir = get_example_config_dir(name)
    with open(os.path.join(example_config_dir, "config.json"), "r") as f:
        example_config = json.load(f)

    configs = CONFIGS[name]
    keys = KEYS[name]

    for hs, nhead, nlayers in configs:
        example_config[keys["hidden_size"]] = hs
        example_config[keys["ffn_dim"]] = hs * 4
        example_config[keys["num_attention_heads"]] = nhead
        example_config[keys["num_hidden_layers"]] = nlayers
        try:
            example_config[keys["word_embed_proj_dim"]] = hs
        except:
            pass

        new_config_dir = get_new_config_dir(name, hs, nlayers)
        os.makedirs(new_config_dir, exist_ok=True)
        shutil.copytree(example_config_dir, new_config_dir, dirs_exist_ok=True)

        new_config_file = os.path.join(new_config_dir, "config.json")
        with open(new_config_file, "w") as f:
            json.dump(example_config, f, indent=4)


if __name__ == "__main__":
    generate_cfg("opt")
    generate_cfg("starcoder")