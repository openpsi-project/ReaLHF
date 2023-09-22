import json
import os
import shutil

example_config_dir = "cfg_templates/opt/"
new_config_base_dir = "/data/meizy/models/cfgonly"

with open(os.path.join(example_config_dir, "config.json"), "r") as f:
    example_config = json.load(f)

configs = [
    (768, 12, 12), # 125m
    (1024, 16, 24), # 350m
    (2048, 32, 24),  # 1.3b
    (3072, 32, 24),  # 2.9b
    (4096, 32, 24),  # 5b
    (4096, 32, 36),  # 7.5b
    (5120, 40, 40),  # 13b
    (5120, 40, 60),  # 19b
    (6144, 48, 40),  # 18.4b
    (6144, 48, 60),  # 27.5b
    (7168, 56, 48),  # 30b
    (9216, 72, 64),  # 66b
]

for hs, nhead, nlayers in configs:
    hidden_size = hs
    ffn_dim = hidden_size * 4
    num_attention_heads = nhead
    num_hidden_layers = nlayers
    word_embed_proj_dim = hidden_size
    max_position_embeddings = 2048

    example_config["hidden_size"] = hidden_size
    example_config["ffn_dim"] = ffn_dim
    example_config["num_attention_heads"] = num_attention_heads
    example_config["num_hidden_layers"] = num_hidden_layers
    example_config["word_embed_proj_dim"] = word_embed_proj_dim
    example_config["max_position_embeddings"] = max_position_embeddings

    new_config_dir = os.path.join(new_config_base_dir, f"opt-{hidden_size}-{num_hidden_layers}")
    # os.makedirs(new_config_dir, exist_ok=True)
    shutil.copytree(example_config_dir, new_config_dir, dirs_exist_ok=True)

    new_config_file = os.path.join(new_config_dir, "config.json")
    with open(new_config_file, "w") as f:
        json.dump(example_config, f, indent=4)
