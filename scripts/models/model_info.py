import os

EXAMPLE_CONFIG_DIR = "scripts/cfg_templates/"
NEW_CONFIG_DIR = "/data/meizy/models/cfgonly"


def get_example_config_dir(name):
    return os.path.join(EXAMPLE_CONFIG_DIR, name)


def get_new_config_dir(name, hidden_size, num_hidden_layers):
    return os.path.join(NEW_CONFIG_DIR, f"{name}-{hidden_size}-{num_hidden_layers}")


CONFIGS = {
    "opt": [
        (768, 12, 12),  # 125m
        (1024, 16, 24),  # 350m
        (2048, 32, 24),  # 1.3b
        (2560, 32, 32),  # 2.7b
        (3072, 32, 24),  # 2.9b
        (4096, 32, 24),  # 5b
        (4096, 32, 32),  # 6.7b
        (4096, 32, 36),  # 7.5b
        (5120, 40, 40),  # 13b
        (5120, 40, 50),
        (5120, 40, 60),  # 19b
        (6144, 48, 40),  # 18.4b
        (6144, 48, 50),
        (6144, 48, 60),  # 27.5b
        (7168, 56, 48),  # 30b
        (9216, 72, 64),  # 66b
    ],
    "starcoder": [
        (1536, 12, 10),
        (6144, 48, 40)  # 15.5b
    ]
}

KEYS = {
    "opt": {
        "hidden_size": "hidden_size",
        "ffn_dim": "ffn_dim",
        "num_attention_heads": "num_attention_heads",
        "num_hidden_layers": "num_hidden_layers",
        "word_embed_proj_dim": "word_embed_proj_dim",
        "max_position_embeddings": "max_position_embeddings",
    },
    "starcoder": {
        "hidden_size": "n_embd",
        "ffn_dim": "n_inner",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
        # "word_embed_proj_dim": "n_embd",
        "max_position_embeddings": "n_positions",
    }
}
