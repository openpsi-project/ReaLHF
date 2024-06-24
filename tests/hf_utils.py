import transformers


def get_llama_hf_config():
    return transformers.LlamaConfig(
        vocab_size=200,
        hidden_size=128,
        intermediate_size=200,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        hidden_act="silu",
        rms_norm_eps=1e-5,
    )


def get_qwen2_hf_config():
    return transformers.Qwen2Config(
        vocab_size=200,
        hidden_size=128,
        intermediate_size=200,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        hidden_act="silu",
        rms_norm_eps=1e-5,
    )


def get_gemma_hf_config():
    return transformers.GemmaConfig(
        **{
            "attention_bias": False,
            "attention_dropout": 0.0,
            "hidden_act": "gelu",
            "hidden_size": 160,
            "intermediate_size": 200,
            "max_position_embeddings": 8192,
            "num_attention_heads": 8,
            "num_hidden_layers": 2,
            "num_key_value_heads": 8,
            "head_dim": 32,
            "rms_norm_eps": 1e-06,
            "rope_theta": 10000.0,
            "vocab_size": 200,
        }
    )


def get_gpt2_config():
    return transformers.GPT2Config(
        vocab_size=200,
        n_positions=200,
        n_embd=128,
        n_layer=2,
        n_head=8,
        n_inner=200,
        activation_function="gelu",
    )


def get_opt_config():
    return transformers.OPTConfig(
        vocab_size=200,
        max_position_embeddings=200,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        ffn_dim=200,
        do_layer_norm_before=True,
    )


def hf_config_factory(model_family_name: str):
    if model_family_name == "llama":
        return get_llama_hf_config()
    elif model_family_name == "qwen2":
        return get_qwen2_hf_config()
    elif model_family_name == "gemma":
        return get_gemma_hf_config()
    elif model_family_name == "gpt2":
        return get_gpt2_config()
    elif model_family_name == "opt":
        return get_opt_config()
    else:
        raise NotImplementedError(model_family_name)
