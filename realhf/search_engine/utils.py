from realhf.api.core.model_api import ReaLModelConfig


def find_factors(n):
    factors = []
    for i in range(1, n + 1):
        if n % i == 0:
            factors.append(i)
    return factors


def make_stats_key(rpc_name, bs, seq_len):
    return f"{rpc_name}|{bs}|{seq_len}"


def parse_stats_key(key):
    rpc_name, bs, seq_len = key.split("|")
    return rpc_name, int(bs), int(seq_len)


def load_model_config(model_class: str, model_path: str) -> ReaLModelConfig:
    from realhf.impl.model.nn.real_llm_api import ReaLModel

    return getattr(ReaLModel, f"config_from_{model_class}")(model_path=model_path)
