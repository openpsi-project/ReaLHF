# TODO: maybe move this file to api
import dataclasses
import json

import torch


@dataclasses.dataclass
class TransformerConfig:
    n_layers: int
    n_heads: int
    head_dim: int
    hidden_dim: int
    intermediate_dim: int  # for mlp, usually 4*h
    vocab_size: int
    n_positions: int
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    activation_function: str = "gelu"

    @staticmethod
    def from_huggingface_config(config_file):
        config_json = json.load(open(config_file, "r"))
        # for starcoder config
        config = TransformerConfig(n_layers=config_json["n_layer"],
                                   n_heads=config_json["n_head"],
                                   head_dim=config_json["n_embd"] // config_json["n_head"],
                                   hidden_dim=config_json["n_embd"],
                                   intermediate_dim=config_json["n_inner"],
                                   vocab_size=config_json["vocab_size"],
                                   n_positions=config_json["n_positions"],
                                   resid_pdrop=config_json["resid_pdrop"],
                                   attn_pdrop=config_json["attn_pdrop"],
                                   embd_pdrop=config_json["embd_pdrop"],
                                   layer_norm_epsilon=config_json["layer_norm_epsilon"],
                                   activation_function="gelu")
        return config


@dataclasses.dataclass
class TransformerData:
    # input, and config
    raw_input_ids: torch.Tensor = None
    raw_attention_mask: torch.Tensor = None
    input_ids: torch.Tensor = None
    attention_mask: torch.Tensor = None
    labels: torch.Tensor = None
    position_ids: torch.Tensor = None
    hidden_states: torch.Tensor = None
    kv_cache: torch.Tensor = None
    head_mask: torch.Tensor = None
    generation_id: int = None  # for kv cache, should > 0
    # outputs
    loss: torch.Tensor = None
    logits: torch.Tensor = None

    def to_tuple(self):
        t = []
        # the order of fields in tuple should be deterministic
        for v in dataclasses.asdict(self).values():
            t.append(to_tensor(v))
        return tuple(t)

    @staticmethod
    def from_tuple(t):
        x = TransformerData()
        for i, f in enumerate(dataclasses.fields(x)):
            setattr(x, f.name, from_tensor(t[i], f.type))
        return x


# TODO: temp solution, all data going through pp models must be non-boolean tensors
# before input to pipe model, convert all data to tensors (# input of pipe model should be tensors)-> convert back to original type
# after output from pipe -> convert all data to tensors


def to_tensor(x):
    device = torch.cuda.current_device()
    if isinstance(x, int) or isinstance(x, bool):
        assert x >= 0
        return torch.tensor(x, dtype=torch.long, device=device)
    elif x is None:
        return torch.tensor(-1, dtype=torch.long, device=device)
    elif torch.is_tensor(x):
        # if x.dtype != torch.bool:
        #     return x.to(device=device)
        # else:
        #     # convert bool tensor to long tensor
        #     return x.to(dtype=torch.long, device=device)
        return x.to(device=device)
    else:
        raise NotImplementedError(f"Cannot convert {x} to tensor")


def from_tensor(x, _type):
    try:
        if int(x) < 0:
            return None
    except:
        pass
    if _type == int:
        return int(x)
    elif _type == bool:
        return bool(x)
    elif _type == torch.Tensor:
        return x
    else:
        raise NotImplementedError(f"Cannot convert tensor to {_type}")
