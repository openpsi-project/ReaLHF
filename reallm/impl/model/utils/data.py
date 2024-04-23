from typing import List, Optional, Tuple, Type, Union
import dataclasses

import torch
import torch.distributed as dist

from reallm.base.monitor import time_mark
import reallm.base.logging as logging

logger = logging.getLogger("Modeling Data Utils")


# TODO: temp solution, all data going through pp models must be non-boolean tensors
# before input to pipe model, convert all data to tensors (# input of pipe model should be tensors)-> convert back to original type
# after output from pipe -> convert all data to tensors
def to_tensor(x: Union[int, bool, torch.Tensor, None]):
    device = torch.cuda.current_device()
    if isinstance(x, int) or isinstance(x, bool):
        assert x >= 0
        return torch.tensor(x, dtype=torch.long, device=device)
    elif x is None:
        return torch.tensor(-1, dtype=torch.long, device=device)
    elif torch.is_tensor(x):
        if x.dtype != torch.bool:
            return x
        else:
            # convert bool tensor to int tensor
            return x.to(dtype=torch.long)  # .to(dtype=torch.long, device=device)
    else:
        raise NotImplementedError(f"Cannot convert {x} to tensor")


def from_tensor(x: torch.Tensor, _type: Type):
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


class TensorDataclassToTupleInterface:

    def to_tuple(self):
        # the first element of the tuple is the length of the tuple
        # sometimes the tuple can be mutliple tensor dataclass instances
        device = torch.cuda.current_device()
        t = []
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            t.append(to_tensor(v))
        t = [torch.tensor(len(t), device=device), torch.tensor(self.encode(), device=device)] + t
        return tuple(t)

    @classmethod
    def from_tuple(cls, t):
        x = cls()
        # logger.info(f"from_tuple debug:: tuple length = {len(t)}, cls={cls}")
        for i, f in enumerate(dataclasses.fields(x)):
            # logger.info(f"from_tuple debug:: f.name={f.name}, i={i}")
            setattr(x, f.name, from_tensor(t[i + 2], f.type))
        return x

    def encode(self):
        return -1


def data_list_to_tensor_tuple(data_list: List[TensorDataclassToTupleInterface]):
    rank = dist.get_rank()
    time_mark("tensor_to_tuple_start", rank)
    res = []
    for data in data_list:
        res += list(data.to_tuple())
    time_mark("tensor_to_tuple_end", rank)
    return tuple(res)


def tensor_tuple_to_data_list(tensor_tuple: tuple):
    rank = dist.get_rank()
    time_mark("tuple_to_tensor_start", rank)
    res = []
    i = 0
    while i < len(tensor_tuple):
        num_fields = tensor_tuple[i]
        type_code = int(tensor_tuple[i + 1])
        if type_code == 0:
            cls_ = PipeTransferData
        elif type_code == 1:
            cls_ = PipeCacheData
        else:
            raise NotImplementedError(f"Unknown type code {type_code}")
        res.append(cls_.from_tuple(tensor_tuple[i:i + num_fields + 2]))
        i += num_fields + 2
    time_mark("tuple_to_tensor_end", rank)
    return res


@dataclasses.dataclass
class PipeTransferData(TensorDataclassToTupleInterface):
    """Data structure for transferring data between stages.

    Each pipeline stage has exactly one PipeTransferData as the input and the output,
    no matter how many layers are in this stage.

    Attributes:
        pp_input: The input to the current stage. Usually hidden states
            with shape [bs, seq_len, hidden_dim].
        pp_output: The output of the current stage, also the input to the next stage.
            Usually hidden states with shape [bs, seq_len, hidden_dim].
        cu_seqlens: The cumulative sequence lengths of packed input_ids.
            Used by flash_attn_varlen_func. Will not be used during generation.
            It's configuration-like data that must be transfered from the first stage
            to the last. Shape [bs + 1].
        max_seqlen: The maximum sequence length of packed input_ids.
            Used by flash_attn_varlen_func. Will not be used during generation.
            It's configuration-like data that must be transfered from the first stage
            to the last.
        store_kv_cache: Whether to store the key and value cache for generation.
        attention_mask: The attention mask of the input, the same as huggingface transformers.
            Used by torch_attn_func to examine the outputs of PyTorch attention and flash
            attention are the same. Only for debugging. Shape [bs, seq_len].
    """

    pp_input: torch.Tensor = None
    pp_output: torch.Tensor = None

    # The followings are "configuration"-like data that should be passed across all stages.
    cu_seqlens: torch.Tensor = None
    max_seqlen: int = None
    store_kv_cache: bool = False

    # Only used for debugging
    attention_mask: torch.Tensor = None

    def encode(self):
        return 0


@dataclasses.dataclass
class PipeCacheData(TensorDataclassToTupleInterface):
    """Data structure for caching data locally that will not be trasferred.

    Each layer has exactly one PipeCacheData as the input.
    If a pipeline stage has multiple layers, a list of PipeCacheData should be passed
    as the input. The cached tensors will be changed in-place.

    Attributes:
        input_ids: The input token ids. Used only at the first stage.
            Can be packed with shape [total_seq_len] or unpacked with shape [bs, seq].
        prompt_mask: Prompt mask used
        position_ids: Input position IDs. Can be resolved automatically in most cases.
            Used only at the first stage. The same shape as input_ids.
            If None, will be resolved automatically.
        k_cache: Key cache used for generation, shape [bs, max_seq, n_kv_heads, head_dim].
            Note that this is the cache for a specific layer, not for all layers.
        v_cache: Value cache used for generation, shape [bs, max_seq, n_kv_heads, head_dim].
            Note that this is the cache for a specific layer, not for all layers.
        cache_seqlens: The sequence lengths of the cached tokens. Used for generation. Shape [bs].
    """

    # Only cached in the first stage.
    input_ids: torch.Tensor = None
    position_ids: torch.Tensor = None
    # Cached in each transformer layer.
    k_cache: torch.Tensor = None
    v_cache: torch.Tensor = None
    cache_seqlens: torch.Tensor = None

    def encode(self):
        return 1


@dataclasses.dataclass
class DuckModelOutput:
    logits: Optional[Union[List[torch.Tensor], torch.Tensor]] = None


@dataclasses.dataclass
class DuckGenerationOutput:
    sequences: torch.Tensor
    scores: Optional[torch.Tensor] = None
    logits_mask: Optional[torch.Tensor] = None
