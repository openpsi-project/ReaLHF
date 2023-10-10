from typing import List, Optional, Tuple, Type, Union
import dataclasses
import logging

import numpy as np
import torch
import torch.distributed as dist
import transformers

logger = logging.getLogger("Data Manipulation")


@torch.no_grad()
def masked_normalization(
    x,
    mask=None,
    dim=None,
    inplace=False,
    unbiased=False,
    eps=1e-5,
    high_precision=True,
):
    """Normalize x with a mask. Typically used in advantage normalization.

    Args:
        x (torch.Tensor):
            Tensor to be normalized.
        mask (torch.Tensor, optional):
            A mask with the same shape as x. Defaults to None.
        dim (int or tuple of ints, optional):
            Dimensions to be normalized. Defaults to None.
        inplace (bool, optional):
            Whether to perform in-place operation. Defaults to False.
        eps (torch.Tensor, optional):
            Minimal denominator. Defaults to 1e-5.

    Returns:
        torch.Tensor:
            Normalized x, with the same shape as x.
    """
    dtype = torch.float64 if high_precision else torch.float32
    x = x.to(dtype)
    if not inplace:
        x = x.clone()
    if dim is None:
        dim = tuple(range(len(x.shape)))
    if mask is None:
        factor = torch.tensor(np.prod([x.shape[d] for d in dim]), dtype=dtype, device=x.device)
    else:
        mask = mask.to(dtype)
        assert len(mask.shape) == len(x.shape), (mask.shape, x.shape, dim)
        for i in range(len(x.shape)):
            if i in dim:
                assert mask.shape[i] == x.shape[i], (mask.shape, x.shape, dim)
            else:
                assert mask.shape[i] == 1, (mask.shape, x.shape, dim)
        x = x * mask
        factor = mask.sum(dim, keepdim=True)
    x_sum = x.sum(dim=dim, keepdim=True)
    x_sum_sq = x.square().sum(dim=dim, keepdim=True)
    if dist.is_initialized():
        dist.all_reduce(factor, op=dist.ReduceOp.SUM)
        dist.all_reduce(x_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(x_sum_sq, op=dist.ReduceOp.SUM)
    mean = x_sum / factor
    meansq = x_sum_sq / factor
    var = meansq - mean**2
    if unbiased:
        var *= factor / (factor - 1)
    return ((x - mean) / (var.sqrt() + eps)).float()


def get_eos_indices(
    input_ids: torch.LongTensor,
    tokenizer: transformers.PreTrainedTokenizerFast,
) -> Tuple[torch.LongTensor, torch.FloatTensor]:
    if torch.any(input_ids[:, 0] == tokenizer.eos_token_id):
        indices = (input_ids[:, 0] == tokenizer.eos_token_id).nonzero().flatten()
        bad_input_ids = input_ids[indices]
        bad_strs = tokenizer.batch_decode(bad_input_ids,
                                          skip_special_tokens=True,
                                          clean_up_tokenization_spaces=True)
        raise RuntimeError(f"Generated sequence terminates unexpectedly early: {bad_strs}")
    seq_len = input_ids.shape[1]
    eos_mask = (input_ids == tokenizer.eos_token_id).float()
    seq_no_eos_mask = (eos_mask.sum(1) == 0).float()
    eos_indices = eos_mask.argmax(1)
    eos_indices = (eos_indices * (1 - seq_no_eos_mask) + seq_no_eos_mask * (seq_len - 1)).long()
    return eos_indices, seq_no_eos_mask


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
        # if x.dtype != torch.bool:
        #     return x.to(device=device)
        # else:
        #     # convert bool tensor to long tensor
        #     return x.to(dtype=torch.long, device=device)
        return x.to(device=device)
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
        t = []
        for v in dataclasses.asdict(self).values():
            t.append(to_tensor(v))
        t = [len(t)] + t
        return tuple(t)

    @classmethod
    def from_tuple(cls, t):
        x = cls()
        for i, f in enumerate(dataclasses.fields(x)):
            setattr(x, f.name, from_tensor(t[i + 1], f.type))
        return x


def tensor_data_list_to_tuple(tensor_data_list: List[TensorDataclassToTupleInterface]):
    res = []
    for tensor_data in tensor_data_list:
        res += list(tensor_data.to_tuple())
    return tuple(res)


def tuple_to_tensor_data_list(t: tuple):
    res = []
    i = 0
    while i < len(t):
        num_fields = t[i]
        res.append(TensorDataclassToTupleInterface.from_tuple(t[i:i + num_fields + 1]))
        i += num_fields + 1
    return res


@torch.jit.script
def upcast_masked_softmax(x: torch.Tensor, mask: torch.Tensor, mask_value: torch.Tensor, scale: float,
                          softmax_dtype: torch.dtype):
    input_dtype = x.dtype
    x = x.to(softmax_dtype) * scale
    x = torch.where(mask, x, mask_value)
    x = torch.nn.functional.softmax(x, dim=-1).to(input_dtype)
    return x


@torch.jit.script
def upcast_softmax(x: torch.Tensor, scale: float, softmax_dtype: torch.dtype):
    input_dtype = x.dtype
    x = x.to(softmax_dtype) * scale
    x = torch.nn.functional.softmax(x, dim=-1).to(input_dtype)
    return x


@torch.jit.script
def masked_softmax(x: torch.Tensor, mask: torch.Tensor, mask_value: torch.Tensor):
    x = torch.where(mask, x, mask_value)
    x = torch.nn.functional.softmax(x, dim=-1)
    return x


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (x[:, :, :, None, :].expand(bs, slen, n_kv_heads, n_rep,
                                       head_dim).reshape(bs, slen, n_kv_heads * n_rep, head_dim))


def mask_eos_token(
    logits: torch.Tensor,
    eos_token_id: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # for min_new_tokens
    if eos_token_id is not None:
        logits[..., eos_token_id] = torch.finfo(logits.dtype).min
    return logits


def build_packed_inputs(input_ids: torch.LongTensor,
                        attention_mask: torch.BoolTensor) -> Tuple[torch.LongTensor, torch.IntTensor, int]:
    bs, prompt_padded_len = input_ids.shape[:2]
    device = input_ids.device
    assert attention_mask.shape == input_ids.shape
    packed_input_ids = []
    input_lens = []
    for i in range(bs):
        if attention_mask is not None:
            start_idx = attention_mask[i].nonzero()[0][0]
            end_idx = prompt_padded_len - attention_mask[i].flip(0).nonzero()[0][0]
        else:
            start_idx, end_idx = 0, prompt_padded_len
        input_lens.append(end_idx - start_idx)
        packed_input_ids.append(input_ids[i, start_idx:end_idx])
    max_seq_len = int(max(input_lens))
    input_lens = torch.tensor(input_lens, dtype=torch.int, device=device)
    packed_input_ids = torch.cat(packed_input_ids, dim=0)
    cu_seqlens = torch.cat([torch.tensor([0], device=device), input_lens.cumsum(-1)]).int()
    return packed_input_ids, cu_seqlens, max_seq_len


def unpack_tensor(packed_x: torch.Tensor, cu_seqlens: torch.IntTensor, padding_side: str):
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    bs = cu_seqlens.shape[0] - 1
    max_seqlen = int(max(seqlens))
    unpacked_x = torch.zeros((bs, max_seqlen, *packed_x.shape[1:]),
                             dtype=packed_x.dtype,
                             device=packed_x.device)
    for i in range(bs):
        if padding_side == 'right':
            unpacked_x[i, :seqlens[i]] = packed_x[cu_seqlens[i]:cu_seqlens[i + 1]]
        elif padding_side == 'left':
            unpacked_x[i, max_seqlen - seqlens[i]:] = packed_x[cu_seqlens[i]:cu_seqlens[i + 1]]
        else:
            raise NotImplementedError()
    return unpacked_x


def gather_shifted_log_probs(logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
    logits = logits[:, :-1]
    labels = labels[:, 1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def gather_packed_shifted_log_probs(logits_: torch.FloatTensor, cu_seqlens: torch.Tensor,
                                    labels_: torch.LongTensor) -> torch.FloatTensor:
    leave_one_indices = torch.cat([
        torch.arange(cu_seqlens[i], cu_seqlens[i + 1] - 1, dtype=torch.long, device=cu_seqlens.device)
        for i in range(cu_seqlens.shape[0] - 1)
    ])
    shift_one_indices = torch.cat([
        torch.arange(cu_seqlens[i] + 1, cu_seqlens[i + 1], dtype=torch.long, device=cu_seqlens.device)
        for i in range(cu_seqlens.shape[0] - 1)
    ])
    logits = logits_[leave_one_indices]
    labels = labels_[shift_one_indices]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    assert (log_probs_labels.shape[0] == logits_.shape[0] - cu_seqlens.shape[0] + 1), (
        log_probs_labels.shape,
        logits_.shape,
        cu_seqlens.shape,
        cu_seqlens,
        shift_one_indices,
    )
    return log_probs_labels
