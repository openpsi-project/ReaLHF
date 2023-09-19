# a terrible greedy generate
import api.model
from typing import Callable, List, Optional, Union, Tuple
import transformers
import dataclasses
import itertools

import torch
import torch.nn.functional as F

from .spec import TransformerData
from impl.model.nn.flash_mqat import PipeCacheData, PipeTransferData, FlashMQATConfig
from impl.model.utils.logits_warper import chained_logits_wraper, TopKLogitsWarper, TopPLogitsWarper, unioned_logits_wraper


@dataclasses.dataclass
class GenerationConfig:
    # generation arguments other than tensors
    min_new_tokens: int = 1
    max_new_tokens: int = 10
    temperature: float = 1.0
    greedy: bool = True
    top_p: float = 1.0
    top_k: int = 0
    num_samples: int = 1


def unioned_top_k_top_p(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
    return unioned_logits_wraper([
        TopKLogitsWarper(top_k=top_k, filter_value=filter_value),
        TopPLogitsWarper(top_p=top_p, filter_value=filter_value)
    ],
                                 filter_value=filter_value)(
                                     None,
                                     logits,
                                     change_logits=True,
                                     change_logits_inplace=True,
                                     change_mask=True,
                                     change_mask_inplace=True,
                                     mask=torch.ones_like(logits),
                                 )


def mask_eos_token(logits, eos_token_id=None):
    # for min_new_tokens
    if eos_token_id is not None:
        logits[..., eos_token_id] = -float("inf")
    return logits


def genstep(
    next_token_logits: torch.Tensor,
    tokenizer: transformers.PreTrainedTokenizerFast,
    unfinished_sequences: torch.Tensor,
    generated_idx: int,
    gconfig: GenerationConfig,
):
    if generated_idx < gconfig.min_new_tokens:
        next_token_logits = mask_eos_token(next_token_logits, eos_token_id=tokenizer.eos_token_id)

    if gconfig.greedy:
        max_iv = next_token_logits.max(-1)
        next_tokens = max_iv.indices
        selected_logits = max_iv.values
        logits_mask = torch.ones_like(next_token_logits)
    else:
        next_token_logits = next_token_logits.float()
        next_token_logits /= gconfig.temperature
        next_token_logits, logits_mask = unioned_top_k_top_p(next_token_logits,
                                                             top_k=gconfig.top_k,
                                                             top_p=gconfig.top_p)
        next_tokens = torch.distributions.Categorical(logits=next_token_logits).sample()
        selected_logits = torch.gather(next_token_logits, -1, next_tokens.unsqueeze(-1)).squeeze(-1)

    if tokenizer.eos_token_id is not None:
        if tokenizer.pad_token_id is None:
            raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
        next_tokens = next_tokens * unfinished_sequences + tokenizer.pad_token_id * (1 - unfinished_sequences)
    unfinished_sequences = next_tokens.ne(tokenizer.eos_token_id).long() * unfinished_sequences

    # terminate check
    terminate = (generated_idx >= gconfig.max_new_tokens - 1) or (unfinished_sequences.max() == 0)

    return next_tokens, selected_logits, logits_mask, terminate, unfinished_sequences


@torch.no_grad()
def generate(
    model: api.model.NeuralNetwork,
    tokenizer: transformers.PreTrainedTokenizerFast,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    k_caches: Optional[List[torch.Tensor]] = None,
    v_caches: Optional[List[torch.Tensor]] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    gconfig: GenerationConfig = dataclasses.field(default_factory=GenerationConfig),
) -> Tuple[torch.Tensor, torch.Tensor]:
    if (k_caches is None) != (v_caches is None) or (k_caches is None) != (cache_seqlens is None):
        raise ValueError("k_cache, v_cache, cache_seqlens must be all None or all not None")
    device = input_ids.device
    mconfig: FlashMQATConfig = model.config
    bs, prompt_padded_len = input_ids.shape[:2]

    terminate = False
    generated_idx = 0
    unfinished_sequences = torch.ones(bs, dtype=torch.long, device=device)

    gen_token_ph = []
    gen_logits_ph = []
    gen_logits_mask_ph = []

    # Prepare inputs for generation iterations
    if k_caches is None:
        # Generate from scratch.
        # Input_ids may have different lengths, we should first pack them into a large batch
        # to use varlen flash attention, then record kv caches for the following inferences.
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
        print("<<", packed_input_ids, cu_seqlens)

        x = PipeTransferData(cu_seqlens=cu_seqlens, max_seqlen=max_seq_len)
        # one embedding layer, n_layers transformer block, one output layer
        ys = [PipeCacheData(input_ids=packed_input_ids)
              ] + [PipeCacheData() for _ in range(mconfig.n_layers + 1)]
        # Model forward will set k/v cache in PipeCacheData.
        logits = model(x, ys).pp_output
        print(logits[..., 0], logits[cu_seqlens[1:] - 1][..., 0])
        logits = logits[cu_seqlens[1:] - 1]
        for y in ys[1:-1]:
            assert y.k_cache is not None and y.v_cache is not None and y.cache_seqlens is not None
            k_cache = torch.zeros((bs, max_seq_len + gconfig.max_new_tokens, *y.k_cache.shape[1:]),
                                  dtype=y.k_cache.dtype,
                                  device=device)
            v_cache = torch.zeros((bs, max_seq_len + gconfig.max_new_tokens, *y.v_cache.shape[1:]),
                                  dtype=y.v_cache.dtype,
                                  device=device)
            for i in range(bs):
                k_cache[i, :input_lens[i]] = y.k_cache[cu_seqlens[i]:cu_seqlens[i + 1]]
                v_cache[i, :input_lens[i]] = y.v_cache[cu_seqlens[i]:cu_seqlens[i + 1]]
            y.k_cache = k_cache
            y.v_cache = v_cache
            y.cache_seqlens = input_lens.clone()
        x = PipeTransferData()
        ys[0].cache_seqlens = input_lens.clone()
        # Next, we will generate the next token after prompts.
        # cache_seqlens is exactly the lengths of prompts.
        next_tokens, selected_logits, logits_mask, terminate, unfinished_sequences = genstep(
            logits, tokenizer, unfinished_sequences, generated_idx, gconfig)
        gen_logits_ph.append(selected_logits)
        gen_token_ph.append(next_tokens)
        gen_logits_mask_ph.append(logits_mask)
        generated_idx += 1
    else:
        # Resume from a previous generation state.
        if prompt_padded_len != 1:
            raise ValueError("prompt_padded_len must be 1 when resuming from a previous generation state.")
        x = PipeTransferData()
        ys = [PipeCacheData(input_ids=input_ids, cache_seqlens=cache_seqlens.clone())] + [
            PipeCacheData(k_cache=k, v_cache=v, cache_seqlens=cache_seqlens.clone())
            for k, v in zip(k_caches, v_caches)
        ] + [PipeCacheData()]
        next_tokens = input_ids[:, -1]

    # The main loop.
    while not terminate:
        # the next round of inference
        ys[0].input_ids = next_tokens.unsqueeze(-1)  # [bs, 1]
        ys[0].position_ids = None
        # K/v cache will be changed in-place with flash attention.
        logits = model(x, ys).pp_output.squeeze()
        for yidx, y in enumerate(ys[:-1]):
            y.cache_seqlens += 1
            ###################################################
            if yidx == 0:
                continue
            for i in range(bs):
                assert (y.k_cache[i, :y.cache_seqlens[i]]
                        != 0).all(), (y.k_cache[i, ..., 0], y.cache_seqlens[i], yidx)
                assert (y.v_cache[i, :y.cache_seqlens[i]] != 0).all()
                assert (y.k_cache[i, y.cache_seqlens[i]:] == 0).all()
                assert (y.v_cache[i, y.cache_seqlens[i]:] == 0).all()
            ###################################################

        next_tokens, selected_logits, logits_mask, terminate, unfinished_sequences = genstep(
            logits, tokenizer, unfinished_sequences, generated_idx, gconfig)
        gen_logits_ph.append(selected_logits)
        gen_token_ph.append(next_tokens)
        gen_logits_mask_ph.append(logits_mask)
        generated_idx += 1

    return torch.stack(gen_token_ph, -1), torch.stack(gen_logits_ph, -2), torch.stack(gen_logits_mask_ph, -2)


def build_packed_inputs(input_ids: torch.LongTensor, attention_mask: torch.BoolTensor,
                        device: torch.device) -> Tuple[torch.LongTensor, torch.IntTensor, int]:
    bs, prompt_padded_len = input_ids.shape[:2]
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


@torch.no_grad()
def vanilla_packed_generate(
    model: api.model.NeuralNetwork,
    tokenizer: transformers.PreTrainedTokenizerFast,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    gconfig: GenerationConfig = dataclasses.field(default_factory=GenerationConfig),
) -> Tuple[torch.Tensor, torch.Tensor]:
    mconfig: FlashMQATConfig = model.config

    terminate = False
    generated_idx = 0
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    gen_token_ph = []
    gen_logits_ph = []
    gen_logits_mask_ph = []

    # The main loop.
    while not terminate:
        packed_input_ids, cu_seqlens, max_seq_len = build_packed_inputs(input_ids, attention_mask,
                                                                        input_ids.device)
        x = PipeTransferData(cu_seqlens=cu_seqlens, max_seqlen=max_seq_len)
        # one embedding layer, n_layers transformer block, one output layer
        ys = [PipeCacheData(input_ids=packed_input_ids)
              ] + [PipeCacheData() for _ in range(mconfig.n_layers + 1)]
        # Model forward will set k/v cache in PipeCacheData.
        logits = model(x, ys).pp_output
        logits = logits[cu_seqlens[1:] - 1]
        # Next, we will generate the next token after prompts.
        # cache_seqlens is exactly the lengths of prompts.
        next_tokens, selected_logits, logits_mask, terminate, unfinished_sequences = genstep(
            logits, tokenizer, unfinished_sequences, generated_idx, gconfig)
        gen_logits_ph.append(selected_logits)
        gen_token_ph.append(next_tokens)
        gen_logits_mask_ph.append(logits_mask)
        generated_idx += 1

        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], 1)
        am = torch.logical_and(
            next_tokens.unsqueeze(-1).not_equal(tokenizer.eos_token_id),
            next_tokens.unsqueeze(-1).not_equal(tokenizer.pad_token_id))
        attention_mask = torch.cat([attention_mask, am], 1)

    return torch.stack(gen_token_ph, -1), torch.stack(gen_logits_ph, -1), torch.stack(gen_logits_mask_ph, -2)


@torch.no_grad()
def vanilla_cpu_generate(
    model: api.model.NeuralNetwork,
    tokenizer: transformers.PreTrainedTokenizerFast,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    gconfig: GenerationConfig = dataclasses.field(default_factory=GenerationConfig),
) -> Tuple[torch.Tensor, torch.Tensor]:
    mconfig: FlashMQATConfig = model.config
    assert str(input_ids.device) == 'cpu'

    terminate = False
    generated_idx = 0
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    gen_token_ph = []
    gen_logits_ph = []
    gen_logits_mask_ph = []

    # The main loop.
    while not terminate:
        x = PipeTransferData(attention_mask=attention_mask)
        # one embedding layer, n_layers transformer block, one output layer
        ys = [PipeCacheData(input_ids=input_ids)] + [PipeCacheData() for _ in range(mconfig.n_layers + 1)]
        # Model forward will set k/v cache in PipeCacheData.
        logits = model(x, ys).pp_output[:, -1, :]
        # Next, we will generate the next token after prompts.
        # cache_seqlens is exactly the lengths of prompts.
        next_tokens, selected_logits, logits_mask, terminate, unfinished_sequences = genstep(
            logits, tokenizer, unfinished_sequences, generated_idx, gconfig)
        gen_logits_ph.append(selected_logits)
        gen_token_ph.append(next_tokens)
        gen_logits_mask_ph.append(logits_mask)
        generated_idx += 1

        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], 1)
        am = torch.logical_and(
            next_tokens.unsqueeze(-1).not_equal(tokenizer.eos_token_id),
            next_tokens.unsqueeze(-1).not_equal(tokenizer.pad_token_id))
        attention_mask = torch.cat([attention_mask, am], 1)

    return torch.stack(gen_token_ph, -1), torch.stack(gen_logits_ph, -1), torch.stack(gen_logits_mask_ph, -2)