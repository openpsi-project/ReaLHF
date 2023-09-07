# a terrible greedy generate

from typing import Callable, List, Optional, Union
import dataclasses
import itertools

import torch
import torch.nn.functional as F

from .spec import TransformerConfig, TransformerData


@dataclasses.dataclass
class GenerationConfig:
    # generation arguments other than tensors
    min_new_tokens: int = 1
    max_new_tokens: int = 10
    min_tokens: int = None
    max_tokens: int = None
    temperature: float = 1.0
    greedy: bool = True
    top_p: float = 1.0
    top_k: int = 0
    num_samples: int = 1
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None


def mask_eos_token(logits, eos_token_id=None):
    # for min_new_tokens
    if eos_token_id is not None:
        logits[..., eos_token_id] = -float("inf")
    return logits


def top_k_logits(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
    """ This function has been mostly taken from huggingface conversational
     ai code at
         https://medium.com/huggingface/how-to-build-a-state-of-the-art-
              conversational-ai-with-transfer-learning-2d818ac26313 """
    # input logits should be a tensor of shape [batch_size, seq_len, vocab_size]
    if top_k > 0:
        # Remove all tokens with a probability less than the
        # last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1, dtype=torch.float32), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] \
            = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        for i, j in itertools.product(range(sorted_indices.size(0)), range(sorted_indices.size(1))):
            indices_to_remove = sorted_indices[i, j, sorted_indices_to_remove[i][j]]
            logits[i, j, indices_to_remove] = filter_value

    return logits


def preprocess_input(x: TransformerData, last_token_only):
    # x = TransformerData()
    position_ids = x.raw_attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(x.raw_attention_mask == 0, 1)

    if last_token_only:
        x.input_ids = x.raw_input_ids[:, -1:]
        x.position_ids = position_ids[:, -1].unsqueeze(-1)
    else:
        x.input_ids = x.raw_input_ids
        x.position_ids = position_ids

    x.attention_mask = x.raw_attention_mask
    return x


def postprocess_output(
    x: TransformerData,
    unfinished_sequences: torch.Tensor,
    args: GenerationConfig,
):
    logits = x.logits
    orig_input_ids = x.raw_input_ids
    orig_attn_mask = x.raw_attention_mask
    cur_seq_len = orig_input_ids.shape[-1]

    # process logits
    next_token_logits = logits[:, -1, :]
    if cur_seq_len < args.min_tokens:
        next_token_logits = mask_eos_token(next_token_logits, args.eos_token_id)

    if args.greedy:
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        print("greedy next tokens", next_tokens, next_tokens.shape)
    else:
        next_token_logits = next_token_logits.float()
        next_token_logits /= args.temperature
        next_token_logits = top_k_logits(next_token_logits, top_k=args.top_k, top_p=args.top_p)
        log_probs = F.softmax(next_token_logits, dim=-1)
        next_tokens = torch.multinomial(log_probs, num_samples=1)
        next_tokens = next_tokens.reshape(-1)

    if args.eos_token_id is not None:
        if args.pad_token_id is None:
            raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
        next_tokens = next_tokens * unfinished_sequences + args.pad_token_id * (1 - unfinished_sequences)
    unfinished_sequences = next_tokens.ne(args.eos_token_id).long() * unfinished_sequences
    x.raw_input_ids = torch.cat([orig_input_ids, next_tokens[:, None]], dim=-1)
    x.raw_attention_mask = torch.cat(
        [orig_attn_mask, orig_attn_mask.new_ones((orig_attn_mask.shape[0], 1))], dim=-1)

    # terminate check
    terminate = (cur_seq_len >= args.max_tokens) or (unfinished_sequences.max() == 0)

    return terminate, unfinished_sequences, x


@torch.no_grad()
def generate(
        model,  # could be a deepspeed pipeline engine 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor = None,
        **kwargs):
    args = GenerationConfig(**kwargs)
    device = input_ids.device
    input_length = input_ids.shape[-1]
    # print(args.min_new_tokens, input_length)
    if args.min_tokens is None:
        args.min_tokens = args.min_new_tokens + input_length
        # print(args.min_tokens)
    if args.max_tokens is None:
        args.max_tokens = args.max_new_tokens + input_length

    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=device)
    first_round = True
    x = TransformerData(raw_input_ids=input_ids, raw_attention_mask=attention_mask)

    while True:
        x = preprocess_input(x, last_token_only=not first_round)
        first_round = False
        x = model(x)
        terminate, unfinished_sequences, x = \
                postprocess_output(x, unfinished_sequences, args)

        if terminate:
            break

    return x.raw_input_ids
