from typing import Callable, Dict, List, Optional, Tuple, Union
import dataclasses
import queue

import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers

from impl.model.nn.flash_mqat.flash_mqat_base import FlashMQATConfig, FlashMQATModel
from impl.model.utils.data import PipeCacheData, PipeTransferData
from impl.model.utils.functional import mask_eos_token
from impl.model.utils.logits_warper import top_k_top_p_logits
import base.constants
import base.logging as logging

try:
    from flash_attn.bert_padding import index_first_axis, unpad_input
except ModuleNotFoundError:
    pass
import base.logging as logging

logger = logging.getLogger("FlashMQAT Generation")


@dataclasses.dataclass
class GenerationConfig:
    min_new_tokens: int = 1
    max_new_tokens: int = 10
    temperature: float = 1.0
    greedy: bool = True
    top_p: float = 1.0
    top_k: int = 0
    num_samples: int = 1


def genstep(
    next_token_logits: torch.Tensor,
    tokenizer: transformers.PreTrainedTokenizerFast,
    unfinished_sequences: torch.Tensor,
    generated_idx: Union[torch.IntTensor, int],
    gconfig: GenerationConfig,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], bool, torch.Tensor]:
    """Advance generation by one step given logits.

    Args:
        next_token_logits (torch.Tensor): Shape [bs, vocab_size].
        tokenizer (transformers.PreTrainedTokenizerFast): .
        unfinished_sequences (torch.Tensor): Bool tensor indicator of whether a sequence is finished.
            Shape [bs].
        generated_idx (int): The token index to be generated.
        gconfig (GenerationConfig): .

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, torch.Tensor]:
        A tuple of
            next_tokens: Shape [bs].
            logprob: The log probability of selected tokens. May be re-normalized
                according to the mask machanism. Shape [bs].
            logits_mask: The mask of logits. Shape [bs, vocab_size].
            terminate: Whether the generation should be terminated.
            unfinished_sequences: Bool tensor indicator of whether a sequence is finished.
                Shape [bs].
    """
    if base.constants.model_parallel_world_size() > 1:
        from impl.model.parallelism.model_parallel.mappings import gather_from_tensor_model_parallel_region
        next_token_logits = gather_from_tensor_model_parallel_region(next_token_logits)

    unfinished_sequences = unfinished_sequences.bool()
    next_token_logits = next_token_logits.float()
    if isinstance(generated_idx, int):
        if generated_idx < gconfig.min_new_tokens:
            next_token_logits = mask_eos_token(next_token_logits, eos_token_id=tokenizer.eos_token_id)
    else:
        assert isinstance(generated_idx, torch.Tensor)
        if (generated_idx < gconfig.min_new_tokens).any():
            _batch_indices = (generated_idx < gconfig.min_new_tokens).unsqueeze(1)
            _vocab_indices = _batch_indices.new_zeros((1, next_token_logits.shape[1]))
            if tokenizer.eos_token_id is not None:
                _vocab_indices[:, tokenizer.eos_token_id] = 1
            next_token_logits.masked_fill_(_batch_indices * _vocab_indices,
                                           torch.finfo(next_token_logits.dtype).min)

    if not gconfig.greedy:
        next_token_logits /= gconfig.temperature
        next_token_logits = top_k_top_p_logits(
            next_token_logits,
            top_k=gconfig.top_k,
            top_p=gconfig.top_p,
            inplace=True,
            ordered=False,
        )

    distrb = torch.distributions.Categorical(logits=next_token_logits)
    next_tokens = distrb.mode if gconfig.greedy else distrb.sample()
    logprob = distrb.log_prob(next_tokens)

    if base.constants.model_parallel_world_size() > 1:
        if base.constants.model_parallel_rank() > 0:
            logprob[:] = 0
            next_tokens[:] = 0
        handle = torch.distributed.all_reduce(logprob,
                                              torch.distributed.ReduceOp.SUM,
                                              async_op=True,
                                              group=base.constants.model_parallel_group())
        torch.distributed.all_reduce(next_tokens,
                                     torch.distributed.ReduceOp.SUM,
                                     group=base.constants.model_parallel_group())

    if tokenizer.eos_token_id is not None:
        if tokenizer.pad_token_id is None:
            raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
        next_tokens.masked_fill_(unfinished_sequences.logical_not(), tokenizer.pad_token_id)
        # next_tokens = next_tokens * unfinished_sequences + tokenizer.pad_token_id * (1 - unfinished_sequences)
    # unfinished_sequences = next_tokens.ne(tokenizer.eos_token_id).long() * unfinished_sequences
    unfinished_sequences.logical_and_(next_tokens.ne(tokenizer.eos_token_id))

    # terminate check
    if isinstance(generated_idx, int):
        terminate = (generated_idx >= gconfig.max_new_tokens - 1) or (unfinished_sequences.max() == 0)
    else:
        unfinished_sequences.logical_and_(generated_idx < gconfig.max_new_tokens - 1)
        terminate = unfinished_sequences.max() == 0

    logits_mask = next_token_logits != torch.finfo(next_token_logits.dtype).min
    if logits_mask.all():
        logits_mask = None

    if base.constants.model_parallel_world_size() > 1:
        handle.wait()

    return next_tokens, logprob, logits_mask, terminate, unfinished_sequences


@torch.no_grad()
def generate(
    model: FlashMQATModel,
    tokenizer: transformers.PreTrainedTokenizerFast,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    k_caches: Optional[List[torch.Tensor]] = None,
    v_caches: Optional[List[torch.Tensor]] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    gconfig: GenerationConfig = dataclasses.field(default_factory=GenerationConfig),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[PipeCacheData], Optional[torch.Tensor]]:
    """Generete a sequence with a FlashMQAT.

    Args:
        model (FlashMQATModel): .
        tokenizer (transformers.PreTrainedTokenizerFast): .
        input_ids (torch.Tensor): Prompts, may be padded. Shape [bs, seqlen].
        attention_mask (Optional[torch.Tensor], optional): The same as huggingface.
            Shape [bs, seqlen]. If None, generate attention mask according to
            pad_token_id and eos_token_id. Defaults to None.
        k_caches (Optional[List[torch.Tensor]], optional): List of k_caches.
            Length equals to the number of transformer layers.
            Each tensor in the list has shape [bs, max_seqlen, #kv, head_dim].
            Used for resuming a previous generation state.
            If None, generate from scratch. Defaults to None.
        v_caches (Optional[List[torch.Tensor]], optional): List of v_caches.
            Length equals to the number of transformer layers.
            Each tensor in the list has shape [bs, max_seqlen, #kv, head_dim].
            Used for resuming a previous generation state.
            If None, generate from scratch. Defaults to None.
        cache_seqlens (Optional[torch.Tensor], optional): Shape [bs].
            Used for resuming a previous generation state. Defaults to None.
        gconfig (GenerationConfig, optional): .

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[PipeCacheData]]:
        The tuple of
            gen_tokens: Generated tokens. Shape [bs * num_samples, #new_tokens].
            log_probs: Log probabilities of generated tokens. Shape [bs * num_samples, #new_tokens].
            mask: The mask of logits. None if no mask otherwise a tensor of
                shape [bs * num_samples, #new_tokens, vocab_size].
                1 if the logits is valid else 0, e.g., should be used as
                `logits.masked_fill_(mask.logical_not(), -1e10)`.
            ys: List of PipeCacheData. Length equals to the number of transformer layers.
                Can be saved for continuing generation.
            prompt_logits: Output logits of prompts. None if k/v caches are passed in.
                Shape [#tot_prompt_tokens * num_samples].
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    if (k_caches is None) != (v_caches is None) or (k_caches is None) != (cache_seqlens is None):
        raise ValueError("k_cache, v_cache, cache_seqlens must be all None or all not None")
    if gconfig.num_samples > 1 and k_caches is None:
        input_ids = input_ids.unsqueeze(1).repeat(1, gconfig.num_samples, 1).flatten(end_dim=1)
        attention_mask = attention_mask.unsqueeze(1).repeat(1, gconfig.num_samples, 1).flatten(end_dim=1)
    elif k_caches is not None:
        for k_cache, v_cache in zip(k_caches, v_caches):
            assert (k_cache.shape[0] == v_cache.shape[0] == input_ids.shape[0] == attention_mask.shape[0] ==
                    cache_seqlens.shape[0])

    device = input_ids.device
    mconfig: FlashMQATConfig = model.config
    bs, prompt_padded_len = input_ids.shape[:2]

    terminate = False
    generated_idx = 0
    unfinished_sequences = torch.ones(bs, dtype=torch.long, device=device)

    gen_token_ph = []
    gen_logprob_ph = []
    gen_logits_mask_ph = []

    prompt_logits = None
    # Prepare inputs for generation iterations
    if k_caches is None:
        # Generate from scratch.
        # Input_ids may have different lengths, we should first pack them into a large batch
        # to use varlen flash attention, then record kv caches for the following inferences.
        packed_input_ids, _, cu_seqlens, max_seq_len = unpad_input(input_ids, attention_mask)
        input_lens = cu_seqlens[1:] - cu_seqlens[:-1]

        x = PipeTransferData(cu_seqlens=cu_seqlens, max_seqlen=max_seq_len, store_kv_cache=True)
        # one embedding layer, n_layers transformer block, one output layer
        ys = [PipeCacheData(input_ids=packed_input_ids)
              ] + [PipeCacheData() for _ in range(mconfig.n_layers + 1)]
        # Model forward will set k/v cache in PipeCacheData.
        prompt_logits = model(x, ys).pp_output
        logits = prompt_logits[cu_seqlens[1:] - 1]
        cache_seqlens = input_lens.clone().to(dtype=torch.int32)
        for y in ys[1:-1]:
            assert y.k_cache is not None and y.v_cache is not None and y.cache_seqlens is not None
            kvcache_seqlen = max(max_seq_len + gconfig.max_new_tokens,
                                 mconfig.hidden_dim // mconfig.head_dim + 10)
            # fix of a flash attention bug
            k_cache = torch.zeros((bs, kvcache_seqlen, *y.k_cache.shape[1:]),
                                  dtype=y.k_cache.dtype,
                                  device=device)
            v_cache = torch.zeros((bs, kvcache_seqlen, *y.v_cache.shape[1:]),
                                  dtype=y.v_cache.dtype,
                                  device=device)
            for i in range(bs):
                k_cache[i, :input_lens[i]] = y.k_cache[cu_seqlens[i]:cu_seqlens[i + 1]]
                v_cache[i, :input_lens[i]] = y.v_cache[cu_seqlens[i]:cu_seqlens[i + 1]]
            y.k_cache = k_cache
            y.v_cache = v_cache
            y.cache_seqlens = cache_seqlens
        x = PipeTransferData(store_kv_cache=True)
        ys[0].cache_seqlens = cache_seqlens
        # Next, we will generate the next token after prompts.
        # cache_seqlens is exactly the lengths of prompts.
        next_tokens, logprob, logits_mask, terminate, unfinished_sequences = genstep(
            logits, tokenizer, unfinished_sequences, generated_idx, gconfig)
        gen_token_ph.append(next_tokens)
        gen_logprob_ph.append(logprob)
        gen_logits_mask_ph.append(logits_mask)
        generated_idx += 1
    else:
        # Resume from a previous generation state.
        if prompt_padded_len != 1:
            raise ValueError("prompt_padded_len must be 1 when resuming from a previous generation state.")
        max_seq_len = gconfig.max_new_tokens + int(max(cache_seqlens)) + 1
        for i in range(len(k_caches)):
            pad = (0, 0, 0, 0, 0, max_seq_len - k_caches[i].shape[1])
            if k_caches[i].shape[1] < max_seq_len:
                k_caches[i] = nn.functional.pad(k_caches[i], pad)
            if v_caches[i].shape[1] < max_seq_len:
                v_caches[i] = nn.functional.pad(v_caches[i], pad)
        x = PipeTransferData(store_kv_cache=torch.tensor(1))
        ys = ([PipeCacheData(cache_seqlens=cache_seqlens)] + [
            PipeCacheData(k_cache=k, v_cache=v, cache_seqlens=cache_seqlens)
            for k, v in zip(k_caches, v_caches)
        ] + [PipeCacheData()])
        next_tokens = input_ids[:, -1]

    # The main loop.
    while not terminate:
        # the next round of inference
        ys[0].input_ids = next_tokens.unsqueeze(-1)  # [bs, 1], seqlen=1
        ys[0].position_ids = None
        # K/v cache will be changed in-place with flash attention.
        logits = model(x, ys).pp_output.squeeze(dim=1)
        cache_seqlens += 1  # The global handle. This will increase all handles in ys by 1.

        next_tokens, logprob, logits_mask, terminate, unfinished_sequences = genstep(
            logits, tokenizer, unfinished_sequences, generated_idx, gconfig)
        gen_token_ph.append(next_tokens)
        gen_logprob_ph.append(logprob)
        gen_logits_mask_ph.append(logits_mask)
        generated_idx += 1

    gen_tokens = torch.stack(gen_token_ph, -1)
    log_probs = torch.stack(gen_logprob_ph, -1)
    if all([m is None for m in gen_logits_mask_ph]):
        logits_mask = None
    else:
        mm = next(m for m in gen_logits_mask_ph if m is not None)
        gen_logits_mask_ph = [torch.ones_like(mm) if m is None else m for m in gen_logits_mask_ph]
        logits_mask = torch.stack(gen_logits_mask_ph, -2)

    return gen_tokens, log_probs, logits_mask, ys[1:-1], prompt_logits


@torch.no_grad()
def vanilla_packed_generate(
    model: FlashMQATModel,
    tokenizer: transformers.PreTrainedTokenizerFast,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    gconfig: GenerationConfig = dataclasses.field(default_factory=GenerationConfig),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Only used for debugging."""
    mconfig: FlashMQATConfig = model.config

    terminate = False
    generated_idx = 0
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    gen_token_ph = []
    gen_logprob_ph = []
    gen_logits_mask_ph = []

    # The main loop.
    while not terminate:
        packed_input_ids, _, cu_seqlens, max_seq_len = unpad_input(input_ids, attention_mask)
        x = PipeTransferData(cu_seqlens=cu_seqlens, max_seqlen=max_seq_len)
        # one embedding layer, n_layers transformer block, one output layer
        ys = [PipeCacheData(input_ids=packed_input_ids)
              ] + [PipeCacheData() for _ in range(mconfig.n_layers + 1)]
        # Model forward will set k/v cache in PipeCacheData.
        logits = model(x, ys).pp_output
        logits = logits[cu_seqlens[1:] - 1]
        # Next, we will generate the next token after prompts.
        # cache_seqlens is exactly the lengths of prompts.
        next_tokens, logprob, logits_mask, terminate, unfinished_sequences = genstep(
            logits, tokenizer, unfinished_sequences, generated_idx, gconfig)
        gen_token_ph.append(next_tokens)
        gen_logprob_ph.append(logprob)
        gen_logits_mask_ph.append(logits_mask)
        generated_idx += 1

        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], 1)
        am = torch.logical_and(
            next_tokens.unsqueeze(-1).not_equal(tokenizer.eos_token_id),
            next_tokens.unsqueeze(-1).not_equal(tokenizer.pad_token_id),
        )
        attention_mask = torch.cat([attention_mask, am], 1)

    gen_tokens = torch.stack(gen_token_ph, -1)
    log_probs = torch.stack(gen_logprob_ph, -1)
    if all([m is None for m in gen_logits_mask_ph]):
        logits_mask = None
    else:
        mm = next(m for m in gen_logits_mask_ph if m is not None)
        gen_logits_mask_ph = [torch.ones_like(mm) if m is None else m for m in gen_logits_mask_ph]
        logits_mask = torch.stack(gen_logits_mask_ph, -2)

    return gen_tokens, log_probs, logits_mask


@torch.no_grad()
def vanilla_cpu_generate(
    model: FlashMQATModel,
    tokenizer: transformers.PreTrainedTokenizerFast,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    gconfig: GenerationConfig = dataclasses.field(default_factory=GenerationConfig),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Only used for debugging."""
    mconfig: FlashMQATConfig = model.config
    assert str(input_ids.device) == "cpu"

    terminate = False
    generated_idx = 0
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    gen_token_ph = []
    gen_logprob_ph = []
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
        next_tokens, logprob, logits_mask, terminate, unfinished_sequences = genstep(
            logits, tokenizer, unfinished_sequences, generated_idx, gconfig)
        gen_token_ph.append(next_tokens)
        gen_logprob_ph.append(logprob)
        gen_logits_mask_ph.append(logits_mask)
        generated_idx += 1

        input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], 1)
        am = torch.logical_and(
            next_tokens.unsqueeze(-1).not_equal(tokenizer.eos_token_id),
            next_tokens.unsqueeze(-1).not_equal(tokenizer.pad_token_id),
        )
        attention_mask = torch.cat([attention_mask, am], 1)

    gen_tokens = torch.stack(gen_token_ph, -1)
    log_probs = torch.stack(gen_logprob_ph, -1)
    if all([m is None for m in gen_logits_mask_ph]):
        logits_mask = None
    else:
        mm = next(m for m in gen_logits_mask_ph if m is not None)
        gen_logits_mask_ph = [torch.ones_like(mm) if m is None else m for m in gen_logits_mask_ph]
        logits_mask = torch.stack(gen_logits_mask_ph, -2)

    return gen_tokens, log_probs, logits_mask


class InflightBatchingGenerator:

    def __init__(
        self,
        inqueue: queue.Queue,
        outqueue: queue.Queue,
        model: FlashMQATModel,
        tokenizer: transformers.PreTrainedTokenizerFast,
        gconfig: GenerationConfig,
        batch_size: int,
        max_prompt_len: int,
    ):
        self.inqueue = inqueue
        self.outqueue = outqueue

        self.model = model
        self.mconfig = mconfig = model.config
        self.tokenizer = tokenizer

        self.gconfig = gconfig
        self.batch_size = batch_size
        self.max_prompt_len = max_prompt_len

        kvcache_seqlen = max(max_prompt_len + gconfig.max_new_tokens,
                             mconfig.hidden_dim // mconfig.head_dim + 10)
        _p = next(self.model.parameters())
        dtype, device = _p.dtype, _p.device

        # Cache
        self.k_caches = [
            torch.zeros(
                (
                    batch_size,
                    kvcache_seqlen,
                    mconfig.n_kv_heads,
                    mconfig.head_dim,
                ),
                dtype=dtype,
                device=device,
            ) for _ in range(self.mconfig.n_layers)
        ]
        self.v_caches = [
            torch.zeros(
                (
                    batch_size,
                    kvcache_seqlen,
                    mconfig.n_kv_heads,
                    mconfig.head_dim,
                ),
                dtype=dtype,
                device=device,
            ) for _ in range(self.mconfig.n_layers)
        ]
        self.cache_seqlens = torch.zeros((batch_size,), dtype=torch.int32, device=device)

        # Input buffers
        self.input_buf = torch.zeros((batch_size, max_prompt_len), dtype=torch.long, device=device)
        self.input_buf_lens = torch.zeros((batch_size,), dtype=torch.int32, device=device)

        # Save prompts for output
        self.prompt_tokens = [None for _ in range(batch_size)]

        # Generation state
        self.generate_idx = torch.zeros((batch_size,), dtype=torch.int32, device=device)
        self.unfinished_sequences = torch.zeros((batch_size,), dtype=torch.float32, device=device)

        self.ys = ([PipeCacheData(cache_seqlens=self.cache_seqlens,)] + [
            PipeCacheData(k_cache=k, v_cache=v, cache_seqlens=self.cache_seqlens)
            for k, v in zip(self.k_caches, self.v_caches)
        ] + [PipeCacheData()])

        # output buffers
        self.output_tokens_buf = [[] for _ in range(batch_size)]
        self.output_logprob_buf = [[] for _ in range(batch_size)]
        self.output_logits_mask = [[] for _ in range(batch_size)]

    def _get_non_eos_logits(self) -> torch.FloatTensor:
        self.ys[0].position_ids = None
        self.ys[0].input_ids = self.input_buf[:, :1]
        logits = self.model(PipeTransferData(), self.ys).pp_output.squeeze(dim=1)

        self.cache_seqlens += 1
        return logits.float()

    def _get_inflight_logits(self) -> torch.FloatTensor:
        finished_sequences = self.unfinished_sequences.logical_not()
        assert finished_sequences.any()

        finish_indices = finished_sequences.nonzero().squeeze(-1).tolist()

        # pop out finished sequences and clear corresponding buffers
        for i in finish_indices:
            prompt_tokens = self.prompt_tokens[i]

            # Used to skip the first call.
            if prompt_tokens is not None:
                gen_tokens = torch.stack(self.output_tokens_buf[i])
                gen_logp = torch.stack(self.output_logprob_buf[i])
                if all([m is None for m in self.output_logits_mask[i]]):
                    gen_logits_mask = None
                else:
                    mm = next(m for m in self.output_logits_mask[i] if m is not None)
                    gen_logits_mask = [
                        torch.ones_like(mm) if m is None else m for m in self.output_logits_mask[i]
                    ]
                    gen_logits_mask = torch.stack(gen_logits_mask, -2)

                res = dict(prompt=prompt_tokens, gen=gen_tokens, logp=gen_logp, logits_mask=gen_logits_mask)
                try:
                    self.outqueue.put_nowait(res)
                except queue.Full as e:
                    raise RuntimeError("Output queue is full. Please set a larger queue size.") from e

            # clear cache
            self.cache_seqlens[i] = 0

            # clear input buffers and prompts
            self.input_buf[i] = 0
            self.input_buf_lens[i] = 0
            self.prompt_tokens[i] = None

            # clear generation state
            self.generate_idx[i] = 0
            self.unfinished_sequences[i] = 1

            self.output_logits_mask[i] = []
            self.output_tokens_buf[i] = []
            self.output_logprob_buf[i] = []

        # build packed input ids with variable lengths for the next-step inference
        for i in range(self.batch_size):
            if i in finish_indices:
                try:
                    prompt = self.inqueue.get_nowait()
                    self.prompt_tokens[i] = prompt
                    self.input_buf[i, :prompt.shape[0]] = prompt
                    self.input_buf_lens[i] = prompt.shape[0]
                except queue.Empty as e:
                    raise RuntimeError("Input queue is empty. This should not happen.") from e

        input_lens = self.input_buf_lens
        valid_input_mask = torch.arange(self.max_prompt_len, device=self.input_buf.device,
                                        dtype=torch.int32).unsqueeze(0) < input_lens.unsqueeze(-1)
        indices = torch.nonzero(valid_input_mask.flatten(), as_tuple=False).flatten()
        packed_input_ids = self.input_buf.flatten()[indices]
        max_seqlen = int(max(input_lens))
        cu_seqlens = torch.nn.functional.pad(input_lens.cumsum(0), (1, 0), value=0).int()

        x = PipeTransferData(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        self.ys[0].position_ids = None
        self.ys[0].input_ids = packed_input_ids
        logits = self.model(x, self.ys).pp_output
        logits = index_first_axis(logits, (cu_seqlens[1:] - 1).long())

        self.cache_seqlens += input_lens

        return logits.float()

    def advance_one_genstep(self):
        if self.unfinished_sequences.logical_not().any():
            logits = self._get_inflight_logits()
        else:
            logits = self._get_non_eos_logits()

        next_tokens, logprob, logits_mask, _, self.unfinished_sequences = genstep(
            logits, self.tokenizer, self.unfinished_sequences, self.generate_idx, self.gconfig)

        for i in range(self.batch_size):
            self.output_tokens_buf[i].append(next_tokens[i].long())
            self.output_logprob_buf[i].append(logprob[i].float())
            if logits_mask is not None:
                self.output_logits_mask[i].append(logits_mask[i].bool())
            else:
                self.output_logits_mask[i].append(None)

        self.generate_idx += 1
        self.input_buf[:, 0] = next_tokens

    def step_for(self, n: int):
        for _ in range(n):
            self.advance_one_genstep()
