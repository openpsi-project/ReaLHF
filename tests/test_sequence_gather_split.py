import dataclasses
import itertools
import random
from typing import *

import pytest
import torch

from realhf.api.core.data_api import SequenceSample, SequenceSplitSpec


def flatten_list(l: List[List]):
    return list(itertools.chain(*l))


def _make_sample_single_sequence(bs):
    keys = [
        "input_ids",
        "rewards",
        "logprobs",
        "logits_mask",
        "prompt_mask",
    ]
    vocab_size = 150
    slens = [torch.randint(1, 100, (1,)).int() for _ in range(bs)]
    input_ids = torch.cat([torch.randint(0, 150, (slen,)) for slen in slens])
    rewards = torch.cat([torch.randn(1) for _ in range(bs)])
    logprobs = torch.cat([torch.randn(slen - 1) for slen in slens])
    logits_mask = torch.cat(
        [torch.randint(0, 2, (slen, vocab_size), dtype=torch.bool) for slen in slens]
    )
    prompt_mask = torch.cat(
        [torch.randint(0, 2, (slen,), dtype=torch.bool) for slen in slens]
    )
    data = dict(
        input_ids=input_ids,
        rewards=rewards,
        logprobs=logprobs,
        logits_mask=logits_mask,
        prompt_mask=prompt_mask,
    )
    ids = list(range(bs))
    trailing_shapes = dict(
        input_ids=(),
        rewards=(),
        logprobs=(),
        logits_mask=(vocab_size,),
        prompt_mask=(),
    )
    dtypes = dict(
        input_ids=torch.long,
        rewards=torch.float,
        logprobs=torch.float,
        logits_mask=torch.bool,
        prompt_mask=torch.bool,
    )
    seqlens = dict(
        input_ids=slens,
        rewards=[torch.tensor([1]).int() for _ in range(bs)],
        logprobs=[x - 1 for x in slens],
        logits_mask=slens,
        prompt_mask=slens,
    )
    return SequenceSample(
        keys=keys,
        ids=ids,
        seqlens=seqlens,
        trailing_shapes=trailing_shapes,
        dtypes=dtypes,
        data=data,
        buf_indices=ids,
        metadata=dict(a=1, b="abc"),
    )


def _make_sample_multiple_sequence(bs):
    keys = [
        "group_factor",
        "seqlogprobs",
        "input_ids",
    ]
    slens = [torch.randint(1, 100, (2,)).int() for _ in range(bs)]
    input_ids = torch.cat(
        flatten_list([[torch.randint(0, 150, (s,)) for s in slen] for slen in slens])
    )
    group_factor = torch.cat([torch.randn(1) for _ in range(bs)])
    seqlogprobs = torch.cat([torch.randn(2, dtype=torch.float16) for _ in range(bs)])

    data = dict(
        input_ids=input_ids,
        group_factor=group_factor,
        seqlogprobs=seqlogprobs,
    )
    ids = list(range(bs))
    trailing_shapes = dict(
        input_ids=(),
        group_factor=(),
        seqlogprobs=(),
    )
    dtypes = dict(
        input_ids=torch.long,
        group_factor=torch.float,
        seqlogprobs=torch.float16,
    )
    seqlens = dict(
        input_ids=slens,
        group_factor=[torch.tensor([1]).int() for _ in range(bs)],
        seqlogprobs=[torch.tensor([1, 1], dtype=torch.int32) for _ in range(bs)],
    )
    return SequenceSample(
        keys=keys,
        ids=ids,
        seqlens=seqlens,
        trailing_shapes=trailing_shapes,
        dtypes=dtypes,
        data=data,
        buf_indices=ids,
        metadata=dict(a=1, b="abc"),
    )


def _make_sample_single_prompt_multi_response(bs):
    keys = [
        "prompt",
        "seq",
        "prompt_mask",
    ]
    n_ans_per_prompt = 5

    prompt_lens = [torch.randint(1, 100, (1,)).int() for _ in range(bs)]
    gen_lens = [torch.randint(1, 100, (n_ans_per_prompt,)).int() for _ in range(bs)]

    prompt = torch.cat([torch.randint(0, 150, (slen,)) for slen in prompt_lens])
    seq = torch.cat(
        flatten_list([[torch.randint(0, 150, (s,)) for s in slen] for slen in gen_lens])
    )
    prompt_mask = torch.randint_like(seq, 0, 2, dtype=torch.bool)

    data = dict(
        prompt=prompt,
        seq=seq,
        prompt_mask=prompt_mask,
    )
    ids = list(range(bs))
    trailing_shapes = dict(
        prompt=(),
        seq=(),
        prompt_mask=(),
    )
    dtypes = dict(
        prompt=torch.long,
        seq=torch.long,
        prompt_mask=torch.bool,
    )
    seqlens = dict(
        prompt=prompt_lens,
        seq=gen_lens,
        prompt_mask=gen_lens,
    )
    return SequenceSample(
        keys=keys,
        ids=ids,
        seqlens=seqlens,
        trailing_shapes=trailing_shapes,
        dtypes=dtypes,
        data=data,
        buf_indices=ids,
    )


def recursive_assert_equal(x1, x2):
    if type(x1) != type(x2):
        raise AssertionError(f"{type(x1)} != {type(x2)}")
    if isinstance(x1, dict):
        assert set(x1.keys()) == set(x2.keys())
        for k in x1.keys():
            recursive_assert_equal(x1[k], x2[k])
    elif dataclasses.is_dataclass(x1):
        for f in dataclasses.fields(x1):
            recursive_assert_equal(getattr(x1, f.name), getattr(x2, f.name))
    elif isinstance(x1, torch.Tensor):
        assert torch.allclose(x1, x2)
    elif isinstance(x1, list):
        assert len(x1) == len(x2)
        for a, b in zip(x1, x2):
            recursive_assert_equal(a, b)
    else:
        assert x1 == x2


@pytest.mark.parametrize("sample_type", ["single", "pair", "multi_sample"])
@pytest.mark.parametrize("dp", [1, 2, 3, 4, 8, 15, 16])
def test_gather_split(sample_type: str, dp: int):
    batch_sizes = [random.randint(1, 10) for _ in range(dp)]
    if sample_type == "single":
        samples = [_make_sample_single_sequence(bs) for bs in batch_sizes]
    elif sample_type == "pair":
        samples = [_make_sample_multiple_sequence(bs) for bs in batch_sizes]
    elif sample_type == "multi_sample":
        samples = [_make_sample_single_prompt_multi_response(bs) for bs in batch_sizes]
    else:
        raise NotImplementedError()

    x = SequenceSample.gather(samples)

    # Test gather-split-gather cosistency
    for k in x.keys:
        y = SequenceSample.gather(x.split(dp, key=k, min_size=1))
        recursive_assert_equal(x, y)

    # Test balanced split
    balanced_size = sum(batch_sizes) // dp
    for k in x.keys:
        splitted = x.split(dp, key=k, min_size=balanced_size)
        assert all(len(s.ids) >= balanced_size for s in splitted)
        y = SequenceSample.gather(splitted)
        recursive_assert_equal(x, y)

    # Test split to original samples
    spec = SequenceSplitSpec(sizes=batch_sizes)
    ss = x.split_with_spec(spec)
    for s1, s2 in zip(samples, ss):
        recursive_assert_equal(s1, s2)

    # Test split to the finest granularity
    total_bs = sum(batch_sizes)
    for k in x.keys:
        ss = x.split(total_bs, key=k, min_size=1)
        assert len(ss) == total_bs
        y = SequenceSample.gather(ss)
        recursive_assert_equal(x, y)
