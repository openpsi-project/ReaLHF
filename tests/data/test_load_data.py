import json
import os
import pathlib
import random
import uuid

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordPieceTrainer
from transformers import PreTrainedTokenizerFast

from realhf.api.core import config as config_api
from realhf.api.core import data_api


def generate_random_sentence(length):
    # A predefined list of common English words
    # fmt: off
    words = [
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "and", "then", "runs", "away", "from", "big", "scary", "bear",
        "in", "the", "forest", "during", "sunny", "day", "while", "birds",
        "sing", "beautiful", "songs", "under", "blue", "sky", "with", "white",
        "clouds", "floating", "gently",
    ]
    # fmt: on

    # Randomly select words to form a sentence
    sentence = " ".join(random.choices(words, k=length))

    return sentence


@pytest.fixture(scope="session")
def dataset_and_tokenizer(request, tmp_path_factory):
    size, max_prompt_len, max_resp_len, vocab_size = request.param
    tmp_dir = tmp_path_factory.mktemp("tokenizer")
    dataset = []
    for i in range(size):
        prompt_len = random.randint(1, max_prompt_len)
        resp_len = random.randint(1, max_resp_len)
        n_pairs = random.randint(1, 5)
        d = dict(
            id=i,
            prompt=generate_random_sentence(prompt_len),
            answer=generate_random_sentence(prompt_len)
            + generate_random_sentence(resp_len),
            pos_answers=[
                generate_random_sentence(prompt_len)
                + generate_random_sentence(resp_len)
                for _ in range(n_pairs)
            ],
            neg_answers=[
                generate_random_sentence(prompt_len)
                + generate_random_sentence(resp_len)
                for _ in range(n_pairs)
            ],
        )
        dataset.append(d)

    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    )

    data = [d["prompt"] + " " + d["answer"] for d in dataset]
    # Train the tokenizer on the sample data
    tokenizer.train_from_iterator(data, trainer)

    tokenizer.save(str(tmp_dir / "tokenizer.json"))

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tmp_dir / "tokenizer.json"))
    tokenizer.add_special_tokens({"pad_token": "[PAD]", "eos_token": "[EOS]"})

    return dataset, tokenizer


def _validate_dataset(cfg: config_api.DatasetAbstraction, tokenizer):
    dataset = data_api.make_dataset(
        cfg,
        seed=1,
        dp_rank=0,
        world_size=1,
        tokenizer_or_tokenizer_name=tokenizer,
        experiment_name=uuid.uuid4(),
        trial_name=uuid.uuid4(),
    )
    dataloader = data_api.PackedDataLoader(dataset)
    for x in dataloader:
        assert isinstance(x, data_api.SequenceSample)
        assert x.data is not None
        for k, v in x.data.items():
            assert v.device == torch.device("cpu")
        bs = len(x.ids)
        for k, vs in x.seqlens.items():
            assert all(isinstance(v, list) for v in vs)
            assert all(all(isinstance(vv, int) for vv in v) for v in vs)
        assert len(x.ids) == len(set(x.ids))
        if x.metadata:
            for k, v in x.metadata.items():
                assert isinstance(v, list), k
        xs = x.split(bs)
        for xx in xs:
            if xx.metadata:
                for k, v in xx.metadata.items():
                    assert isinstance(v, list), k
                    assert len(v) == 1


@pytest.mark.parametrize("dataset_and_tokenizer", [(1000, 128, 512, 3)], indirect=True)
@pytest.mark.parametrize("max_length", [128, 256, 1024])
def test_prompt_answer_dataset(dataset_and_tokenizer, max_length: int):
    # NOTE: import all dataset implementations
    import realhf.impl.dataset

    raw_dataset, tokenizer = dataset_and_tokenizer
    cfg = config_api.DatasetAbstraction(
        type_="prompt_answer",
        args=dict(max_length=max_length, dataset_builder=lambda: raw_dataset),
    )
    _validate_dataset(cfg, tokenizer)


@pytest.mark.parametrize("dataset_and_tokenizer", [(1000, 128, 512, 3)], indirect=True)
@pytest.mark.parametrize("max_length", [128, 256, 1024])
def test_prompt_only_dataset(
    dataset_and_tokenizer,
    max_length: int,
):
    # NOTE: import all dataset implementations
    import realhf.impl.dataset

    raw_dataset, tokenizer = dataset_and_tokenizer
    cfg = config_api.DatasetAbstraction(
        type_="prompt",
        args=dict(
            max_length=max_length,
            dataset_builder=lambda: raw_dataset,
        ),
    )
    _validate_dataset(cfg, tokenizer)


@pytest.mark.parametrize("dataset_and_tokenizer", [(1000, 128, 512, 3)], indirect=True)
@pytest.mark.parametrize("max_length", [128, 256, 1024])
@pytest.mark.parametrize("max_pairs_per_prompt", [1, 3, 10])
def test_paired_rw_dataset(
    dataset_and_tokenizer, max_length: int, max_pairs_per_prompt: int
):
    # NOTE: import all dataset implementations
    import realhf.impl.dataset

    raw_dataset, tokenizer = dataset_and_tokenizer
    cfg = config_api.DatasetAbstraction(
        type_="rw_pair",
        args=dict(
            max_length=max_length,
            dataset_builder=lambda: raw_dataset,
            max_pairs_per_prompt=max_pairs_per_prompt,
        ),
    )
    _validate_dataset(cfg, tokenizer)
