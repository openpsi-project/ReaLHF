from typing import Callable, Dict, List, Optional
import itertools
import json

import numpy as np
import torch
import torch.utils.data

from realhf.api.core import data_api
from realhf.base.namedarray import NamedArray


class RewardModelingPairedDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        util: data_api.DatasetUtility,
        max_length: int,
        max_pairs_per_prompt: int = 2,
        dataset_path: Optional[str] = None,
        dataset_builder: Optional[Callable[[], List[Dict]]] = None,
    ):
        """Dataset used for reward modeling. Each sample consists of a prompt, several positive answers, and several negative answers.

        Args:
            util (api.data.DatasetUtility): Dataset utility.
            max_length (int): The maximum sequence length. Sequences will be right-padded to this length
            dataset_path (Optional[str], optional): Path to the dataset json/jsonl file.
                The json/jsonl file should be a list of dictionary. Each element in the list should have
                the key "prompt", "pos_answers", and "neg_answers". Each "pos_answer" must correspond to
                one and only one "neg_answer" (i.e., they are one-to-one pairs). Defaults to None.
            dataset_builder (Optional[Callable[[], List[Dict]]], optional): Alternative to dataset_path.
                A callable that returns a list of dictionary. Defaults to None.
        """
        self.util = util
        seed = self.util.seed
        tokenizer = self.util.tokenizer

        self.max_pairs_per_prompt = max_pairs_per_prompt

        self.rng = np.random.RandomState(seed=seed)
        data = data_api.load_shuffle_split_dataset(
            util, dataset_path, dataset_builder
        )

        prompts = [x["prompt"] for x in data]

        pos_answers = [
            [x["prompt"] + c + tokenizer.eos_token for c in x["pos_answers"]]
            for x in data
        ]
        neg_answers = [
            [x["prompt"] + c + tokenizer.eos_token for c in x["neg_answers"]]
            for x in data
        ]

        for a, b in zip(pos_answers, neg_answers):
            if len(a) != len(b):
                raise RuntimeError(
                    "pos_answers and neg_answers must be one-to-one pairs."
                )
            if len(a) == 0:
                raise RuntimeError(
                    "pos_answers and neg_answers must be non-empty."
                )

        group_sizes = [len(x) for x in pos_answers]

        self.prompt_tokens = tokenizer(
            prompts,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_length=True,
        )
        _pos_answer_tokens = tokenizer(
            list(itertools.chain.from_iterable(pos_answers)),
            max_length=max_length,
            padding=False,
            truncation=True,
            return_length=True,
        )
        _neg_answer_tokens = tokenizer(
            list(itertools.chain.from_iterable(neg_answers)),
            max_length=max_length,
            padding=False,
            truncation=True,
            return_length=True,
        )

        pos_answer_tokens = []
        neg_answer_tokens = []
        offset = 0
        for g in group_sizes:
            pos_answer_tokens.append(
                {
                    k: v[offset : offset + g]
                    for k, v in _pos_answer_tokens.items()
                }
            )
            neg_answer_tokens.append(
                {
                    k: v[offset : offset + g]
                    for k, v in _neg_answer_tokens.items()
                }
            )
            offset += g

        self.pos_answer_tokens: List[List[int]] = pos_answer_tokens
        self.neg_answer_tokens: List[List[int]] = neg_answer_tokens
        assert (
            len(self.prompt_tokens["input_ids"])
            == len(self.pos_answer_tokens)
            == len(self.neg_answer_tokens)
        )

    def __len__(self):
        return len(self.pos_answer_tokens)

    def __getitem__(self, idx):
        prompt_len = self.prompt_tokens["length"][idx]
        n_pairs_this_prompt = len(self.pos_answer_tokens[idx]["input_ids"])
        group_size = min(self.max_pairs_per_prompt, n_pairs_this_prompt)
        pair_indices = self.rng.choice(
            n_pairs_this_prompt, group_size, replace=False
        )

        packed_input_ids = []
        for i in pair_indices:
            packed_input_ids += self.pos_answer_tokens[idx]["input_ids"][i]
            packed_input_ids += self.neg_answer_tokens[idx]["input_ids"][i]

        pos_input_lens = [
            len(self.pos_answer_tokens[idx]["input_ids"][i])
            for i in pair_indices
        ]
        neg_input_lens = [
            len(self.neg_answer_tokens[idx]["input_ids"][i])
            for i in pair_indices
        ]

        input_lens = [x + y for x, y in zip(pos_input_lens, neg_input_lens)]
        assert sum(input_lens) == len(packed_input_ids)

        x = NamedArray(
            packed_input_ids=torch.tensor(packed_input_ids, dtype=torch.long),
            pos_input_lens=torch.tensor(pos_input_lens, dtype=torch.int32),
            group_factor=torch.tensor(
                [1 / group_size for _ in range(group_size)], dtype=torch.float32
            ),
            prompt_lens=torch.tensor(
                [prompt_len for _ in range(group_size)], dtype=torch.int32
            ),
        )
        assert (
            x["pos_input_lens"].shape
            == x["group_factor"].shape
            == x["prompt_lens"].shape
        )
        assert x["pos_input_lens"].shape[0] == len(input_lens)
        x.register_metadata(seqlens=input_lens)
        return x


data_api.register_dataset("rw_pair", RewardModelingPairedDataset)
