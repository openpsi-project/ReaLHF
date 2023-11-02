from typing import Callable, Dict, List, Optional
import itertools
import json

import torch
import torch.utils.data
import numpy as np
import api.data
from base.datapack import min_abs_diff_partition


class RewardModelingPairedDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        util: api.data.DatasetUtility,
        max_seq_len: int,
        pad_to_max_length: bool = False,
        max_pairs_per_prompt: Optional[int] = None,
        dataset_path: Optional[str] = None,
        dataset_builder: Optional[Callable[[], List[Dict]]] = None,
    ):
        """Dataset used for reward modeling. Each sample consists of a prompt, several positive answers, and several negative answers.

        Args:
            util (api.data.DatasetUtility): Dataset utility.
            max_seq_len (int): The maximum sequence length. Sequences will be right-padded to this length
            dataset_path (Optional[str], optional): Path to the dataset json/jsonl file.
                The json/jsonl file should be a list of dictionary. Each element in the list should have
                the key "prompt", "pos_answers", and "neg_answers". Each "pos_answer" must correspond to
                one and only one "neg_answer" (i.e., they are one-to-one pairs). Defaults to None.
            dataset_builder (Optional[Callable[[], List[Dict]]], optional): Alternative to dataset_path.
                A callable that returns a list of dictionary. Defaults to None.
        """
        self.util = util
        seed = self.util.seed
        world_size = self.util.world_size
        tokenizer = self.util.tokenizer
        ddp_rank = self.util.ddp_rank

        self.rng = np.random.RandomState(seed=seed)

        if dataset_path is not None:
            if dataset_path.endswith(".jsonl"):
                with open(dataset_path, 'r') as f:
                    data = [json.loads(ff) for ff in f]
            elif dataset_path.endswith(".json"):
                with open(dataset_path, 'r') as f:
                    data = json.load(f)
            else:
                raise NotImplementedError(f"Unkown dataset extension: {dataset_path}")
        else:
            assert dataset_builder is not None
            data = dataset_builder()

        shuffle_indices = api.data.get_shuffle_indices(seed, len(data))
        data = [data[i] for i in shuffle_indices]
        print(">>>>", len(data))

        if max_pairs_per_prompt is not None:
            all_group_sizes = [min(max_pairs_per_prompt, len(x['pos_answers'])) for x in data]
        else:
            all_group_sizes = [len(x['pos_answers']) for x in data]
        start, end = min_abs_diff_partition(all_group_sizes, world_size)[ddp_rank]

        data: List[Dict[str, str]] = data[start:end]
        print("<<<<", len(data))

        prompts = [x['prompt'] for x in data]
        if max_pairs_per_prompt is None:
            pos_answers = [[x['prompt'] + c + tokenizer.eos_token for c in x['pos_answers']] for x in data]
            neg_answers = [[x['prompt'] + c + tokenizer.eos_token for c in x['neg_answers']] for x in data]
        else:
            pos_answers = [[
                x['prompt'] + c + tokenizer.eos_token for c in self.rng.choice(
                    x['pos_answers'], min(len(x['pos_answers']), max_pairs_per_prompt), replace=False)
            ] for x in data]
            neg_answers = [[
                x['prompt'] + c + tokenizer.eos_token for c in self.rng.choice(
                    x['neg_answers'], min(len(x['neg_answers']), max_pairs_per_prompt), replace=False)
            ] for x in data]

        for a, b in zip(pos_answers, neg_answers):
            if len(a) != len(b):
                raise RuntimeError("pos_answers and neg_answers must be one-to-one pairs.")
            if len(a) == 0:
                raise RuntimeError("pos_answers and neg_answers must be non-empty.")

        group_sizes = [len(x) for x in pos_answers]

        self.prompt_tokens = tokenizer(
            prompts,
            max_length=max_seq_len,
            truncation=True,
            padding=False,
            return_length=True,
        )
        _pos_answer_tokens = tokenizer(list(itertools.chain.from_iterable(pos_answers)),
                                       max_length=max_seq_len,
                                       padding="max_length" if pad_to_max_length else True,
                                       truncation=True,
                                       return_tensors="pt")
        _neg_answer_tokens = tokenizer(list(itertools.chain.from_iterable(neg_answers)),
                                       max_length=max_seq_len,
                                       padding="max_length" if pad_to_max_length else True,
                                       truncation=True,
                                       return_tensors="pt")

        pos_answer_tokens = []
        neg_answer_tokens = []
        offset = 0
        for g in group_sizes:
            pos_answer_tokens.append({k: v[offset:offset + g] for k, v in _pos_answer_tokens.items()})
            neg_answer_tokens.append({k: v[offset:offset + g] for k, v in _neg_answer_tokens.items()})
            offset += g

        self.pos_answer_tokens = pos_answer_tokens
        self.neg_answer_tokens = neg_answer_tokens
        assert len(self.prompt_tokens['input_ids']) == len(self.pos_answer_tokens) == len(self.neg_answer_tokens)

    def __len__(self):
        return len(self.pos_answer_tokens)

    def __getitem__(self, idx):
        group_size = self.pos_answer_tokens[idx]['input_ids'].shape[0]
        prompt_len = self.prompt_tokens['length'][idx]
        return {
            "pos_input_ids": self.pos_answer_tokens[idx]['input_ids'],
            "pos_attention_mask": self.pos_answer_tokens[idx]['attention_mask'],
            "neg_input_ids": self.neg_answer_tokens[idx]['input_ids'],
            'neg_attention_mask': self.neg_answer_tokens[idx]['attention_mask'],
            'group_factor': torch.tensor([1 / group_size for _ in range(group_size)]),
            'prompt_lens': torch.tensor([prompt_len for _ in range(group_size)]),
        }


api.data.register_dataset("rw_pair", RewardModelingPairedDataset)