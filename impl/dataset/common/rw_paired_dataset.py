from typing import Callable, Dict, List, Optional
import itertools
import json

import torch
import torch.utils.data

import api.data


class RewardModelingPairedDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        util: api.data.DatasetUtility,
        max_seq_len: int,
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

        datasize_per_rank = len(data) // world_size
        shuffle_indices = api.data.get_shuffle_indices(seed, datasize_per_rank * world_size)
        subset_indices = shuffle_indices[ddp_rank * datasize_per_rank:(ddp_rank + 1) * datasize_per_rank]
        data: List[Dict[str, str]] = [data[i] for i in subset_indices]

        self.prompts = [x['prompt'] for x in data]
        pos_answers = [[x['prompt'] + c + tokenizer.eos_token for c in x['pos_answers']] for x in data]
        neg_answers = [[x['prompt'] + c + tokenizer.eos_token for c in x['neg_answers']] for x in data]

        for a, b in zip(pos_answers, neg_answers):
            if len(a) != len(b):
                raise RuntimeError("pos_answers and neg_answers must be one-to-one pairs.")
            if len(a) == 0:
                raise RuntimeError("pos_answers and neg_answers must be non-empty.")

        group_sizes = [len(x) for x in pos_answers]

        _pos_answer_tokens = tokenizer(list(itertools.chain.from_iterable(pos_answers)),
                                       max_length=max_seq_len,
                                       padding="max_length",
                                       truncation=True,
                                       return_tensors="pt")
        _neg_answer_tokens = tokenizer(list(itertools.chain.from_iterable(neg_answers)),
                                       max_length=max_seq_len,
                                       padding="max_length",
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

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {
            "pos_input_ids": self.pos_answer_tokens[idx]['input_ids'],
            "pos_attention_mask": self.pos_answer_tokens[idx]['attention_mask'],
            "neg_input_ids": self.neg_answer_tokens[idx]['input_ids'],
            'neg_attention_mask': self.neg_answer_tokens[idx]['attention_mask'],
        }


api.data.register_dataset("rw_pair", RewardModelingPairedDataset)