from typing import Callable, Dict, List, Optional
import itertools
import json

import torch
import torch.utils.data

import api.data


class RewardModelingGTLabelDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        util: api.data.DatasetUtility,
        max_seq_len: int,
        dataset_path: Optional[str] = None,
        dataset_builder: Optional[Callable[[], List[Dict]]] = None,
    ):
        """Dataset used for reward modeling. Each sample consists of a prompt, several answers, and corresponding labels.
        
        Basically a dataset for binary classification. Labels are lists of 0 (incorrect) or 1 (correct).

        Args:
            util (api.data.DatasetUtility): Dataset utility.
            max_seq_len (int): The maximum sequence length. Sequences will be right-padded to this length
            dataset_path (Optional[str], optional): Path to the dataset json/jsonl file.
                The json/jsonl file should be a list of dictionary. Each element in the list should have
                the key "prompt", "answers", and "labels". Each "answer" must correspond to
                one and only one "label". Defaults to None.
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
        answers = [[x['prompt'] + c + tokenizer.eos_token for c in x['answers']] for x in data]
        self.labels = labels = [x['labels'] for x in data]

        for a, b in zip(answers, labels):
            if len(a) != len(b):
                raise ValueError("Each answer must have a label")
            if len(a) == 0:
                raise ValueError("Each example must have at least one answer")

        group_sizes = [len(x) for x in labels]

        _answer_tokens = tokenizer(list(itertools.chain.from_iterable(answers)),
                                   max_length=max_seq_len,
                                   padding="max_length",
                                   truncation=True,
                                   return_tensors="pt")
        answer_tokens = []
        offset = 0
        for g in group_sizes:
            answer_tokens.append({k: v[offset:offset + g] for k, v in _answer_tokens.items()})
            offset += g

        self.answer_tokens = answer_tokens

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {
            "input_ids": self.answer_tokens["input_ids"][idx],
            "attention_mask": self.answer_tokens["attention_mask"][idx],
            "correctness_labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


api.data.register_dataset("rw_gtlabel", RewardModelingGTLabelDataset)