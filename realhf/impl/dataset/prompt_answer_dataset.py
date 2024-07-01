import json
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.utils.data

from realhf.api.core import data_api
from realhf.base import logging, namedarray

logger = logging.getLogger("Prompt Answer Dataset")


class PromptAnswerDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        util: data_api.DatasetUtility,
        max_length: int,
        dataset_path: Optional[str] = None,
        dataset_builder: Optional[Callable[[], List[Dict]]] = None,
    ):
        """A dataset with prompts and corresponding answers. Usually used for SFT.

        Args:
            util (api.data.DatasetUtility): .
            max_length (Optional[int], optional): The maximum length of each sequence in the batch.
            dataset_path (Optional[str], optional): Path to the dataset json/jsonl file.
                The json/jsonl file should be a list of dictionary. Each element in the list should have
                a key "prompt" and a key "answer". Defaults to None.
            dataset_builder (Optional[Callable[[], List[Dict]]], optional): Alternative to dataset_path.
                A callable that returns a list of dictionary. Defaults to None.
        """
        self._util = util
        tokenizer = self.util.tokenizer

        data = data_api.load_shuffle_split_dataset(util, dataset_path, dataset_builder)

        seqs = [x["prompt"] + x["answer"] + tokenizer.eos_token for x in data]
        prompts = [x["prompt"] for x in data]

        self.tokens = tokenizer(
            seqs,
            padding=False,
            truncation=True,
            max_length=max_length,
            return_length=True,
            return_attention_mask=False,
        )
        prompt_tokens = tokenizer(
            prompts,
            padding=False,
            truncation=True,
            return_length=True,
            max_length=max_length,
            return_attention_mask=False,
        )

        prompt_lengths = prompt_tokens["length"]
        seq_lengths = self.tokens["length"]
        prompt_masks = []
        for i in range(len(self)):
            prompt_len = prompt_lengths[i]
            seqlen = self.tokens["length"][i]
            # seq = self.tokens["input_ids"][i]
            # prompt = prompt_tokens["input_ids"][i]
            # assert seq[:prompt_len] == prompt, (seq, prompt, prompt_len, seqlen)
            assert seqlen >= prompt_len, (seqlen, prompt_len)
            prompt_mask = [1] * prompt_len + [0] * (seqlen - prompt_len)
            prompt_masks.append(prompt_mask)

        self.prompt_masks = prompt_masks

        logger.info(
            f"Loaded Prompt Answer Dataset with INFO: "
            f"#seqs={len(self)}, "
            f"truncation length={max_length}, "
            f"avg prompt length={np.mean(prompt_lengths):.1f}, "
            f"avg answer length={np.mean(seq_lengths) - np.mean(prompt_lengths):.1f}",
        )

    @property
    def util(self):
        return self._util

    def __len__(self):
        return len(self.tokens["input_ids"])

    def __getitem__(self, idx):
        d = {
            "packed_input_ids": torch.tensor(
                self.tokens["input_ids"][idx], dtype=torch.long
            ),
            "prompt_mask": torch.tensor(self.prompt_masks[idx], dtype=torch.bool),
        }
        assert len(d["packed_input_ids"]) == len(d["prompt_mask"])
        seqlen = len(d["packed_input_ids"])
        x = namedarray.from_dict(d)
        x.register_metadata(seqlens=[seqlen])
        return x


data_api.register_dataset("prompt_answer", PromptAnswerDataset)
