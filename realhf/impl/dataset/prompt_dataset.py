import uuid
from typing import Callable, Dict, List, Optional

import torch.utils.data

from realhf.api.core import data_api
from realhf.base import logging

logger = logging.getLogger("Prompt Dataset")


class PromptDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        util: data_api.DatasetUtility,
        max_length: Optional[int] = None,
        dataset_path: Optional[str] = None,
        dataset_builder: Optional[Callable[[], List[Dict]]] = None,
        pad_to_max_length: bool = False,
    ):
        """A dataset with prompts. Usually used for PPO.

        Args:
            util (api.data.DatasetUtility): .
            max_length (Optional[int], optional): The maximum length of each sequence in the batch.
            dataset_path (Optional[str], optional): Path to the dataset json/jsonl file.
                The json/jsonl file should be a list of dictionary. Each element in the list should have
                a key "prompt". Defaults to None.
            dataset_builder (Optional[Callable[[], List[Dict]]], optional): Alternative to dataset_path.
                A callable that returns a list of dictionary. Defaults to None.
            pad_to_max_length (bool): Whether to pad prompts to the maximum length.
                Used only for benchmarking. If True, all mini-batches created by the DP balanced partition
                algorithm will have the same number of tokens, making MFC time predictable. Defaults to False.
        """
        self._util = util
        self.max_length = max_length

        data = data_api.load_shuffle_split_dataset(util, dataset_path, dataset_builder)

        prompts_str = [x["prompt"] for x in data]
        self.ids = [x["id"] for x in data]
        util.tokenizer.padding_side = "left"
        prompt_encodings = util.tokenizer(
            prompts_str,
            truncation=True,
            max_length=max_length,
            padding=pad_to_max_length,
            return_length=True,
            return_attention_mask=False,
        )

        self.prompt_lengths = prompt_encodings["length"]
        self.prompts = prompt_encodings["input_ids"]
        assert all(len(x) == l for x, l in zip(self.prompts, self.prompt_lengths))

        logger.info(f"Number of prompts in the dataset: {len(self.prompts)}")

    @property
    def util(self):
        return self._util

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return data_api.SequenceSample.from_default(
            ids=[self.ids[idx]],
            seqlens=[self.prompt_lengths[idx]],
            data=dict(packed_prompts=torch.tensor(self.prompts[idx], dtype=torch.long)),
            metadata=dict(random_id=[uuid.uuid4()]),
        )


data_api.register_dataset("prompt", PromptDataset)
