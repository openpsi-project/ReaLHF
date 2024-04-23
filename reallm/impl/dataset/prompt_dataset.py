from typing import Callable, Dict, List, Optional
import json

import torch
import torch.utils.data

import reallm.api.data
import reallm.api.huggingface
import reallm.base.logging as logging

logger = logging.getLogger("Prompt Dataset")


class PromptDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        util: reallm.api.data.DatasetUtility,
        max_prompt_len: int,
        pad_to_max_length: bool = False,
        dataset_path: Optional[str] = None,
        dataset_builder: Optional[Callable[[], List[Dict]]] = None,
    ):
        """Prompt dataset used for RLHF (PPO).

        Args:
            util (api.data.DatasetUtility): Dataset utility.
            max_prompt_len (int): The maximum prompt length. Prompts will be truncated and left-padded to this length.
            dataset_path (Optional[str], optional): Path to the dataset json/jsonl file.
                The json/jsonl file should be a list of dictionary. Each element in the list should have
                the key "prompt". Defaults to None.
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
        shuffle_indices = reallm.api.data.get_shuffle_indices(seed, datasize_per_rank * world_size)
        subset_indices = shuffle_indices[ddp_rank * datasize_per_rank:(ddp_rank + 1) * datasize_per_rank]
        data: List[Dict[str, str]] = [data[i] for i in subset_indices]

        prompts = [x['prompt'] for x in data]

        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = 'left'
        prompt_tokens = tokenizer(prompts,
                                  return_tensors='pt',
                                  padding=True if not pad_to_max_length else 'max_length',
                                  truncation=True,
                                  max_length=max_prompt_len)
        tokenizer.padding_side = original_padding_side

        self.prompt_tokens = prompt_tokens

    def __len__(self):
        return len(self.prompt_tokens['input_ids'])

    def __getitem__(self, idx):
        return {
            "prompts": self.prompt_tokens["input_ids"][idx],
            "prompt_att_mask": self.prompt_tokens["attention_mask"][idx],
        }


api.data.register_dataset("prompt", PromptDataset)
