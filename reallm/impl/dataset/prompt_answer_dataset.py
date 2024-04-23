from typing import Callable, Dict, List, Optional
import json

import torch
import torch.utils.data

import api.data
import api.huggingface
import reallm.base.logging as logging

logger = logging.getLogger("Prompt Dataset")


class PromptAnswerDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        util: api.data.DatasetUtility,
        max_seq_len: int,
        dataset_path: Optional[str] = None,
        dataset_builder: Optional[Callable[[], List[Dict]]] = None,
    ):
        """A dataset with prompts and corresponding answers. Usually used for SFT.

        Args:
            util (api.data.DatasetUtility): .
            n_tokens_per_batch (int, optional): The number of tokens in the batch.
            max_length (Optional[int], optional): The maximum length of each sequence in the batch. Defaults to n_tokens_per_batch.
            dataset_path (Optional[str], optional): Path to the dataset json/jsonl file.
                The json/jsonl file should be a list of dictionary. Each element in the list should have
                a key "prompt" and a key "answer". Defaults to None.
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

        seqs = [x['prompt'] + x['answer'] + tokenizer.eos_token for x in data]
        prompts = [x['prompt'] for x in data]

        self.tokens = tokenizer(seqs,
                                return_tensors='pt',
                                padding=True,
                                truncation=True,
                                max_length=max_seq_len)
        prompt_tokens = tokenizer(prompts,
                                  padding=False,
                                  truncation=True,
                                  max_length=max_seq_len,
                                  return_length=True,
                                  return_attention_mask=False)
        prompt_lengths = prompt_tokens['length']
        prompt_masks = []
        for i in range(len(self)):
            seq = self.tokens['input_ids'][i]
            prompt_len = prompt_lengths[i]
            prompt = prompt_tokens['input_ids'][i]
            attention_mask = self.tokens['attention_mask'][i]
            seqlen = attention_mask.sum()
            assert seq[:prompt_len] == prompt
            assert seqlen >= prompt_len, (seqlen, prompt_len)
            prompt_masks.append(torch.tensor([1] * prompt_len + [0] * (seqlen - prompt_len)))

        self.prompt_masks = prompt_masks

    def __len__(self):
        return len(self.tokens['input_ids'])

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokens['input_ids'][idx],
            "attention_mask": self.tokens['attention_mask'][idx],
            "prompt_mask": self.prompt_masks[idx],
        }


api.data.register_dataset("prompt_answer", PromptAnswerDataset)
