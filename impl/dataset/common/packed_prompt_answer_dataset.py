from typing import Callable, Dict, List, Optional, Tuple
import itertools
import json
import logging

import numpy as np
import torch.utils.data

from base.datapack import ffd_with_result_unsorted
import api.data


class PackedPromptAnswerDataset(torch.utils.data.IterableDataset):

    def __init__(
        self,
        util: api.data.DatasetUtility,
        n_tokens_per_batch: int,
        max_length: Optional[int] = None,
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
        if max_length is None:
            max_length = n_tokens_per_batch
        if max_length > n_tokens_per_batch:
            raise ValueError(
                f"max_length ({max_length}) must be smaller than n_tokens_per_batch ({n_tokens_per_batch}).")
        self.max_length = max_length

        self.util = util
        if self.util.tokenizer.pad_token_id is None:
            self.util.tokenizer.pad_token_id = self.util.tokenizer.eos_token_id
            if self.util.tokenizer.eos_token_id is None:
                raise ValueError("eos_token_id must be defined.")

        if dataset_path is not None:
            if dataset_path.endswith(".jsonl"):
                with open(dataset_path, 'r') as f:
                    data = [json.loads(ff) for ff in f]
            elif dataset_path.endswith(".json"):
                with open(dataset_path, 'r') as f:
                    data = json.load(f)
            else:
                raise NotImplementedError(f"Unkown extention: {dataset_path}")
        else:
            assert dataset_builder is not None
            data = dataset_builder()

        dataset_size = len(data)
        dataset_size_per_rank = dataset_size // util.world_size
        shuffle_indices = api.data.get_shuffle_indices(util.seed, dataset_size)
        subset_indices = shuffle_indices[util.ddp_rank * dataset_size_per_rank:(util.ddp_rank + 1) *
                                         dataset_size_per_rank]
        data = [data[i] for i in subset_indices]

        tokenizer = util.tokenizer

        prompts_str = [x['prompt'] for x in data]
        prompt_chosen_str = [x['answer'] + tokenizer.eos_token for x in data]

        prompt_encodings = tokenizer(prompts_str,
                                     truncation=True,
                                     max_length=max_length,
                                     padding=False,
                                     return_length=True,
                                     return_attention_mask=False)
        prompt_lengths = prompt_encodings['length']
        prompts = prompt_encodings['input_ids']

        prompt_chosen_encodings = tokenizer(prompt_chosen_str,
                                            truncation=True,
                                            max_length=max_length,
                                            padding=False,
                                            return_length=True,
                                            return_attention_mask=False)
        seqlens = prompt_chosen_encodings['length']
        seqs = prompt_chosen_encodings['input_ids']

        prompt_masks = []
        for seq, prompt, seqlen, prompt_len in zip(seqs, prompts, seqlens, prompt_lengths):
            assert seq[:prompt_len] == prompt
            assert seqlen >= prompt_len, (seqlen, prompt_len)
            prompt_masks.append([1] * prompt_len + [0] * (seqlen - prompt_len))

        self.seqlens = seqlens
        self.prompt_lengths = prompt_lengths
        self.seqs = seqs
        self.prompts = prompts
        self.prompt_masks = prompt_masks

        self.n_tokens_per_batch = n_tokens_per_batch

        self.shuffle_cnt = 0

        self.rng = np.random.RandomState(seed=util.seed)

        self._shuffle()
        assert all(seq <= self.n_tokens_per_batch for seq in self.seqlens)
        self.__batch_indices = ffd_with_result_unsorted(np.array(self.seqlens), self.n_tokens_per_batch)
        self.rng.shuffle(self.__batch_indices)

    def _shuffle(self):
        shuffle_indices = api.data.get_shuffle_indices(
            self.util.seed + self.shuffle_cnt * 7 + self.util.ddp_rank * 3, len(self.seqlens))

        self.seqlens = [self.seqlens[i] for i in shuffle_indices]
        self.prompt_lengths = [self.prompt_lengths[i] for i in shuffle_indices]
        self.seqs = [self.seqs[i] for i in shuffle_indices]
        self.prompts = [self.prompts[i] for i in shuffle_indices]
        self.prompt_masks = [self.prompt_masks[i] for i in shuffle_indices]

        self.shuffle_cnt += 1

    def __len__(self):
        return len(self.__batch_indices)

    def __iter__(self):
        for indices in self.__batch_indices:
            seqlens = [self.seqlens[i] for i in indices]
            seqs = [self.seqs[i] for i in indices]
            prompt_masks = [self.prompt_masks[i] for i in indices]

            total_seqlen = sum(seqlens)
            assert total_seqlen <= self.n_tokens_per_batch, (total_seqlen, self.n_tokens_per_batch)

            seqlens = torch.tensor(seqlens, dtype=torch.int32)
            cu_seqlens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(seqlens, dim=0)])
            prompt_masks = torch.cat([torch.tensor(m, dtype=torch.bool) for m in prompt_masks])
            packed_input_ids = torch.cat([torch.tensor(p, dtype=torch.long) for p in seqs])

            assert packed_input_ids.shape[0] == prompt_masks.shape[0], (packed_input_ids.shape[0],
                                                                        prompt_masks.shape[0], total_seqlen)

            yield dict(
                packed_input_ids=packed_input_ids,
                prompt_mask=prompt_masks,
                cu_seqlens=cu_seqlens,
            )
        self._shuffle()
        self.__batch_indices = ffd_with_result_unsorted(np.array(self.seqlens), self.n_tokens_per_batch)
        self.rng.shuffle(self.__batch_indices)


api.data.register_dataset("packed_prompt_answer", PackedPromptAnswerDataset)
