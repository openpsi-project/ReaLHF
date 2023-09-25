from typing import Dict, List, Optional
import json
import logging
import random

import numpy as np
import torch.utils.data

from impl.data.utils.pack import ffd_with_result_unsorted
import api.data

logger = logging.getLogger("WPSFormulaPackedDataset")

PROMPT_FORMAT = ("Below is an instruction that describes a task. "
                 "Write a response that appropriately completes the request.\n\n"
                 "### Instruction:\n{input}\n\n### Response:\n{output}")


def get_prompt(entry: Dict) -> str:
    return PROMPT_FORMAT.format(input=entry['input'], output="")


def get_prompt_and_chosen(entry: Dict) -> str:
    return PROMPT_FORMAT.format(input=entry['input'], output=entry['output'])


class WPSFormulaPackedDataset(torch.utils.data.IterableDataset):

    def __init__(
        self,
        util: api.data.DatasetUtility,
        n_tokens_per_batch: int = 2048,
        max_n_seqs_per_batch: int = 40,
        max_length: Optional[int] = None,
        json_path: str = "/data/aigc/public/wps-excel/train-0908-formula-psi.json",
    ):
        if max_length is None:
            max_length = n_tokens_per_batch
        if max_length > n_tokens_per_batch:
            raise ValueError(
                f"max_length ({max_length}) must be smaller than n_tokens_per_batch ({n_tokens_per_batch}).")

        self.util = util
        if self.util.tokenizer.pad_token_id is None:
            self.util.tokenizer.pad_token_id = self.util.tokenizer.eos_token_id
            if self.util.tokenizer.eos_token_id is None:
                raise ValueError("eos_token_id must be defined.")

        with open(json_path) as f:
            data: List[Dict] = json.load(f)
            dataset_size = len(data)
            dataset_size_per_rank = dataset_size // util.world_size
            shuffle_indices = api.data.get_shuffle_indices(util.seed, dataset_size)
            subset_indices = shuffle_indices[util.ddp_rank * dataset_size_per_rank:(util.ddp_rank + 1) *
                                             dataset_size_per_rank]
            data = [data[i] for i in subset_indices]

        tokenizer = util.tokenizer

        prompts_str = []
        prompt_chosen_str = []
        for entry in data:
            prompts_str.append(get_prompt(entry))
            prompt_chosen_str.append(get_prompt_and_chosen(entry) + tokenizer.eos_token)

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
        self.max_n_seqs_per_batch = max_n_seqs_per_batch

        self.shuffle_cnt = 0

        self._shuffle()
        assert all(seq <= self.n_tokens_per_batch for seq in self.seqlens)
        self.__batch_indices = ffd_with_result_unsorted(np.array(self.seqlens), self.n_tokens_per_batch)
        random.shuffle(self.__batch_indices)

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
            prompt_lengths = [self.prompt_lengths[i] for i in indices]
            seqs = [self.seqs[i] for i in indices]
            prompts = [self.prompts[i] for i in indices]
            prompt_masks = [self.prompt_masks[i] for i in indices]

            total_seqlen = sum(seqlens)
            assert total_seqlen <= self.n_tokens_per_batch, (total_seqlen, self.n_tokens_per_batch)
            if total_seqlen < self.n_tokens_per_batch:
                seqlen_to_pad = self.n_tokens_per_batch - total_seqlen
                seqlens.append(seqlen_to_pad)
                prompt_lengths.append(seqlen_to_pad)
                seqs.append([self.util.tokenizer.pad_token_id] * seqlen_to_pad)
                prompts.append([self.util.tokenizer.pad_token_id] * seqlen_to_pad)
                prompt_masks.append([1] * seqlen_to_pad)

            seqlens = torch.tensor(seqlens, dtype=torch.int32)
            cu_seqlens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(seqlens, dim=0)])
            assert cu_seqlens[-1] == self.n_tokens_per_batch, (cu_seqlens[-1], self.n_tokens_per_batch)
            prompt_masks = torch.cat([torch.tensor(m, dtype=torch.bool) for m in prompt_masks])
            packed_input_ids = torch.cat([torch.tensor(p, dtype=torch.long) for p in seqs])

            if cu_seqlens.shape[0] > self.max_n_seqs_per_batch + 1:
                raise RuntimeError(
                    f"cu_seqlens.shape[0] ({cu_seqlens.shape[0]}) > max_n_seqs_per_batch + 1 ({self.max_n_seqs_per_batch + 1}), "
                    f"please set a larger max_n_seqs_per_batch.")

            cu_seqlens = torch.nn.functional.pad(cu_seqlens,
                                                 (0, self.max_n_seqs_per_batch + 1 - cu_seqlens.shape[0]),
                                                 value=-1)
            assert packed_input_ids.shape[0] == prompt_masks.shape[0] == self.n_tokens_per_batch, (
                packed_input_ids.shape[0], prompt_masks.shape[0], self.n_tokens_per_batch, total_seqlen)

            yield dict(
                packed_input_ids=packed_input_ids.unsqueeze(0),
                prompt_mask=prompt_masks.unsqueeze(0),
                cu_seqlens=cu_seqlens.unsqueeze(0),
            )
        self._shuffle()
        self.__batch_indices = ffd_with_result_unsorted(np.array(self.seqlens), self.n_tokens_per_batch)
        random.shuffle(self.__batch_indices)


api.data.register_dataset("wps_formula_packed", WPSFormulaPackedDataset)

# if __name__ == "__main__":
#     import transformers
#     util = api.data.DatasetUtility(1,
#                                    0,
#                                    1,
#                                    tokenizer=transformers.AutoTokenizer.from_pretrained(
#                                        "/lustre/meizy/backup_zy/model_saves/four_layers_starcoder"))
#     dataset = WPSFormulaPackedDataset(
#         util=util,
#         json_path="/data/aigc/llm/datasets/wps-formula-sft/dllm-train-0908-formula-psi.json",
#         n_tokens_per_batch=768,
#         max_length=768,
#     )
#     for idx in range(5):
#         print(idx)
#         for x in iter(dataset):
#             pass
#     import numpy as np
#     cap = 768
#     a = np.random.randint(256, cap, (60000, ))
#     random.shuffle(a)
#     result = ffd_with_result_unsorted(a, cap)
#     for indices in result:
#         assert sum(a[indices]) <= cap
