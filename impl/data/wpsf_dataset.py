from typing import Dict, List, Optional, Tuple
import itertools
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


class WPSFormulaPackedSFTDataset(torch.utils.data.IterableDataset):

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
        self.max_length = max_length

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
            prompts_str.append(PROMPT_FORMAT.format(input=entry['input'], output=""))
            prompt_chosen_str.append(
                PROMPT_FORMAT.format(input=entry['input'], output=entry['output']) + tokenizer.eos_token)

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
                n_pads = (seqlen_to_pad + self.max_length - 1) // self.max_length
                for j in range(n_pads):
                    padlen = self.max_length if j < n_pads - 1 else seqlen_to_pad % self.max_length
                    seqlens.append(padlen)
                    prompt_lengths.append(padlen)
                    seqs.append([self.util.tokenizer.pad_token_id] * padlen)
                    prompts.append([self.util.tokenizer.pad_token_id] * padlen)
                    prompt_masks.append([1] * padlen)

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


api.data.register_dataset("wpsf_sft_packed", WPSFormulaPackedSFTDataset)

RUBBISH_CODE_COLLECTIONS = ['\n\n\n', '=\n', '=====']


class WPSFormulaPackedRWDataset(torch.utils.data.IterableDataset):

    def __init__(
        self,
        util: api.data.DatasetUtility,
        contrastive_dim: int,
        enforce_one_or_less_pos: bool,
        n_tokens_per_batch: int = 2048,
        max_n_seqs_per_batch: int = 40,
        max_length: Optional[int] = None,
        json_path: str = "/data/aigc/llm/datasets/wps-formula-rw/dataset_train.jsonl",
    ):
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

        with open(json_path, 'r') as f:
            data: List[Dict] = [json.loads(line) for line in f.readlines()]
            dataset_size = len(data)
            dataset_size_per_rank = dataset_size // util.world_size
            shuffle_indices = api.data.get_shuffle_indices(util.seed, dataset_size)
            subset_indices = shuffle_indices[util.ddp_rank * dataset_size_per_rank:(util.ddp_rank + 1) *
                                             dataset_size_per_rank]
            self_data = [data[i] for i in subset_indices]

            if util.world_size > 1:
                other_indices = np.concatenate([
                    shuffle_indices[rank * dataset_size_per_rank:(rank + 1) * dataset_size_per_rank]
                    for rank in range(util.world_size) if rank != util.ddp_rank
                ])
                global_data = self_data + [data[j] for j in other_indices]
            else:
                global_data = self_data

        tokenizer = util.tokenizer

        all_seqs = [[
            PROMPT_FORMAT.format(input=d['task'], output=x['code']) + tokenizer.eos_token
            for x in d['labeled_codes']
        ] for d in global_data]
        n_labeled_codes = [len(x) for x in all_seqs]
        all_encodings = tokenizer(list(itertools.chain.from_iterable(all_seqs)),
                                  truncation=True,
                                  max_length=max_length,
                                  padding=False,
                                  return_attention_mask=False)
        all_input_ids = all_encodings['input_ids']

        global_input_ids = []
        start = 0
        for n in n_labeled_codes:
            global_input_ids.append(all_input_ids[start:start + n])
            start += n
        global_correctness_labels = [[x['correctness_label'] for x in d['labeled_codes']]
                                     for d in global_data]

        # the most inner list is token ids, the middle list is answers for the same prompt, the outer list is prompts
        self.global_input_ids: List[List[List[int]]] = global_input_ids
        self.global_correctness_labels: List[List[int]] = global_correctness_labels

        self.local_input_ids = global_input_ids[:dataset_size_per_rank]
        self.local_correctness_labels = global_correctness_labels[:dataset_size_per_rank]

        # some rubbish code for robustness
        rubbish_encodings = tokenizer([x + tokenizer.eos_token for x in RUBBISH_CODE_COLLECTIONS],
                                      truncation=True,
                                      max_length=max_length,
                                      padding=False,
                                      return_attention_mask=False)
        self.rubbish_input_ids = rubbish_encodings['input_ids']

        # attributes for generating batches
        self.contrastive_dim = contrastive_dim
        self.enforce_one_or_less_pos = enforce_one_or_less_pos

        self.n_tokens_per_batch = n_tokens_per_batch
        self.max_n_seqs_per_batch = max_n_seqs_per_batch

        self.shuffle_cnt = 0

        # build packed inputs for flash attention
        self.codes = self.labels = self.lengths = None
        self._build_shuffled_contrastive_tuples()
        assert all(length <= self.n_tokens_per_batch
                   for length in self.lengths), (max(self.lengths), self.n_tokens_per_batch)
        self.__batch_indices = ffd_with_result_unsorted(np.array(self.lengths), self.n_tokens_per_batch)
        random.shuffle(self.__batch_indices)

    def _build_shuffled_contrastive_tuples(self):
        codes, labels, lengths = list(
            zip(*list(map(self._build_one_contrastive_tuple, range(len(self.local_input_ids))))))
        shuffle_indices = api.data.get_shuffle_indices(
            self.util.seed + self.shuffle_cnt * 7 + self.util.ddp_rank * 3, len(codes))

        self.codes: List[List[int]] = [codes[i] for i in shuffle_indices]
        self.labels: List[torch.Tensor] = [labels[i] for i in shuffle_indices]
        self.lengths: List[int] = [lengths[i] for i in shuffle_indices]  # length of each contrastive batch

        self.shuffle_cnt += 1

    def _build_one_contrastive_tuple(self, idx: int) -> Tuple[List[torch.Tensor], torch.Tensor, int]:
        # For the idx-th prompt, we have many answers and corresponding correctness labels (0 or 1).
        # Remark the number of positive samples as n_pos, and the number of negative samples as n_neg.
        # We sample as many as positive samples from the answers, and sample (contrastive_dim - n_pos) negative samples.
        # These samples build a contrastive batch. The reward model will score each individual sample in it.
        # The probability of that an individual sample is correct is exp(logits) / (\sum exp(all logits) + 1).
        # If there is no positive samples, the label is [1, 0, ..., 0].
        # Otherwise, the label is [0, 1 / n_pos, ..., 1 / n_pos, 0, ..., 0].
        assert 0 <= idx < len(self.local_input_ids)
        code_input_ids = self.local_input_ids[idx]
        correctness_labels = self.local_correctness_labels[idx]

        n_pos = sum(correctness_labels)
        existing_neg_codes = [x for i, x in enumerate(code_input_ids) if not correctness_labels[i]]
        pos_codes = [x for i, x in enumerate(code_input_ids) if correctness_labels[i]]
        if n_pos == 0:
            n_required_neg_codes = self.contrastive_dim
        elif self.enforce_one_or_less_pos and n_pos > 1:
            n_required_neg_codes = self.contrastive_dim - 1
        else:
            n_required_neg_codes = max(0, self.contrastive_dim - n_pos)

        # If n_pos + n_neg < contrastive_dim, we need to sample other negative samples.
        # These samples include answers of other prompts and rubbish codes.
        sampled_other_code_indices = []
        sampled_rubbish_code_indices = []
        while len(existing_neg_codes) < n_required_neg_codes:
            if np.random.random() < 0.1:
                rubbish_code_idx = np.random.choice(len(RUBBISH_CODE_COLLECTIONS))
                if rubbish_code_idx in sampled_rubbish_code_indices:
                    continue
                existing_neg_codes.append(self.rubbish_input_ids[rubbish_code_idx])
                sampled_rubbish_code_indices.append(rubbish_code_idx)
                continue

            other_data_idx = np.random.choice(len(self.global_input_ids))
            if other_data_idx == idx:
                continue
            other_code_idx = np.random.choice(len(self.global_input_ids[other_data_idx]))
            if (other_data_idx, other_code_idx) in sampled_other_code_indices:
                continue
            other_code = self.global_input_ids[other_data_idx][other_code_idx]
            existing_neg_codes.append(other_code)
            sampled_other_code_indices.append((other_data_idx, other_code_idx))

        # randomly discard negative codes if too many
        # this may happend when contrastive_dim is set too small
        if len(existing_neg_codes) > n_required_neg_codes:
            random.shuffle(existing_neg_codes)
            existing_neg_codes = existing_neg_codes[:n_required_neg_codes]

        if self.enforce_one_or_less_pos and n_pos > 1:
            codes = [random.choice(pos_codes)] + existing_neg_codes
            label = [0] + [1] + [0] * len(existing_neg_codes)
        elif n_pos > 0:
            pos_code_indices = np.random.choice(len(pos_codes), size=self.contrastive_dim - len(existing_neg_codes), replace=False)
            pos_codes = [pos_codes[kk] for kk in pos_code_indices]
            codes = pos_codes + existing_neg_codes
            label = [0] + [1 / len(pos_codes)] * len(pos_codes) + [0] * len(existing_neg_codes)
        else:
            codes = existing_neg_codes
            label = [1] + [0] * len(existing_neg_codes)
        assert len(codes) == self.contrastive_dim, len(codes)
        assert len(label) == self.contrastive_dim + 1, len(label)

        return codes, torch.tensor(label, dtype=torch.float32), sum(len(c) for c in codes)

    def __len__(self):
        return len(self.__batch_indices)

    def __iter__(self):
        for indices in self.__batch_indices:
            code_lengths = [[len(x) for x in self.codes[i]] for i in indices]
            codes = [self.codes[i] for i in indices]

            total_seqlen = sum(itertools.chain.from_iterable(code_lengths))
            assert total_seqlen <= self.n_tokens_per_batch, (total_seqlen, self.n_tokens_per_batch)
            if total_seqlen < self.n_tokens_per_batch:
                seqlen_to_pad = self.n_tokens_per_batch - total_seqlen
                n_pads = (seqlen_to_pad + self.max_length - 1) // self.max_length
                for j in range(n_pads):
                    padlen = self.max_length if j < n_pads - 1 else seqlen_to_pad % self.max_length
                    code_lengths.append([padlen])
                    codes.append([[self.util.tokenizer.pad_token_id] * padlen])

            seqlens = torch.tensor(list(itertools.chain.from_iterable(code_lengths)), dtype=torch.int32)
            cu_seqlens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(seqlens, dim=0)])
            assert cu_seqlens[-1] == self.n_tokens_per_batch, (cu_seqlens[-1], self.n_tokens_per_batch)
            packed_input_ids = torch.cat(
                [torch.tensor(c, dtype=torch.long) for c in itertools.chain.from_iterable(codes)])
            labels = torch.stack([self.labels[i] for i in indices], dim=0)

            if cu_seqlens.shape[0] > self.max_n_seqs_per_batch + 1:
                raise RuntimeError(
                    f"cu_seqlens.shape[0] ({cu_seqlens.shape[0]}) > max_n_seqs_per_batch + 1 ({self.max_n_seqs_per_batch + 1}), "
                    f"please set a larger max_n_seqs_per_batch.")

            # pad this to max_n_seqs_per_batch, such that this tensor has the same shape in different data workers
            cu_seqlens = torch.nn.functional.pad(cu_seqlens,
                                                 (0, self.max_n_seqs_per_batch + 1 - cu_seqlens.shape[0]),
                                                 value=-1)
            n_c_batches = torch.tensor([labels.shape[0]], dtype=torch.long)
            contrastive_dim = torch.tensor([self.contrastive_dim], dtype=torch.long)
            assert labels.shape[-1] == self.contrastive_dim + 1
            labels = labels.flatten()
            if labels.shape[0] > self.max_n_seqs_per_batch:
                raise RuntimeError("Please set a larger max_n_seqs_per_batch.")
            labels = torch.nn.functional.pad(labels, (0, self.max_n_seqs_per_batch - labels.shape[0]),
                                             value=-1)

            yield dict(
                # unsqueeze for convinient batching and splitting in master worker
                packed_input_ids=packed_input_ids.unsqueeze(0),
                cu_seqlens=cu_seqlens.unsqueeze(0),
                labels=labels.unsqueeze(0),
                n_contrastive_batches=n_c_batches.unsqueeze(0),
                contrastive_dim=contrastive_dim.unsqueeze(0),
            )
        self._build_shuffled_contrastive_tuples()
        assert all(length <= self.n_tokens_per_batch for length in self.lengths)
        self.__batch_indices = ffd_with_result_unsorted(np.array(self.lengths), self.n_tokens_per_batch)
        random.shuffle(self.__batch_indices)


api.data.register_dataset("wpsf_plrw_packed", WPSFormulaPackedRWDataset)
