from typing import Callable, Dict, List, Optional, Tuple
import itertools
import json

import numpy as np
import torch.utils.data

from reallm.base.datapack import ffd_with_result_unsorted, min_abs_diff_partition
import reallm.api.data
import reallm.base.logging as logging

logger = logging.getLogger("Packed Prompt Dataset")


class PackedPromptAnswerDataset(torch.utils.data.IterableDataset):

    def __init__(
        self,
        util: reallm.api.data.DatasetUtility,
        n_tokens_per_batch: int,
        min_seqs_per_batch: int = 1,
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

        shuffle_indices = reallm.api.data.get_shuffle_indices(util.seed, len(data))
        data = [data[i] for i in shuffle_indices]
        for x in data:
            if x['answer'].startswith(x['prompt']):
                raise ValueError("Answer should not start with prompt.")

        all_prompt_str = [x['prompt'] for x in data]
        all_prompt_chosen_str = [x['prompt'] + x['answer'] + util.tokenizer.eos_token for x in data]
        all_prompt_encodings = util.tokenizer(all_prompt_str,
                                              truncation=True,
                                              max_length=max_length,
                                              padding=False,
                                              return_length=True,
                                              return_attention_mask=False)
        all_prompt_chosen_encodings = util.tokenizer(all_prompt_chosen_str,
                                                     truncation=True,
                                                     max_length=max_length,
                                                     padding=False,
                                                     return_length=True,
                                                     return_attention_mask=False)

        start, end = min_abs_diff_partition(np.array(all_prompt_chosen_encodings['length']),
                                            util.world_size)[util.ddp_rank]

        prompt_lengths = all_prompt_encodings['length'][start:end]
        prompts = all_prompt_encodings['input_ids'][start:end]

        seqlens = all_prompt_chosen_encodings['length'][start:end]
        seqs = all_prompt_chosen_encodings['input_ids'][start:end]

        prompts_str = all_prompt_str[start:end]
        prompt_chosen_str = all_prompt_chosen_str[start:end]

        indices_to_pop = []
        prompt_masks = []
        for ii, (seq, prompt, seqlen, prompt_len, pstr, pcstr) in enumerate(
                zip(seqs, prompts, seqlens, prompt_lengths, prompts_str, prompt_chosen_str)):
            try:
                assert seq[:prompt_len] == prompt, (prompt, seq, prompt_len, pstr, pcstr)
                assert seqlen >= prompt_len, (seqlen, prompt_len)
            except AssertionError:
                indices_to_pop.append(ii)
            prompt_masks.append([1] * prompt_len + [0] * (seqlen - prompt_len))

        for ii in reversed(indices_to_pop):
            seqlens.pop(ii)
            seqs.pop(ii)
            prompts.pop(ii)
            prompt_lengths.pop(ii)
            prompt_masks.pop(ii)

        self.seqlens = seqlens
        self.prompt_lengths = prompt_lengths
        self.min_seqs_per_batch = min_seqs_per_batch
        self.seqs = seqs
        self.prompts = prompts
        self.prompt_masks = prompt_masks

        assert len(self.seqlens) == len(self.prompt_lengths) == len(self.prompts) == len(
            self.prompt_masks) == len(self.seqs)

        logger.info(f"Number of sequences in the dataset: {len(self.seqs)}")

        self.n_tokens_per_batch = n_tokens_per_batch

        self.shuffle_cnt = 0

        self.rng = np.random.RandomState(seed=util.seed)

        self._shuffle()
        assert all(seq <= self.n_tokens_per_batch for seq in self.seqlens)
        self.__batch_indices = ffd_with_result_unsorted(np.array(self.seqlens), self.n_tokens_per_batch)
        self.__batch_indices = list(filter(lambda x: len(x) >= self.min_seqs_per_batch, self.__batch_indices))
        tokens_in_batches = sum([sum([self.seqlens[i] for i in indices]) for indices in self.__batch_indices])
        tokens_in_dataset = sum(self.seqlens)
        if tokens_in_batches < 0.5 * tokens_in_dataset:
            raise ValueError(
                f"After dynamic batch allocation, #tokens contained in batches ({tokens_in_batches}) "
                f"is less than half of the original dataset ({tokens_in_dataset}) because "
                f"min_seqs_per_batch ({self.min_seqs_per_batch}) is too large or max_length ({self.max_length}) is too large. "
                "There are not enough sequences to be dispatched to model workers in allocated batches. "
                f"Please lower max_length ({self.max_length}) or increase n_tokens_per_batch ({self.n_tokens_per_batch}). "
                f"Current #seqs per batch: {[len(x) for x in self.__batch_indices]}, "
                f"current #tokens per batch: {[sum([self.seqlens[i] for i in indices]) for indices in self.__batch_indices]}. "
            )
        self.rng.shuffle(self.__batch_indices)

    def _shuffle(self):
        shuffle_indices = reallm.api.data.get_shuffle_indices(
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
        self.__batch_indices = list(filter(lambda x: len(x) >= self.min_seqs_per_batch, self.__batch_indices))
        tokens_in_batches = sum([sum([self.seqlens[i] for i in indices]) for indices in self.__batch_indices])
        tokens_in_dataset = sum(self.seqlens)
        if tokens_in_batches < 0.5 * tokens_in_dataset:
            raise ValueError(
                f"After dynamic batch allocation, #tokens contained in batches ({tokens_in_batches}) "
                f"is less than half of the original dataset ({tokens_in_dataset}) because "
                f"min_seqs_per_batch ({self.min_seqs_per_batch}) is too large or max_length ({self.max_length}) is too large. "
                "There are not enough sequences to be dispatched to model workers in allocated batches. "
                f"Please lower max_length ({self.max_length}) or increase n_tokens_per_batch ({self.n_tokens_per_batch}). "
                f"Current #seqs per batch: {[len(x) for x in self.__batch_indices]}, "
                f"current #tokens per batch: {[sum([self.seqlens[i] for i in indices]) for indices in self.__batch_indices]}. "
            )
        self.rng.shuffle(self.__batch_indices)


if __name__ != "__main__":
    reallm.api.data.register_dataset("packed_prompt_answer", PackedPromptAnswerDataset)
else:
    import transformers

    from reallm.base.dataparallel import PackedParallelDataBroker
    from reallm.base.namedarray import from_dict

    def have_common_prefix_at_least(a, b, n):
        return (a[:n] == b[:n]).all()

    tokenizer = transformers.AutoTokenizer.from_pretrained("/lustre/fw/pretrained/gpt2-large")
    ddp_rank = 0
    world_size = 1
    seed = 1

    util = reallm.api.data.DatasetUtility(tokenizer=tokenizer,
                                          ddp_rank=ddp_rank,
                                          world_size=world_size,
                                          seed=seed)

    n_dp = 100
    dataset = PackedPromptAnswerDataset(
        util,
        max_length=10240,
        min_seqs_per_batch=n_dp,
        n_tokens_per_batch=40960,
        dataset_path="/lustre/fw/datasets/imdb/rl/sft_pos-train.jsonl",
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)
    for _ in range(10):
        print("dataset iteration")
        for x in dataloader:
            datas = PackedParallelDataBroker.scatter_to(from_dict(x), n_dp)
            continue
