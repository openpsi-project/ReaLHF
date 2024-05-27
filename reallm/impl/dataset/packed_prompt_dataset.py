from typing import Callable, Dict, List, Optional, Tuple
import itertools
import json

import numpy as np
import torch.utils.data

from reallm.api.core import data_api
from reallm.base.datapack import ffd_with_result_unsorted, min_abs_diff_partition
import reallm.base.dataparallel as dataparallel
import reallm.base.logging as logging
import reallm.base.namedarray as namedarray

logger = logging.getLogger("Packed Prompt Dataset")


def split_packed_batch_into_seqs(
    sample: namedarray.NamedArray,
    input_lens: Optional[torch.Tensor] = None,
    return_seqlens: bool = False,
) -> List[namedarray.NamedArray]:
    if input_lens is None:
        if "input_lens" in sample:
            input_lens = sample["input_lens"]
        elif "prompt_lens" in sample:
            input_lens = sample["prompt_lens"]
        elif "cu_seqlens" in sample:
            input_lens = sample["cu_seqlens"][1:] - sample["cu_seqlens"][:-1]
        elif "prompt_cu_seqlens" in sample:
            input_lens = sample["prompt_cu_seqlens"][1:] - sample["prompt_cu_seqlens"][:-1]

    partitions = [(i, i + 1) for i in range(input_lens.shape[0])]
    sample["input_lens"] = input_lens
    sample.register_metadata(seqlens=input_lens.cpu().numpy().tolist())
    res = dataparallel.PackedParallelDataBroker.scatter_to(sample,
                                                           n_dp=len(input_lens),
                                                           partitions=partitions)
    if not return_seqlens:
        return res
    else:
        return res, input_lens


class PackedPromptDataset(torch.utils.data.IterableDataset):

    def __init__(
        self,
        util: data_api.DatasetUtility,
        n_tokens_per_batch: int,
        min_seqs_per_batch: int = 1,
        max_length: Optional[int] = None,
        dataset_path: Optional[str] = None,
        dataset_builder: Optional[Callable[[], List[Dict]]] = None,
        pad_to_max_length: bool = False,
    ):
        """A dataset with prompts. Usually used for PPO.

        Args:
            util (api.data.DatasetUtility): .
            n_tokens_per_batch (int, optional): The number of tokens in the batch.
            max_length (Optional[int], optional): The maximum length of each sequence in the batch. Defaults to n_tokens_per_batch.
            dataset_path (Optional[str], optional): Path to the dataset json/jsonl file.
                The json/jsonl file should be a list of dictionary. Each element in the list should have
                a key "prompt". Defaults to None.
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
                with open(dataset_path, "r") as f:
                    data = [json.loads(ff) for ff in f]
            elif dataset_path.endswith(".json"):
                with open(dataset_path, "r") as f:
                    data = json.load(f)
            else:
                raise NotImplementedError(f"Unkown extention: {dataset_path}")
        else:
            assert dataset_builder is not None
            data = dataset_builder()

        shuffle_indices = data_api.get_shuffle_indices(util.seed, len(data))
        data = [data[i] for i in shuffle_indices]

        all_prompt_str = [x["prompt"] for x in data]
        util.tokenizer.padding_side = "left"
        all_prompt_encodings = util.tokenizer(
            all_prompt_str,
            truncation=True,
            max_length=max_length,
            padding=pad_to_max_length,
            return_length=True,
            return_attention_mask=False,
        )

        start, end = min_abs_diff_partition(np.array(all_prompt_encodings["length"]),
                                            util.world_size)[util.ddp_rank]

        prompt_lengths = all_prompt_encodings["length"][start:end]
        prompts = all_prompt_encodings["input_ids"][start:end]

        prompts_str = all_prompt_str[start:end]

        self.prompt_lengths = prompt_lengths
        self.min_seqs_per_batch = min_seqs_per_batch
        self.prompts = prompts

        assert len(self.prompt_lengths) == len(self.prompts)

        logger.info(f"Number of prompts in the dataset: {len(self.prompts)}")

        self.n_tokens_per_batch = n_tokens_per_batch

        self.shuffle_cnt = 0

        self.rng = np.random.RandomState(seed=util.seed)

        self._shuffle()
        assert all(seq <= self.n_tokens_per_batch for seq in self.prompt_lengths)
        self.__batch_indices = ffd_with_result_unsorted(np.array(self.prompt_lengths),
                                                        self.n_tokens_per_batch)
        self.__batch_indices = list(filter(lambda x: len(x) >= self.min_seqs_per_batch, self.__batch_indices))
        tokens_in_batches = sum(
            [sum([self.prompt_lengths[i] for i in indices]) for indices in self.__batch_indices])
        tokens_in_dataset = sum(self.prompt_lengths)
        if tokens_in_batches < 0.5 * tokens_in_dataset:
            raise ValueError(
                f"After dynamic batch allocation, #tokens contained in batches ({tokens_in_batches}) "
                f"is less than half of the original dataset ({tokens_in_dataset}) because "
                f"min_seqs_per_batch ({self.min_seqs_per_batch}) is too large or max_length ({self.max_length}) is too large. "
                "There are not enough sequences to be dispatched to model workers in allocated batches. "
                f"Please lower max_length ({self.max_length}) or increase n_tokens_per_batch ({self.n_tokens_per_batch}). "
                f"Current #seqs per batch: {[len(x) for x in self.__batch_indices]}, "
                f"current #tokens per batch: {[sum([self.prompt_lengths[i] for i in indices]) for indices in self.__batch_indices]}. "
            )
        self.rng.shuffle(self.__batch_indices)

    def _shuffle(self):
        shuffle_indices = data_api.get_shuffle_indices(
            self.util.seed + self.shuffle_cnt * 7 + self.util.ddp_rank * 3, len(self.prompt_lengths))

        self.prompt_lengths = [self.prompt_lengths[i] for i in shuffle_indices]
        self.prompts = [self.prompts[i] for i in shuffle_indices]
        self.shuffle_cnt += 1

    @property
    def max_seqlen(self):
        return max(self.prompt_lengths)

    def __len__(self):
        return len(self.__batch_indices)

    def __iter__(self):
        for indices in self.__batch_indices:
            seqlens = [self.prompt_lengths[idx] for idx in indices]
            # assert all(x == seqlens[0] for x in seqlens), seqlens
            prompts = [self.prompts[idx] for idx in indices]
            total_seqlen = sum(seqlens)
            assert total_seqlen <= self.n_tokens_per_batch, (total_seqlen, self.n_tokens_per_batch)

            seqlens = torch.tensor(seqlens, dtype=torch.int32)
            cu_seqlens = torch.cat([torch.tensor([0], dtype=torch.int32), torch.cumsum(seqlens, dim=0)])
            packed_input_ids = torch.cat([torch.tensor(p, dtype=torch.long) for p in prompts])

            yield dict(
                packed_prompts=packed_input_ids,
                # prompt_cu_seqlens=cu_seqlens,
                prompt_lens=seqlens,
            )
        self._shuffle()
        assert all(seq <= self.n_tokens_per_batch for seq in self.prompt_lengths)
        self.__batch_indices = ffd_with_result_unsorted(np.array(self.prompt_lengths),
                                                        self.n_tokens_per_batch)
        self.__batch_indices = list(filter(lambda x: len(x) >= self.min_seqs_per_batch, self.__batch_indices))
        tokens_in_batches = sum(
            [sum([self.prompt_lengths[i] for i in indices]) for indices in self.__batch_indices])
        tokens_in_dataset = sum(self.prompt_lengths)
        if tokens_in_batches < 0.5 * tokens_in_dataset:
            raise ValueError(
                f"After dynamic batch allocation, #tokens contained in batches ({tokens_in_batches}) "
                f"is less than half of the original dataset ({tokens_in_dataset}) because "
                f"min_seqs_per_batch ({self.min_seqs_per_batch}) is too large or max_length ({self.max_length}) is too large. "
                "There are not enough sequences to be dispatched to model workers in allocated batches. "
                f"Please lower max_length ({self.max_length}) or increase n_tokens_per_batch ({self.n_tokens_per_batch}). "
                f"Current #seqs per batch: {[len(x) for x in self.__batch_indices]}, "
                f"current #tokens per batch: {[sum([self.prompt_lengths[i] for i in indices]) for indices in self.__batch_indices]}. "
            )
        self.rng.shuffle(self.__batch_indices)


if __name__ != "__main__":
    data_api.register_dataset("packed_prompt", PackedPromptDataset)
else:
    import transformers

    from reallm.base.dataparallel import PackedParallelDataBroker
    from reallm.base.namedarray import from_dict

    def have_common_prefix_at_least(a, b, n):
        return (a[:n] == b[:n]).all()

    tokenizer = transformers.AutoTokenizer.from_pretrained("/lustre/fw/pretrained/gpt2-large")
    ddp_rank = 0
    world_size = 1
    seed = 5

    util = data_api.DatasetUtility(tokenizer=tokenizer, ddp_rank=ddp_rank, world_size=world_size, seed=seed)

    n_dp = 2
    n_pp = 1
    dataset = PackedPromptDataset(
        util,
        max_length=128,
        # min_seqs_per_batch=n_dp * n_pp,
        n_tokens_per_batch=128 * 128,
        dataset_path="/lustre/meizy/data/antropic-hh/ppo_prompt_only_short.jsonl",
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)

    num_iter = 1
    for i in range(1):
        hash_sum = 0
        print("dataset iteration")
        for x in dataloader:
            # datas, sizes = PackedParallelDataBroker.scatter_to(from_dict(x), n_dp, return_sizes=True)
            # for data in datas:
            #     PackedParallelDataBroker.scatter_to(data, n_pp)
            print(x)
            print(len(x["prompt_lens"]), x["packed_prompts"].shape)
            # print(hash(x["packed_prompts"]))
            # import hashlib
            # hash_val = int(hashlib.md5(x["packed_prompts"].numpy().tobytes()).hexdigest(), 16)
            # print(hash_val)
            a = split_packed_batch_into_seqs(namedarray.from_dict(x))
            print(len(a))
            print(a[0])
            print(hash(a[0]))
            # for aa in a:
            # print(aa)
            # hash_sum += hash(aa) % 1000
            # print(hash(aa))
            break
        # print(f"hash sum of iter {i}: {hash_sum}")
