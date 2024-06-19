from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import collections
import dataclasses
import functools
import inspect
import json
import os
import time

import numpy as np
import torch.utils.data
import transformers

from realhf.api.core import model_api, system_api
from realhf.base import datapack, logging, namedarray
from realhf.base.cluster import spec as cluster_spec

logger = logging.getLogger("api.data")


@dataclasses.dataclass
class DataBatchMeta:
    dp_rank: int
    keys: List[str]
    seqlens: List[int]
    epoch: int
    is_final_batch: bool
    hash_vals: List[int]


@dataclasses.dataclass
class DatasetUtility:
    seed: int
    ddp_rank: int
    world_size: int
    tokenizer: transformers.PreTrainedTokenizerFast

    def __post_init__(self):
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            if self.tokenizer.eos_token_id is None:
                raise ValueError("eos_token_id of tokenizer must be defined.")


def unpack_data_batch(x: DataBatchMeta) -> List[DataBatchMeta]:
    return [
        DataBatchMeta(x.dp_rank, x.keys, [seqlen], x.epoch, False, [hash_val])
        for seqlen, hash_val in zip(x.seqlens, x.hash_vals)
    ]


def get_shuffle_indices(seed: int, size: int):
    """Generate shuffled indices given seed and (dataset) size."""
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx


def load_shuffle_split_dataset(
    util: DatasetUtility,
    dataset_path: str,
    dataset_builder: Optional[Callable[[], List[Dict[str, str]]]] = None,
):
    if dataset_path is not None:
        if dataset_path.endswith(".jsonl"):
            with open(dataset_path, "r") as f:
                data = [json.loads(ff) for ff in f]
        elif dataset_path.endswith(".json"):
            with open(dataset_path, "r") as f:
                data = json.load(f)
        else:
            raise NotImplementedError(
                f"Unkown dataset extension: {dataset_path}"
            )
    else:
        assert dataset_builder is not None
        data = dataset_builder()

    datasize_per_rank = len(data) // util.world_size
    shuffle_indices = get_shuffle_indices(
        util.seed, datasize_per_rank * util.world_size
    )
    subset_indices = shuffle_indices[
        util.ddp_rank
        * datasize_per_rank : (util.ddp_rank + 1)
        * datasize_per_rank
    ]
    data: List[Dict[str, str]] = [data[i] for i in subset_indices]

    return data


ALL_DATASET_CLASSES = {}


def register_dataset(name, dataset_cls):
    assert name not in ALL_DATASET_CLASSES
    assert "/" not in name
    # call_arg_names = list(inspect.signature(dataset_cls).parameters.keys())
    # if 'util' not in call_arg_names:
    #     raise KeyError(
    #         f"'util' must be one of the arguments in __init__, which is an instance of DatasetUtility. "
    #         f"Existing arguments: {call_arg_names}.")
    ALL_DATASET_CLASSES[name] = dataset_cls


def make_dataset(
    cfg: Union[str, system_api.Dataset],
    seed: int,
    ddp_rank: int,
    world_size: int,
    tokenizer_or_tokenizer_name: Union[
        transformers.PreTrainedTokenizerFast, str
    ],
    experiment_name: str,
    trial_name: str,
    cache_root: Optional[str] = None,
) -> torch.utils.data.Dataset:
    if isinstance(cfg, str):
        cfg = system_api.Dataset(type_=cfg)

    if isinstance(tokenizer_or_tokenizer_name, str):
        tokenizer = model_api.load_hf_tokenizer(tokenizer_or_tokenizer_name)
    elif tokenizer_or_tokenizer_name is None:
        raise RuntimeError("tokenizer_or_tokenizer_name cannot be None.")
    else:
        tokenizer = tokenizer_or_tokenizer_name
    util = DatasetUtility(
        seed,
        ddp_rank,
        world_size,
        tokenizer,
    )

    if cache_root is None:
        dataset_cls = ALL_DATASET_CLASSES[cfg.type_]
        return dataset_cls(util=util, **cfg.args)

    # Create and check cache path.
    if not cache_root.startswith(
        cluster_spec.fileroot
    ) and not cache_root.startswith("/home"):
        raise ValueError(
            f"Data cache path {cache_root} should be /home or under {cluster_spec.fileroot}."
        )
    if "_" in experiment_name or "_" in trial_name:
        raise ValueError(f"Invalid experiment/trial name.")

    output_path = os.path.join(
        cache_root,
        experiment_name,
        trial_name,
        cfg.type_,
        f"seed{seed}",
        f"world_size{world_size}",
        f"rank{ddp_rank}",
    )
    os.makedirs(output_path, exist_ok=True)

    fname = "dataset.pt"
    cache_found = os.path.isfile(os.path.join(output_path, fname))

    tik = time.perf_counter()
    if not cache_found:
        logger.info(
            f"No data cache found for rank {ddp_rank}. Create it from scratch."
        )
        dataset = ALL_DATASET_CLASSES[cfg.type_](
            seed, ddp_rank, world_size, **cfg.args
        )
        torch.save(dataset, os.path.join(output_path, fname))
    else:
        logger.info(f"Rank {ddp_rank} find existing data cache, load it.")
        dataset = torch.load(os.path.join(output_path, fname))
    logger.info(
        f"Dataset creation/loading time: {time.perf_counter() - tik:.3f}s"
    )

    return dataset


ALL_DATALOADER_CLASSES = {}


def register_dataloader(name, dataloader_cls):
    assert name not in ALL_DATALOADER_CLASSES
    ALL_DATALOADER_CLASSES[name] = dataloader_cls


def make_dataloader(
    cfg: Union[str, system_api.DataLoader], dataset: torch.utils.data.Dataset
) -> torch.utils.data.DataLoader:
    if isinstance(cfg, str):
        cfg = system_api.DataLoader(type_=cfg)
    dataloader_cls = ALL_DATALOADER_CLASSES[cfg.type_]
    return dataloader_cls(dataset, **cfg.args)


def PackedDataLoader(dataset, *args, **kwargs):
    return torch.utils.data.DataLoader(
        dataset,
        *args,
        collate_fn=gather_sequences,
        # NOTE: This is *NOT* the actual batch size for training.
        # It is just a proper size to load data to workers.
        batch_size=512,
        shuffle=True,
        **kwargs,
    )


def PackedEvalDataLoader(dataset, *args, **kwargs):
    return torch.utils.data.DataLoader(
        dataset,
        *args,
        collate_fn=gather_sequences,
        shuffle=False,
        **kwargs,
    )


register_dataloader("packed", PackedDataLoader)
register_dataloader("packed_eval", PackedEvalDataLoader)


def split_sequences(
    src: namedarray.NamedArray,
    n_dp: int = None,
    return_sizes: bool = False,
    partitions: Optional[List[Tuple[int, int]]] = None,
    min_size: int = 1,
    return_partitions=False,
) -> List[namedarray.NamedArray]:
    # FIXME: remove cu_seqlens here
    if src.metadata.get("seqlens", None) is None:
        raise ValueError(
            "seqlens must be in the metadata of the input namedarray."
        )

    seqlens = src.metadata["seqlens"]

    if n_dp is None:
        partitions = [(i, i + 1) for i in range(len(seqlens))]
        n_dp = len(seqlens)

    if partitions is None:
        partitions = datapack.min_abs_diff_partition(seqlens, n_dp, min_size)

    batch_sizes = [end - start for start, end in partitions]

    # These are used by log probabilities, which are one-step shorter than packed inputed ids.
    # We use numpy/list for indexing to avoid host-device synchronization
    batch_seqlens = [sum(seqlens[start:end]) for start, end in partitions]
    offsets = [0] + np.cumsum(batch_seqlens).tolist()[:-1]
    short1batch_seqlens = [
        sum(seqlens[start:end]) - (end - start) for start, end in partitions
    ]
    short1offsets = [0] + np.cumsum(short1batch_seqlens).tolist()[:-1]

    splitted_data = [collections.defaultdict() for _ in range(n_dp)]
    for k, v in src.items():
        for i, sp in enumerate(splitted_data):
            # NOTE: This is a terrible implementation, because we must know how to split each tensor according to its semantics.
            # Different tensor has different shape and semantics,
            # e.g., packed_seq has shape [tot_seqlen] and should be splited according to cumulative lengths,
            # packed_logprobs has shape [tot_seqlen - bs] (each sequence is one-step shorter) and should be splitted
            # according to short-1 cumulative lengths, seq_no_eos_mask has shape [bs] and performs similar as input_lens, ...
            # so we must enumerate each possible key and deal with them separately, etc.
            if v is None:
                sp[k] = None
            elif k in ["prompt_cu_seqlens", "cu_seqlens"]:
                continue
            elif k in [
                "prompt_lens",
                "input_lens",
                "seq_no_eos_mask",
                "rewards",
                "reward_score",
                "group_factor",
                "prompt_lens",
                "pos_input_lens",
                "group_input_lens",
                "seqlogp",
            ]:
                start, end = partitions[i]
                sp[k] = v[start:end]
            elif k in [
                "packed_seq",
                "packed_logits_mask",
                "prompt_mask",
                "packed_input_ids",
                "values",
                "logits_mask",
                "packed_prompts",
            ]:
                sp[k] = v[offsets[i] : offsets[i] + batch_seqlens[i]]
            elif k in [
                "packed_logprobs",
                "packed_ref_logprobs",
                "old_logp",
                "ref_logp",
                "advantages",
                "ppo_loss_mask",
                "kl_rewards",
                "returns",
            ]:
                sp[k] = v[
                    short1offsets[i] : short1offsets[i] + short1batch_seqlens[i]
                ]
            elif not torch.is_tensor(src[k]):
                # for constant, preserve value for each splitted instance
                sp[k] = src[k]
            else:
                raise RuntimeError(
                    f"Unknown key {k} in packed data. We don't know how to split it. "
                    f"Check api/core/data_api.py for implemented keys."
                )

    if "cu_seqlens" in src.keys():
        for x, (start, end) in zip(splitted_data, partitions):
            slens = torch.tensor(
                seqlens[start:end],
                dtype=torch.int32,
                device=src["cu_seqlens"].device,
            )
            x["cu_seqlens"] = torch.nn.functional.pad(
                slens.cumsum(dim=0), (1, 0)
            ).int()

    if "prompt_cu_seqlens" in src.keys():
        raw_prompt_lens = (
            src["prompt_cu_seqlens"][1:] - src["prompt_cu_seqlens"][:-1]
        )
        all_prompt_lens: List[torch.IntTensor] = [
            raw_prompt_lens[start:end].int() for start, end in partitions
        ]
        for x, pslens in zip(splitted_data, all_prompt_lens):
            x["prompt_cu_seqlens"] = torch.nn.functional.pad(
                pslens.cumsum(dim=0), (1, 0)
            ).int()

    splitted_data = [namedarray.from_dict(dict(x)) for x in splitted_data]
    for x, (start, end) in zip(splitted_data, partitions):
        x.register_metadata(seqlens=seqlens[start:end])

    res = [splitted_data]
    if return_sizes:
        res.append(batch_sizes)
    if return_partitions:
        res.append(partitions)

    if not return_sizes and not return_partitions:
        return res[0]
    else:
        return res


def gather_sequences(src: List[namedarray.NamedArray]) -> namedarray.NamedArray:
    seqlens = []
    for x in src:
        if x.metadata.get("seqlens", None) is None:
            raise ValueError(
                "seqlens must be in the metadata of the input namedarray."
            )
        seqlens += x.metadata["seqlens"]

    res = namedarray.recursive_aggregate(src, lambda x: torch.cat(x, dim=0))

    if "cu_seqlens" in src[0]:
        slens = torch.cat(
            [x["cu_seqlens"][1:] - x["cu_seqlens"][:-1] for x in src], dim=0
        )
        res["cu_seqlens"] = torch.nn.functional.pad(
            slens.cumsum(dim=0), (1, 0)
        ).int()

    if "prompt_cu_seqlens" in src[0]:
        slens = torch.cat(
            [
                x["prompt_cu_seqlens"][1:] - x["prompt_cu_seqlens"][:-1]
                for x in src
            ],
            dim=0,
        )
        res["prompt_cu_seqlens"] = torch.nn.functional.pad(
            slens.cumsum(dim=0), (1, 0)
        ).int()

    res.register_metadata(seqlens=seqlens)
    return res


def get_shape_from_key_and_seqlen(k: str, seqlen: int, vocab_size: int):
    if k in [
        "input_lens",
        "prompt_lens",
        "seq_no_eos_mask",
        "rewards",
        "reward_score",
        "group_factor",
        "pos_input_lens",
    ]:
        shape = (1,)
    elif k in ["cu_seqlens", "prompt_cu_seqlens"]:
        shape = (2,)
    # FIXME: problem here if we use groups instead of pairs?
    elif k in ["seqlogp"]:
        shape = (1, 2)
    elif k in [
        "packed_seq",
        "prompt_mask",
        "packed_input_ids",
        "values",
        "packed_prompts",
    ]:
        shape = (seqlen,)
    elif k in [
        "packed_logprobs",
        "packed_ref_logprobs",
        "old_logp",
        "ref_logp",
        "advantages",
        "ppo_loss_mask",
        "kl_rewards",
        "returns",
    ]:
        shape = (seqlen - 1,)
    elif k in ["logits_mask", "packed_logits_mask"]:
        shape = (seqlen, vocab_size)
    else:
        raise NotImplementedError(f"Unknown key {k} in packed data.")
    return shape


def get_dtype_from_key(k: str):
    if k in [
        "seq_no_eos_mask",
        "ppo_loss_mask",
        "prompt_mask",
        "logits_mask",
        "packed_logits_mask",
    ]:
        dtype = torch.bool
    elif k in [
        "reward_score",
        "packed_ref_logprobs",
        "old_logp",
        "ref_logp",
        "advantages",
        "kl_rewards",
        "returns",
        "values",
    ]:
        dtype = torch.float16
    elif k in [
        "input_lens",
        "prompt_lens",
        "cu_seqlens",
        "prompt_cu_seqlens",
        "pos_input_lens",
    ]:
        dtype = torch.int32
    elif k in ["packed_seq", "packed_input_ids", "packed_prompts"]:
        dtype = torch.int64
    elif k in ["rewards", "packed_logprobs", "group_factor", "seqlogp"]:
        dtype = torch.float32
    else:
        raise NotImplementedError(f"Unknown key {k} in packed data.")
    return dtype
