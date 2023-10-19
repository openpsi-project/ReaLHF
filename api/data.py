from typing import Dict, Iterable, Optional, Union
import dataclasses
import functools
import inspect
import logging
import os
import time

import numpy as np
import torch.utils.data
import transformers

from base.cluster import spec as cluster_spec
import api.config
import api.huggingface

logger = logging.getLogger("api.data")


@dataclasses.dataclass
class DataBatch:
    # Object returned by data workers.
    data: Dict[str, torch.Tensor]
    epoch: int
    epoch_step: int
    global_step: int


def get_shuffle_indices(seed: int, size: int):
    """Generate shuffled indices given seed and (dataset) size."""
    np_rng = np.random.RandomState(seed=seed)
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx


@dataclasses.dataclass
class DatasetUtility:
    seed: int
    ddp_rank: int
    world_size: int
    tokenizer: transformers.PreTrainedTokenizerFast


ALL_DATASET_CLASSES = {}


def register_dataset(name, dataset_cls):
    assert name not in ALL_DATASET_CLASSES
    assert '/' not in name
    # call_arg_names = list(inspect.signature(dataset_cls).parameters.keys())
    # if 'util' not in call_arg_names:
    #     raise KeyError(
    #         f"'util' must be one of the arguments in __init__, which is an instance of DatasetUtility. "
    #         f"Existing arguments: {call_arg_names}.")
    ALL_DATASET_CLASSES[name] = dataset_cls


def make_dataset(
    cfg: Union[str, api.config.Dataset],
    seed: int,
    ddp_rank: int,
    world_size: int,
    tokenizer_or_tokenizer_name: Union[transformers.PreTrainedTokenizerFast, str],
    experiment_name: str,
    trial_name: str,
    cache_root: Optional[str] = None,
) -> torch.utils.data.Dataset:
    if isinstance(cfg, str):
        cfg = api.config.Dataset(type_=cfg)

    if isinstance(tokenizer_or_tokenizer_name, str):
        tokenizer = api.huggingface.load_hf_tokenizer(tokenizer_or_tokenizer_name)
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
    if (not cache_root.startswith(cluster_spec.fileroot) and not cache_root.startswith("/home")):
        raise ValueError(f"Data cache path {cache_root} should be /home or under {cluster_spec.fileroot}.")
    if "_" in experiment_name or "_" in trial_name:
        raise ValueError(f"Invalid experiment/trial name.")

    output_path = os.path.join(cache_root, experiment_name, trial_name, cfg.type_, f"seed{seed}",
                               f"world_size{world_size}", f"rank{ddp_rank}")
    os.makedirs(output_path, exist_ok=True)

    fname = "dataset.pt"
    cache_found = os.path.isfile(os.path.join(output_path, fname))

    tik = time.perf_counter()
    if not cache_found:
        logger.info(f"No data cache found for rank {ddp_rank}. Create it from scratch.")
        dataset = ALL_DATASET_CLASSES[cfg.type_](seed, ddp_rank, world_size, **cfg.args)
        torch.save(dataset, os.path.join(output_path, fname))
    else:
        logger.info(f"Rank {ddp_rank} find existing data cache, load it.")
        dataset = torch.load(os.path.join(output_path, fname))
    logger.info(f"Dataset creation/loading time: {time.perf_counter() - tik:.3f}s")

    return dataset


ALL_DATALOADER_CLASSES = {}


def register_dataloader(name, dataloader_cls):
    assert name not in ALL_DATALOADER_CLASSES
    ALL_DATALOADER_CLASSES[name] = dataloader_cls


def make_dataloader(cfg: Union[str, api.config.DataLoader],
                    dataset: torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
    if isinstance(cfg, str):
        cfg = api.config.DataLoader(type_=cfg)
    dataloader_cls = ALL_DATALOADER_CLASSES[cfg.type_]
    return dataloader_cls(dataset, **cfg.args)


register_dataloader(
    'default',
    functools.partial(
        torch.utils.data.DataLoader,
        shuffle=True,
        drop_last=True,
        batch_size=8,
        collate_fn=transformers.default_data_collator,
    ),
)
register_dataloader(
    'default_eval',
    functools.partial(
        torch.utils.data.DataLoader,
        shuffle=False,
        drop_last=False,
        batch_size=16,
        collate_fn=transformers.default_data_collator,
    ),
)
register_dataloader(
    'iterable_dataset_loader',
    functools.partial(
        torch.utils.data.DataLoader,
        batch_size=None,
    ),
)
