from typing import Dict, Iterable, Optional, Union
import dataclasses
import functools
import logging
import os
import time

import numpy as np
import torch.utils.data
import transformers

import api.config
import api.utils

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


class Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        seed: int,
        ddp_rank: int,
        world_size: int,
        tokenizer: transformers.PreTrainedTokenizerFast,
    ):
        """Initialize the subset of dataset belonging to the current worker.
        
        Dataset split is done by the __init__ function such that different workers
        can only load different subsets instead of the whole dataset, which is usually
        faster (e.g. parallel tokenization) and more memory efficient.

        Args:
            seed (int): Random seed used to create split indices.
            ddp_rank (int): ...
            world_size (int): ...
            tokenizer (transformers.PreTrainedTokenizerFast): ...
        """
        super().__init__()
        self.__seed = seed
        self.__ddp_rank = ddp_rank
        self.__world_size = world_size
        self.__tokenizer = tokenizer

    @property
    def seed(self) -> int:
        return self.__seed

    @property
    def ddp_rank(self) -> int:
        return self.__ddp_rank

    @property
    def world_size(self) -> int:
        return self.__world_size

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizerFast:
        return self.__tokenizer

    def __len__(self) -> int:
        raise NotImplementedError()

    def __getitem__(self, index: int) -> Dict:
        raise NotImplementedError()


class ConcatDataset(torch.utils.data.ConcatDataset):

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)
        for key in ['seed', 'ddp_rank', 'world_size']:
            if len(set([getattr(d, key) for d in datasets])) != 1:
                raise ValueError(
                    f"All datasets must have the same {key}. Given: {[getattr(d, key) for d in datasets]}")
        if len(set([d.tokenizer.__class__.__name__ for d in datasets])) != 1:
            raise ValueError("All datasets must have the same tokenizer.")

    @property
    def seed(self) -> int:
        return self.datasets[0].seed

    @property
    def ddp_rank(self) -> int:
        return self.datasets[0].ddp_rank

    @property
    def world_size(self) -> int:
        return self.datasets[0].world_size

    @property
    def tokenizer(self) -> transformers.PreTrainedTokenizerFast:
        return self.datasets[0].tokenizer


ALL_DATASET_CLASSES = {}


def register_dataset(name, dataset_cls):
    assert name not in ALL_DATASET_CLASSES
    assert '/' not in name
    ALL_DATASET_CLASSES[name] = dataset_cls


def make_dataset(
    cfg: Union[str, api.config.Dataset],
    seed: int,
    ddp_rank: int,
    world_size: int,
    experiment_name: str,
    trial_name: str,
    cache_root: Optional[str] = None,
    is_eval: bool = False,
) -> Dataset:
    if isinstance(cfg, str):
        cfg = api.config.Dataset(type_=cfg)

    if cache_root is None:
        dataset_cls = ALL_DATASET_CLASSES[cfg.type_]
        return dataset_cls(seed, ddp_rank, world_size, **cfg.args)

    # Create and check cache path.
    if (not cache_root.startswith("/data") and not cache_root.startswith("/hddlustre")):
        raise ValueError(f"Data cache path {cache_root} is not on NFS"
                         " (either '/data' or '/hddlustre').")
    if "_" in experiment_name or "_" in trial_name:
        raise ValueError(f"Invalid experiment/trial name.")

    output_path = os.path.join(cache_root, experiment_name, trial_name, cfg.type_, f"seed{seed}",
                               f"world_size{world_size}", f"rank{ddp_rank}")
    if is_eval:
        output_path = os.path.join(output_path, "eval")
    else:
        output_path = os.path.join(output_path, "train")
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


def make_dataloader(cfg: Union[str, api.config.DataLoader], dataset: Dataset) -> torch.utils.data.DataLoader:
    if isinstance(cfg, str):
        cfg = api.config.DataLoader(type_=cfg)
    dataloader_cls = ALL_DATALOADER_CLASSES[cfg.type_]
    return dataloader_cls(dataset, **cfg.args)


class TrivialDataset(Dataset):

    def __init__(self, seed, ddp_rank, world_size, tokenizer_name_or_path):
        tokenizer = api.utils.load_hf_tokenizer(tokenizer_name_or_path)
        super().__init__(seed, ddp_rank, world_size, tokenizer)

    def __len__(self):
        return 1024

    def __getitem__(self, index):
        return dict(x=torch.rand(1), y=torch.full((1,), index))


register_dataset('trivial', TrivialDataset)

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
