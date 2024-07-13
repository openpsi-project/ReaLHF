import dataclasses
import json
import os
import random
import time

# NOTE: We don't sue wildcard importing here because the type
# `Sequence` has a very similar name to `SequenceSample`.
# We don't want to confuse them.
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.utils.data
import transformers

# NOTE: We only use pandatic dataclasses for SequenceSample
# such that it will perform automatic checks.
from pydantic import Field
from pydantic import dataclasses as pdclasses
from pydantic import field_validator, model_validator

from realhf.api.core import config as config_api
from realhf.base import datapack, logging
from realhf.base.cluster import spec as cluster_spec

logger = logging.getLogger("api.data")


def load_hf_tokenizer(
    model_name_or_path: str,
    fast_tokenizer=True,
    padding_side: Optional[str] = None,
) -> transformers.PreTrainedTokenizerFast:
    kwargs = {}
    if padding_side is not None:
        kwargs["padding_side"] = padding_side
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path, fast_tokenizer=fast_tokenizer, **kwargs
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


@pdclasses.dataclass
class SequenceSplitSpec:
    partitions: Optional[List[Tuple[int, int]]] = None
    sizes: Optional[List[int]] = None

    @model_validator(mode="after")
    def _validate_partitions(self) -> "SequenceSplitSpec":
        if self.partitions is not None:
            bound = 0
            for start, end in self.partitions:
                if start >= end:
                    raise ValueError(f"Partition {start}-{end} is empty.")
                if start != bound:
                    raise ValueError(f"Partition {start}-{end} is not contiguous.")
                bound = end

        if self.sizes is None and self.partitions is None:
            raise ValueError("Either sizes or partitions must be provided.")
        elif self.sizes is not None and self.partitions is not None:
            if len(self.sizes) != len(self.partitions):
                raise ValueError("Sizes and partitions are not the consistent.")
            if self.sizes != [end - start for start, end in self.partitions]:
                raise ValueError("Sizes and partitions are not the consistent.")
        elif self.sizes is None:
            self.sizes = [end - start for start, end in self.partitions]
        elif self.partitions is None:
            offsets = np.cumsum([0] + self.sizes)
            self.partitions = [
                (offsets[i], offsets[i + 1]) for i in range(len(self.sizes))
            ]

        return self


@pdclasses.dataclass(config=dict(arbitrary_types_allowed=True))
class SequenceSample:
    keys: Set[str]
    trailing_shapes: Dict[str, torch.Size | Tuple | None]
    dtypes: Dict[str, torch.dtype | None]

    ids: List[Hashable]

    seqlens: Dict[str, List[torch.Tensor]]

    data: Optional[Dict[str, torch.Tensor | None]] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("ids")
    @classmethod
    def _validate_ids(cls, ids: List[Hashable]) -> List[Hashable]:
        if len(ids) != len(set(ids)):
            raise ValueError(f"IDs contain duplicates: {ids}.")
        return ids

    @field_validator("keys")
    @classmethod
    def _validate_keys_type(cls, keys: Iterable) -> Set[str]:
        keys_ = set(keys)
        if len(keys_) != len(keys):
            raise ValueError(f"Keys contain duplicates: {keys}.")
        return keys_

    @field_validator("seqlens")
    @classmethod
    def _validate_seqlens_device_dtype(
        cls, seqlens: Dict[str, List[torch.Tensor]]
    ) -> Dict[str, List[torch.Tensor]]:
        for k, lens in seqlens.items():
            assert isinstance(lens, list)
            assert all(isinstance(l, torch.Tensor) for l in lens)
            for i, lens_ in enumerate(lens):
                if str(lens_.device) != "cpu":
                    logger.warning(
                        "The device of seqlens is not cpu. "
                        "Transfering data between host and "
                        "device will cause additional overheads."
                    )
                    lens[i] = lens_.cpu()
                if lens_.dtype != torch.int32:
                    logger.warning(
                        "The dtype of seqlens is not int32. " "Converting to int32."
                    )
                    lens[i] = lens_.to(torch.int32)
                if len(lens_.shape) != 1:
                    raise ValueError(f"Seqlens should be 1D tensors: {lens_}.")
        return seqlens

    @model_validator(mode="after")
    def _validate_list_length(self) -> "SequenceSample":
        cond = True
        l = len(self.ids)
        cond &= all(len(lens) == l for lens in self.seqlens.values())
        if not cond:
            raise ValueError(
                f"Lengths of ids({len(self.ids)})"
                f"/seqlens({self.seqlens}) "
                "are not the same."
            )

        return self

    @model_validator(mode="after")
    def _validate_keys(self) -> "SequenceSample":
        cond = True
        cond &= self.keys == set(self.seqlens.keys())
        cond &= self.keys == set(self.trailing_shapes.keys())
        cond &= self.keys == set(self.dtypes.keys())
        if self.data is not None:
            cond &= self.keys == set(self.data.keys())
        if not cond:
            err = (
                f"Keys are mismatched. "
                f"keys={self.keys}, "
                f"seqlens keys={set(self.seqlens.keys())}, "
                f"trailing_shapes keys={set(self.trailing_shapes.keys())}, "
                f"dtypes keys={set(self.dtypes.keys())}"
            )
            if self.data is not None:
                err += f", data keys={set(self.data.keys())}"
            raise KeyError(err)
        return self

    @model_validator(mode="after")
    def _validate_shapes(self) -> "SequenceSample":
        if self.data is None:
            return self
        acc_seqlen = {
            k: sum(lens.sum() for lens in lens_list)
            for k, lens_list in self.seqlens.items()
        }
        for k, v in self.data.items():
            if v is None:
                continue
            if v.shape != (acc_seqlen[k], *self.trailing_shapes[k]):
                raise ValueError(
                    f"Key: {k}, Data shape {v.shape} does not match "
                    f"configured shape {(acc_seqlen[k], *self.trailing_shapes[k])}."
                )
        return self

    @model_validator(mode="after")
    def _validate_dtypes(self) -> "SequenceSample":
        if self.data is None:
            return self
        for k, v in self.data.items():
            if v is None:
                continue
            if v.dtype != self.dtypes[k]:
                raise ValueError(
                    f"Data dtype {v.dtype} "
                    f"does not match configured "
                    f"dtype {self.dtypes[k]}."
                )
        return self

    @classmethod
    def gather(cls, samples: List["SequenceSample"], keys: Optional[List[str]] = None):
        if keys is None:
            for sample in samples:
                if sample.keys != samples[0].keys:
                    raise ValueError("Keys of samples are not the same.")
            keys = samples[0].keys
        else:
            for k in keys:
                assert all(k in s.keys for s in samples)

        seqlens = {k: sum([s.seqlens[k] for s in samples], []) for k in keys}
        if samples[0].data is not None:
            data = {
                k: (
                    torch.cat([s.data[k] for s in samples], dim=0)
                    if samples[0].data[k] is not None
                    else None
                )
                for k in keys
            }
        else:
            assert all(s.data is None for s in samples)
            data = None
        id_ = sum([s.ids for s in samples], [])
        metadata = {}
        for sample in samples:
            for k, v in sample.metadata.items():
                if k in metadata and metadata[k] != v:
                    raise ValueError(
                        f"Metadata {k} is not the same: {v} and {metadata[k]}."
                    )
                metadata[k] = v
        return cls(
            keys=keys,
            dtypes={key: samples[0].dtypes[key] for key in keys},
            trailing_shapes={key: samples[0].trailing_shapes[key] for key in keys},
            ids=id_,
            seqlens=seqlens,
            data=data,
            metadata=metadata,
        )

    def _get_split_key(self):
        acc_seqlen = {k: sum(l.sum() for l in lens) for k, lens in self.seqlens.items()}
        return max(acc_seqlen, key=acc_seqlen.get)

    def get_split_spec(
        self, k: int, key: Optional[str] = None, min_size: int = 1
    ) -> SequenceSplitSpec:
        if key is None:
            key = self._get_split_key()
        lens = [lens.sum() for lens in self.seqlens[key]]
        partitions = datapack.min_abs_diff_partition(lens, k, min_size)
        return SequenceSplitSpec(partitions=partitions)

    def split_with_spec(self, spec: SequenceSplitSpec) -> List["SequenceSample"]:
        samples = []
        data_offset = {k: 0 for k in self.keys}
        for start, end in spec.partitions:
            new_seqlens = {
                k: lens_list[start:end] for k, lens_list in self.seqlens.items()
            }
            _data_len = {
                k: sum(lens.sum() for lens in lens_list)
                for k, lens_list in new_seqlens.items()
            }
            if self.data is not None:
                new_data = {
                    k: (
                        v[data_offset[k] : _data_len[k] + data_offset[k]]
                        if v is not None
                        else None
                    )
                    for k, v in self.data.items()
                }
            else:
                new_data = None
            for k in self.keys:
                data_offset[k] += _data_len[k]
            new_id = self.ids[start:end]
            samples.append(
                SequenceSample(
                    dtypes=self.dtypes,
                    trailing_shapes=self.trailing_shapes,
                    keys=self.keys,
                    ids=new_id,
                    seqlens=new_seqlens,
                    data=new_data,
                    metadata=self.metadata,
                )
            )
        return samples

    def split(
        self,
        k: int,
        key: Optional[str] = None,
        min_size: int = 1,
    ) -> List["SequenceSample"]:
        spec = self.get_split_spec(k, key, min_size)
        return self.split_with_spec(spec)

    def unpack(self):
        return self.split(self.bs, min_size=1)

    def cuda(self):
        if self.data is None:
            return self
        self.data = {
            k: v.cuda() if v is not None else None for k, v in self.data.items()
        }
        return self

    @property
    def bs(self):
        return len(self.ids)

    def meta(self) -> "SequenceSample":
        """Create a new SequenceSample that does not contain any data."""
        return SequenceSample(
            keys=self.keys,
            trailing_shapes=self.trailing_shapes,
            dtypes=self.dtypes,
            ids=self.ids,
            data=None,
            seqlens=self.seqlens,
            metadata=self.metadata,
        )

    def update_(self, other: "SequenceSample"):
        self.keys = self.keys.union(other.keys)
        self.trailing_shapes.update(other.trailing_shapes)
        self.dtypes.update(other.dtypes)
        assert self.ids == other.ids, (self.ids, other.ids)
        if self.data is not None:
            self.data.update(other.data)
        self.seqlens.update(other.seqlens)
        self.metadata.update(other.metadata)

    @staticmethod
    def _resolve_seqlen_from_key(key, seqlens: List[int]) -> List[torch.Tensor]:
        if key in [
            "seq_no_eos_mask",
            "greedy_seq_no_eos_mask",
            "loss_mask",
            "rewards",
            "greedy_rewards",
        ]:
            return [torch.tensor([1], dtype=torch.int32) for _ in seqlens]
        elif key in [
            "input_ids",
            "packed_seq",
            "seq",
            "packed_logits_mask",
            "logits_mask",
            "prompt_mask",
            "greedy_prompt_mask",
            "packed_input_ids",
            "greedy_packed_input_ids",
            "values",
            "packed_prompts",
        ]:
            return [torch.tensor([seqlen], dtype=torch.int32) for seqlen in seqlens]
        elif key in [
            "packed_logprobs",
            "logprobs",
            "packed_ref_logprobs",
            "ref_logprobs",
            "old_logp",
            "ref_logp",
            "advantages",
            "ppo_loss_mask",
            "kl_rewards",
            "returns",
        ]:
            return [torch.tensor([seqlen - 1], dtype=torch.int32) for seqlen in seqlens]
        else:
            raise NotImplementedError(
                f"Seqlen could not be resolved given key {key}. "
                f"Please explicltly construct the `SequenceSample` object"
                " without using the `from_default` method."
            )

    @classmethod
    def from_default(
        cls,
        seqlens: List[int],
        ids: List[Hashable],
        data: Dict[str, torch.Tensor],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        keys = set(data.keys())
        seqlens = [int(seqlen) for seqlen in seqlens]
        seqlens = {key: cls._resolve_seqlen_from_key(key, seqlens) for key in keys}
        trailing_shapes = {
            key: data[key].shape[1:] if data[key] is not None else None for key in keys
        }
        dtypes = {
            key: data[key].dtype if data[key] is not None else None for key in keys
        }
        return cls(
            keys=keys,
            ids=ids,
            seqlens=seqlens,
            trailing_shapes=trailing_shapes,
            dtypes=dtypes,
            data=data,
            metadata=metadata if metadata is not None else {},
        )


@dataclasses.dataclass
class DataBatchMeta:
    dp_rank: int
    meta_sample: SequenceSample
    epoch: int
    is_final_batch: bool

    def __post_init__(self):
        if self.meta_sample.data is not None:
            raise ValueError("meta_sample should not contain any data.")


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
            raise NotImplementedError(f"Unkown dataset extension: {dataset_path}")
    else:
        assert dataset_builder is not None
        data = dataset_builder()

    datasize_per_rank = len(data) // util.world_size
    shuffle_indices = get_shuffle_indices(
        util.seed, datasize_per_rank * util.world_size
    )
    subset_indices = shuffle_indices[
        util.ddp_rank * datasize_per_rank : (util.ddp_rank + 1) * datasize_per_rank
    ]
    data: List[Dict[str, str]] = [data[i] for i in subset_indices]

    return data


ALL_DATASET_CLASSES = {}


def register_dataset(name, dataset_cls):
    assert name not in ALL_DATASET_CLASSES
    assert "/" not in name
    ALL_DATASET_CLASSES[name] = dataset_cls


def make_dataset(
    cfg: Union[str, config_api.DatasetAbstraction],
    seed: int,
    ddp_rank: int,
    world_size: int,
    tokenizer_or_tokenizer_name: Union[transformers.PreTrainedTokenizerFast, str],
    experiment_name: str,
    trial_name: str,
    cache_root: Optional[str] = None,
) -> torch.utils.data.Dataset:
    if isinstance(cfg, str):
        cfg = config_api.DatasetAbstraction(type_=cfg)

    if isinstance(tokenizer_or_tokenizer_name, str):
        tokenizer = load_hf_tokenizer(tokenizer_or_tokenizer_name)
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
    if not cache_root.startswith(cluster_spec.fileroot) and not cache_root.startswith(
        "/home"
    ):
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


def make_dataloader(
    cfg: Union[str, config_api.DataLoaderAbstraction], dataset: torch.utils.data.Dataset
) -> torch.utils.data.DataLoader:
    if isinstance(cfg, str):
        cfg = config_api.DataLoaderAbstraction(type_=cfg)
    dataloader_cls = ALL_DATALOADER_CLASSES[cfg.type_]
    return dataloader_cls(dataset, **cfg.args)


def PackedDataLoader(dataset, *args, **kwargs):
    if not isinstance(getattr(dataset, "util", None), DatasetUtility):
        raise ValueError("Dataset must have a `util` attribute of type DatasetUtility.")
    g = torch.Generator()
    g.manual_seed(dataset.util.seed)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return torch.utils.data.DataLoader(
        dataset,
        *args,
        collate_fn=SequenceSample.gather,
        # NOTE: This is *NOT* the actual batch size for training.
        # It is just a proper size to load data to workers.
        batch_size=512,
        shuffle=True,
        generator=g,
        worker_init_fn=seed_worker,
        **kwargs,
    )


def PackedEvalDataLoader(dataset, *args, **kwargs):
    if not isinstance(getattr(dataset, "util", None), DatasetUtility):
        raise ValueError("Dataset must have a `util` attribute of type DatasetUtility.")
    return torch.utils.data.DataLoader(
        dataset,
        *args,
        collate_fn=SequenceSample.gather,
        shuffle=False,
        **kwargs,
    )


register_dataloader("packed", PackedDataLoader)
register_dataloader("packed_eval", PackedEvalDataLoader)
