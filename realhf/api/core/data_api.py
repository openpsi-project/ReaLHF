import dataclasses
import json
import os
import random
import time
from contextlib import contextmanager

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
        model_name_or_path,
        fast_tokenizer=fast_tokenizer,
        trust_remote_code=True,
        **kwargs,
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
    """The data structure used to represent sequence data.

    Each piece of data is assumed to have several "keys" (like a dictionary),
    with each key potentially corresponding to multiple sequences.

    For example, when running PPO, multiple responses can be generated for each prompt.
    If there are 2 prompts, each with 3 responses, the batch might look like:

    .. code-block:: console

        >>> s = SequenceSample(...)
        >>> s.keys
        {'resp', 'prompt'}
        >>> s.seqlens
        {'prompt': [[13], [6]], 'resp': [[6, 17, 15], [13, 15, 13]]}
        >>> s.data
        {'prompt': torch.tensor([...]), 'resp': torch.tensor([...])}

    Key points:

    - Data with different batch indices can have varying lengths (e.g., the first prompt has a length of 13
      while the second has a length of 6).

    - A key (e.g., "response") can correspond to multiple sequences with different lengths.
      Additionally, the number of sequences for each key can differ from the number of sequences for the data.
      For example, the first prompt may have 2 responses, and the second may have 3.

    - Regardless of the batch size or the number of sequences stored for each key,
      the data is concatenated into a 1D tensor. The outer dimension represents the batch size,
      and the inner dimension represents the number of sequences for the key.

    This data structure facilitates easy gathering, splitting,
    and transferring of non-padded batches between different GPUs.

    :param keys: The keys of the data.
    :type keys: Set[str]
    :param trailing_shapes: The trailing shapes of the data,
        excluding the first dimension, which must be the sequence length.
        Used to construct the receiving buffer for data transfer.
    :type trailing_shapes: Dict[str, torch.Size | Tuple | None]
    :param dtypes: The types of the data. Used to construct
        the receiving buffer for data transfer.
    :type dtypes: Dict[str, torch.dtype | None]
    :param ids: Unique identifiers for each piece of data.
        Should be provided in the dataset implementation.
        Used to append new data to the buffer after a model function call.
    :type ids: List[Hashable]
    :param seqlens: The sequence lengths of each sequence in the data. For a given key,
        this should be a list of lists of integers. The outer list represents the batch size,
        while the inner lists represent the sequence lengths for this key.
        Python-native lists are used here because (1) pickling torch.Tensor or numpy array is inefficient,
        and (2) the size of the inner lists can vary across the batch, making 2D arrays impractical.
    :type seqlens: Dict[str, List[List[int]]]
    :param data: The actual concatenated data. If this is None,
        the sample is a metadata-only sample used by the master worker.
        The specification of the data should be consistent with the seqlens,
        dtypes, and trailing_shapes.
    :type data: Optional[Dict[str, torch.Tensor | None]]
    :param metadata: Metadata for the sample. It should be a
        dictionary of lists, provided in the dataset implementation.
        Note that adding metadata can slow down data transfer.
    :type metadata: Dict[str, List[Any]]
    """

    keys: Set[str]
    trailing_shapes: Dict[str, torch.Size | Tuple | None]
    dtypes: Dict[str, torch.dtype | None]

    ids: List[Hashable]

    seqlens: Dict[str, List[List[int]]]

    data: Optional[Dict[str, torch.Tensor | None]] = None

    metadata: Dict[str, List[Any]] = Field(default_factory=dict)

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
            assert all(isinstance(l, list) for l in lens)
            for i, lens_ in enumerate(lens):
                assert all(isinstance(l_, int) for l_ in lens_)
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
            k: sum(sum(lens) for lens in lens_list)
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
        """Gather a list of SequenceSample objects into a single batch.

        :param samples: A list of SequenceSample objects to be gathered.
        :type samples: List[SequenceSample]
        :param keys: The keys to be gathered. Only a subset of keys can
            be gathered. If None, the keys from the first sample will be
            used.
        :type keys: Optional[List[str]]
        """
        if keys is None:
            keys = samples[0].keys
        else:
            keys = set(keys)

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
            data = None
        id_ = sum([s.ids for s in samples], [])
        metadata = {
            k: sum([s.metadata[k] for s in samples], []) for k in samples[0].metadata
        }
        with cls.disable_validation():
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
        acc_seqlen = {k: sum(sum(l) for l in lens) for k, lens in self.seqlens.items()}
        return max(acc_seqlen, key=acc_seqlen.get)

    def get_split_spec(
        self, k: int, key: Optional[str] = None, min_size: int = 1
    ) -> SequenceSplitSpec:
        """Get the partition specification for splitting the data into `k`
        parts using a dynamic programming algorithm to achieve the most
        balanced partitioning.

        :param k: The number of parts to split the data into.
        :type k: int
        :param key: The key to be used for splitting. If None, the key
            with the largest total sequence length will be used.
        :type key: Optional[str]
        :param min_size: The minimum size of each partition.
        :type min_size: int
        :return: A SequenceSplitSpec object representing the
            partitioning specification.
        :rtype: SequenceSplitSpec
        """
        if key is None:
            key = self._get_split_key()
        lens = [sum(lens) for lens in self.seqlens[key]]
        partitions = datapack.min_abs_diff_partition(lens, k, min_size)
        return SequenceSplitSpec(partitions=partitions)

    def split_with_spec(self, spec: SequenceSplitSpec) -> List["SequenceSample"]:
        """Split the data according to the given spec."""
        samples = []
        data_offset = {k: 0 for k in self.keys}
        for start, end in spec.partitions:
            new_seqlens = {
                k: lens_list[start:end] for k, lens_list in self.seqlens.items()
            }
            _data_len = {
                k: sum(sum(lens) for lens in lens_list)
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
            for k, v in self.metadata.items():
                if not isinstance(v, list):
                    raise ValueError(
                        f"Unknown how to split non-list metadata: ({k}, {v})."
                    )
            with self.disable_validation():
                samples.append(
                    SequenceSample(
                        dtypes=self.dtypes,
                        trailing_shapes=self.trailing_shapes,
                        keys=self.keys,
                        ids=new_id,
                        seqlens=new_seqlens,
                        data=new_data,
                        metadata={k: v[start:end] for k, v in self.metadata.items()},
                    )
                )
        return samples

    def split(
        self,
        k: int,
        key: Optional[str] = None,
        min_size: int = 1,
    ) -> List["SequenceSample"]:
        """Split the data into `k` parts.

        This method uses the specified key or the key with the largest total sequence length
        to split the data into `k` parts. The partitioning ensures that each part meets the
        minimum size requirement.

        :param k: The number of parts to split the data into.
        :type k: int
        :param key: The key to use for splitting. If None, the key with the largest
            total sequence length will be used.
        :type key: Optional[str]
        :param min_size: The minimum size of each partition.
        :type min_size: int
        :return: A list of `SequenceSample` objects, each representing a part of the split data.
        :rtype: List[SequenceSample]
        """
        spec = self.get_split_spec(k, key, min_size)
        return self.split_with_spec(spec)

    def unpack(self):
        """Unpack a batch of data into individual pieces of data."""
        partitions = [(i, i + 1) for i in range(self.bs)]
        return self.split_with_spec(SequenceSplitSpec(partitions=partitions))

    def cuda(self):
        """Move the data to GPU inplace."""
        if self.data is None:
            return self
        self.data = {
            k: v.cuda() if v is not None else None for k, v in self.data.items()
        }
        return self

    @property
    def bs(self):
        """The batch size or the number of data pieces in the sample."""
        return len(self.ids)

    def meta(self) -> "SequenceSample":
        """Create a new SequenceSample that does not contain any data."""
        with self.disable_validation():
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
        """Inplace update data from another SequenceSample.

        Used to amend newly produced data after a model function call.
        """
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
            return [[1] for _ in seqlens]
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
            return [[seqlen] for seqlen in seqlens]
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
            return [[seqlen - 1] for seqlen in seqlens]
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
        """Construct a `SequenceSample` object from default parameters.

        This helper function is intended for cases where each piece of data has
        a single sequence length (e.g., a single response for each prompt).
        The sequence lengths for different keys are resolved automatically
        according to the rules in ``_resolve_seqlen_from_key``. While this function
        can reduce boilerplate code, it may introduce potential bugs, so it should
        be used with caution.

        :param seqlens: The sequence lengths of each piece of data. This represents
            the length of the main attribute (e.g., `packed_input_ids`). Sequence lengths
            for other attributes (e.g., rewards and logprobs) are computed from this parameter.
            It is **NOT** the actual length of rewards or logprobs even if it is the only key
            in the data.
        :type seqlens: List[int]
        :param ids: Unique identifiers for each piece of data.
        :type ids: List[Hashable]
        :param data: The actual data.
        :type data: Dict[str, torch.Tensor]
        :param metadata: Metadata for the sample. Should be a dictionary where each value
            is a list with a length equal to the number of sequence lengths.
        :type metadata: Optional[Dict[str, Any]]
        """
        if metadata is None:
            metadata = {}
        for k, v in metadata.items():
            if not isinstance(v, list) or len(v) != len(seqlens):
                raise ValueError(
                    f"Metadata `{k}` should be a list of length {len(seqlens)}: {v}."
                )
        keys = set(data.keys())
        if isinstance(seqlens[0], list):
            assert len(seqlens[0]) == 1
            seqlens = [seqlen[0] for seqlen in seqlens]
        else:
            assert all(isinstance(seqlen, int) for seqlen in seqlens)
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
            metadata=metadata,
        )

    def remap_keys_(self, remap: Dict[str, str]):
        """Inplace remap keys of the data.

        Useful for reusing the same interface implementation in
        different algorithms, where the data can be named differently.
        """
        for k in self.keys:
            if k in remap:
                new_k = remap[k]
                self.seqlens[new_k] = self.seqlens.pop(k)
                self.trailing_shapes[new_k] = self.trailing_shapes.pop(k)
                self.dtypes[new_k] = self.dtypes.pop(k)
                if self.data is not None:
                    self.data[new_k] = self.data.pop(k)
        self.keys = set(remap.get(k, k) for k in self.keys)

    @classmethod
    @contextmanager
    def disable_validation(cls):
        """Disable the expensive pydantic validation within this context.

        Used to accelerate gather/split/transfer operations since we
        have ensured that the data created in datasets and interfaces
        are valid.
        """
        original_init = cls.__init__

        def no_validation_init(self, *args, **kwargs):
            kwargs["keys"] = set(kwargs["keys"])
            self.__dict__.update(kwargs)

        cls.__init__ = no_validation_init
        try:
            yield
        finally:
            cls.__init__ = original_init


@dataclasses.dataclass
class DataBatchMeta:
    dp_rank: int
    meta_sample: SequenceSample
    epoch: int
    is_final_batch: bool


@dataclasses.dataclass
class DatasetUtility:
    seed: int
    dp_rank: int
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
            raise NotImplementedError(f"Unknown dataset extension: {dataset_path}")
    else:
        assert dataset_builder is not None
        data = dataset_builder()

    if any("id" not in d for d in data):
        logger.warning(
            f'Key "id" not found in the dataset. Use indices as dataset IDs.'
        )
        for idx, d in enumerate(data):
            d["id"] = idx

    datasize_per_rank = len(data) // util.world_size
    shuffle_indices = get_shuffle_indices(
        util.seed, datasize_per_rank * util.world_size
    )
    subset_indices = shuffle_indices[
        util.dp_rank * datasize_per_rank : (util.dp_rank + 1) * datasize_per_rank
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
    dp_rank: int,
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
        dp_rank,
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
        f"rank{dp_rank}",
    )
    os.makedirs(output_path, exist_ok=True)

    fname = "dataset.pt"
    cache_found = os.path.isfile(os.path.join(output_path, fname))

    tik = time.perf_counter()
    if not cache_found:
        logger.info(f"No data cache found for rank {dp_rank}. Create it from scratch.")
        dataset = ALL_DATASET_CLASSES[cfg.type_](seed, dp_rank, world_size, **cfg.args)
        torch.save(dataset, os.path.join(output_path, fname))
    else:
        logger.info(f"Rank {dp_rank} find existing data cache, load it.")
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
