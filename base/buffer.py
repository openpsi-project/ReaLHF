from dataclasses import dataclass, field
from typing import *
import asyncio
import bisect
import time

import numpy as np

import api.dfg
import base.dataparallel as dataparallel
import base.logging as logging
import base.namedarray as namedarray

logger = logging.getLogger("buffer")


class BufferFull(Exception):
    pass


@dataclass
class _ReplayEntry:
    reuses_left: int
    receive_time: float
    sample: namedarray.NamedArray


def _get_seqlen_from_sample(sample: namedarray.NamedArray) -> int:
    assert ("input_lens" in sample.keys() or "cu_seqlens" in sample.keys()
            or "prompt_cu_seqlens" in sample.keys() or "prompt_lens" in sample.keys()), (
                list(sample.keys()),
                sample,
            )
    if "input_lens" in sample.keys():
        return sample["input_lens"].item()
    elif "cu_seqlens" in sample.keys():
        return int(sample["cu_seqlens"][1] - 1)
    # NOTE: The order matters. We should first try to get the length of the generated text rather than prompts.
    elif "prompt_lens" in sample.keys():
        return sample["prompt_lens"].item()
    elif "prompt_cu_seqlens" in sample.keys():
        return int(sample["prompt_cu_seqlens"][1] - 1)
    else:
        return None


class _TensorDictSequenceBuffer:
    """An thread-unsafe buffer implementation based on list.

    Used as an internal buffer object in asyncio-based SequenceBuffer.
    Can be replaced with a more efficient C++ implementation based on std vector.

    Methods starting with _ should be called in a locked context.
    """

    def __init__(self, keys: List[str], max_size: int, reuses: int):
        # Fixed-size storage, storing pointers, but sequeces in dict have variable lengths.
        self.__storage: List[_ReplayEntry] = [None for _ in range(max_size)]

        # Some states of the storage. Read/Write applied to them should be locked.
        self.__seqlens = np.zeros(max_size, dtype=np.int32)
        self.__has_keys = np.zeros((max_size, len(keys)), dtype=bool)

        self.__keys = keys
        self.__reuses = reuses

    def _update_seqlen(self, indices: int):
        self.__seqlens[indices] = [_get_seqlen_from_sample(self.__storage[idx].sample) for idx in indices]

    def _get_seqlen(self, indices: int) -> np.ndarray:
        return self.__seqlens[indices]

    def _update_has_keys(self, indices: List[int]):
        for idx in indices:
            x = self.__storage[idx].sample
            self.__has_keys[idx] = [k in x.keys() and x[k] is not None for k in self.__keys]

    def _get_has_keys(self, indices):
        return self.__has_keys[indices, :]

    def put_batch(self, indices: List[int], xs: List[namedarray.NamedArray]):
        assert len(indices) == len(xs)
        # Can be parallelized.
        for idx, x in zip(indices, xs):
            self.__storage[idx] = _ReplayEntry(
                reuses_left=self.__reuses,
                sample=x,
                receive_time=time.time(),
            )

    def amend_batch(self, indices: List[int], new_datas: namedarray.NamedArray):
        assert len(indices) == len(new_datas)
        # Can be parallelized.
        for idx, new_data in zip(indices, new_datas):
            d = self.__storage[idx].sample.to_dict()
            d.update(new_data)
            self.__storage[idx].sample = namedarray.from_dict(d)

    def get_batch(self, indices: List[int]) -> List[_ReplayEntry]:
        # Can be parallelized.
        res = []
        for idx in indices:
            r = self.__storage[idx]
            r.reuses_left -= 1
            res.append(r)
        return res

    def pop_batch(self, indices: List[int]):
        res = []
        for idx in indices:
            r = self.__storage[idx]
            self.__storage[idx] = None
            self.__seqlens[idx] = 0
            self.__has_keys[idx] = False
            res.append(r)
        return res


@dataclass
class SequenceSample:
    data: namedarray.NamedArray
    indices: List[int]
    seqlens: List[int]


class AsyncIOSequenceBuffer:

    def __init__(
        self,
        rpcs: List[api.dfg.ModelRPC],
        max_size: int,
        fetch_ctl: asyncio.Queue,
        fetch_master_ctl: asyncio.Queue,
    ):
        self._lock = asyncio.Condition(asyncio.Lock())

        # Both are queues of size 1.
        self._fetch_ctl = fetch_ctl
        self._fetch_master_ctl = fetch_master_ctl

        # Buffer indicators, should be locked by self._lock.
        # Put, amend, ready, idle, and empty are mutually exclusive.
        self._is_being_put = np.zeros(max_size, dtype=bool)
        self._is_being_amended = np.zeros(max_size, dtype=bool)
        self._is_being_read = np.zeros(max_size, dtype=bool)
        self._is_idle = np.zeros(max_size, dtype=bool)
        self._is_empty = np.ones(max_size, dtype=bool)
        self._buf_size = 0
        self._n_tokens = 0
        # We allow concurrent amenders and readers.
        self._n_amenders = np.zeros(max_size, dtype=int)
        self._n_readers = np.zeros(max_size, dtype=int)

        self._ready_for_rpcs = np.zeros((max_size, len(rpcs)), dtype=bool)
        self._completed_rpc = np.zeros((max_size, len(rpcs)), dtype=bool)

        rpc_data_keys = list(set().union(*[rpc.input_data for rpc in rpcs]))
        # We can efficiently compute whether an RPC is ready using this mask
        self._rpc_key_mask = np.stack(
            [np.array([k in rpc.input_data for k in rpc_data_keys], dtype=bool) for rpc in rpcs], axis=1)
        self._rpc_names = [rpc.name for rpc in rpcs]

        # The internal buffer implementation.
        self.__max_size = max_size
        self.__buffer = _TensorDictSequenceBuffer(keys=rpc_data_keys, max_size=max_size, reuses=len(rpcs))

    @property
    def lock(self):
        return self._lock

    @property
    def n_rpcs(self):
        return len(self._rpc_names)

    def _assert_valid_indicator(self):
        assert (self._is_being_put + self._is_being_amended + self._is_being_read +
                self._is_idle).sum() == self._buf_size
        assert (self._is_empty.sum() + self._buf_size) == self.__max_size
        assert ((self._n_amenders > 0) == self._is_being_amended).all()
        assert (self._n_amenders >= 0).all()
        assert ((self._n_readers > 0) == self._is_being_read).all()
        assert (self._n_readers >= 0).all()
        assert (self._is_empty[:, None] * self._ready_for_rpcs).sum() == 0
        assert (self._is_empty[:, None] * self._completed_rpc).sum() == 0

    async def put_batch(self, samples: List[namedarray.NamedArray]):
        async with self._lock:
            self._assert_valid_indicator()
            n = len(samples)
            indices = np.where(self._is_empty)[0][:n]
            if len(indices) < n:
                raise BufferFull("Please set a larger buffer size")
            self._is_empty[indices] = False
            self._is_being_put[indices] = True

        self.__buffer.put_batch(indices, samples)

        async with self._lock:
            self.__buffer._update_has_keys(indices)
            self.__buffer._update_seqlen(indices)

            has_keys = self.__buffer._get_has_keys(indices)  # [bs, #keys]
            rpc_key_mask = self._rpc_key_mask  # [#keys, #rpcs]
            self._ready_for_rpcs[indices] = (has_keys[:, :, None] >= rpc_key_mask[None, :, :]).all(axis=1)

            self._is_being_put[indices] = False
            self._is_idle[indices] = True

            self._buf_size += len(samples)
            self._n_tokens += self.__buffer._get_seqlen(indices).sum()

    async def amend_batch(self, indices: List[int], new_datas: List[namedarray.NamedArray]):
        async with self._lock:
            await self._lock.wait_for(
                lambda: (self._is_idle[indices] | self._is_being_amended[indices]).all(),)
            self._assert_valid_indicator()
            self._is_idle[indices] = False
            self._is_being_amended[indices] = True
            self._n_amenders[indices] += 1
        self.__buffer.amend_batch(indices, new_datas)
        async with self._lock:
            self._n_tokens -= self.__buffer._get_seqlen(indices).sum()
            self.__buffer._update_has_keys(indices)
            self.__buffer._update_seqlen(indices)
            self._n_tokens += self.__buffer._get_seqlen(indices).sum()

            has_keys = self.__buffer._get_has_keys(indices)  # [bs, #keys]
            rpc_key_mask = self._rpc_key_mask  # [#keys, #rpcs]
            self._ready_for_rpcs[indices] = (has_keys[:, :, None] >= rpc_key_mask[None, :, :]).all(axis=1)

            self._n_amenders[indices] -= 1
            self._is_being_amended[indices] = self._n_amenders[indices] > 0
            self._is_idle[indices] = np.logical_not(self._is_being_amended[indices])
            if self._is_idle[indices].any():
                self._lock.notify(len(self._rpc_names))

    async def get_batch_for_rpc(self, rpc: api.dfg.ModelRPC) -> SequenceSample:
        rpc_idx = self._rpc_names.index(rpc.name)

        def _can_do_rpc() -> bool:
            ready_indices = np.nonzero((self._is_idle | self._is_being_read)
                                       & self._ready_for_rpcs[:, rpc_idx]
                                       & ~self._completed_rpc[:, rpc_idx])[0]
            if len(ready_indices) < rpc.min_n_seqs:
                return False
            seqlens = self.__buffer._get_seqlen(ready_indices)
            if seqlens.sum() < rpc.min_n_tokens:
                return False
            return True

        async with self._lock:
            if rpc.is_src and (self._buf_size < rpc.min_n_seqs or self._n_tokens < rpc.min_n_tokens):
                try:
                    self._fetch_ctl.put_nowait(1)
                except asyncio.QueueFull:
                    pass
                try:
                    self._fetch_master_ctl.put_nowait(1)
                except asyncio.QueueFull:
                    pass

            while not _can_do_rpc():
                await self._lock.wait()
            # await self._lock.wait_for(_can_do_rpc)
            self._assert_valid_indicator()

            ready_indices = np.nonzero((self._is_idle | self._is_being_read)
                                       & self._ready_for_rpcs[:, rpc_idx]
                                       & ~self._completed_rpc[:, rpc_idx])[0]
            seqlens = self.__buffer._get_seqlen(ready_indices)

            token_cumsum = np.cumsum(seqlens, axis=0)
            token_valid_mask = (token_cumsum >= rpc.min_n_tokens) & (token_cumsum <= rpc.max_n_tokens)
            token_intervals = token_valid_mask.nonzero()[0]
            if len(token_intervals) < 1:
                raise RuntimeError(
                    "No valid token intervals found. Please set a smaller min_n_tokens and a larger max_n_tokens. "
                    f"Current values min_n_tokens={rpc.min_n_tokens}, max_n_tokens={rpc.max_n_tokens}.")
            token_start, token_end = token_intervals[0], token_intervals[-1]

            n_seqs_cumsum = np.arange(1, len(ready_indices) + 1)
            n_seqs_valid_mask = (n_seqs_cumsum >= rpc.min_n_seqs) & (n_seqs_cumsum <= rpc.max_n_seqs)
            n_seqs_intervals = n_seqs_valid_mask.nonzero()[0]
            n_seqs_start, n_seqs_end = n_seqs_intervals[0], n_seqs_intervals[-1]

            if token_start > n_seqs_end or token_end < n_seqs_start:
                raise RuntimeError("Tokens and n_seqs interval are not overlapped. "
                                   f"Please set proper batch sizes in RPC {rpc.name}.")

            indices = ready_indices[:min(token_end, n_seqs_end) + 1]
            seqlens = seqlens[:min(token_end, n_seqs_end) + 1]
            assert rpc.min_n_tokens <= seqlens.sum() <= rpc.max_n_tokens
            assert rpc.min_n_seqs <= len(indices) <= rpc.max_n_seqs

            self._is_idle[indices] = False
            self._is_being_read[indices] = True
            self._n_readers[indices] += 1

        entries = self.__buffer.get_batch(indices)
        assert all([entry.reuses_left >= 0 for entry in entries])
        pop_indices = [idx for idx, entry in zip(indices, entries) if entry.reuses_left == 0]
        # The following call is safe because no more RPC will write to popped data.
        pop_tokens = self.__buffer._get_seqlen(pop_indices).sum()
        if len(pop_indices) > 0:
            self.__buffer.pop_batch(pop_indices)
        data = dataparallel.PackedParallelDataBroker.gather_from(
            [rpc.remap_input_keys(x.sample) for x in entries])

        async with self._lock:
            self._n_readers[indices] -= 1
            self._is_being_read[indices] = self._n_readers[indices] > 0
            self._is_idle[indices] = self._n_readers[indices] == 0
            self._completed_rpc[indices, rpc_idx] = True

            assert (self._n_readers[pop_indices] == 0).all()
            assert (self._n_amenders[pop_indices] == 0).all()
            self._is_empty[pop_indices] = True
            self._is_idle[pop_indices] = False
            self._completed_rpc[pop_indices] = False
            self._ready_for_rpcs[pop_indices] = False
            self._buf_size -= len(pop_indices)
            self._n_tokens -= pop_tokens

            if self._is_idle[indices].any():
                self._lock.notify(len(self._rpc_names))
        return SequenceSample(data=data, indices=indices, seqlens=seqlens)
