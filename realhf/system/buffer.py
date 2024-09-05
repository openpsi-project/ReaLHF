import asyncio
import copy
import time
from dataclasses import dataclass, field
from typing import *

import numpy as np

import realhf.api.core.dfg as dfg
import realhf.base.logging as logging
from realhf.api.core.data_api import SequenceSample

logger = logging.getLogger("buffer")


def _extract_intervals(arr):
    if len(arr) == 0:
        return []

    # Initialize the list to hold the intervals
    intervals = []

    # Start of the first interval
    start = arr[0]

    for i in range(1, len(arr)):
        # Check if the current element is not contiguous with the previous one
        if arr[i] != arr[i - 1] + 1:
            # End of the current interval
            end = arr[i - 1]
            # Add the interval as a tuple
            intervals.append((start, end + 1))
            # Start a new interval
            start = arr[i]

    # Add the last interval
    intervals.append((start, arr[-1] + 1))

    return intervals


class BufferFull(Exception):
    pass


@dataclass
class _ReplayEntry:
    reuses_left: int
    receive_time: float
    sample: SequenceSample


class _TensorDictSequenceBuffer:
    """An thread-unsafe buffer implementation based on list.

    Used as an internal buffer object in asyncio-based SequenceBuffer.
    Can be replaced with a more efficient C++ implementation based on
    std vector.

    Methods starting with _ should be called in a locked context.
    """

    def __init__(self, keys: List[str], max_size: int, reuses: int):
        # Fixed-size storage, storing pointers, but sequeces in dict have variable lengths.
        self.__storage: List[_ReplayEntry] = [None for _ in range(max_size)]

        # Some states of the storage. Read/Write applied to them should be locked.
        self.__has_keys = np.zeros((max_size, len(keys)), dtype=bool)

        self.__keys = keys
        self.__reuses = reuses

    def _update_has_keys(self, indices: List[int]):
        for idx in indices:
            self.__has_keys[idx] = [
                k in self.__storage[idx].sample.keys for k in self.__keys
            ]

    def _get_has_keys(self, indices):
        return self.__has_keys[indices, :]

    def put_batch(self, indices: List[int], xs: List[SequenceSample]):
        assert len(indices) == len(xs)
        # Can be parallelized.
        for idx, x in zip(indices, xs):
            self.__storage[idx] = _ReplayEntry(
                reuses_left=self.__reuses,
                receive_time=time.time(),
                sample=x,
            )

    def amend_batch(self, indices: List[int], xs: List[SequenceSample]):
        assert len(indices) == len(xs)
        # Can be parallelized.
        for idx, x in zip(indices, xs):
            self.__storage[idx].sample.update_(x)

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
            self.__has_keys[idx] = False
            res.append(r)
        return res


class AsyncIOSequenceBuffer:

    def __init__(
        self,
        rpcs: List[dfg.MFCDef],
        max_size: int,
        fetch_ctl: asyncio.Queue,
        fetch_master_ctl: asyncio.Queue,
    ):
        self._lock = asyncio.Condition(asyncio.Lock())

        # Both are queues of size 1.
        self._fetch_ctl = fetch_ctl
        self._fetch_master_ctl = fetch_master_ctl
        self._load_data_requested = False

        # Buffer indicators, should be locked by self._lock.
        # Put, amend, ready, idle, and empty are mutually exclusive.
        self._is_being_put = np.zeros(max_size, dtype=bool)
        self._is_being_amended = np.zeros(max_size, dtype=bool)
        self._is_being_read = np.zeros(max_size, dtype=bool)
        self._is_idle = np.zeros(max_size, dtype=bool)
        self._is_empty = np.ones(max_size, dtype=bool)

        self._buf_size = 0

        # We allow concurrent amenders and readers.
        self._n_amenders = np.zeros(max_size, dtype=int)
        self._n_readers = np.zeros(max_size, dtype=int)

        self._ready_for_rpcs = np.zeros((max_size, len(rpcs)), dtype=bool)
        self._completed_rpc = np.zeros((max_size, len(rpcs)), dtype=bool)

        self._rpc_data_keys = rpc_data_keys = list(
            set().union(*[rpc.input_keys for rpc in rpcs])
        )
        # We can efficiently compute whether an RPC is ready using this mask
        self._rpc_key_mask = np.stack(
            [
                np.array([k in rpc.input_keys for k in rpc_data_keys], dtype=bool)
                for rpc in rpcs
            ],
            axis=1,
        )
        self._rpc_names = [rpc.name for rpc in rpcs]

        # The internal buffer implementation.
        self.__max_size = max_size
        self.__buffer = _TensorDictSequenceBuffer(
            keys=rpc_data_keys, max_size=max_size, reuses=len(rpcs)
        )

    @property
    def lock(self):
        return self._lock

    @property
    def n_rpcs(self):
        return len(self._rpc_names)

    def _assert_valid_indicator(self):
        assert (
            self._is_being_put
            + self._is_being_amended
            + self._is_being_read
            + self._is_idle
        ).sum() == self._buf_size
        assert (self._is_empty.sum() + self._buf_size) == self.__max_size
        assert ((self._n_amenders > 0) == self._is_being_amended).all()
        assert (self._n_amenders >= 0).all()
        assert ((self._n_readers > 0) == self._is_being_read).all()
        assert (self._n_readers >= 0).all()
        assert (self._is_empty[:, None] * self._ready_for_rpcs).sum() == 0
        assert (self._is_empty[:, None] * self._completed_rpc).sum() == 0

    async def put_batch(self, samples: List[SequenceSample]):
        async with self._lock:
            self._assert_valid_indicator()
            n = len(samples)

            # Find the last contiguous empty block to write.
            indices = np.where(self._is_empty)[0]
            indices = np.arange(*_extract_intervals(indices)[-1])[:n]

            if len(indices) < n:
                raise BufferFull("Please set a larger buffer size")
            self._is_empty[indices] = False
            self._is_being_put[indices] = True

        self.__buffer.put_batch(indices, samples)

        async with self._lock:
            self.__buffer._update_has_keys(indices)

            has_keys = self.__buffer._get_has_keys(indices)  # [bs, #keys]
            rpc_key_mask = self._rpc_key_mask  # [#keys, #rpcs]
            self._ready_for_rpcs[indices] = (
                has_keys[:, :, None] >= rpc_key_mask[None, :, :]
            ).all(axis=1)

            self._is_being_put[indices] = False
            self._is_idle[indices] = True

            self._buf_size += len(samples)
            if self._buf_size >= 0.95 * self.__max_size:
                logger.warning(
                    f"Buffer is 95% full. The current buffer size is {self._buf_size} "
                    f"while the maximum size is {self.__max_size}. "
                    f"If your dataset has more than 1M sequences, consider enlarge "
                    f"the default batch size in the master worker."
                )
            self._load_data_requested = False
        return indices

    async def amend_batch(self, indices: List[int], samples: List[SequenceSample]):
        async with self._lock:
            await self._lock.wait_for(
                lambda: (
                    self._is_idle[indices] | self._is_being_amended[indices]
                ).all(),
            )
            self._assert_valid_indicator()
            self._is_idle[indices] = False
            self._is_being_amended[indices] = True
            self._n_amenders[indices] += 1

        self.__buffer.amend_batch(indices, samples)

        async with self._lock:
            self.__buffer._update_has_keys(indices)

            has_keys = self.__buffer._get_has_keys(indices)  # [bs, #keys]
            rpc_key_mask = self._rpc_key_mask  # [#keys, #rpcs]
            self._ready_for_rpcs[indices] = (
                has_keys[:, :, None] >= rpc_key_mask[None, :, :]
            ).all(axis=1)

            self._n_amenders[indices] -= 1
            self._is_being_amended[indices] = self._n_amenders[indices] > 0
            self._is_idle[indices] = np.logical_not(self._is_being_amended[indices])
            if self._is_idle[indices].any():
                self._lock.notify(len(self._rpc_names))

    def _request_load_data(self):
        if not self._load_data_requested:
            try:
                self._fetch_ctl.put_nowait(1)
            except asyncio.QueueFull:
                pass
            try:
                self._fetch_master_ctl.put_nowait(1)
            except asyncio.QueueFull:
                pass
            self._load_data_requested = True

    async def get_batch_for_rpc(
        self, rpc: dfg.MFCDef
    ) -> Tuple[List[int], SequenceSample]:
        logger.info(
            f"MFC {rpc.name} is waiting for its input keys: {rpc.input_keys}..."
        )
        rpc_idx = self._rpc_names.index(rpc.name)

        def _can_do_rpc() -> bool:
            ready_indices = np.nonzero(
                (self._is_idle | self._is_being_read)
                & self._ready_for_rpcs[:, rpc_idx]
                & ~self._completed_rpc[:, rpc_idx]
            )[0]
            if len(ready_indices) < rpc.n_seqs:
                return False
            return True

        async with self._lock:
            # await self._lock.wait_for(_can_do_rpc)
            if rpc.is_src:
                ready_indices = np.nonzero(
                    (self._is_idle | self._is_being_read)
                    & self._ready_for_rpcs[:, rpc_idx]
                    & ~self._completed_rpc[:, rpc_idx]
                )[0]
                # *2 because we want to fetch new data as long as the *next* RPC does not have enough data.
                if len(ready_indices) < rpc.n_seqs * 2:
                    self._request_load_data()

            while not _can_do_rpc():
                await self._lock.wait()

            logger.info(f"Input keys ({rpc.input_keys}) for MFC {rpc.name} are ready!")
            self._assert_valid_indicator()

            ready_indices = np.nonzero(
                (self._is_idle | self._is_being_read)
                & self._ready_for_rpcs[:, rpc_idx]
                & ~self._completed_rpc[:, rpc_idx]
            )[0]

            indices = ready_indices[: rpc.n_seqs]

            self._is_idle[indices] = False
            self._is_being_read[indices] = True
            self._n_readers[indices] += 1

        entries = self.__buffer.get_batch(indices)
        assert all([entry.reuses_left >= 0 for entry in entries])
        pop_indices = [
            idx for idx, entry in zip(indices, entries) if entry.reuses_left == 0
        ]
        # The following call is safe because no more RPC will write to popped data.
        if len(pop_indices) > 0:
            self.__buffer.pop_batch(pop_indices)

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

            if self._is_idle[indices].any():
                self._lock.notify(len(self._rpc_names))
        return indices, SequenceSample.gather(
            [e.sample for e in entries], keys=rpc.input_keys
        )
