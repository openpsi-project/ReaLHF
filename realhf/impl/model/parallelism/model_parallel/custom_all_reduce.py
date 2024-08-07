# Copied from the vLLM project: https://github.com/vllm-project/vllm
from contextlib import contextmanager
from typing import Any, List, Optional

import torch
import torch.distributed
import torch.distributed as dist

import realhf.base.constants as constants
import realhf.base.logging as logging

try:
    import pynvml

    from realhf._C.custom_all_reduce import custom_ar

    @contextmanager
    def _nvml():
        try:
            pynvml.nvmlInit()
            yield
        finally:
            pynvml.nvmlShutdown()

except ImportError:
    # For AMD GPUs
    custom_ar = None
    pynvml = None

    @contextmanager
    def _nvml():
        try:
            yield
        finally:
            pass


logger = logging.getLogger(__name__)

_CA_HANDLE = None
_SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]


def init_custom_ar() -> None:
    global _CA_HANDLE
    if _CA_HANDLE is not None:
        return
    mp_rank = constants.model_parallel_rank()
    mp_world_size = constants.model_parallel_world_size()
    if mp_world_size == 1:
        # No need to initialize custom allreduce for single GPU case.
        return

    full_nvlink = _is_full_nvlink(mp_rank, mp_world_size)
    if mp_world_size > 2 and not full_nvlink:
        logger.warning(
            "Custom allreduce is disabled because it's not supported on"
            " more than two PCIe-only GPUs. "
        )
        return

    if mp_world_size not in _SUPPORTED_WORLD_SIZES:
        logger.warn(
            "Custom allreduce is disabled due to an unsupported world size: "
            "%d. Supported world sizes: %s.",
            mp_world_size,
            str(_SUPPORTED_WORLD_SIZES),
        )
        return

    if not _can_p2p():
        logger.warn(
            "Custom allreduce is disabled because your platform lacks GPU P2P"
            " capability. "
        )
        return

    _CA_HANDLE = CustomAllreduce(mp_rank, mp_world_size, full_nvlink=full_nvlink)


def get_handle() -> Optional["CustomAllreduce"]:
    return _CA_HANDLE


def is_initialized() -> bool:
    return _CA_HANDLE is not None


# query if the set of gpus are fully connected by nvlink (1 hop)
@_nvml()
def _is_full_nvlink(rank, world_size):
    handle = pynvml.nvmlDeviceGetHandleByIndex(rank)
    for i in range(world_size):
        if i != rank:
            try:
                link_state = pynvml.nvmlDeviceGetNvLinkState(handle, i)
                if not link_state:
                    return False
            except pynvml.NVMLError as error:
                logger.info(
                    f'NVLink detection failed with message "{str(error)}". '
                    "This is normal if your machine has no NVLink equipped"
                )
                return False
    return True


def _can_p2p() -> bool:
    parallelism_world_size = constants.parallelism_group_size()
    rank = constants.parallelism_rank()
    assert parallelism_world_size % 2 == 0
    if parallelism_world_size == 1:
        return True
    buf = torch.zeros(1, dtype=torch.int32, device="cuda")
    try:
        if rank % 2 == 0:
            dist.send(
                torch.tensor([1], device="cuda"), constants.to_global_pg_rank(rank + 1)
            )
        else:
            dist.recv(buf, constants.to_global_pg_rank(rank - 1))
    except Exception as e:
        logger.warn(e)
        return False
    return True


class CustomAllreduce:

    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]

    # max_size: max supported allreduce size
    def __init__(
        self,
        rank: int,
        world_size: int,
        full_nvlink: bool = True,
        max_size: int = 2**31
        - 2,  # originally 8MB not enough, here is max possible max_size as input
    ) -> None:
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the CustomAllreduce to. If None,
                it will be bind to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self._IS_CAPTURING = False
        self.group = constants.model_parallel_cpu_group()

        assert (
            dist.get_backend(self.group) != dist.Backend.NCCL
        ), "CustomAllreduce should be attached to a non-NCCL group."

        rank = dist.get_rank(group=self.group)
        world_size = dist.get_world_size(group=self.group)
        self.device = torch.device("cuda")

        # buffers memory are owned by this Python class and passed to C++
        # meta data composes of two parts: meta data for synchronization
        # (256 bytes) and a temporary buffer for storing intermediate
        # allreduce results.
        self.meta = torch.zeros(
            custom_ar.meta_size() + max_size,
            dtype=torch.uint8,
            device=self.device,
        )
        # This is a pre-registered IPC buffer. In eager mode, input tensors
        # are first copied into this buffer before allreduce is performed
        self.buffer = torch.empty(max_size, dtype=torch.uint8, device=self.device)
        # This is a buffer for storing the tuples of pointers pointing to
        # IPC buffers from all ranks. Each registered tuple has size of
        # 8*world_size bytes where world_size is at most 8. Allocating 8MB
        # is enough for 131072 such tuples. The largest model I've seen only
        # needs less than 10000 of registered tuples.
        self.rank_data = torch.empty(
            8 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self.max_size = max_size
        self.rank = rank
        self.world_size = world_size
        handles, offsets = self._get_ipc_meta(self.meta)
        self.full_nvlink = full_nvlink
        self._ptr = custom_ar.init_custom_ar(
            self.meta, self.rank_data, handles, offsets, rank, self.full_nvlink
        )
        self.register_buffer(self.buffer)

    @contextmanager
    def capture(self):
        """The main responsibility of this context manager is the
        `register_graph_buffers` call at the end of the context.

        It records all the buffer addresses used in the CUDA graph.
        """
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            self.register_graph_buffers()

    def _get_ipc_meta(self, inp: torch.Tensor):
        data = inp.untyped_storage()._share_cuda_()
        shard_data = (
            data[1],  # ipc handle to base ptr
            data[3],  # offset of base ptr
        )
        return self._gather_ipc_meta(shard_data)

    def _gather_ipc_meta(self, shard_data):
        # Note: don't use `[[None]] * self.world_size` here
        # because it will create a list of the same reference
        all_data: List[Optional[Any]] = [[None] for i in range(self.world_size)]
        all_data[self.rank][0] = shard_data

        ranks = dist.get_process_group_ranks(group=self.group)
        ranks.sort()
        for i, rank in enumerate(ranks):
            dist.broadcast_object_list(
                all_data[i], src=rank, group=self.group, device="cpu"
            )

        # we cannot directly use `dist.all_gather_object` here
        # because it is incompatible with `gloo` backend under inference mode.
        # see https://github.com/pytorch/pytorch/issues/126032 for details.

        handles = []
        offsets = []
        for i in range(len(all_data)):
            handles.append(all_data[i][0][0])  # type: ignore
            offsets.append(all_data[i][0][1])  # type: ignore
        return handles, offsets

    def register_buffer(self, inp: torch.Tensor):
        handles, offsets = self._get_ipc_meta(inp)
        custom_ar.register_buffer(self._ptr, inp, handles, offsets)

    def register_graph_buffers(self):
        handle, offset = custom_ar.get_graph_buffer_ipc_meta(self._ptr)
        handles, offsets = self._gather_ipc_meta((bytes(handle), offset))
        logger.debug("Registering %d cuda graph addresses", len(offset))
        custom_ar.register_graph_buffers(self._ptr, handles, offsets)

    def should_custom_ar(self, inp: torch.Tensor):
        should_custom_ar = custom_ar.should_custom_ar(
            inp, self.max_size, self.world_size, self.full_nvlink
        )
        return should_custom_ar

    # all reduce, assuming inp tensor is IPC registered with register_buffer,
    # or, in the context of cuda graphs, register_graph_buffers
    def all_reduce_reg(self, inp: torch.Tensor, out: torch.Tensor = None):
        if out is None:
            out = torch.empty_like(inp)
        custom_ar.all_reduce_reg(self._ptr, inp, out)
        return out

    # all reduce, assuming inp tensor is NOT IPC registered
    def all_reduce_unreg(self, inp: torch.Tensor, out: torch.Tensor = None):
        if out is None:
            out = torch.empty_like(inp)
        custom_ar.all_reduce_unreg(self._ptr, inp, self.buffer, out)
        return out

    def custom_all_reduce(self, input: torch.Tensor) -> Optional[torch.Tensor]:
        # when custom allreduce is disabled, this will be None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                if self.should_custom_ar(input):
                    return self.all_reduce_reg(input)
            else:
                if self.should_custom_ar(input):
                    # if warm up, mimic the allocation pattern
                    # since custom allreduce is out-of-place
                    return torch.empty_like(input)
        else:
            # note: outside of cuda graph context,
            # custom allreduce incurs a cost of cudaMemcpy, which should
            # be small(<=1% of overall latency) compared to the performance
            # gains of using custom kernels
            if self.should_custom_ar(input):
                return self.all_reduce_unreg(input)
        return None

    def close(self):
        if self._ptr:
            custom_ar.dispose(self._ptr)
            self._ptr = 0

    def __del__(self):
        self.close()
