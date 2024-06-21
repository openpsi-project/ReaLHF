# Copied from https://github.com/microsoft/DeepSpeed
import torch
import torch.distributed as dist
from packaging.version import Version

import realhf.base.constants as constants

ID_TO_DTYPE = [
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
    torch.float16,
    torch.bfloat16,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
]
DTYPE_TO_ID = {dtype: id_ for id_, dtype in enumerate(ID_TO_DTYPE)}


def _tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def can_send_recv() -> bool:
    # torch_version = Version(torch_info["version"])
    torch_version = Version(torch.__version__)
    sendrecv_min = Version("1.8")
    return torch_version >= sendrecv_min


assert can_send_recv()


def _is_valid_send_recv(src_stage, dest_stage):
    first_stage = 0
    last_stage = constants.grid().pipe_parallel_size - 1
    assert (
        abs(src_stage - dest_stage) == 1
        or (src_stage == first_stage and dest_stage == last_stage)
        or (src_stage == last_stage and dest_stage == first_stage)
    ), f"Functionality currently limited to send and receive between adjacent ranks only (src={src_stage}, dst={dest_stage})"


def send(tensor, dest_stage, async_op=False):
    # NOTE: The input is the stage id rather than the global rank
    src_stage = constants.grid().get_stage_id()
    _is_valid_send_recv(src_stage, dest_stage)

    dest_rank = constants.grid().stage_to_global(stage_id=dest_stage)
    send_method = dist.isend if async_op else dist.send
    return send_method(tensor, constants.to_global_pg_rank(dest_rank))


def recv(tensor, src_stage, async_op=False):
    # NOTE: The input is the stage id rather than the global rank
    dest_stage = constants.grid().get_stage_id()
    _is_valid_send_recv(src_stage, dest_stage)

    src_rank = constants.grid().stage_to_global(stage_id=src_stage)
    recv_method = dist.irecv if async_op else dist.recv
    return recv_method(tensor, constants.to_global_pg_rank(src_rank))
