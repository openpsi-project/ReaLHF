# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pickle
import typing

from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.git_version_info import torch_info
# To query whether we have send/recv support
from packaging.version import Version
import torch

_groups = None
_grid = None

_async = []

ID_TO_DTYPE = [
    torch.float32, torch.float64, torch.complex64, torch.complex128, torch.float16, torch.bfloat16,
    torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.bool
]
DTYPE_TO_ID = {dtype: id_ for id_, dtype in enumerate(ID_TO_DTYPE)}


def _tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def can_send_recv() -> bool:
    torch_version = Version(torch_info['version'])
    sendrecv_min = Version('1.8')
    return torch_version >= sendrecv_min


#initializes adjacent process groups
#run this only after deepspeed.init_distributed() has been called
def init_process_groups(grid):
    global _groups, _grid
    _grid = grid

    assert _grid.pipe_parallel_size > 1, "There is no pipeline parallelism"

    if not can_send_recv():
        _groups = [dist.new_group(ranks=group) for group in _grid.p2p_groups]


def _is_valid_send_recv(src_stage, dest_stage):
    first_stage = 0
    last_stage = _grid.pipe_parallel_size - 1
    assert abs(src_stage-dest_stage) == 1 or \
        (src_stage == first_stage and dest_stage == last_stage) or \
        (src_stage == last_stage and dest_stage == first_stage), \
    "Functionality currently limited to send and receive between adjacent ranks only"


def send(tensor, dest_stage, async_op=False):
    global _groups
    assert async_op == False, "Doesn't support async_op true"
    src_stage = _grid.get_stage_id()
    _is_valid_send_recv(src_stage, dest_stage)

    dest_rank = _grid.stage_to_global(stage_id=dest_stage)
    if async_op:
        global _async
        op = dist.isend(tensor, dest_rank)
        _async.append(op)
    else:

        if can_send_recv():
            return dist.send(tensor, dest_rank)
        else:
            group = _get_send_recv_group(src_stage, dest_stage)
            src_rank = _grid.stage_to_global(stage_id=src_stage)
            return dist.broadcast(tensor, src_rank, group=group, async_op=async_op)


def recv(tensor, src_stage, async_op=False):
    global _groups
    assert async_op == False, "Doesn't support async_op true"
    dest_stage = _grid.get_stage_id()
    _is_valid_send_recv(src_stage, dest_stage)

    src_rank = _grid.stage_to_global(stage_id=src_stage)

    if async_op:
        global _async
        op = dist.irecv(tensor, src_rank)
        _async.append(op)
    else:
        if can_send_recv():
            return dist.recv(tensor, src_rank)
        else:
            group = _get_send_recv_group(src_stage, dest_stage)
            return dist.broadcast(tensor, src_rank, group=group, async_op=async_op)


def wait():
    global _async
    for op in _async:
        op.wait()
    _async = []

    get_accelerator().synchronize()


def send_obj(msg: typing.Any, dest: int):
    """Send an arbitrary python object to ``dest``.

    Note: ``msg`` must be pickleable.

    WARN: This incurs a CPU -> GPU transfer and should be used sparingly
    for performance reasons.

    Args:
        msg (typing.Any): The object to send.
        dest (int): Destination rank.
    """
    # serialize the message
    msg = pickle.dumps(msg)
    # construct a tensor to send
    msg = torch.ByteTensor(torch.ByteStorage.from_buffer(msg)).to(get_accelerator().device_name())

    # Send meta and message
    length_tensor = torch.tensor([len(msg)], dtype=torch.long).to(get_accelerator().device_name())
    dist.send(length_tensor, dst=dest)
    dist.send(msg, dst=dest)


def recv_obj(sender: int) -> typing.Any:
    """Receive an arbitrary python object from ``sender``.

    WARN: This incur a CPU <-> GPU transfers and should be used sparingly
    for performance reasons.

    Args:
        sender (int): The rank sending the message.
    """
    # Get message meta
    length = torch.tensor([0], dtype=torch.long).to(get_accelerator().device_name())
    dist.recv(length, src=sender)

    # Receive and deserialize
    msg = torch.empty(length.item(), dtype=torch.uint8).to(get_accelerator().device_name())
    dist.recv(msg, src=sender)

    msg = pickle.loads(msg.cpu().numpy().tobytes())

    def _to(x):
        """Recursively move to the current device."""
        if torch.is_tensor(x):
            return x.to(get_accelerator().device_name())
        if isinstance(x, (tuple, list)):
            ret = [_to(x_) for x_ in x]
            if isinstance(x, tuple):
                ret = tuple(ret)
            return ret
        # handle kwargs
        if isinstance(x, dict):
            ret = dict()
            for key, val in x.items():
                ret[_to(key)] = _to(val)
            return ret

        # Anything else is a no-op
        return x

    msg = _to(msg)
    return msg


def _get_send_recv_group(src_stage, dest_stage):
    '''the group id is always the smaller rank unless its a wrap around'''

    stage_id = None

    first_stage = 0
    last_stage = _grid.pipe_parallel_size - 1

    if (src_stage == first_stage and dest_stage == last_stage
            or dest_stage == first_stage and src_stage == last_stage):
        stage_id = last_stage
    elif src_stage > dest_stage:
        stage_id = dest_stage
    else:
        stage_id = src_stage
    '''group_id corresponds to group of [group_id, group_id+1]
     unless group_id is the rank of the last stage
     in which case group_id corresponds to group[group_id-num_stages+1, group_id]
     '''
    group_id = _grid.stage_to_global(stage_id=stage_id)

    return _groups[group_id]


def send_tensor_tuple_meta(tensor_tuple, recv_stage):
    """ Communicate metadata about upcoming p2p transfers.

    Metadata is communicated in this order:
        * num_tensors in tuple
        foreach tensor in tensor_tuple:
            * ndims
            * shape
    """
    count_tensor = torch.LongTensor(data=[len(tensor_tuple)]).cuda()
    send(count_tensor, recv_stage)
    for idx, tensor in enumerate(tensor_tuple):
        if isinstance(tensor, torch.Tensor):
            send_shape = torch.LongTensor(data=tensor.size()).cuda()
            send_ndims = torch.LongTensor(data=[len(tensor.size())]).cuda()
            send_dtype = torch.LongTensor(data=[DTYPE_TO_ID[tensor.dtype]]).cuda()
            send(send_dtype, recv_stage)
            send(send_ndims, recv_stage)
            send(send_shape, recv_stage)
        elif tensor is None:
            send_dtype = torch.LongTensor(data=[-1]).cuda()
            send(send_dtype, recv_stage)


def recv_tensor_tuple_meta(send_stage):
    """Receive metadata about upcoming p2p transfers and return allocated buffers.

    Metadata is communicated in this order:
        * num_tensors in tensor_tuple
        foreach tensor in buffer:
            * ndims
            * shape

    Returns:
        Allocated buffer for receiving from send_stage.
    """
    count_tensor = torch.LongTensor(data=[0]).cuda()
    recv(count_tensor, send_stage)
    num_tensors = count_tensor.item()
    recv_shapes_and_dtypes = []
    for idx in range(num_tensors):
        recv_dtype = torch.LongTensor(data=[0]).cuda()
        recv(recv_dtype, send_stage)
        if recv_dtype.item() == -1:
            recv_shapes_and_dtypes.append((None, None))
        else:
            recv_dtype = ID_TO_DTYPE[recv_dtype.item()]
            recv_ndims = torch.LongTensor(data=[0]).cuda()
            recv(recv_ndims, send_stage)
            recv_ndims = recv_ndims.item()
            recv_shape = torch.LongTensor([1] * recv_ndims).cuda()
            recv(recv_shape, send_stage)
            recv_shapes_and_dtypes.append((recv_shape.tolist(), recv_dtype))

    buffers = allocate_buffers(recv_shapes_and_dtypes)
    buffers = tuple(buffers)
    return buffers


def allocate_buffers(shapes_and_dtypes, requires_grad=False):
    buffer = []
    for shape, dtype in shapes_and_dtypes:
        if shape is None:
            buffer.append(None)
        else:
            buffer.append(
                torch.zeros(shape,
                            dtype=dtype,
                            requires_grad=requires_grad,
                            device=torch.cuda.current_device()))
    return buffer
