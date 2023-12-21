# adopted from megatron
import torch

import base.constants


def assert_viewless_tensor(tensor, extra_msg=None):
    '''Assert that a tensor is not a view (i.e., its '._base' field is
    not set).'''
    if isinstance(tensor, list):
        [assert_viewless_tensor(t) for t in tensor]
        return tensor
    if not isinstance(tensor, torch.Tensor):
        return tensor
    assert tensor._base is None, ("Ensure tensor._base is None before setting tensor.data or storing "
                                  "tensor to memory buffer. Otherwise, a memory leak will occur (and "
                                  "likely accumulate over iterations). %s") % extra_msg
    return tensor


def safely_set_viewless_tensor_data(tensor, new_data_tensor):
    '''Safely set tensor's '.data' field.

    Check first that the tensor is viewless (i.e., '._base' not set). If not,
    raise an exception.
    '''
    assert_viewless_tensor(
        tensor,
        extra_msg="FYI, tensor._base has shape %s, and new_data_tensor has shape %s." %
        ("--" if tensor._base is None else tensor._base.shape, new_data_tensor.shape),
    )
    tensor.data = new_data_tensor


def split_tensor_into_1d_equal_chunks(tensor, new_buffer=False):
    """ Break a tensor into equal 1D chunks across tensor parallel ranks.

        Returns a Tensor or View with this rank's portion of the data.

        Arguments:
            tensor: The tensor to split

        Keyword Arguments:
            new_buffer (bool): If True, returns a new Tensor.
                               If False, returns a view into the existing Tensor.
                               Default is False

    """
    partition_size = torch.numel(tensor) // base.constants.model_parallel_world_size()
    start_index = partition_size * base.constants.model_parallel_rank()
    end_index = start_index + partition_size
    if new_buffer:
        data = torch.empty(
            partition_size,
            dtype=tensor.dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )
        data.copy_(tensor.view(-1)[start_index:end_index])
    else:
        data = tensor.view(-1)[start_index:end_index]
    return data


def gather_split_1d_tensor(tensor):
    """ Opposite of split_tensor_into_1d_equal_chunks. Gather values from tensor
        model parallel ranks.

        Returns a new Tensor with the gathered data.

        Arguments:
            tensor: A Tensor or view of this rank's portion of the data.
    """
    numel_gathered = torch.numel(tensor) * base.constants.model_parallel_world_size()
    gathered = torch.empty(numel_gathered,
                           dtype=tensor.dtype,
                           device=torch.cuda.current_device(),
                           requires_grad=False)
    # TODO: This API is experimental in pytorch (as of Feb 2022) and
    # this might break in future pytorch releases. We chose this API
    # as opposed to torch.distributed.all_gather for efficiency reasons.
    # This API calls directly NCCL all-gather versus the former does
    # internal copies and can potentially cause slow down.
    torch.distributed._all_gather_base(gathered, tensor, group=base.constants.model_parallel_group())
    return gathered


def pad_sequence_parallel_input(packed_input_ids: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int):
    """ Sequence parallel requires packed_input_ids has a shape of 1 dimension [total_seq_len], and 
    total_seq_len should be divisible by model_parallel_world_size. This function is used to pad packed_input_ids
    to suitable length with an empty sequence, and return new packed_input_ids, cu_seqlens and max_seqlen.
    
    Args:
        packed_input_ids (torch.Tensor): unpadded packed_input_ids
        cu_seqlens (torch.Tensor): unpadded cu_seqlens
        max_seqlen (int): unpadded max_seqlen
    
    Returns:
        (torch.Tensor, torch.Tensor, int, int): padded (packed_input_ids, cu_seqlens, max_seqlen, pad_size)
    """
    mp_world_size = base.constants.model_parallel_world_size()
    pad_size = 0
    if len(packed_input_ids) % mp_world_size != 0:
        pad_size = mp_world_size - len(packed_input_ids) % mp_world_size
        packed_input_ids = torch.nn.functional.pad(packed_input_ids, (0, pad_size), value=1)
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (0, 1), value=len(packed_input_ids))
        max_seqlen = max_seqlen if pad_size < max_seqlen else pad_size
    return packed_input_ids, cu_seqlens, max_seqlen, pad_size


def pad_sequence_parallel_generate_input(packed_input_ids: torch.Tensor, cu_seqlens: torch.Tensor,
                                         max_seqlen: int):
    """ Only for pipeline generate input when model+seq parallel is enabled. To make sure inputs for seq parallel model 
    have a shape with first dimension divisible by model_parallel_world_size, the packed_input_ids should have length 
    divisible by model_parallel_world_size, and contains number of sequences divisible by model_parallel_world_size.
    
    Args:
        packed_input_ids (torch.Tensor): unpadded packed_input_ids
        cu_seqlens (torch.Tensor): unpadded cu_seqlens
        max_seqlen (int): unpadded max_seqlen
    
    Returns:
        (torch.Tensor, torch.Tensor, int, int, int): padded (packed_input_ids, cu_seqlens, max_seqlen, pad_size, pad_seq_size)
    """
    mp_world_size = base.constants.model_parallel_world_size()
    pad_size, pad_seq_size = 0, 0
    if len(packed_input_ids) % mp_world_size != 0 or (len(cu_seqlens) - 1) % mp_world_size != 0:
        pad_size = mp_world_size - len(packed_input_ids) % mp_world_size
        pad_seq_size = mp_world_size - (len(cu_seqlens) - 1) % mp_world_size
        if pad_size < pad_seq_size:
            pad_size += mp_world_size
        pad_cu_seqlens = torch.tensor(list(range(1, pad_seq_size)) + [pad_size]) + len(packed_input_ids)
        pad_cu_seqlens = pad_cu_seqlens.to(dtype=cu_seqlens.dtype, device=cu_seqlens.device)
        packed_input_ids = torch.nn.functional.pad(packed_input_ids, (0, pad_size), value=1)
        cu_seqlens = torch.cat([cu_seqlens, pad_cu_seqlens], dim=0)
        max_seqlen = max_seqlen if (pad_size - pad_seq_size + 1) < max_seqlen else (pad_size - pad_seq_size +
                                                                                    1)
        print(f"padded cu_seqlens = {cu_seqlens}")
    return packed_input_ids, cu_seqlens, max_seqlen, pad_size, pad_seq_size
