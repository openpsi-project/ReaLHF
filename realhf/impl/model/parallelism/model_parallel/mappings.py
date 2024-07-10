# Copied from Megatron-LM: https://github.com/NVIDIA/Megatron-LM
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch
import torch.distributed

import realhf.impl.model.parallelism.model_parallel.custom_all_reduce as custom_all_reduce
from realhf.base import constants

from .utils import split_tensor_along_last_dim


def _reduce(input_):
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if constants.model_parallel_world_size() == 1:
        return input_

    # All-reduce.
    if custom_all_reduce.is_initialized():
        out = custom_all_reduce.get_handle().custom_all_reduce(input_)
        if out is not None:
            return out
        # else:
        #     print("inside _reduce custom all reduce return None")
    torch.distributed.all_reduce(input_, group=constants.model_parallel_group())
    return input_


def _split_along_last_dim(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = constants.model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = constants.model_parallel_rank()
    output = input_list[rank].contiguous()

    return output


def _split_along_first_dim(input_):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""

    world_size = constants.model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along first dimension.
    dim_size = input_.size()[0]
    assert (
        dim_size % world_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"
    local_dim_size = dim_size // world_size
    rank = constants.model_parallel_rank()
    dim_offset = rank * local_dim_size

    output = input_[dim_offset : dim_offset + local_dim_size].contiguous()

    return output


def _gather_along_last_dim(input_):
    """Gather tensors and concatinate along the last dimension."""

    world_size = constants.model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = constants.model_parallel_rank()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(
        tensor_list, input_, group=constants.model_parallel_group()
    )

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output


def _gather_along_first_dim(input_):
    """Gather tensors and concatinate along the first dimension."""

    world_size = constants.model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(
        dim_size, dtype=input_.dtype, device=torch.cuda.current_device()
    )
    torch.distributed._all_gather_base(
        output, input_.contiguous(), group=constants.model_parallel_group()
    )

    return output


def _gather_along_first_dim_moe(input_, use_global_buffer=False):
    """Gather tensors and concatenate along the first dimension."""
    group = constants.expert_and_model_parallel_group()
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    if use_global_buffer:
        output = constants.get_global_memory_buffer().get_tensor(
            dim_size, input_.dtype, "mpu"
        )
    else:
        output = torch.empty(
            dim_size, dtype=input_.dtype, device=torch.cuda.current_device()
        )
    torch.distributed._all_gather_base(output, input_.contiguous(), group=group)

    return output


def _reduce_scatter_along_first_dim_moe(input_, use_global_buffer=False):
    """Reduce-scatter the input tensor across model parallel group."""
    group = constants.expert_and_model_parallel_group()
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    assert dim_size[0] % world_size == 0
    dim_size[0] = dim_size[0] // world_size

    if use_global_buffer:
        output = constants.get_global_memory_buffer().get_tensor(
            dim_size, input_.dtype, "mpu"
        )
    else:
        output = torch.empty(
            dim_size, dtype=input_.dtype, device=torch.cuda.current_device()
        )
    torch.distributed._reduce_scatter_base(output, input_.contiguous(), group=group)
    return output


def _reduce_scatter_along_last_dim(input_):
    """Reduce-scatter tensors on the last dimension."""
    world_size = constants.model_parallel_world_size()
    target_shape = list(input_.size())
    target_shape[-1] = target_shape[-1] // world_size
    input_ = input_.reshape(-1, input_.shape[-1])
    split_tensors = torch.split(
        input_, split_size_or_sections=input_.shape[-1] // world_size, dim=1
    )
    concat_tensor = torch.cat(split_tensors, dim=0)
    output = _reduce_scatter_along_first_dim(concat_tensor).reshape(target_shape)
    return output


def _gather_along_first_dim_expert_parallel(input_):
    """Gather tensors and concatinate along the first dimension."""

    world_size = constants.expert_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(
        dim_size, dtype=input_.dtype, device=torch.cuda.current_device()
    )
    torch.distributed._all_gather_base(
        output, input_.contiguous(), group=constants.expert_parallel_group()
    )
    return output


def _reduce_scatter_along_first_dim(input_):
    """Reduce-scatter the input tensor across model parallel group."""
    world_size = constants.model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    assert (
        dim_size[0] % world_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"

    dim_size[0] = dim_size[0] // world_size

    output = torch.empty(
        dim_size, dtype=input_.dtype, device=torch.cuda.current_device()
    )
    torch.distributed._reduce_scatter_base(
        output, input_.contiguous(), group=constants.model_parallel_group()
    )
    return output


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_last_dim(grad_output)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split_along_last_dim(grad_output)


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_, model_parallel_output_grad=True):
        return _gather_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_, model_parallel_output_grad=True):
        ctx.model_parallel_output_grad = model_parallel_output_grad
        return _gather_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        model_parallel_output_grad = ctx.model_parallel_output_grad

        # If the computation graph after the gather operation is
        # in the tensor parallel mode, output gradients need to reduce
        # scattered and whereas if the computation is duplicated,
        # output gradients need to be scattered.
        if model_parallel_output_grad:
            return _reduce_scatter_along_first_dim(grad_output), None
        else:
            return _split_along_first_dim(grad_output), None


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce_scatter_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce_scatter_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)


class _GatherFromSequenceParallelRegionToMOE(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate."""  # TODO

    @staticmethod
    def symbolic(graph, input_, use_global_buffer=False):
        return _gather_along_first_dim_moe(input_, use_global_buffer)

    @staticmethod
    def forward(ctx, input_, use_global_buffer=False):
        ctx.use_global_buffer = use_global_buffer
        return _gather_along_first_dim_moe(input_, use_global_buffer)

    @staticmethod
    def backward(ctx, grad_output):
        use_global_buffer = ctx.use_global_buffer
        return _reduce_scatter_along_first_dim_moe(grad_output, use_global_buffer), None


class _ReduceScatterToSequenceParallelRegionFromMOE(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, use_global_buffer=False):
        return _reduce_scatter_along_first_dim_moe(input_, use_global_buffer)

    @staticmethod
    def forward(ctx, input_, use_global_buffer=False):
        ctx.use_global_buffer = use_global_buffer
        return _reduce_scatter_along_first_dim_moe(input_, use_global_buffer)

    @staticmethod
    def backward(ctx, grad_output):
        use_global_buffer = ctx.use_global_buffer
        return _gather_along_first_dim_moe(grad_output, use_global_buffer), None


class _AllGatherFromTensorParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate."""

    @staticmethod
    def symbolic(graph, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _gather_along_last_dim(
            input_,
        )

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce_scatter_along_last_dim(grad_output)


class _ReduceScatterToTensorParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce_scatter_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce_scatter_along_last_dim(
            input_,
        )

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_last_dim(grad_output)


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = torch.distributed.get_world_size(group=group)
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input

        input = input.contiguous()
        if output_split_sizes is None:
            # Equal split (all2all)
            output = torch.empty_like(input)
        else:
            # Unequal split (all2all-v)
            output = input.new_empty(
                size=[sum(output_split_sizes)] + list(input.size()[1:]),
                dtype=input.dtype,
                device=torch.cuda.current_device(),
            )
        torch.distributed.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        return (
            None,
            _AllToAll.apply(
                ctx.group, *grad_output, ctx.input_split_sizes, ctx.output_split_sizes
            ),
            None,
            None,
        )


# -----------------
# Helper functions.
# -----------------


def copy_to_tensor_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_tensor_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_tensor_model_parallel_region(input_):
    return _ScatterToModelParallelRegion.apply(input_)


def gather_from_tensor_model_parallel_region(input_):
    return _GatherFromModelParallelRegion.apply(input_)


def scatter_to_sequence_parallel_region(input_):
    return _ScatterToSequenceParallelRegion.apply(input_)


def gather_from_sequence_parallel_region(input_, model_parallel_output_grad=True):
    return _GatherFromSequenceParallelRegion.apply(input_, model_parallel_output_grad)


def reduce_scatter_to_sequence_parallel_region(input_):
    return _ReduceScatterToSequenceParallelRegion.apply(input_)


def gather_from_sequence_parallel_region_to_moe(input_, use_global_buffer=False):
    return _GatherFromSequenceParallelRegionToMOE.apply(input_, use_global_buffer)


def reduce_scatter_to_sequence_parallel_region_from_moe(
    input_, use_global_buffer=False
):
    return _ReduceScatterToSequenceParallelRegionFromMOE.apply(
        input_, use_global_buffer
    )


def all_gather_last_dim_from_tensor_parallel_region(input_):
    return _AllGatherFromTensorParallelRegion.apply(input_)


def reduce_scatter_last_dim_to_tensor_parallel_region(input_):
    return _ReduceScatterToTensorParallelRegion.apply(input_)


def all_to_all(group, input_, output_split_sizes_=None, input_split_sizes_=None):
    return _AllToAll.apply(group, input_, output_split_sizes_, input_split_sizes_)


def all_to_all_sp2hp(input_):
    """
    Perform AlltoAll communication on tensor parallel group, transform the input tensor from shape [num_tokens/TP, H] to [num_tokens, H/TP].

    Args:
        input_ (torch.Tensor): The input tensor which has been distributed along the sequence dimension.

    Returns:
        torch.Tensor: The output tensor with shape [num_tokens, H/TP].

    """
    world_size = constants.model_parallel_world_size()
    tp_group = constants.model_parallel_group()
    input_ = input_.reshape(-1, input_.shape[-1])
    split_tensors = torch.split(
        input_, split_size_or_sections=input_.shape[-1] // world_size, dim=1
    )
    concat_tensor = torch.cat(split_tensors, dim=0)
    output = all_to_all(tp_group, concat_tensor)
    return output


def all_to_all_hp2sp(input_):
    """
    Perform AlltoAll communication on tensor parallel group, transform the input tensor from shape [num_tokens, H/TP] to [num_tokens/TP, H].

    Args:
        input_ (torch.Tensor): The input tensor which has been distributed along the hidden dimension.

    Returns:
        torch.Tensor: The output tensor with shape [num_tokens/TP, H].
    """
    world_size = constants.model_parallel_world_size()
    input_ = input_.reshape(-1, input_.shape[-1])
    tp_group = constants.model_parallel_group()
    input_exchanged = all_to_all(tp_group, input_)
    input_reshaped = input_exchanged.reshape(-1, input_exchanged.shape[-1])
    split_tensors = torch.split(
        input_reshaped,
        split_size_or_sections=input_reshaped.shape[0] // world_size,
        dim=0,
    )
    output = torch.cat(split_tensors, dim=-1)
    return output
