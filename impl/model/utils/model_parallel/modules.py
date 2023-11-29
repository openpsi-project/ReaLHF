from typing import Optional, Union
import os
import warnings

from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from base.constants import *
from impl.model.utils.data import PipeCacheData, PipeTransferData
from impl.model.utils.model_parallel.mappings import *
from impl.model.utils.model_parallel.utils import (_initialize_affine_weight_cpu,
                                                   _initialize_affine_weight_gpu, divide,
                                                   set_tensor_model_parallel_attributes, VocabUtility)

_grad_accum_fusion_available = True
try:
    import fused_weight_gradient_mlp_cuda
except ImportError:
    _grad_accum_fusion_available = False

import base.logging as logging

logger = logging.getLogger("model_parallel.modules")


class ParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.

    Keyword Arguments:
        init_method: method to initialize weights.
        perform_initialization
    """

    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            init_method=init.xavier_normal_,
            # params_dtype: torch.dtype=torch.float32,
            perform_initialization: bool = True,
            dtype: Optional[torch.dtype] = None,
            device: Optional[Union[str, torch.device]] = None):
        super(ParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.tensor_model_parallel_size = model_parallel_world_size()
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = \
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings, model_parallel_rank(),
                self.tensor_model_parallel_size)
        self.num_embeddings_per_partition = self.vocab_end_index - \
            self.vocab_start_index

        logger.info(
            f"ParallelEmbedding: num_embeddings={num_embeddings}, per_partition={self.num_embeddings_per_partition}, embedding_dim={embedding_dim},"
            f"tp_rank={model_parallel_rank()},tp_world_size={model_parallel_world_size()}")
        # Allocate weights and initialize.
        self.weight = Parameter(
            torch.empty(self.num_embeddings_per_partition, self.embedding_dim, device=device, dtype=dtype))
        if perform_initialization:
            _initialize_affine_weight_gpu(self.weight, init_method, partition_dim=0, stride=1)

    def forward(self, input_):
        if self.tensor_model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | \
                         (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight, self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq, self.sparse)
        # Mask the output embedding.
        if self.tensor_model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_from_tensor_model_parallel_region(output_parallel)
        return output


class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """See linear_with_grad_accumulation_and_async_allreduce"""

    @staticmethod
    @custom_fwd
    def forward(ctx,
                input,
                weight,
                bias,
                gradient_accumulation_fusion,
                async_grad_allreduce,
                sequence_parallel=False):
        # disable sequence parallel for now for it requires a global buffer
        assert sequence_parallel == False
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.sequence_parallel = sequence_parallel

        # if sequence_parallel:
        #     world_size = get_tensor_model_parallel_world_size()
        #     dim_size = list(input.size())
        #     dim_size[0] = dim_size[0] * world_size

        #     all_gather_buffer = \
        #         get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
        #     torch.distributed._all_gather_base(
        #         all_gather_buffer,
        #         input,
        #         group=get_tensor_model_parallel_group())
        #     total_input = all_gather_buffer
        # else:
        #     total_input = input
        total_input = input

        output = torch.matmul(total_input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        assert ctx.sequence_parallel == False

        # if ctx.sequence_parallel:
        #     world_size = get_tensor_model_parallel_world_size()
        #     dim_size = list(input.size())
        #     dim_size[0] = dim_size[0] * world_size

        #     all_gather_buffer = \
        #         get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
        #     handle = torch.distributed._all_gather_base(
        #         all_gather_buffer,
        #         input,
        #         group=get_tensor_model_parallel_group(), async_op=True)

        #     # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
        #     # gather is scheduled before the input gradient computation
        #     total_input = all_gather_buffer
        # else:
        #     total_input = input
        total_input = input
        grad_input = grad_output.matmul(weight)

        # if ctx.sequence_parallel:
        #     handle.wait()

        # Doing gather + slicing during the NeMo forward pass can make this tensor
        # not be contiguous. PyTorch only checks if the tensor is contiguous, and only
        # clones it if it's not contiguous:
        # https://github.com/pytorch/pytorch/blob/c47cf9bc7f9e02f649ab4ed53fe4d35732c92ab6/torch/_refs/__init__.py#L2761
        grad_output = grad_output.contiguous()
        # Convert the tensor shapes to 2D for execution compatibility
        grad_output = grad_output.view(grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2])
        total_input = total_input.view(total_input.shape[0] * total_input.shape[1], total_input.shape[2])

        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(grad_input, group=model_parallel_group(), async_op=True)
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # all-reduce is scheduled before the weight gradient computation

        # if ctx.sequence_parallel:
        #     assert not ctx.async_grad_allreduce
        #     dim_size = list(input.size())
        #     sub_grad_input = torch.empty(dim_size, dtype=input.dtype,
        #                                  device=torch.cuda.current_device(),
        #                                  requires_grad=False)
        #     # reduce_scatter
        #     handle = torch.distributed._reduce_scatter_base(sub_grad_input, grad_input,
        #                                                     group=get_tensor_model_parallel_group(),
        #                                                     async_op=True)
        #     # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
        #     # reduce scatter is scheduled before the weight gradient computation

        if ctx.gradient_accumulation_fusion:
            if weight.main_grad.dtype == torch.float32:
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(total_input, grad_output,
                                                                     weight.main_grad)
            elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(total_input, grad_output,
                                                                     weight.main_grad)
            else:
                raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")
            grad_weight = None
        else:
            grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        # if ctx.sequence_parallel:
        #     handle.wait()
        #     return sub_grad_input, grad_weight, grad_bias, None, None, None

        if ctx.async_grad_allreduce:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None


def linear_with_grad_accumulation_and_async_allreduce(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool,
    async_grad_allreduce: bool,
    sequence_parallel_enabled: bool,
) -> torch.Tensor:
    """Linear layer execution with asynchronous communication and
    gradient accumulation fusion in backprop.

    This has the option to accumulate the result of backprop
    calculation into an existing gradient buffer, preventing the need
    to do an additional addition kernel after the gradient
    calculation.

    Additionally, the tensor parallel all reduce of the input
    gradients can be done asynchronously with the calculation of
    the weight gradients.

    In the case of sequence parallelism, the reduce scatter of the
    input gradients is done asynchronously with the calcluation of the
    weight gradients.

    Use of this module requires that the environment variable
    CUDA_DEVICE_MAX_CONNECTIONS=1. There are a few collective
    operations, noted in the code, that should be scheduled before
    compute kernels to overlap the communication with the computation,
    which is necessary for a speedup but not for correctness so that
    ordering isn't imposed by the scheduler. Setting
    CUDA_DEVICE_MAX_CONNECTIONS=1 forces the kernels to be scheduled
    in the order they are called.

    Arguments:

    input (torch.Tensor required): input like torch.nn.functional.linear

    weight (torch.Tensor required): weight like torch.nn.functional.linear

    bias (torch.Tensor optional): bias like torch.nn.functional.linear

    gradient_accumulation_fusion (bool required): Perform the gradient
        accumulation fusion, requires the custom CUDA extension
        fused_weight_gradient_mlp_cuda module. To use
        gradient_accumulation_fusion you must install APEX with
        --cpp_ext and --cuda_ext. For example: "pip install
        --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\"
        " Note that the extension requires CUDA>=11. Otherwise, you
        must turn off gradient accumulation fusion."

    async_grad_allreduce (bool required): Do the allreduce of input
        gradients asyncronously with the computation of weight
        gradients. If sequence_parallel_enabled is True, this must be
        False, as no all reduce is performed.

    sequence_parallel_enabled (bool required): Indicates that sequence
        parallelism is used and thus in the forward pass the input is
        all gathered, and the backward pass the input gradients are
        reduce scattered.
    """
    args = [
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        async_grad_allreduce,
        sequence_parallel_enabled,
    ]

    if not linear_with_grad_accumulation_and_async_allreduce.warned:
        if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":
            if sequence_parallel_enabled:
                warnings.warn("When using sequence parallelism it is recommended to set the "
                              "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                              "maximum speedup")
                linear_with_grad_accumulation_and_async_allreduce.warned = True

            if async_grad_allreduce:
                warnings.warn("When using async grad allreduce it is recommended to set the "
                              "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                              "maximum speedup")
                linear_with_grad_accumulation_and_async_allreduce.warned = True

    return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)


linear_with_grad_accumulation_and_async_allreduce.warned = False


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.

    Keyword Arguments
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
        async_tensor_model_parallel_allreduce:
        params_dtype:
        gradient_accumulation_fusion:
    """

    def __init__(
        self,
        input_size,
        output_size,
        bias=True,
        gather_output=True,
        init_method=init.xavier_normal_,
        stride=1,
        skip_bias_add=False,
        async_tensor_model_parallel_allreduce=True,
        perform_initialization=True,
        gradient_accumulation_fusion=False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        logger.info(
            f"ColumnLinear: input_size={input_size}, output_size={output_size}, output_size_per_partition={self.output_size_per_partition}"
        )
        self.weight = Parameter(
            torch.empty(self.output_size_per_partition, self.input_size, device=device, dtype=dtype))
        if perform_initialization:
            _initialize_affine_weight_gpu(self.weight, init_method, partition_dim=0, stride=stride)

        if bias:
            self.bias = Parameter(torch.empty(self.output_size_per_partition, device=device, dtype=dtype))
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

        self.async_tensor_model_parallel_allreduce = (async_tensor_model_parallel_allreduce
                                                      and world_size > 1)

        if gradient_accumulation_fusion:
            if not _grad_accum_fusion_available:
                raise RuntimeError(
                    "ColumnParallelLinear was called with gradient_accumulation_fusion set "
                    "to True but the custom CUDA extension fused_weight_gradient_mlp_cuda "
                    "module is not found. To use gradient_accumulation_fusion you must "
                    "install APEX with --cpp_ext and --cuda_ext. For example: "
                    "pip install --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\" "
                    "Note that the extension requires CUDA>=11. Otherwise, you must turn off "
                    "gradient accumulation fusion.")
        self.gradient_accumulation_fusion = gradient_accumulation_fusion

    def forward(self, input_):
        """Forward of ColumnParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """
        bias = self.bias if not self.skip_bias_add else None

        if self.async_tensor_model_parallel_allreduce:
            input_parallel = input_
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = linear_with_grad_accumulation_and_async_allreduce(
            input=input_parallel,
            weight=self.weight,
            bias=bias,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
            sequence_parallel_enabled=False,
        )
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.

    Keyword Arguments:
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimization where bias
                       can be fused with other elementwise operations. We skip
                       adding bias but instead return it.
        params_dtype:
        use_cpu_initialization:
        perform_initialization:
        gradient_accumulation_fusion:
        sequence_parallel_enabled:
    """

    def __init__(
        self,
        input_size,
        output_size,
        bias=True,
        input_is_parallel=True,
        init_method=init.xavier_normal_,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        perform_initialization=True,
        gradient_accumulation_fusion=False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add
        self.gradient_accumulation_fusion = gradient_accumulation_fusion

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        self.weight = Parameter(
            torch.empty(self.output_size, self.input_size_per_partition, device=device, dtype=dtype))
        if perform_initialization:
            _initialize_affine_weight_gpu(self.weight, init_method, partition_dim=1, stride=stride)
        if bias:
            self.bias = Parameter(torch.empty(self.output_size, device=device, dtype=dtype))
            setattr(self.bias, 'sequence_parallel', False)

            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

    def forward(self, input_):
        """Forward of RowParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = linear_with_grad_accumulation_and_async_allreduce(
            input=input_parallel,
            weight=self.weight,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=False,
            sequence_parallel_enabled=False,
        )

        # All-reduce across all the partitions.
        output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias


class LayerNormColumnLinear(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layer_norm_epsilon: float,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if dtype is None:
            dtype = torch.float16
        self.ln = nn.LayerNorm(input_dim, eps=layer_norm_epsilon, dtype=dtype, device=device)
        self.linear = ColumnParallelLinear(input_dim, output_dim, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.ln(x))


class LayerNormParallelMLP(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        resid_pdrop: float,
        activation_function: str,
        layer_norm_epsilon: float,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        if dtype is None:
            dtype = torch.float16

        self.ln = nn.LayerNorm(hidden_dim, eps=layer_norm_epsilon, dtype=dtype, device=device)
        self.c_fc = ColumnParallelLinear(hidden_dim, intermediate_dim, dtype=dtype, device=device)
        self.c_proj = RowParallelLinear(intermediate_dim, hidden_dim, dtype=dtype, device=device)
        if activation_function == "gelu":
            self.act = nn.functional.gelu
        elif activation_function == 'gelu_new':
            from .activations import new_gelu_activation
            self.act = new_gelu_activation
        else:
            raise NotImplementedError("Only \"gelu\" activation function is available.")
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.ln(hidden_states)
        hidden_states, _ = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.c_proj(hidden_states)
        return self.dropout(hidden_states)
