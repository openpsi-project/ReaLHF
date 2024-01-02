from typing import Callable, Optional, Tuple, Union
import os
import warnings

from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn.parameter import Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from base.constants import *
from impl.model.utils.model_parallel.mappings import *
from impl.model.utils.model_parallel.utils import (_initialize_affine_weight_cpu,
                                                   _initialize_affine_weight_gpu, divide,
                                                   set_tensor_model_parallel_attributes, VocabUtility)
from impl.model.utils.modules import LlamaRMSNorm

_grad_accum_fusion_available = True
try:
    import fused_weight_gradient_mlp_cuda
except ImportError:
    _grad_accum_fusion_available = False

import base.logging as logging

logger = logging.getLogger("model_parallel.modules")


def get_activation_fn(activation_function: str) -> Callable:
    if activation_function == "gelu":
        return nn.functional.gelu
    elif activation_function == "gelu_new":
        from impl.model.utils.modules.activations import new_gelu_activation

        return new_gelu_activation
    elif activation_function == "silu":
        return nn.SiLU()
    else:
        raise NotImplementedError('Only "gelu" activation function is available.')


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
        device: Optional[Union[str, torch.device]] = None,
    ):
        super(ParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.tensor_model_parallel_size = model_parallel_world_size()
        # Divide the weight matrix along the vocaburaly dimension.
        self.vocab_start_index, self.vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
            self.num_embeddings, model_parallel_rank(), self.tensor_model_parallel_size)
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index

        logger.debug(
            f"ParallelEmbedding: num_embeddings={num_embeddings}, per_partition={self.num_embeddings_per_partition}, embedding_dim={embedding_dim},"
            f"tp_rank={model_parallel_rank()},tp_world_size={model_parallel_world_size()}")
        # Allocate weights and initialize.
        self.weight = Parameter(
            torch.empty(self.num_embeddings_per_partition, self.embedding_dim, device=device, dtype=dtype))
        if perform_initialization:
            _initialize_affine_weight_gpu(self.weight, init_method, partition_dim=0, stride=1)

    def forward(self, input_) -> torch.Tensor:
        if self.tensor_model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            # Get the embeddings.

        output_parallel = F.embedding(
            masked_input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
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
    def forward(ctx, input, weight, bias, gradient_accumulation_fusion, async_grad_allreduce,
                sequence_parallel):
        # disable sequence parallel for now for it requires a global buffer
        ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.sequence_parallel = sequence_parallel

        if sequence_parallel:
            assert (not ctx.async_grad_allreduce
                    ), "async_grad_allreduce and sequence_parallel can not be both True"
            world_size = model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = torch.empty(*dim_size, dtype=input.dtype, device=input.device)
            torch.distributed._all_gather_base(all_gather_buffer, input, group=model_parallel_group())
            total_input = all_gather_buffer
        else:
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

        if ctx.sequence_parallel:
            world_size = model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = torch.empty(*dim_size, dtype=input.dtype, device=torch.cuda.current_device())
            handle = torch.distributed._all_gather_base(all_gather_buffer,
                                                        input,
                                                        group=model_parallel_group(),
                                                        async_op=True)

            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # gather is scheduled before the input gradient computation
            total_input = all_gather_buffer
        else:
            total_input = input
        grad_input = grad_output.matmul(weight)

        if ctx.sequence_parallel:
            handle.wait()

        # Doing gather + slicing during the NeMo forward pass can make this tensor
        # not be contiguous. PyTorch only checks if the tensor is contiguous, and only
        # clones it if it's not contiguous:
        # https://github.com/pytorch/pytorch/blob/c47cf9bc7f9e02f649ab4ed53fe4d35732c92ab6/torch/_refs/__init__.py#L2761
        grad_output = grad_output.contiguous()
        # Convert the tensor shapes to 2D for execution compatibility
        grad_output = grad_output.view(-1, grad_output.shape[-1])
        total_input = total_input.view(-1, total_input.shape[-1])

        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(grad_input, group=model_parallel_group(), async_op=True)
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # all-reduce is scheduled before the weight gradient computation

        if ctx.sequence_parallel:
            assert not ctx.async_grad_allreduce
            dim_size = list(input.size())
            sub_grad_input = torch.empty(dim_size,
                                         dtype=input.dtype,
                                         device=torch.cuda.current_device(),
                                         requires_grad=False)
            # reduce_scatter
            handle = torch.distributed._reduce_scatter_base(sub_grad_input,
                                                            grad_input,
                                                            group=model_parallel_group(),
                                                            async_op=True)
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # reduce scatter is scheduled before the weight gradient computation

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

        if ctx.sequence_parallel:
            handle.wait()
            return sub_grad_input, grad_weight, grad_bias, None, None, None

        if ctx.async_grad_allreduce:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None


def linear_with_grad_accumulation_and_async_allreduce(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool,
    async_grad_allreduce: bool,
    sequence_parallel: bool,
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

    sequence_parallel (bool required): Indicates that sequence
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
        sequence_parallel,
    ]

    if not linear_with_grad_accumulation_and_async_allreduce.warned:
        if os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") != "1":
            if sequence_parallel:
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
        gather_output=False,
        init_method=init.xavier_normal_,
        stride=1,
        skip_bias_add=False,
        async_tensor_model_parallel_allreduce=True,
        perform_initialization=True,
        gradient_accumulation_fusion=False,
        sequence_parallel=False,
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
        # TODO: add fused bias add later
        assert skip_bias_add is False

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        # logger.info(
        #     f"ColumnLinear: input_size={input_size}, output_size={output_size}, output_size_per_partition={self.output_size_per_partition}"
        # )
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
            self.register_parameter("bias", None)

        self.async_tensor_model_parallel_allreduce = async_tensor_model_parallel_allreduce and world_size > 1
        self.sequence_parallel = sequence_parallel

        if gradient_accumulation_fusion:
            if not _grad_accum_fusion_available:
                raise RuntimeError("ColumnParallelLinear was called with gradient_accumulation_fusion set "
                                   "to True but the custom CUDA extension fused_weight_gradient_mlp_cuda "
                                   "module is not found. To use gradient_accumulation_fusion you must "
                                   "install APEX with --cpp_ext and --cuda_ext. For example: "
                                   'pip install --global-option="--cpp_ext" --global-option="--cuda_ext ." '
                                   "Note that the extension requires CUDA>=11. Otherwise, you must turn off "
                                   "gradient accumulation fusion.")
        self.gradient_accumulation_fusion = gradient_accumulation_fusion

        if self.async_tensor_model_parallel_allreduce and self.sequence_parallel:
            raise RuntimeError("`async_tensor_model_parallel_allreduce` and `sequence_parallel` "
                               "cannot be enabled at the same time.")

    def forward(self, input_) -> torch.Tensor:
        """Forward of ColumnParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """
        bias = self.bias if not self.skip_bias_add else None

        if self.async_tensor_model_parallel_allreduce or self.sequence_parallel:
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
            sequence_parallel=self.sequence_parallel,
        )
        if self.gather_output:
            # All-gather across the partitions.
            assert not self.sequence_parallel
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        # output_bias = self.bias if self.skip_bias_add else None
        return output


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
        skip_bias_add=False,
        perform_initialization=True,
        gradient_accumulation_fusion=False,
        sequence_parallel=False,
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
        assert self.skip_bias_add is False
        self.gradient_accumulation_fusion = gradient_accumulation_fusion
        self.sequence_parallel = sequence_parallel
        if self.sequence_parallel and not self.input_is_parallel:
            raise RuntimeError("To enable `sequence_parallel`, `input_is_parallel` must be `True`")

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
            setattr(self.bias, "sequence_parallel", False)

            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

    def forward(self, input_) -> torch.Tensor:
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
            assert not self.sequence_parallel
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = linear_with_grad_accumulation_and_async_allreduce(
            input=input_parallel,
            weight=self.weight,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=False,
            sequence_parallel=False,  # Here false because we do not need allreduce grad in backward here
        )

        # All-reduce across all the partitions.
        if self.sequence_parallel:
            output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
        else:
            output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        output = output_ + self.bias if self.bias is not None else output_
        return output


class LayerNormColumnLinear(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layer_norm_epsilon: float,
        use_attention_bias: bool,
        sequence_parallel: Optional[bool] = False,
        layer_norm_type: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if dtype is None:
            dtype = torch.float16
        if layer_norm_type is None:
            layer_norm_fn = nn.LayerNorm
        elif layer_norm_type == "rms":
            layer_norm_fn = LlamaRMSNorm
        self.ln = layer_norm_fn(input_dim, eps=layer_norm_epsilon, dtype=dtype, device=device)
        self.linear = ColumnParallelLinear(
            input_dim,
            output_dim,
            async_tensor_model_parallel_allreduce=not sequence_parallel,
            sequence_parallel=sequence_parallel,
            bias=use_attention_bias,
            dtype=dtype,
            device=device,
        )

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
        sequence_parallel: Optional[bool] = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        if dtype is None:
            dtype = torch.float16

        self.ln = nn.LayerNorm(hidden_dim, eps=layer_norm_epsilon, dtype=dtype, device=device)
        self.c_fc = ColumnParallelLinear(
            hidden_dim,
            intermediate_dim,
            async_tensor_model_parallel_allreduce=not sequence_parallel,
            sequence_parallel=sequence_parallel,
            dtype=dtype,
            device=device,
        )
        self.c_proj = RowParallelLinear(intermediate_dim,
                                        hidden_dim,
                                        sequence_parallel=sequence_parallel,
                                        dtype=dtype,
                                        device=device)
        self.act = get_activation_fn(activation_function)
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.ln(hidden_states)
        hidden_states, _ = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.c_proj(hidden_states)
        return self.dropout(hidden_states)


class LlamaLayerNormParallelMLP(nn.Module):

    def __init__(
        self,
        hidden_dim: int,
        intermediate_dim: int,
        activation_function: str,
        layer_norm_epsilon: float,
        sequence_parallel: Optional[bool] = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        if dtype is None:
            dtype = torch.float16
        self.hidden_size = hidden_dim
        self.intermediate_size = intermediate_dim
        self.ln = LlamaRMSNorm(hidden_dim, eps=layer_norm_epsilon, dtype=dtype, device=device)
        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            async_tensor_model_parallel_allreduce=not sequence_parallel,
            sequence_parallel=sequence_parallel,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.up_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            async_tensor_model_parallel_allreduce=not sequence_parallel,
            sequence_parallel=sequence_parallel,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            sequence_parallel=sequence_parallel,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.act_fn = get_activation_fn(activation_function)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def parallel_lm_logits(
    input_: torch.HalfTensor,
    word_embeddings_weight: torch.HalfTensor,
    parallel_output: bool = False,
    async_tensor_model_parallel_allreduce: bool = False,
    sequence_parallel: bool = False,
    gradient_accumulation_fusion: bool = False,
    bias=None,
):
    """LM logits using word embedding weights."""
    # Parallel logits.
    if async_tensor_model_parallel_allreduce or sequence_parallel:
        input_parallel = input_
        model_parallel = model_parallel_world_size() > 1
        async_grad_allreduce = (async_tensor_model_parallel_allreduce and model_parallel
                                and not sequence_parallel)
    else:
        input_parallel = copy_to_tensor_model_parallel_region(input_)
        async_grad_allreduce = False

    # Matrix multiply.
    logits_parallel = linear_with_grad_accumulation_and_async_allreduce(
        input=input_parallel,
        weight=word_embeddings_weight,
        bias=bias,
        gradient_accumulation_fusion=gradient_accumulation_fusion,
        async_grad_allreduce=async_grad_allreduce,
        sequence_parallel=sequence_parallel,
    )
    # Gather if needed.

    if parallel_output:
        return logits_parallel

    return gather_from_tensor_model_parallel_region(logits_parallel)


class _VocabParallelCrossEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, label_smoothing=0.0):
        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        torch.distributed.all_reduce(logits_max,
                                     op=torch.distributed.ReduceOp.MAX,
                                     group=model_parallel_group())
        # Subtract the maximum value.
        vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(dim=-1)

        # Get the partition's vocab indecies
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = model_parallel_rank()
        world_size = model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(
            predicted_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=model_parallel_group(),
        )

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        torch.distributed.all_reduce(
            sum_exp_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=model_parallel_group(),
        )

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits

        # Normalize and optionally smooth logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        vocab_size = exp_logits.size(-1)
        if label_smoothing > 0:
            """
            We'd like to assign 1 / (K - 1) probability mass to every index that is not the ground truth.
            = (1 - alpha) * y_gt + alpha * mean(y_{i for i != gt})
            = (1 - alpha) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = ((K - 1) * (1 - alpha) / (K - 1)) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = (K * (1 - alpha) - 1) / (K - 1)) * y_gt  + (alpha / (K - 1)) * \sum_{i} y_i
            = (1 - (alpha * K) / (K - 1)) * y_gt + ( (alpha * K) / (K - 1) ) * \sum_{i} y_i / K
            From: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/losses/smoothed_cross_entropy.py
            """
            assert 1.0 > label_smoothing > 0.0
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)

            # Exp logits at this point are normalized probabilities. So we can just take the log to get log-probs.
            log_probs = torch.log(exp_logits)
            mean_log_probs = log_probs.mean(dim=-1)
            loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs

        ctx.label_smoothing, ctx.vocab_size = label_smoothing, vocab_size

        # Store softmax, target-mask and masked-target for backward pass.
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors
        label_smoothing, vocab_size = ctx.label_smoothing, ctx.vocab_size

        # All the inputs have softmax as thier gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)

        softmax_update = 1.0 - target_mask.view(-1).float()

        if label_smoothing > 0:
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)
            grad_2d[arange_1d, masked_target_1d] -= (1.0 - smoothing) * softmax_update
            average_grad = 1 / vocab_size
            grad_2d[arange_1d, :] -= smoothing * average_grad
        else:
            grad_2d[arange_1d, masked_target_1d] -= softmax_update

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None, None


def vocab_parallel_cross_entropy(vocab_parallel_logits, target, label_smoothing=0.0):
    """
    Performs cross entropy loss when logits are split across tensor parallel ranks

    Arguments:
        vocab_parallel_logits: logits split across tensor parallel ranks
                               dimension is [sequence_length, batch_size, hidden_size]

        target: correct vocab ids of dimseion [sequence_length, micro_batch_size]

        lobal_smoothing: smoothing factor, must be in range [0.0, 1.0)
                         default is no smoothing (=0.0)
    """
    return _VocabParallelCrossEntropy.apply(vocab_parallel_logits, target, label_smoothing)
