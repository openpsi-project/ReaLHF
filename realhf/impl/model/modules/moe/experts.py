# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# adopted from megatron

from typing import Callable, Optional, Union

import torch
import torch.nn.init as init
from torch.nn.parameter import Parameter

import realhf.base.constants as constants
import realhf.impl.model.utils.grouped_gemm as gg
from realhf.api.core.model_api import ReaLModelConfig
from realhf.impl.model.modules.mlp import LlamaLayerNormMLP, get_activation_fn
from realhf.impl.model.parallelism.model_parallel.utils import divide
from realhf.impl.model.utils.random import _initialize_affine_weight_gpu


class SequentialMLP(torch.nn.Module):
    """An implementation of the Experts layer using a sequence of MLP layers.

    This class executes each expert sequentially.
    """

    def __init__(
        self,
        num_local_experts: int,
        config: ReaLModelConfig,
        add_bias_linear: bool = False,  # FIXME: currently bias is not supported
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.config = config
        self.add_bias = add_bias_linear

        self.num_local_experts = num_local_experts
        self.local_experts = torch.nn.ModuleList()

        for _ in range(self.num_local_experts):
            expert = LlamaLayerNormMLP(
                hidden_dim=config.hidden_dim,
                intermediate_dim=config.intermediate_dim,
                activation_function=config.activation_function,
                use_layer_norm=False,  # layer norm is in the moe layer
                model_parallel=constants.model_parallel_world_size() > 1,
                gradient_accumulation_fusion=config.gradient_accumulation_fusion,
                dtype=dtype,
                device=device,
            )
            self.local_experts.append(expert)

    def forward(self, permuted_local_hidden_states, tokens_per_expert):

        output_local = torch.zeros_like(permuted_local_hidden_states)
        output_bias_local = None
        if self.add_bias:
            output_bias_local = torch.zeros_like(permuted_local_hidden_states)

        cumsum_num_tokens = torch.cumsum(tokens_per_expert, dim=0)
        # Insert zero at the begining for offset index's convenience
        zero_tensor = torch.zeros(1, dtype=torch.long, device=cumsum_num_tokens.device)
        cumsum_num_tokens = torch.cat((zero_tensor, cumsum_num_tokens))
        for expert_num, expert in enumerate(self.local_experts):
            start = cumsum_num_tokens[expert_num]
            end = cumsum_num_tokens[expert_num + 1]
            hidden = permuted_local_hidden_states[start:end]
            output = expert(hidden)
            output_local[start:end] = output

        return output_local, output_bias_local


class GroupedMLP(torch.nn.Module):
    """An efficient implementation of the Experts layer using CUTLASS GroupedGEMM.

    This class is designed to execute multiple experts in parallel, thereby maximizing computational efficiency.
    """

    def __init__(
        self,
        num_local_experts: int,
        config: ReaLModelConfig,
        init_method: Callable = init.xavier_normal_,
        add_bias_linear: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.device = device
        self.num_local_experts = num_local_experts
        self.expert_parallel = constants.expert_parallel_world_size() > 1

        gg.assert_grouped_gemm_is_available()
        assert (
            add_bias_linear == False
        ), "bias in the expert layer is not supported in Grouped GEMM yet, please set '--disable-bias-linear' instead."

        self.activation_func = get_activation_fn(self.config.activation_function)

        # How many feature each rank holds for fc1 and fc2, respectively.
        tp_size = constants.model_parallel_world_size()

        fc1_output_size = self.config.intermediate_dim * self.num_local_experts
        fc1_output_size_per_partition = divide(fc1_output_size, tp_size)

        fc2_input_size = self.config.intermediate_dim * self.num_local_experts
        fc2_input_size_per_partition = divide(fc2_input_size, tp_size)

        # Note: The current kernel implementations of grouped_gemm
        # does not support transposition with CUTLASS grouped GEMM
        # (https://github.com/fanshiqing/grouped_gemm/blob/main/csrc/grouped_gemm.cu#L355-L358)
        # and as a result we avoid allocate the transpose of weights.
        # Initialize weight.
        self.weight1 = Parameter(
            torch.empty(
                self.config.hidden_dim,
                fc1_output_size_per_partition,
                device=self.device,
                dtype=self.dtype,
            )
        )
        self.weight2 = Parameter(
            torch.empty(
                fc2_input_size_per_partition,
                self.config.hidden_dim,
                device=self.device,
                dtype=self.dtype,
            )
        )
        _initialize_affine_weight_gpu(
            self.weight1,
            init_method,
            partition_dim=1,
            expert_parallel=self.expert_parallel,
        )
        _initialize_affine_weight_gpu(
            self.weight2,
            init_method,
            partition_dim=0,
            expert_parallel=self.expert_parallel,
        )
        # setattr(self.weight1, "allreduce", not self.expert_parallel)
        # setattr(self.weight2, "allreduce", not self.expert_parallel)

        # def remove_extra_states_check(self, incompatible_keys):
        #     """
        #     Remove _extra_state from unexpected keys.
        #     These keys are for dist ckpt compatibility with SequentialMLP.
        #     """
        #     keys = deepcopy(incompatible_keys.unexpected_keys)
        #     for key in keys:
        #         if "_extra_state" in key:
        #             incompatible_keys.unexpected_keys.remove(key)

        # self.register_load_state_dict_post_hook(remove_extra_states_check)

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: int,
    ):
        if permuted_local_hidden_states.nelement() != 0:
            # Reshape the weights for the grouped GEMMs.
            w1 = self.weight1.view(self.num_local_experts, self.config.hidden_dim, -1)
            w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_dim)

            fc1_output = gg.ops.gmm(
                permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False
            )

            intermediate_parallel = self.activation_func(fc1_output)

            fc2_output = gg.ops.gmm(
                intermediate_parallel, w2, tokens_per_expert, trans_b=False
            )
        else:
            # No token is allocated for local experts.
            assert torch.count_nonzero(tokens_per_expert) == 0

            # Make sure parameters still have gradients when no tokens are routed to this set of experts.
            w1 = self.weight1.view(self.config.hidden_dim, -1)
            w2 = self.weight2.view(-1, self.config.hidden_dim)
            h = torch.matmul(permuted_local_hidden_states, w1)
            h = self.activation_func(h)
            h = torch.matmul(h, w2)

            fc2_output = h

        return fc2_output, None
