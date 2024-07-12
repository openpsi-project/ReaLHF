# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# adopted from megatron

from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn.init as init

import realhf.base.constants as constants
from realhf.api.core.model_api import ReaLModelConfig
from realhf.impl.model.parallelism.model_parallel.mappings import (
    gather_from_sequence_parallel_region,
)
from realhf.impl.model.utils.moe import (
    MoEAuxLossAutoScaler,
    save_to_aux_losses_tracker,
    sinkhorn,
    switch_load_balancing_loss_func,
    topk_softmax_with_capacity,
    z_loss_func,
)
from realhf.impl.model.utils.random import (
    get_cuda_rng_tracker,
    get_data_parallel_rng_tracker_name,
)


class Router(ABC, torch.nn.Module):
    """Base Router class"""

    def __init__(
        self,
        hidden_dim: int,
        config: ReaLModelConfig,
        init_method: Callable = init.xavier_normal_,
    ) -> None:
        """
        Initialize the Router module.

        Args:
            config (TransformerConfig): Configuration object for the Transformer model.
        """
        super().__init__()
        self.config = config
        self.num_experts = self.config.num_experts
        self.moe_aux_loss_func = None
        self.layer_number = None
        self.use_sequence_parallel = constants.sequence_parallel()

        # Initialize the gate weights.
        self.weight = torch.nn.Parameter(torch.empty((self.num_experts, hidden_dim)))
        # FIXME:
        # with get_cuda_rng_tracker().fork(get_data_parallel_rng_tracker_name()):
        init_method(self.weight)

    def gating(self, input: torch.Tensor):
        """Forward pass of the router gate.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Logits tensor.
        """
        logits = torch.nn.functional.linear(input, self.weight)
        return logits

    @abstractmethod
    def routing(self, logits: torch.Tensor):
        """Routing function.

        Args:
            logits (torch.Tensor): Logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of tensors representing max probs and the indices.
        """
        raise NotImplementedError("Routing function not implemented.")

    @abstractmethod
    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        raise NotImplementedError("Forward function not implemented.")


class TopKRouter(Router):
    """Route each token to the top-k experts."""

    def __init__(
        self,
        hidden_dim: int,
        config: ReaLModelConfig,
        init_method: Callable = init.xavier_normal_,
        layer_idx: int = 0,  # FIXME
        num_layers: int = 1,  # FIXME
    ) -> None:
        """Initialize the zero token dropping router.

        Args:
            config (TransformerConfig): The configuration for the transformer model.
        """
        super().__init__(hidden_dim, config, init_method)
        self.input_jitter = None
        self.layer_idx = layer_idx
        self.num_layers = num_layers

    def sinkhorn_load_balancing(self, logits: torch.Tensor):
        """Apply sinkhorn routing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            torch.Tensor: The logits tensor after applying sinkhorn routing.
        """

        def _sinkhorn_activation(logits):
            if self.config.moe_top_k == 1:
                logits = torch.sigmoid(logits)
            else:  # k > 1
                logits = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(
                    logits
                )
            return logits

        if self.training:
            with torch.no_grad():
                norm_logits = sinkhorn(
                    logits.to(dtype=torch.float32)
                )  # explicit fp32 conversion for stability
                _, indices = torch.topk(norm_logits, k=self.config.moe_top_k, dim=1)
            logits = _sinkhorn_activation(logits)
            scores = torch.gather(logits, 1, indices)
        else:
            logits = _sinkhorn_activation(logits)
            scores, indices = torch.topk(logits, k=self.config.moe_top_k, dim=1)
        return scores, indices

    def aux_loss_load_balancing(self, logits: torch.Tensor):
        """Apply loss-based load balancing to the logits tensor.

        Args:
            logits (torch.Tensor): the logits tensor after gating, shape: [num_tokens, num_experts].

        Returns:
            probs (torch.Tensor): the probabilities tensor after load balancing.
            indices (torch.Tensor): the indices tensor after top-k selection.
        """
        probs, indices, tokens_per_expert = topk_softmax_with_capacity(
            logits,
            self.config.moe_top_k,
            capacity_factor=self.config.capacity_factor,
            pad_to_capacity=self.config.pad_to_capacity,
            drop_policy=self.config.token_drop_policy,
        )

        # Apply load balancing loss
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
        probs = self.apply_load_balancing_loss(
            scores, tokens_per_expert, activation=probs
        )
        return probs, indices

    def apply_load_balancing_loss(
        self,
        probs: torch.Tensor,
        num_local_tokens_per_expert: torch.Tensor,
        activation: torch.Tensor,
    ):
        """Applies auxiliary loss to the MoE layer.

        Args:
            probs (torch.Tensor): The probs output by the router for each token. [num_tokens, num_experts]
            num_local_tokens_per_expert (torch.Tensor): The number of tokens per expert. [num_experts]
            activation (torch.Tensor): The activation tensor to attach the gradient function to.

        Returns:
            torch.Tensor: The activation tensor with the attached gradient function.
        """
        moe_aux_loss_coeff = self.config.aux_loss_coeff
        scale_for_logging = 1.0
        sequence_partition_group = None
        if self.config.token_dispatcher_type == "allgather":
            sequence_partition_group = constants.model_parallel_group()
        elif self.config.token_dispatcher_type == "alltoall":
            moe_aux_loss_coeff /= constants.model_parallel_world_size()

        if sequence_partition_group is not None:
            scale_for_logging *= constants.model_parallel_world_size()

        aux_loss = switch_load_balancing_loss_func(
            probs,
            num_local_tokens_per_expert,
            self.config.moe_top_k,
            moe_aux_loss_coeff,
            sequence_partition_group=sequence_partition_group,
        )
        # FIXME
        # save_to_aux_losses_tracker(
        #     "load_balancing_loss",
        #     aux_loss / moe_aux_loss_coeff * scale_for_logging,
        #     self.layer_number,
        #     self.num_layers,
        # )
        activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
        return activation

    def apply_z_loss(self, logits):
        """Encourages the router's logits to remain small to enhance stability.
        Please refer to the ST-MoE paper (https://arxiv.org/pdf/2202.08906.pdf) for details.

        Args:
            logits (torch.Tensor): The logits of the router.

        Returns:
            torch.Tensor: The logits after applying the z-loss.
        """
        if self.config.z_loss_coeff is not None:
            moe_z_loss_coeff = (
                self.config.z_loss_coeff / constants.model_parallel_world_size()
            )
            z_loss = z_loss_func(logits, moe_z_loss_coeff)
            logits = MoEAuxLossAutoScaler.apply(logits, z_loss)
            # FIXME
            # save_to_aux_losses_tracker(
            #     "z_loss",
            #     z_loss / moe_z_loss_coeff,
            #     self.layer_number,
            #     self.num_layers,
            # )
        return logits

    def apply_input_jitter(self, input: torch.Tensor):
        """Add noise to the input tensor.
        Refer to https://arxiv.org/abs/2101.03961.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Jittered input.
        """
        if self.config.input_jitter_eps is not None:
            eps = self.config.input_jitter_eps
            if self.input_jitter is None:
                self.input_jitter = torch.distributions.uniform.Uniform(
                    torch.tensor(1.0 - eps, device=input.device),
                    torch.tensor(1.0 + eps, device=input.device),
                ).rsample

            input = (input * self.input_jitter(input.shape)).to(input.dtype)
        return input

    def routing(self, logits: torch.Tensor):
        """Top-k routing function

        Args:
            logits (torch.Tensor): Logits tensor after gating.

        Returns:
            probs (torch.Tensor): the probabilities tensor after load balancing.
            indices (torch.Tensor): the indices tensor after top-k selection.
        """
        logits = logits.view(-1, self.num_experts)

        # Apply Z-Loss
        logits = self.apply_z_loss(logits)

        if (
            constants.model_parallel_world_size() > 1
            and self.config.token_dispatcher_type == "alltoall"
        ):
            # Gather the logits from the TP region
            logits = gather_from_sequence_parallel_region(logits)

        if self.config.routing_type == "sinkhorn":
            scores, indices = self.sinkhorn_load_balancing(logits)
        elif self.config.routing_type == "aux_loss":
            scores, indices = self.aux_loss_load_balancing(logits)
        elif self.config.routing_type == "none":
            # A naive top-k routing without load balancing
            scores, indices, _ = topk_softmax_with_capacity(
                logits,
                self.config.moe_top_k,
                capacity_factor=self.config.capacity_factor,
                pad_to_capacity=self.config.pad_to_capacity,
                drop_policy=self.config.token_drop_policy,
            )
        else:
            raise ValueError(
                f"Unsupported MoE routing type: {self.config.routing_type}"
            )

        return scores, indices

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the router.

        Args:
            input (torch.Tensor): Input tensor.
        """
        self.hidden = input.shape[-1]
        # Apply input jitter
        input = self.apply_input_jitter(input)
        logits = self.gating(input)
        logits = logits.view(-1, self.num_experts)
        scores, indices = self.routing(logits)
        return scores, indices
