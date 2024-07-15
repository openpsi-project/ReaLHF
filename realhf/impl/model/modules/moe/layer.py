# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# adopted from megatron
from typing import Optional, Union

import torch
import torch.nn as nn

import realhf.base.constants as constants
from realhf.api.core.model_api import ReaLModelConfig
from realhf.impl.model.modules.mlp import GemmaRMSNorm, LlamaRMSNorm
from realhf.impl.model.modules.moe.experts import GroupedMLP, SequentialMLP
from realhf.impl.model.modules.moe.router import TopKRouter
from realhf.impl.model.modules.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
)


class LayerNormMoELayer(torch.nn.Module):
    """Mixture of experts Layer **currently only supports no token dropping**."""

    def __init__(
        self,
        config: ReaLModelConfig,
        use_grouped_gemm: bool,
        layer_idx: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super(LayerNormMoELayer, self).__init__()

        self.config = config
        self.dtype = dtype
        self.device = device
        self.expert_parallel_size = constants.expert_parallel_world_size()
        assert (
            self.expert_parallel_size > 0
        ), "Expected non-negative expert parallel size"

        assert self.config.num_experts % self.expert_parallel_size == 0
        self.num_local_experts = self.config.num_experts // self.expert_parallel_size
        local_expert_indices_offset = (
            constants.expert_parallel_rank() * self.num_local_experts
        )

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(
            map(lambda x: x < self.config.num_experts, self.local_expert_indices)
        )
        self.layer_idx = layer_idx

        if config.layer_norm_type is None:
            layer_norm_fn = nn.LayerNorm
        elif config.layer_norm_type == "rms":
            layer_norm_fn = LlamaRMSNorm
        elif config.layer_norm_type == "gemma":
            layer_norm_fn = GemmaRMSNorm
        self.ln = layer_norm_fn(
            config.hidden_dim, eps=config.layer_norm_epsilon, dtype=dtype, device=device
        )

        self.router = TopKRouter(config.hidden_dim, config=self.config)
        if use_grouped_gemm:
            self.experts = GroupedMLP(
                self.num_local_experts, self.config, dtype=dtype, device=device
            )
        else:
            self.experts = SequentialMLP(
                self.num_local_experts, self.config, dtype=dtype, device=device
            )

        if self.config.token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {self.config.token_dispatcher_type}"
            )

    def forward(self, hidden_states: torch.Tensor):
        if (
            self.training
            and constants.model_parallel_world_size() > 1
            and not constants.sequence_parallel()
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # process MoE
        def custom_forward(hidden_states):
            probs, indices = self.router(hidden_states)
            (dispatched_input, tokens_per_expert) = (
                self.token_dispatcher.token_permutation(hidden_states, probs, indices)
            )
            expert_output = self.experts(dispatched_input, tokens_per_expert)
            output = self.token_dispatcher.token_unpermutation(
                expert_output,
            )
            return output

        hidden_states = self.ln(hidden_states)
        output = custom_forward(hidden_states)
        return output
