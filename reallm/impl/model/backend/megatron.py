from reallm.base import constants
from megatron.core import parallel_state
import torch.distributed as dist
from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel
from megatron.core.transformer.transformer_config import TransformerConfig
from contextlib import contextmanager
from typing import Dict, Optional
from reallm.impl.model.modules.mlp import get_activation_fn
from reallm.api.core import model_api
from reallm.impl.model.nn.real_llm_api import ReaLModel
import torch.nn as nn
import torch
from megatron.core.distributed.param_and_grad_buffer import ParamAndGradBuffer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.optimizer import get_megatron_optimizer, MegatronOptimizer
import dataclasses
from megatron.core.distributed.finalize_model_grads import finalize_model_grads

WITHIN_MEGATRON_CONTEXT = False


@contextmanager
def megatron_ctx():
    global WITHIN_MEGATRON_CONTEXT
    if WITHIN_MEGATRON_CONTEXT:
        raise RuntimeError("Megatron context is already set up. Destroy it first.")

    WITHIN_MEGATRON_CONTEXT = True

    grid = constants.grid()
    # TODO: implement context parallel.
    # TODO: implement expert parallel.

    # Build the data-parallel groups.
    g = constants.data_parallel_group()
    parallel_state._DATA_PARALLEL_GROUP = g
    parallel_state._DATA_PARALLEL_GROUP_GLOO = grid.get_data_parallel_group_gloo()
    parallel_state._DATA_PARALLEL_GLOBAL_RANKS = dist.get_process_group_ranks(g)
    parallel_state._DATA_PARALLEL_GROUP_WITH_CP = g
    parallel_state._DATA_PARALLEL_GROUP_WITH_CP_GLOO = grid.get_data_parallel_group_gloo()
    parallel_state._DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = dist.get_process_group_ranks(g)

    # Build the context-parallel groups.
    parallel_state._CONTEXT_PARALLEL_GROUP = constants.self_group()
    parallel_state._CONTEXT_PARALLEL_GLOBAL_RANKS = [dist.get_rank()]

    # Build the model-parallel groups.
    parallel_state._MODEL_PARALLEL_GROUP = grid.get_model_parallel_group()

    # Build the tensor model-parallel groups.
    g = constants.model_parallel_group()
    parallel_state._TENSOR_MODEL_PARALLEL_GROUP = g

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    g = constants.pipe_parallel_group()
    parallel_state._PIPELINE_MODEL_PARALLEL_GROUP = g
    parallel_state._PIPELINE_GLOBAL_RANKS = dist.get_process_group_ranks(g)
    parallel_state._EMBEDDING_GROUP = grid.embedding_proc_group
    parallel_state._EMBEDDING_GLOBAL_RANKS = dist.get_process_group_ranks(grid.embedding_proc_group)
    parallel_state._POSITION_EMBEDDING_GROUP = grid.position_embedding_proc_group
    parallel_state._POSITION_EMBEDDING_GLOBAL_RANKS = dist.get_process_group_ranks(
        grid.position_embedding_proc_group
    )

    # Build the tensor + data parallel groups.
    parallel_state._TENSOR_AND_DATA_PARALLEL_GROUP = grid.tp_dp_proc_group
    parallel_state._TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = grid.tp_dp_proc_group

    # Build the tensor + expert parallel groups
    parallel_state._EXPERT_MODEL_PARALLEL_GROUP = constants.self_group()
    parallel_state._TENSOR_AND_EXPERT_PARALLEL_GROUP = constants.model_parallel_group()
    g = constants.data_parallel_group()
    parallel_state._DATA_MODULO_EXPERT_PARALLEL_GROUP = g
    parallel_state._DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = grid.get_data_parallel_group_gloo()

    yield
    WITHIN_MEGATRON_CONTEXT = False


def get_megatron_transformer_config(mconfig: model_api.ReaLModelConfig) -> TransformerConfig:
    nq = mconfig.hidden_dim // mconfig.head_dim
    n_group = nq // mconfig.n_kv_heads
    return TransformerConfig(
        num_layers=mconfig.n_layers,
        hidden_size=mconfig.hidden_dim,
        num_attention_heads=nq,
        num_query_groups=n_group,
        ffn_hidden_size=mconfig.intermediate_dim,
        kv_channels=mconfig.n_kv_heads,
        hidden_dropout=0.0,
        attention_dropout=mconfig.attn_pdrop,
        layernorm_epsilon=mconfig.layer_norm_epsilon,
        add_qkv_bias=mconfig.use_attention_bias,
        activation_func=get_activation_fn(mconfig.activation_function),
        rotary_interleaved=mconfig.rotary_interleaved,
        normalization="RMSNorm" if mconfig.layer_norm_type == "rms" else "LayerNorm",
        attention_softmax_in_fp32=True,
        apply_query_key_layer_scaling=mconfig.scale_attn_by_inverse_layer_idx,
    )


@dataclasses.dataclass
class MegatronModelRunner:
    megatron_module: DistributedDataParallel
    optimizer: MegatronOptimizer

    @property
    def module(self):
        return self.megatron_module.module

    def zero_grad(self, set_to_none=True):
        with megatron_ctx():
            self.megatron_module.zero_grad_buffer()
            self.optimizer.zero_grad(set_to_none=set_to_none)

    def __call__(self, *args, **kwargs):
        with megatron_ctx():
            return self.megatron_module(*args, **kwargs)

    def forward(self, *args, **kwargs):
        with megatron_ctx():
            return self.megatron_module(*args, **kwargs)

    def backward(self, loss: torch.Tensor):
        with megatron_ctx():
            loss.backward()
            finalize_model_grads([self.megatron_module])

    def step(self):
        with megatron_ctx():
            # TODO: lr scheduler here
            return self.optimizer.step()
    
    def train(self, *args, **kwargs):
        with megatron_ctx():
            return self.megatron_module.train(*args, **kwargs)

    def eval(self, *args, **kwargs):
        with megatron_ctx():
            return self.megatron_module.eval(*args, **kwargs)
    


@dataclasses.dataclass
class MegatronTrainBackend(model_api.ModelBackend):
    optimizer_name: str = dataclasses.field(
        metadata={"choices": ["adam", "sgd"]},
        default="adam",
    )
    optimizer_config: dict = dataclasses.field(
        default_factory=lambda: dict(lr=1e-5, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-5)
    )
    lr_scheduler_type: str = "cosine"
    warmup_steps_proportion: float = 0.0
    min_lr_ratio: float = 0.0  # will be used for linear and cosine schedule
    enable_fp16: bool = True
    enable_bf16: bool = False
    zero_stage: int = 2
    accumulate_allreduce_grads_in_fp32: bool = False
    # addtional args
    additional_config: Dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.zero_stage > 2:
            raise ValueError("Zero stage for Megatron must be in [0, 1, 2].")

    def _initialize(self, model: model_api.Model, spec: model_api.FinetuneSpec) -> model_api.Model:
        module = model.module
        if not isinstance(module, ReaLModel):
            raise ValueError("MegatronTrainBackend only supports ReaLModel.")
        with megatron_ctx():
            module = DistributedDataParallel(
                config=get_megatron_transformer_config(module.config),
                module=module,
                data_parallel_group=constants.data_parallel_group(),
                accumulate_allreduce_grads_in_fp32=self.accumulate_allreduce_grads_in_fp32,
                overlap_grad_reduce=self.zero_stage > 1,
                use_distributed_optimizer=self.zero_stage > 0,
                expert_data_parallel_group=None,
                disable_bucketing=False,
                check_for_nan_in_grad=False,
            )

        # Remap parameters.
        assert len(module.buffers) == 1
        param_grad_buf: ParamAndGradBuffer = module.buffers[0]

        # Map Megatron flattened parameters to ReaLModel!
        real_model: ReaLModel = module.module
        real_model.contiguous_param = param_grad_buf.param_data

        # Sanity checks.
        assert real_model._param_size == param_grad_buf.numel
        for n, p in real_model.layers.named_parameters():
            idx_start, idx_end, _ = param_grad_buf.param_index_map[p]
            assert real_model._param_spec[n].start_idx == idx_start
            assert real_model._param_spec[n].end_idx == idx_end
            assert real_model._param_spec[n].shape == p.shape
            assert torch.allclose(p, real_model.contiguous_param[idx_start:idx_end].view(p.shape))

        betas = self.optimizer_config.get("betas", (0.9, 0.95))
        opt_cfg = OptimizerConfig(
            optimizer=self.optimizer_name,
            fp16=self.enable_fp16,
            bf16=self.enable_bf16,
            lr=self.optimizer_config["lr"],
            min_lr=self.min_lr_ratio * self.optimizer_config["lr"],
            weight_decay=self.optimizer_config.get("weight_decay", 0.1),
            params_dtype=real_model.dtype,
            initial_loss_scale=4096.0,
            adam_beta1=betas[0],
            adam_beta2=betas[1],
            adam_eps=self.optimizer_config.get("eps", 1e-5),
            sgd_momentum=self.optimizer_config.get("momentum", 0.9),
            use_distributed_optimizer=self.zero_stage > 0,
            overlap_grad_reduce=self.zero_stage > 1,
            overlap_param_gather=self.zero_stage > 1,
            clip_grad=self.optimizer_config.get("clip_grad", 1.0),
        )

        with megatron_ctx():
            # no_weight_decay_cond and scale_lr_cond have the following signature:
            # foo(name: str, param: torch.Tensor) -> bool
            optimizer = get_megatron_optimizer(
                opt_cfg,
                [module],
                no_weight_decay_cond=None,
                scale_lr_cond=None,
                lr_mult=1.0,
            )
        # TODO: add lr scheduler
        model.module = MegatronModelRunner(module, optimizer)
        return model
