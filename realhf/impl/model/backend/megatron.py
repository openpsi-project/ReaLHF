from contextlib import contextmanager
from typing import *
import dataclasses

try:
    from megatron.core import parallel_state
    from megatron.core.distributed.distributed_data_parallel import (
        DistributedDataParallel,
    )
    from megatron.core.distributed.finalize_model_grads import (
        finalize_model_grads,
    )
    from megatron.core.distributed.param_and_grad_buffer import (
        ParamAndGradBuffer,
    )
    from megatron.core.optimizer import get_megatron_optimizer
    from megatron.core.optimizer.optimizer_config import OptimizerConfig
    from megatron.core.transformer.transformer_config import TransformerConfig
except ImportError or ModuleNotFoundError:
    # dummy class for type hint, due to missing files in megatron CPU installation
    class TransformerConfig:
        pass


import torch
import torch.distributed as dist
import torch.nn as nn
import transformers

from realhf.api.core import model_api
from realhf.base import constants, logging
from realhf.impl.model.backend.pipe_runner import PipelineRunner
from realhf.impl.model.backend.utils import (
    MegatronEngine,
    OptimizerParamScheduler,
)
from realhf.impl.model.modules.mlp import get_activation_fn
from realhf.impl.model.nn.real_llm_api import ReaLModel
from realhf.impl.model.nn.real_llm_generate import GenerationConfig

WITHIN_MEGATRON_CONTEXT = False

logger = logging.getLogger("Megatron Backend", "benchmark")


@contextmanager
def megatron_ctx():
    global WITHIN_MEGATRON_CONTEXT
    if WITHIN_MEGATRON_CONTEXT:
        raise RuntimeError(
            "Megatron context is already set up. Destroy it first."
        )

    WITHIN_MEGATRON_CONTEXT = True

    grid = constants.grid()
    # TODO: implement context parallel.
    # TODO: implement expert parallel.

    # Build the data-parallel groups.
    g = constants.data_parallel_group()
    parallel_state._DATA_PARALLEL_GROUP = g
    parallel_state._DATA_PARALLEL_GROUP_GLOO = (
        grid.get_data_parallel_group_gloo()
    )
    parallel_state._DATA_PARALLEL_GLOBAL_RANKS = dist.get_process_group_ranks(g)
    parallel_state._DATA_PARALLEL_GROUP_WITH_CP = g
    parallel_state._DATA_PARALLEL_GROUP_WITH_CP_GLOO = (
        grid.get_data_parallel_group_gloo()
    )
    parallel_state._DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = (
        dist.get_process_group_ranks(g)
    )

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
    parallel_state._EMBEDDING_GLOBAL_RANKS = (
        dist.get_process_group_ranks(grid.embedding_proc_group)
        if grid.embedding_proc_group is not None
        else list(range(dist.get_world_size()))
    )
    parallel_state._POSITION_EMBEDDING_GROUP = (
        grid.position_embedding_proc_group
    )
    parallel_state._POSITION_EMBEDDING_GLOBAL_RANKS = (
        dist.get_process_group_ranks(grid.position_embedding_proc_group)
        if grid.position_embedding_proc_group is not None
        else list(range(dist.get_world_size()))
    )

    # Build the tensor + data parallel groups.
    parallel_state._TENSOR_AND_DATA_PARALLEL_GROUP = grid.tp_dp_proc_group
    parallel_state._TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = (
        grid.tp_dp_proc_group
    )

    # Build the tensor + expert parallel groups
    parallel_state._EXPERT_MODEL_PARALLEL_GROUP = constants.self_group()
    parallel_state._TENSOR_AND_EXPERT_PARALLEL_GROUP = (
        constants.model_parallel_group()
    )
    g = constants.data_parallel_group()
    parallel_state._DATA_MODULO_EXPERT_PARALLEL_GROUP = g
    parallel_state._DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = (
        grid.get_data_parallel_group_gloo()
    )

    # Remove the global memory buffer for megatron to save GPU memory.
    parallel_state._GLOBAL_MEMORY_BUFFER = None
    yield
    WITHIN_MEGATRON_CONTEXT = False


def get_megatron_transformer_config(
    mconfig: model_api.ReaLModelConfig,
) -> TransformerConfig:
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
        normalization=(
            "RMSNorm" if mconfig.layer_norm_type == "rms" else "LayerNorm"
        ),
        attention_softmax_in_fp32=True,
        apply_query_key_layer_scaling=mconfig.scale_attn_by_inverse_layer_idx,
    )


class ReaLMegatronEngine:
    # TODO: merge this with deepspeed and inference engine

    def __init__(self, module: ReaLModel, megatron_engine: MegatronEngine):
        self.module = module

        self.device = module.device
        self.dtype = module.dtype

        self.engine = megatron_engine

        if constants.pipe_parallel_world_size() > 1:
            self.pipe_runner = PipelineRunner(module)

            model_parameters = filter(
                lambda p: p.requires_grad, self.module.parameters()
            )
            num_params = sum([p.numel() for p in model_parameters])
            unique_params = num_params

            params_tensor = torch.LongTensor(
                data=[num_params, unique_params]
            ).to(self.device)
            dist.all_reduce(
                params_tensor, group=constants.grid().get_model_parallel_group()
            )
            params_tensor = params_tensor.tolist()
            total_params = params_tensor[0]
            unique_params = params_tensor[1]

            if constants.parallelism_rank() == 0:
                logger.info(
                    f"CONFIG: default_train_mbs={self.pipe_runner.default_train_mbs} "
                    f"default_inf_mbs={self.pipe_runner.default_inf_mbs} "
                    f"num_layers(this stage)={self.module.num_layers} "
                    f"pp_size={constants.pipe_parallel_world_size()} "
                    f"dp_size={constants.data_parallel_world_size()} "
                    f"mp_size={constants.model_parallel_world_size()} "
                )
            if constants.data_parallel_rank() == 0:
                logger.info(
                    f"rank={constants.parallelism_rank()} "
                    f"stage={constants.pipe_parallel_rank()} "
                    f"layers={self.module.num_layers} "
                    f"[{self.module.layer_idx_start}, {self.module.layer_idx_end}) "
                    f"stage_params={num_params} ({num_params/1e6:0.3f}M) "
                    f"total_params={total_params} ({total_params/1e6:0.3f}M) "
                    f"unique_params={unique_params} ({unique_params/1e6:0.3f}M)"
                )

    def train(self, mode: bool = True):
        self.module.train(mode)
        self.engine.ddp.train(mode)
        return self

    def eval(self):
        self.module.eval()
        self.engine.ddp.eval()
        return self

    def train_batch(
        self,
        seqlens_cpu: List[int],
        packed_input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        loss_fn: Callable,
        version_steps: int,
        input_lens_for_partition: Optional[torch.Tensor] = None,
        num_micro_batches: Optional[int] = None,
        **loss_fn_kwargs,
    ):
        with megatron_ctx():
            self.engine.zero_grad()
            if constants.pipe_parallel_world_size() > 1:
                return self.pipe_runner.train_batch(
                    engine=self.engine,
                    seqlens_cpu=seqlens_cpu,
                    packed_input_ids=packed_input_ids,
                    cu_seqlens=cu_seqlens,
                    loss_fn=loss_fn,
                    version_steps=version_steps,
                    input_lens_for_partition=input_lens_for_partition,
                    num_micro_batches=num_micro_batches,
                    **loss_fn_kwargs,
                )
            else:
                # FIXME: num_micro_batches is ignored here
                max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))
                model_output = self.engine.ddp(
                    packed_input_ids=packed_input_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                ).logits
                loss, stat = loss_fn(
                    model_output, packed_input_ids, cu_seqlens, **loss_fn_kwargs
                )
                self.engine.optim.scale_loss(loss).backward()
                finalize_model_grads([self.engine.ddp])
                update_successful, grad_norm, _ = self.engine.optim.step()
                if update_successful:
                    self.engine.lr_scheduler.step_absolute(version_steps)
                if (
                    constants.data_parallel_rank() == 0
                    and constants.model_parallel_rank() == 0
                ):
                    logger.info(
                        f"Megatron backend update success? {update_successful}. "
                        f"Grad Norm: {grad_norm}. "
                        f"Current loss scale: {self.engine.optim.get_loss_scale()}. "
                    )
                return stat

    @torch.no_grad()
    def eval_batch(
        self,
        seqlens_cpu: List[int],
        packed_input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        loss_fn: Callable,
        input_lens_for_partition: Optional[torch.Tensor] = None,
        num_micro_batches: Optional[int] = None,
        **loss_fn_kwargs,
    ):
        with megatron_ctx():
            if constants.pipe_parallel_world_size() > 1:
                return self.pipe_runner.eval_batch(
                    seqlens_cpu=seqlens_cpu,
                    packed_input_ids=packed_input_ids,
                    cu_seqlens=cu_seqlens,
                    loss_fn=loss_fn,
                    input_lens_for_partition=input_lens_for_partition,
                    num_micro_batches=num_micro_batches,
                    **loss_fn_kwargs,
                )
            else:
                max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))
                model_output = self.engine.ddp(
                    packed_input_ids=packed_input_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                ).logits
                _, stat = loss_fn(
                    model_output, packed_input_ids, cu_seqlens, **loss_fn_kwargs
                )
                return stat

    def forward(
        self,
        seqlens_cpu: List[int],
        packed_input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        num_micro_batches: Optional[int] = None,
    ):
        with megatron_ctx():
            if constants.pipe_parallel_world_size() > 1:
                return self.pipe_runner.forward(
                    packed_input_ids=packed_input_ids,
                    cu_seqlens=cu_seqlens,
                    seqlens_cpu=seqlens_cpu,
                    num_micro_batches=num_micro_batches,
                )
            else:
                max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))
                return self.engine.ddp(
                    packed_input_ids=packed_input_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                ).logits

    @torch.no_grad()
    def generate(
        self,
        seqlens_cpu: List[int],
        packed_input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        tokenizer: transformers.PreTrainedTokenizerFast,
        gconfig: GenerationConfig = dataclasses.field(
            default_factory=GenerationConfig
        ),
        num_micro_batches: Optional[int] = None,
    ):
        with megatron_ctx():
            if constants.pipe_parallel_world_size() > 1:
                return self.pipe_runner.generate(
                    seqlens_cpu=seqlens_cpu,
                    num_micro_batches=num_micro_batches,
                    tokenizer=tokenizer,
                    packed_input_ids=packed_input_ids,
                    cu_seqlens=cu_seqlens,
                    gconfig=gconfig,
                )
            else:
                max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))
                res = self.module.generate(
                    tokenizer=tokenizer,
                    packed_input_ids=packed_input_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    gconfig=gconfig,
                )
                return res.sequences, res.scores, res.logits_mask


@dataclasses.dataclass
class MegatronTrainBackend(model_api.ModelBackend):
    optimizer_name: str = dataclasses.field(
        metadata={"choices": ["adam", "sgd"]},
        default="adam",
    )
    optimizer_config: dict = dataclasses.field(
        default_factory=lambda: dict(
            lr=1e-5, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-5
        )
    )
    lr_scheduler_type: str = "cosine"
    warmup_steps_proportion: float = 0.0
    min_lr_ratio: float = 0.0  # will be used for linear and cosine schedule
    enable_fp16: bool = True
    enable_bf16: bool = False
    use_zero_optimization: bool = True
    overlap_grad_reduce: bool = True
    overlap_param_gather: bool = True
    accumulate_allreduce_grads_in_fp32: bool = False
    initial_loss_scale: float = 4096.0
    gradient_clipping: float = 1.0
    # addtional args
    additional_config: Dict = dataclasses.field(default_factory=dict)

    def _initialize(
        self, model: model_api.Model, spec: model_api.FinetuneSpec
    ) -> model_api.Model:
        module = model.module
        if not isinstance(module, ReaLModel):
            raise ValueError("MegatronTrainBackend only supports ReaLModel.")
        with megatron_ctx():
            module = DistributedDataParallel(
                config=get_megatron_transformer_config(module.config),
                module=module,
                data_parallel_group=constants.data_parallel_group(),
                accumulate_allreduce_grads_in_fp32=self.accumulate_allreduce_grads_in_fp32,
                overlap_grad_reduce=self.overlap_grad_reduce,
                use_distributed_optimizer=self.use_zero_optimization,
                expert_data_parallel_group=None,
                disable_bucketing=False,
                check_for_nan_in_grad=False,
            )

        real_model: ReaLModel = module.module
        if self.use_zero_optimization:
            # Remap parameters.
            assert len(module.buffers) == 1
            param_grad_buf: ParamAndGradBuffer = module.buffers[0]

            # Map Megatron flattened parameters to ReaLModel!
            real_model.contiguous_param = param_grad_buf.param_data

            # Sanity checks.
            assert real_model._param_size == param_grad_buf.numel
            for n, p in real_model.layers.named_parameters():
                n = ".".join(
                    [
                        str(real_model.layer_idx_start + int(n.split(".")[0])),
                        n.split(".", 1)[1],
                    ]
                )
                idx_start, idx_end, _ = param_grad_buf.param_index_map[p]
                assert real_model._param_spec[n].start_idx == idx_start
                assert real_model._param_spec[n].end_idx == idx_end
                assert real_model._param_spec[n].shape == p.shape
                assert torch.allclose(
                    p,
                    real_model.contiguous_param[idx_start:idx_end].view(
                        p.shape
                    ),
                )

        betas = self.optimizer_config.get("betas", (0.9, 0.95))
        wd = self.optimizer_config.get("weight_decay", 0.1)
        lr = self.optimizer_config["lr"]
        opt_cfg = OptimizerConfig(
            optimizer=self.optimizer_name,
            fp16=self.enable_fp16,
            bf16=self.enable_bf16,
            lr=lr,
            min_lr=self.min_lr_ratio * lr,
            weight_decay=wd,
            params_dtype=real_model.dtype,
            initial_loss_scale=self.initial_loss_scale,
            adam_beta1=betas[0],
            adam_beta2=betas[1],
            adam_eps=self.optimizer_config.get("eps", 1e-5),
            sgd_momentum=self.optimizer_config.get("momentum", 0.9),
            use_distributed_optimizer=self.use_zero_optimization,
            overlap_grad_reduce=self.overlap_grad_reduce,
            overlap_param_gather=self.overlap_param_gather,
            clip_grad=self.gradient_clipping,
        )

        with megatron_ctx():
            # no_weight_decay_cond and scale_lr_cond have the following signature:
            # foo(name: str, param: torch.Tensor) -> bool
            optimizer = get_megatron_optimizer(
                opt_cfg,
                [module],
                no_weight_decay_cond=lambda n, p: any(
                    k in n for k in ["bias", "ln.weight", "ln_f.weight"]
                ),
                scale_lr_cond=None,
                lr_mult=1.0,
            )

            warmup_steps = int(
                self.warmup_steps_proportion * spec.total_train_steps
            )
            lr_scheduler = OptimizerParamScheduler(
                optimizer,
                init_lr=0.0 if self.warmup_steps_proportion > 0 else lr,
                max_lr=lr,
                min_lr=self.min_lr_ratio * lr,
                lr_warmup_steps=warmup_steps,
                lr_decay_steps=spec.total_train_steps - warmup_steps,
                lr_decay_style=self.lr_scheduler_type,
                start_wd=wd,
                end_wd=wd,
                wd_incr_steps=spec.total_train_steps,
                wd_incr_style="constant",
            )

        mg_engine = MegatronEngine(module, optimizer, lr_scheduler)
        model.module = ReaLMegatronEngine(real_model, mg_engine)
        return model


model_api.register_backend("megatron", MegatronTrainBackend)
