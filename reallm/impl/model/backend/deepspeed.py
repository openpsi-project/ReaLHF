from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import dataclasses
import functools
import math

from deepspeed.runtime import zero
from deepspeed.runtime.config import DeepSpeedConfig
from deepspeed.runtime.engine import DeepSpeedEngine, DeepSpeedOptimizerCallable, DeepSpeedSchedulerCallable
import deepspeed
import torch
import torch.distributed

from reallm.base.constants import (data_parallel_world_size, model_parallel_world_size,
                                   pipe_parallel_world_size)
from reallm.impl.model.backend.pipe_engine import DeepSpeedPipelineEngine
import reallm.api.core.model_api as model_api
import reallm.base.constants as constants
import reallm.base.logging as logging

DEFAULT_TRAIN_MICRO_BATCH_SIZE_PER_GPU = 32  # A place-holder for inference.
logger = logging.getLogger("DeepSpeed Backend")


def get_train_ds_config(
    offload_param: bool = False,
    offload_optimizer_state: bool = False,
    enable_fp16: bool = True,
    enable_bf16: bool = False,
    stage: int = 2,
    **kwargs,
):
    if enable_bf16 and enable_fp16:
        raise ValueError("Cannot enable both fp16 and bf16 at the same time.")
    zero_opt_dict = {
        "stage": stage,
        "overlap_comm": True,
        "round_robin_gradients": True,
        "offload_param": {
            "device": "cpu" if offload_param else "none",
            "pin_memory": True,
        },
        "offload_optimizer": {
            "device": "cpu" if offload_optimizer_state else "none",
            "pin_memory": True,
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False,
    }
    return {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "fp16": {
            "enabled": enable_fp16,
            "loss_scale_window": 40,
            "initial_scale_power": 12,
        },
        "bf16": {
            "enabled": enable_bf16,
        },
        "data_types": {
            "grad_accum_dtype": "fp32" if enable_bf16 else "fp16",
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "gradient_predevide_factor": 1.0,
        "wall_clock_breakdown": False,
        **kwargs,
    }


def get_eval_ds_config(offload=False, stage=0, enable_fp16: bool = True, enable_bf16: bool = False, **kwargs):
    if enable_bf16 and enable_fp16:
        raise ValueError("Cannot enable both fp16 and bf16 at the same time.")
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device,
            "pin_memory": True,
        },
        "memory_efficient_linear": False,
    }
    return {
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "train_micro_batch_size_per_gpu": DEFAULT_TRAIN_MICRO_BATCH_SIZE_PER_GPU,
        "train_batch_size": torch.distributed.get_world_size(group=constants.data_parallel_group()) *
        DEFAULT_TRAIN_MICRO_BATCH_SIZE_PER_GPU,
        "fp16": {
            "enabled": enable_fp16,
        },
        "bf16": {
            "enabled": enable_bf16,
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        **kwargs,
    }


def get_optimizer_grouped_parameters(
    model: torch.nn.Module,
    weight_decay: float,
    no_decay_name_list: List[str] = ["bias", "LayerNorm.weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def deepspeed_initialize(
    model: torch.nn.Module,
    config: Dict,
    engine_type: str = "deepspeed",
    optimizer: Optional[Union[torch.optim.Optimizer, DeepSpeedOptimizerCallable]] = None,
    model_parameters: Optional[torch.nn.Module] = None,
    lr_scheduler: Optional[Union[torch.optim.lr_scheduler._LRScheduler, DeepSpeedSchedulerCallable]] = None,
    mpu=None,
    enable_async_p2p_communication: Optional[bool] = False,
    enable_async_instruction: Optional[bool] = False,
    instruction_sync: Optional[bool] = False,
) -> Tuple[DeepSpeedEngine, torch.optim.Optimizer, Any, Any]:
    """A simple wrapper around deepspeed.initialize."""
    if mpu is None:
        mpu = constants.grid()
    if engine_type == "deepspeed":
        # Disable zero.Init context if it's currently enabled
        zero.partition_parameters.shutdown_init_context()

        from deepspeed import comm as dist

        deepspeed.dist = dist

        config_class = DeepSpeedConfig(config, mpu)
        engine = DeepSpeedEngine(
            args=None,
            model=model,
            optimizer=optimizer,
            model_parameters=model_parameters,
            lr_scheduler=lr_scheduler,
            mpu=mpu,
            dist_init_required=False,
            config=config,
            config_class=config_class,
            # dont_change_device=True,
        )

        # Restore zero.Init context if necessary
        zero.partition_parameters.restore_init_context()
        logger.info(f"Deepspeed Engine initialze finished.")
        return_items = [engine, engine.optimizer, engine.training_dataloader, engine.lr_scheduler]
    elif engine_type == "pipe":
        # mpu = model.mpu()
        config_class = DeepSpeedConfig(config, mpu)
        engine_cls = DeepSpeedPipelineEngine
        engine = engine_cls(
            enable_async_p2p_communication=enable_async_p2p_communication,
            enable_async_instruction=enable_async_instruction,
            model=model,
            args=None,
            config=config,
            config_class=config_class,
            mpu=mpu,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            dist_init_required=False,
            instruction_sync=instruction_sync,
        )
        logger.info(f"Deepspeed Pipeline Engine initialze finished.")
        return_items = [engine, engine.optimizer, engine.training_dataloader, engine.lr_scheduler]
    return tuple(return_items)


@dataclasses.dataclass
class DeepspeedTrainBackend(model_api.ModelBackend):
    optimizer_name: str = "adam"
    optimizer_config: dict = dataclasses.field(
        default_factory=lambda: dict(lr=1e-5, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-5))
    lr_scheduler_type: str = "cosine"
    warmup_steps_proportion: float = 0.0
    min_lr_ratio: float = 0.0  # will be used for linear and cosine schedule
    gradient_checkpointing: bool = False  # FIXME:
    offload_param: bool = False
    offload_optimizer_state: bool = False
    enable_fp16: bool = True
    enable_bf16: bool = False
    zero_stage: int = 2
    # hybrid engine args
    enable_hybrid_engine: bool = False
    max_out_tokens: int = 512
    inference_tp_size: int = 1
    release_inference_cache: bool = False
    pin_parameters: bool = True
    tp_gather_partition_size: int = 8
    # addtional deepspeed args
    additional_ds_config: Dict = dataclasses.field(default_factory=dict)
    engine_type: str = "deepspeed"
    # parallelism args
    enable_async_p2p_communication: bool = False  # FIXME:
    enable_async_instruction: bool = False
    instruction_sync: bool = False
    # selective gradient ckpt, only effective when gradient_checkpointing is True
    ckpt_attn: bool = False  # checkpoint attn only
    ckpt_mlp: bool = False  # checkpoint mlp only
    # stream pipe engine require model configs
    max_seq_len: int = 512
    max_new_tokens: int = 512
    max_mb_size: int = 32

    def __post_init__(self):
        if self.engine_type == "pipe":
            assert self.zero_stage < 2
            assert self.enable_hybrid_engine is False

    def _initialize(self, model: model_api.Model, spec: model_api.FinetuneSpec):
        deepspeed.init_distributed(auto_mpi_discovery=False)
        module = model.module
        weight_decay = self.optimizer_config.get("weight_decay", 0.0)
        if self.optimizer_name == "adam":
            if not self.offload_param and not self.offload_optimizer_state:
                optim_cls = deepspeed.ops.adam.FusedAdam
            else:
                optim_cls = deepspeed.ops.adam.DeepSpeedCPUAdam
            optimizer = optim_cls(
                get_optimizer_grouped_parameters(module, weight_decay),
                **self.optimizer_config,
            )
        else:
            raise NotImplementedError(f"Unsupported optimizer: {self.optimizer_name}.")

        hybrid_engine_args = dict(
            enabled=self.enable_hybrid_engine,
            max_out_tokens=self.max_out_tokens,
            inference_tp_size=self.inference_tp_size,
            release_inference_cache=self.release_inference_cache,
            pin_parameters=self.pin_parameters,
            tp_gather_partition_size=self.tp_gather_partition_size,
        )

        ds_config = get_train_ds_config(
            offload_param=self.offload_param,
            offload_optimizer_state=self.offload_optimizer_state,
            stage=self.zero_stage,
            enable_bf16=self.enable_bf16,
            enable_fp16=self.enable_fp16,
            hybrid_engine_args=hybrid_engine_args,
            **self.additional_ds_config,
        )

        ds_config["train_micro_batch_size_per_gpu"] = spec.batch_size_per_device
        ds_config["train_batch_size"] = (spec.batch_size_per_device * data_parallel_world_size() *
                                         model_parallel_world_size() * pipe_parallel_world_size())

        def warmup_then_cosine_anneal(step, warmup_steps_proportion, total_steps, min_lr_ratio):
            warmup_steps = max(5, int(total_steps * warmup_steps_proportion))
            cosine_steps = total_steps - warmup_steps
            if step < warmup_steps:
                return 1.0 / warmup_steps * step
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1 + math.cos(
                (step - warmup_steps) / cosine_steps * math.pi))

        def warmup_then_linear_anneal(step, warmup_steps_proportion, total_steps, min_lr_ratio):
            warmup_steps = max(5, int(total_steps * warmup_steps_proportion))
            linear_steps = total_steps - warmup_steps
            if step < warmup_steps:
                return 1.0 / warmup_steps * step
            return 1.0 - (1.0 - min_lr_ratio) / linear_steps * (step - warmup_steps)

        def warmup_then_constant_anneal(step, warmup_steps_proportion, total_steps, min_lr_ratio):
            warmup_steps = max(5, int(total_steps * warmup_steps_proportion))
            if step < warmup_steps:
                return 1.0 / warmup_steps * step
            return 1.0

        if self.lr_scheduler_type == "cosine":
            lr_scheduler_fn = warmup_then_cosine_anneal
        elif self.lr_scheduler_type == "linear":
            lr_scheduler_fn = warmup_then_linear_anneal
        elif self.lr_scheduler_type == "constant":
            lr_scheduler_fn = warmup_then_constant_anneal
        else:
            raise NotImplementedError(f"Unknown lr_scheduler_type {self.lr_scheduler_type}.")

        lr_lambda = functools.partial(
            lr_scheduler_fn,
            warmup_steps_proportion=self.warmup_steps_proportion,
            total_steps=spec.total_train_steps,
            min_lr_ratio=self.min_lr_ratio,
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        if self.gradient_checkpointing:
            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable(self.ckpt_attn, self.ckpt_mlp)

        module, *_ = deepspeed_initialize(
            model=module,
            optimizer=optimizer,
            config=ds_config,
            lr_scheduler=lr_scheduler,
            engine_type=self.engine_type,
            enable_async_p2p_communication=self.enable_async_p2p_communication,
            enable_async_instruction=self.enable_async_instruction,
            instruction_sync=self.instruction_sync,
        )

        if self.engine_type == "pipe":
            # log pipeline infos
            assert isinstance(module, DeepSpeedPipelineEngine)
            logger.info(f"PipelineEngine:: ddp rank = {torch.distributed.get_rank()}; "
                        f"pipe id = {module.stage_id}; dp id = {module.dp_id};")

        model.module = module
        return model


model_api.register_backend("ds_train", DeepspeedTrainBackend)
