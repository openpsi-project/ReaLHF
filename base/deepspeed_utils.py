from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from deepspeed.accelerator import get_accelerator
from deepspeed.runtime import zero
from deepspeed.runtime.config import DeepSpeedConfig
from deepspeed.runtime.engine import DeepSpeedEngine, DeepSpeedOptimizerCallable, DeepSpeedSchedulerCallable
import deepspeed
import torch
import torch.distributed

from impl.model.backend.pipe_engine import DeepSpeedPipelineEngine, StreamPipeEngine
import base.constants
import base.logging as logging

DEFAULT_TRAIN_MICRO_BATCH_SIZE_PER_GPU = 32  # A place-holder for inference.

logger = logging.getLogger("DeepSpeed Utils")


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
        "train_batch_size": torch.distributed.get_world_size(group=base.constants.data_parallel_group()) *
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
    num_pipeline_micro_batches: Optional[int] = None,
    num_inf_pipeline_mbs: Optional[int] = None,
    sequence_parallel: Optional[bool] = None,
) -> Tuple[DeepSpeedEngine, torch.optim.Optimizer, Any, Any]:
    """A simple wrapper around deepspeed.initialize."""
    if mpu is None:
        mpu = base.constants.grid()
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
        engine = DeepSpeedPipelineEngine(
            num_micro_batches=num_pipeline_micro_batches,
            num_inf_micro_batches=num_inf_pipeline_mbs,
            sequence_parallel=sequence_parallel,
            model=model,
            args=None,
            config=config,
            config_class=config_class,
            mpu=mpu,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            dist_init_required=False,
        )
        logger.info(f"Deepspeed Pipeline Engine initialze finished.")
        return_items = [engine, engine.optimizer, engine.training_dataloader, engine.lr_scheduler]
    elif engine_type == "stream_pipe":
        # mpu = model.mpu()
        config_class = DeepSpeedConfig(config, mpu)
        engine = StreamPipeEngine(
            model=model,
            args=None,
            config=config,
            config_class=config_class,
            mpu=mpu,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            dist_init_required=False,
        )
        return_items = [engine, engine.optimizer, engine.training_dataloader, engine.lr_scheduler]

    return tuple(return_items)
