from typing import *

from reallm.api.core.config import ModelBackend
from reallm.api.quickstart.model import get_real_model_config, ModelTrainEvalConfig, ParallelismConfig
from reallm.base.topology import PipeModelDataParallelTopology


def get_topo(
    parallel: ParallelismConfig,
    gradient_checkpointing: bool,
    max_prompt_len: Optional[int] = None,
) -> PipeModelDataParallelTopology:
    return PipeModelDataParallelTopology(
        num_mp=parallel.model_parallel_size,
        num_pp=parallel.pipeline_parallel_size,
        num_dp=parallel.data_parallel_size,
        sequence_parallel=parallel.use_sequence_parallel,
        gradient_checkpointing=gradient_checkpointing,
        max_prompt_len=max_prompt_len,
    )


def get_world_size(parallel: ParallelismConfig) -> int:
    return parallel.model_parallel_size * parallel.pipeline_parallel_size * parallel.data_parallel_size


def make_train_backend_config(cfg: ModelTrainEvalConfig):
    parallel = cfg.parallel
    if parallel.pipeline_parallel_size > 1:
        engine_type = "pipe"
    else:
        engine_type = "deepspeed"
    return ModelBackend(
        "ds_train",
        args=dict(
            optimizer_name="adam",
            optimizer_config=dict(
                lr=cfg.optimizer.lr,
                weight_decay=cfg.optimizer.weight_decay,
                eps=cfg.optimizer.eps,
                betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
            ),
            lr_scheduler_type=cfg.optimizer.lr_scheduler_type,
            warmup_steps_proportion=cfg.optimizer.warmup_steps_proportion,
            min_lr_ratio=cfg.optimizer.min_lr_ratio,
            zero_stage=(cfg.zero_stage if parallel.pipeline_parallel_size == 1 else min(cfg.zero_stage, 1)),
            engine_type=engine_type,
            offload_optimizer_state=cfg.optimizer.offload,
            offload_param=cfg.offload,
            enable_bf16=cfg.enable_bf16,
            enable_fp16=cfg.enable_fp16,
        ),
    )


def make_inf_backend_config(cfg: ModelTrainEvalConfig):
    if cfg.parallel.pipeline_parallel_size > 1:
        return ModelBackend("pipe_inference")
    else:
        return ModelBackend("null")


def make_model_config(cfg: ModelTrainEvalConfig):
    return get_real_model_config(
        model_path=cfg.path,
        hf_model_family=cfg.type._class,
        is_critic=cfg.type.is_critic,
        init_critic_from_actor=False,
        dtype="bf16" if cfg.enable_bf16 else "fp16",
        lora=cfg.lora,
    )
