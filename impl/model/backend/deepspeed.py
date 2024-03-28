from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import dataclasses
import functools
import math

import deepspeed
import torch

from base.constants import data_parallel_world_size, model_parallel_world_size, pipe_parallel_world_size
from impl.model.backend.pipe_engine import DeepSpeedPipelineEngine, StreamPipeEngine
import api.model
import base.constants
import base.deepspeed_utils as deepspeed_utils
import base.logging as logging

logger = logging.getLogger("DeepSpeed Backend")


@dataclasses.dataclass
class DeepspeedTrainBackend(api.model.ModelBackend):
    optimizer_name: str = 'adam'
    optimizer_config: dict = dataclasses.field(
        default_factory=lambda: dict(lr=1e-5, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-5))
    lr_scheduler_type: str = 'cosine'
    warmup_steps_proportion: float = 0.0
    min_lr_ratio: float = 0.0  # will be used for linear and cosine schedule
    gradient_checkpointing: bool = False
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
    sequence_parallel: bool = False
    enable_async_p2p_communication: bool = False
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
        if base.constants.model_parallel_world_size() == 1 and self.sequence_parallel:
            logger.warning("Sequence parallel only works with tensor model parallelism, but currently "
                           f"model_parallel_world_size = {base.constants.model_parallel_world_size()}. ")
            self.sequence_parallel = False
        if self.engine_type == "pipe" or self.engine_type == "stream_pipe":
            assert self.zero_stage < 2
            assert self.enable_hybrid_engine is False

    def _initialize(self, model: api.model.Model, spec: api.model.FinetuneSpec):
        deepspeed.init_distributed(auto_mpi_discovery=False)
        module = model.module
        weight_decay = self.optimizer_config.get('weight_decay', 0.0)
        if self.optimizer_name == 'adam':
            if not self.offload_param and not self.offload_optimizer_state:
                optim_cls = deepspeed.ops.adam.FusedAdam
            else:
                optim_cls = deepspeed.ops.adam.DeepSpeedCPUAdam
            optimizer = optim_cls(deepspeed_utils.get_optimizer_grouped_parameters(module, weight_decay),
                                  **self.optimizer_config)
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

        ds_config = deepspeed_utils.get_train_ds_config(
            offload_param=self.offload_param,
            offload_optimizer_state=self.offload_optimizer_state,
            stage=self.zero_stage,
            enable_bf16=self.enable_bf16,
            enable_fp16=self.enable_fp16,
            hybrid_engine_args=hybrid_engine_args,
            **self.additional_ds_config,
        )

        ds_config['train_micro_batch_size_per_gpu'] = spec.batch_size_per_device
        ds_config['train_batch_size'] = spec.batch_size_per_device * data_parallel_world_size() \
                                        * model_parallel_world_size() * pipe_parallel_world_size()

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

        if self.lr_scheduler_type == 'cosine':
            lr_scheduler_fn = warmup_then_cosine_anneal
        elif self.lr_scheduler_type == 'linear':
            lr_scheduler_fn = warmup_then_linear_anneal
        elif self.lr_scheduler_type == 'constant':
            lr_scheduler_fn = warmup_then_constant_anneal
        else:
            raise NotImplementedError(f"Unknown lr_scheduler_type {self.lr_scheduler_type}.")

        lr_lambda = functools.partial(lr_scheduler_fn,
                                      warmup_steps_proportion=self.warmup_steps_proportion,
                                      total_steps=spec.total_train_steps,
                                      min_lr_ratio=self.min_lr_ratio)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        if self.gradient_checkpointing:
            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable(self.ckpt_attn, self.ckpt_mlp)

        module, *_ = deepspeed_utils.deepspeed_initialize(
            model=module,
            optimizer=optimizer,
            config=ds_config,
            lr_scheduler=lr_scheduler,
            engine_type=self.engine_type,
            sequence_parallel=self.sequence_parallel,
            enable_async_p2p_communication=self.enable_async_p2p_communication,
            enable_async_instruction=self.enable_async_instruction,
            instruction_sync=self.instruction_sync,
        )

        if self.engine_type == "pipe" or self.engine_type == "stream_pipe":
            # log pipeline infos
            assert isinstance(module, DeepSpeedPipelineEngine) or isinstance(module, StreamPipeEngine)
            logger.info(f"PipelineEngine:: ddp rank = {torch.distributed.get_rank()}; "
                        f"pipe id = {module.stage_id}; dp id = {module.dp_id};")

        model.module = module
        return model


@dataclasses.dataclass
class DeepspeedInferenceBackend(api.model.ModelBackend):
    offload: bool = False
    zero_stage: int = 0
    enable_fp16: bool = True
    enable_bf16: bool = False
    additional_ds_config: Dict = dataclasses.field(default_factory=dict)
    # pipeline inference
    engine_type: str = "deepspeed"
    sequence_parallel: bool = False
    enable_async_p2p_communication: bool = False

    def __post_init__(self):
        if base.constants.model_parallel_world_size() == 1 and self.sequence_parallel:
            logger.warning("Sequence parallel only works with tensor model parallelism, but currently "
                           f"model_parallel_world_size = {base.constants.model_parallel_world_size()}. ")
            self.sequence_parallel = False

    def _initialize(self, model: api.model.Model, spec: api.model.FinetuneSpec):
        deepspeed.init_distributed(auto_mpi_discovery=False)
        module = model.module
        ds_config = deepspeed_utils.get_eval_ds_config(offload=self.offload,
                                                       stage=self.zero_stage,
                                                       enable_bf16=self.enable_bf16,
                                                       enable_fp16=self.enable_fp16,
                                                       **self.additional_ds_config)
        module, *_ = deepspeed_utils.deepspeed_initialize(
            model=module,
            config=ds_config,
            engine_type=self.engine_type,
            sequence_parallel=self.sequence_parallel,
            enable_async_p2p_communication=self.enable_async_p2p_communication,
        )
        model.module = module
        return model


api.model.register_backend("ds_train", DeepspeedTrainBackend)
api.model.register_backend("ds_inference", DeepspeedInferenceBackend)
