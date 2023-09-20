from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import dataclasses
import functools
import logging
import math

import deepspeed
import torch

import api.model
import base.deepspeed_utils as deepspeed_utils

logger = logging.getLogger("DeepSpeed Backend")


@dataclasses.dataclass
class DeepspeedTrainBackend(api.model.ModelBackend):
    optimizer_name: str = 'adam'
    optimizer_config: dict = dataclasses.field(
        default_factory=lambda: dict(lr=1e-5, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-5))
    lr_scheduler_type: str = 'cosine'
    warmup_steps_proportion: float = 0.0
    min_lr_ratio: float = 0.0  # will be used for linear and cosine schedule
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    offload_param: bool = False
    offload_optimizer_state: bool = False
    enable_fp16: bool = True
    zero_stage: int = 2
    additional_ds_config: Dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        assert self.zero_stage == 2

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

        ds_config = deepspeed_utils.get_train_ds_config(
            offload_param=self.offload_param,
            offload_optimizer_state=self.offload_optimizer_state,
            stage=self.zero_stage,
            enable_fp16=self.enable_fp16,
            **self.additional_ds_config,
        )

        ds_config['train_micro_batch_size_per_gpu'] = spec.batch_size_per_device
        ds_config['train_batch_size'] = spec.batch_size_per_device * torch.distributed.get_world_size(
        ) * self.gradient_accumulation_steps

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

        module, *_ = deepspeed_utils.deepspeed_initialize(
            model=module,
            optimizer=optimizer,
            config=ds_config,
            lr_scheduler=lr_scheduler,
        )

        if self.gradient_checkpointing:
            module.gradient_checkpointing_enable()

        model.module = module
        return model


@dataclasses.dataclass
class DeepspeedInferenceBackend(api.model.ModelBackend):
    offload: bool = False
    zero_stage: int = 0
    enable_fp16: bool = True
    additional_ds_config: Dict = dataclasses.field(default_factory=dict)

    def _initialize(self, model: api.model.Model, spec: api.model.FinetuneSpec):
        deepspeed.init_distributed(auto_mpi_discovery=False)
        module = model.module
        ds_config = deepspeed_utils.get_eval_ds_config(offload=self.offload,
                                                       stage=self.zero_stage,
                                                       enable_fp16=self.enable_fp16,
                                                       **self.additional_ds_config)
        module, *_ = deepspeed_utils.deepspeed_initialize(model=module, config=ds_config)
        model.module = module
        return model


api.model.register_backend("ds_train", DeepspeedTrainBackend)
api.model.register_backend("ds_inference", DeepspeedInferenceBackend)
