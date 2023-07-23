from typing import Callable
import dataclasses
import functools
import math

import deepspeed
import torch

import api.model
import base.deepspeed_utils as deepspeed_utils


@dataclasses.dataclass
class DeepspeedTrainBackend(api.model.ModelBackend):
    optimizer_name: str
    optimizer_config: dict
    warmup_steps_proportion: float
    min_lr_ratio: float
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    offload: bool = False
    zero_stage: int = 2
    enable_hybrid_engine: bool = False
    inference_tp_size: int = 1
    release_inference_cache: bool = False
    pin_parameters: bool = True
    tp_gather_partition_size: int = 8
    max_out_tokens: int = 512

    def __post_init__(self):
        assert self.zero_stage == 2
        if self.inference_tp_size > 1 and self.zero_stage != 3:
            raise ValueError(f"Zero stage 3 must be used to do Tensor sharding in the hybrid engine.")

    def _initialize(self, model: api.model.Model, spec: api.model.FinetuneSpec):
        module = model.module
        weight_decay = self.optimizer_config.get('weight_decay', 0.0)
        if self.optimizer_name == 'adam':
            optimizer = deepspeed.ops.adam.FusedAdam(
                deepspeed_utils.get_optimizer_grouped_parameters(module, weight_decay),
                **self.optimizer_config)
        else:
            raise NotImplementedError(f"Unsupported optimizer: {self.optimizer_name}.")

        ds_config = deepspeed_utils.get_train_ds_config(
            offload=self.offload,
            stage=self.zero_stage,
            enable_hybrid_engine=self.enable_hybrid_engine,
            inference_tp_size=self.inference_tp_size,
            release_inference_cache=self.release_inference_cache,
            pin_parameters=self.pin_parameters,
            tp_gather_partition_size=self.tp_gather_partition_size,
            max_out_tokens=self.max_out_tokens,
        )

        ds_config['train_micro_batch_size_per_gpu'] = spec.batch_size_per_device
        ds_config['train_batch_size'] = spec.batch_size_per_device * torch.distributed.get_world_size(
        ) * self.gradient_accumulation_steps

        # TODO: implement other scheduler other than cosine
        def warmup_then_consine_anneal(step, warmup_steps_proportion, total_steps, min_lr_ratio):
            warmup_steps = max(5, int(total_steps * warmup_steps_proportion))
            consine_steps = total_steps - warmup_steps
            if step < warmup_steps:
                return 1.0 / warmup_steps * step
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1 + math.cos(
                (step - warmup_steps) / consine_steps * math.pi))

        lr_lambda = functools.partial(warmup_then_consine_anneal,
                                      warmup_steps_proportion=self.warmup_steps_proportion,
                                      total_steps=spec.total_train_steps,
                                      min_lr_ratio=self.min_lr_ratio)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        module, *_ = deepspeed.initialize(model=module,
                                          optimizer=optimizer,
                                          config=ds_config,
                                          lr_scheduler=lr_scheduler,
                                          dist_init_required=False)

        if self.gradient_checkpointing:
            module.gradient_checkpointing_enable()

        model.module = module
        return model


@dataclasses.dataclass
class DeepspeedInferenceBackend(api.model.ModelBackend):
    offload: bool = False
    zero_stage: int = 0

    def _initialize(self, model: api.model.Model, spec: api.model.FinetuneSpec):
        module = model.module
        ds_config = deepspeed_utils.get_eval_ds_config(
            offload=self.offload,
            stage=self.zero_stage,
        )
        module, *_ = deepspeed.initialize(model=module, config=ds_config)
        model.module = module
        return model


api.model.register_backend("ds_train", DeepspeedTrainBackend)
api.model.register_backend("ds_inference", DeepspeedInferenceBackend)