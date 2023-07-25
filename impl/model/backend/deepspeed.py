from typing import Callable
import dataclasses
import functools
import logging
import math

import deepspeed
import torch

import api.model

logger = logging.getLogger("DeepSpeed Backend")

from typing import List

import torch

DEFAULT_TRAIN_MICRO_BATCH_SIZE_PER_GPU = 32  # A place-holder for inference.


def get_train_ds_config(offload=False,
                        stage=2,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=512):

    # TODO: more versatile config
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False
    }
    return {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "fp16": {
            "enabled": True,
            "loss_scale_window": 100
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        }
    }


def get_eval_ds_config(offload=False, stage=0):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device
        },
        "memory_efficient_linear": False
    }
    return {
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "train_micro_batch_size_per_gpu": DEFAULT_TRAIN_MICRO_BATCH_SIZE_PER_GPU,
        "train_batch_size": torch.distributed.get_world_size() * DEFAULT_TRAIN_MICRO_BATCH_SIZE_PER_GPU,
        "fp16": {
            "enabled": True
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False
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
        deepspeed.init_distributed(auto_mpi_discovery=False)
        module = model.module
        weight_decay = self.optimizer_config.get('weight_decay', 0.0)
        if self.optimizer_name == 'adam':
            optimizer = deepspeed.ops.adam.FusedAdam(get_optimizer_grouped_parameters(module, weight_decay),
                                                     **self.optimizer_config)
        else:
            raise NotImplementedError(f"Unsupported optimizer: {self.optimizer_name}.")

        ds_config = get_train_ds_config(
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

        import os
        import socket
        logger.info(
            f">>>>>> {model.device}, {socket.gethostname()}, {os.environ['CUDA_VISIBLE_DEVICES']}, {os.environ['LOCAL_RANK']}"
        )
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
        deepspeed.init_distributed(auto_mpi_discovery=False)
        module = model.module
        ds_config = get_eval_ds_config(
            offload=self.offload,
            stage=self.zero_stage,
        )
        module, *_ = deepspeed.initialize(model=module, config=ds_config)
        model.module = module
        return model


api.model.register_backend("ds_train", DeepspeedTrainBackend)
api.model.register_backend("ds_inference", DeepspeedInferenceBackend)