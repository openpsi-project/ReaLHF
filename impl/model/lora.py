from typing import List, Optional, Union
import logging
import math
import os

from torch import nn
import deepspeed
import deepspeed.compression.helper
import torch
import torch.nn.functional as F
import transformers

import api.config
import api.model
import api.utils

logger = logging.getLogger("LoRA")


class LinearLayer_LoRA(nn.Module):
    # an simple implementation of LoRA
    # for now only support Linear Layer
    def __init__(self, weight, lora_dim=0, lora_scaling=1, lora_dropout=0, bias=None):
        super(LinearLayer_LoRA, self).__init__()
        self.weight = weight
        self.bias = bias

        if lora_dim <= 0:
            raise ValueError("You are training to use LoRA, whose reduced dim should be larger than 1")

        try:
            # for zero stage 3
            rows, columns = weight.ds_shape
        except:
            rows, columns = weight.shape
        self.lora_right_weight = nn.Parameter(torch.zeros(
            columns, lora_dim))  # apply transpose so in forward we do not need to
        self.lora_left_weight = nn.Parameter(torch.zeros(lora_dim, rows))
        self.lora_scaling = lora_scaling / lora_dim

        if lora_dropout > 0:
            self.lora_dropout = nn.Dropout(lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        self.reset_parameters()
        # disable the original weight gradient
        self.weight.requires_grad = False
        # fuse LoRA to the original weight
        self.fuse_lora = False

    def eval(self):
        self.lora_dropout.eval()

    #   self.fuse_lora_weight()

    def train(self, mode=True):
        self.lora_dropout.train(mode)
        # self.unfuse_lora_weight()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_right_weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_left_weight)

    def fuse_lora_weight(self):
        if not self.fuse_lora:
            self.weight.data += self.lora_scaling * torch.matmul(self.lora_left_weight.t(),
                                                                 self.lora_right_weight.t())
        self.fuse_lora = True

    def unfuse_lora_weight(self):
        if self.fuse_lora:
            self.weight.data -= self.lora_scaling * torch.matmul(self.lora_left_weight.t(),
                                                                 self.lora_right_weight.t())
        self.fuse_lora = False

    def forward(self, input):
        if self.fuse_lora:
            return F.linear(input, self.weight, self.bias)
        else:
            return F.linear(input, self.weight, self.bias) + (
                self.lora_dropout(input) @ self.lora_right_weight @ self.lora_left_weight) * self.lora_scaling


# convert the linear layer to LoRA
def convert_linear_layer_to_lora(model,
                                 part_module_name,
                                 lora_dim=0,
                                 lora_scaling=1,
                                 lora_dropout=0,
                                 lora_exclude_module_names=None):
    if lora_exclude_module_names is None:
        lora_exclude_module_names = []

    replace_name = []
    for name, module in model.named_modules():
        if (isinstance(module, nn.Linear) and part_module_name in name
                and not any(x in name for x in lora_exclude_module_names)):
            replace_name.append(name)

    for name in replace_name:
        module = deepspeed.compression.helper.recursive_getattr(model, name)
        tmp = LinearLayer_LoRA(module.weight, lora_dim, lora_scaling, lora_dropout,
                               module.bias).to(module.weight.device).to(module.weight.dtype)
        deepspeed.compression.helper.recursive_setattr(model, name, tmp)
    return model


def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list if hasattr(p, 'ds_id')
        and p.ds_status == deepspeed.runtime.zero.partition_parameters.ZeroParamStatus.NOT_AVAILABLE
    ]


# convert the LoRA layer to linear layer
def convert_lora_to_linear_layer(model):
    repalce_name = []
    for name, module in model.named_modules():
        if isinstance(module, LinearLayer_LoRA):
            repalce_name.append(name)
    for name in repalce_name:
        module = deepspeed.compression.helper.recursive_getattr(model, name)
        zero_stage_3 = hasattr(module.weight, 'ds_id')
        with deepspeed.zero.GatheredParameters(_z3_params_to_fetch(
            [module.weight, module.bias, module.lora_left_weight, module.lora_right_weight]),
                                               modifier_rank=0,
                                               enabled=zero_stage_3):
            module.fuse_lora_weight()
    return model


def unfuse_lora_after_saving(model):
    repalce_name = []
    for name, module in model.named_modules():
        if isinstance(module, LinearLayer_LoRA):
            repalce_name.append(name)
    for name in repalce_name:
        module = deepspeed.compression.helper.recursive_getattr(model, name)
        zero_stage_3 = hasattr(module.weight, 'ds_id')
        with deepspeed.zero.GatheredParameters(_z3_params_to_fetch(
            [module.weight, module.bias, module.lora_left_weight, module.lora_right_weight]),
                                               modifier_rank=0,
                                               enabled=zero_stage_3):
            module.unfuse_lora_weight()
    return model


def only_optimize_lora_parameters(model: nn.Module, additional_module_names_to_opt: List[str]):
    # turn off the gradient of all the parameters except the LoRA parameters
    for name, param in model.named_parameters():
        requires_grad = "lora_right_weight" in name or "lora_left_weight" in name
        for x in additional_module_names_to_opt:
            requires_grad |= x in name
        param.requires_grad = requires_grad
    logger.info(
        f"Parameter names to be optimized: {list(n for n, p in model.named_parameters() if p.requires_grad)}."
    )
    return model


def lora_wrap_fn(cls_):

    def wrapped_cls(lora_dim,
                    lora_module_name,
                    lora_exclude_module_names=None,
                    additional_module_names_to_opt=None,
                    lora_scaling=1.0,
                    lora_dropout=0.0,
                    **kwargs):
        model: api.model.Model = cls_(**kwargs)

        if additional_module_names_to_opt is None:
            additional_module_names_to_opt = []
        elif isinstance(additional_module_names_to_opt, str):
            additional_module_names_to_opt = [additional_module_names_to_opt]
        elif not isinstance(additional_module_names_to_opt, list):
            raise RuntimeError(f"additional_module_names_to_opt should be a "
                               f"list of strings. {type(additional_module_names_to_opt)}")

        if lora_exclude_module_names is None:
            lora_exclude_module_names = []
        elif isinstance(lora_exclude_module_names, str):
            lora_exclude_module_names = [lora_exclude_module_names]
        elif not isinstance(lora_exclude_module_names, list):
            raise RuntimeError(f"lora_exclude_module_names should be a "
                               f"list of strings. {type(lora_exclude_module_names)}")

        model.module = convert_linear_layer_to_lora(model.module,
                                                    lora_module_name,
                                                    lora_dim=lora_dim,
                                                    lora_scaling=lora_scaling,
                                                    lora_dropout=lora_dropout,
                                                    lora_exclude_module_names=lora_exclude_module_names)
        model.module = only_optimize_lora_parameters(
            model.module, additional_module_names_to_opt=additional_module_names_to_opt)
        model.module = model.module.to(model.device)
        return model

    return wrapped_cls


existing_model_classes = api.model.ALL_MODEL_CLASSES.copy()
for k, cls_ in existing_model_classes.items():
    api.model.register_model(f"{k}_lora", lora_wrap_fn(cls_))


class LinearLayer_OptionalLoRA(nn.Module):

    def __init__(self,
                 weight: nn.Parameter,
                 lora_dim=0,
                 lora_scaling=1,
                 lora_dropout=0,
                 bias: Optional[nn.Parameter] = None):
        super().__init__()
        self.weight = weight
        self.bias = bias

        if lora_dim <= 0:
            raise ValueError("You are training to use LoRA, whose reduced dim should be larger than 1")

        rows, columns = weight.shape
        self.lora_right_weight = nn.Parameter(torch.zeros(
            columns, lora_dim))  # apply transpose so in forward we do not need to
        self.lora_left_weight = nn.Parameter(torch.zeros(lora_dim, rows))

        self.lora_scaling = lora_scaling / lora_dim

        if lora_dropout > 0:
            self.lora_dropout = nn.Dropout(lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        self.reset_parameters()
        self.weight.requires_grad = False

        self._lora_on = False

    def turn_on_lora(self):
        self._lora_on = True

    def turn_off_lora(self):
        self._lora_on = False

    def eval(self):
        self.lora_dropout.eval()

    def train(self, mode=True):
        self.lora_dropout.train(mode)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_right_weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_left_weight)

    def forward(self, input):
        o = F.linear(input, self.weight, self.bias)
        if not self._lora_on:
            return o
        return o + self.lora_scaling * (
            self.lora_dropout(input) @ self.lora_right_weight @ self.lora_left_weight)
