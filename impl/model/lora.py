from typing import List, Optional, Union
import logging
import math

from torch import nn
import deepspeed
import deepspeed.compression.helper
import torch
import torch.nn.functional as F
import transformers
import os

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


class LoRAContrastiveRewardModel(nn.Module):

    def __init__(
        self,
        base_model_name_or_path: str,
        load_state_dict: bool,
        disable_dropout: bool,
        lora_dim: int,
        embed_dim: int,
        lora_module_name: str,
        lora_scaling: float = 1.0,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        base_model = api.utils.create_hf_nn(
            transformers.AutoModel,
            model_name_or_path=base_model_name_or_path,
            disable_dropout=disable_dropout,
            init_from_scratch=load_state_dict,
        )
        self.config = base_model.config
        # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
        self.config.n_embd = self.config.hidden_size if hasattr(self.config,
                                                                "hidden_size") else self.config.n_embd
        # base_model is created by `AutoModel` instead of `AutoModelForCausalLM`
        # its forward function outputs the last hidden state instead of logits
        self.rwtranrsformer = base_model

        self.embed_dim = embed_dim
        self.prompt_linear_proj = nn.Linear(self.config.n_embd, embed_dim, bias=False)
        self.response_linear_proj = nn.Linear(self.config.n_embd, embed_dim, bias=False)
        self.temperature = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=True)

        model = self.rwtranrsformer
        replace_name = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and lora_module_name in name:
                replace_name.append(name)
        self._lora_module_names = replace_name
        for name in replace_name:
            module = deepspeed.compression.helper.recursive_getattr(model, name)
            tmp = LinearLayer_OptionalLoRA(module.weight, lora_dim, lora_scaling, lora_dropout,
                                           module.bias).to(module.weight.device).to(module.weight.dtype)
            deepspeed.compression.helper.recursive_setattr(model, name, tmp)

        if load_state_dict:
            model_ckpt_path = os.path.join(base_model_name_or_path, 'pytorch_model.bin')
            assert os.path.exists(model_ckpt_path), f"Cannot find model checkpoint at {model_ckpt_path}"
            model_ckpt = torch.load(model_ckpt_path, map_location='cpu')
            self.load_state_dict(model_ckpt)

    def turn_on_lora(self):
        for name in self._lora_module_names:
            deepspeed.compression.helper.recursive_getattr(self.rwtranrsformer, name).turn_on_lora()

    def turn_off_lora(self):
        for name in self._lora_module_names:
            deepspeed.compression.helper.recursive_getattr(self.rwtranrsformer, name).turn_off_lora()

    def gradient_checkpointing_enable(self):
        self.rwtranrsformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtranrsformer.gradient_checkpointing_disable()

    def forward(
        self,
        prompts=None,
        responses=None,
        prompt_attention_mask=None,
        response_attention_mask=None,
        eos_indices=None,
        use_cache=False,
    ):
        self.turn_off_lora()

        with torch.inference_mode():
            transformer_outputs = self.rwtranrsformer(prompts,
                                                      attention_mask=prompt_attention_mask,
                                                      use_cache=use_cache)
            prompt_hidden_states: torch.FloatTensor = transformer_outputs[0][:, -2:-1]  # left pad, [bs, 1, D]

        prompt_hidden_states = self.prompt_linear_proj(prompt_hidden_states.clone()).float()

        self.turn_on_lora()

        assert len(responses.shape) == 2 or len(responses.shape) == 3, len(responses.shape)
        has_multiple_responses = len(responses.shape) == 3
        if has_multiple_responses:
            bs, c_dim = shape = responses.shape[:2]
            responses = responses.flatten(end_dim=1)
            response_attention_mask = response_attention_mask.flatten(end_dim=1)
            eos_indices = eos_indices.flatten(end_dim=1)

        code_outputs = self.rwtranrsformer(responses,
                                           attention_mask=response_attention_mask,
                                           use_cache=use_cache)
        code_hidden_states: torch.FloatTensor = code_outputs[0]
        eos_indices = eos_indices.view(-1, 1, 1).repeat(1, 1, code_hidden_states.shape[-1])
        code_hidden_states = self.response_linear_proj(code_hidden_states.gather(
            -2, eos_indices)).float()  # [bs * c_dim, 1, D]

        prompt_hidden_states = torch.nn.functional.normalize(prompt_hidden_states, dim=-1)
        code_hidden_states = torch.nn.functional.normalize(code_hidden_states, dim=-1)

        if has_multiple_responses:
            code_hidden_states = code_hidden_states.view(*shape, *code_hidden_states.shape[1:])
            scores = torch.matmul(prompt_hidden_states.unsqueeze(1),
                                  code_hidden_states.transpose(-1, -2)).view(bs, c_dim)
        else:
            assert code_hidden_states.shape == prompt_hidden_states.shape, (code_hidden_states.shape,
                                                                            prompt_hidden_states.shape)
            scores = torch.matmul(prompt_hidden_states, code_hidden_states.transpose(-1, -2)).view(-1)

        scores = scores * torch.exp(self.temperature)

        return scores


def create_contrastive_reward_model(
    name: str,
    model_name_or_path: str,
    load_state_dict: bool,
    disable_dropout: bool,
    lora_dim: int,
    embed_dim: int,
    lora_module_name: str,
    device: Union[str, torch.device],
    lora_scaling: float = 1.0,
    lora_dropout: float = 0.0,
):
    module = LoRAContrastiveRewardModel(
        base_model_name_or_path=model_name_or_path,
        load_state_dict=load_state_dict,
        disable_dropout=disable_dropout,
        lora_dim=lora_dim,
        embed_dim=embed_dim,
        lora_scaling=lora_scaling,
        lora_dropout=lora_dropout,
        lora_module_name=lora_module_name,
    )
    tokenizer = api.utils.load_hf_tokenizer(model_name_or_path)
    return api.model.Model(name, module, tokenizer, device)


api.model.register_model("lora_contrastive_reward", create_contrastive_reward_model)