from typing import Any, Dict, List, Literal, Optional, Union
import dataclasses
import gc

from torch import nn
import bitsandbytes as bnb
import deepspeed
import deepspeed.compression.helper
import torch
import torch.nn.functional as F

from impl.model.utils.save_load import load_from_disk
import api.config.config_system
import api.huggingface
import api.model
import base.logging as logging

logger = logging.getLogger("LoRA")


@dataclasses.dataclass
class LoRA8bitConfig:
    trainable: bool
    threshold: float


class LinearLoRA(nn.Module):
    """Taken from DeepSpeedChat.
    """

    def __init__(
        self,
        linear: Union[nn.Linear, bnb.nn.Linear8bitLt],
        lora_dim: int = 0,
        lora_scaling: float = 1,
        lora_dropout: float = 0,
        bnb_8bit_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(LinearLoRA, self).__init__()

        bnb_8bit_config = LoRA8bitConfig(**bnb_8bit_kwargs) if bnb_8bit_kwargs is not None else None

        # sanity checks
        if lora_dim <= 0:
            raise ValueError("You are training to use LoRA, whose reduced dim should be larger than 1")

        self.linear = linear

        rows, columns = self.linear.weight.shape
        dtype = torch.float16

        self.use_bnb_8bit = (bnb_8bit_config is not None)
        if bnb_8bit_config is not None:
            self.lora_right = bnb.nn.Linear8bitLt(
                lora_dim,
                rows,
                bias=False,
                has_fp16_weights=bnb_8bit_config.trainable,
                threshold=bnb_8bit_config.threshold,
                device=self.linear.weight.device,
            ).to(dtype=dtype)
            self.lora_left = bnb.nn.Linear8bitLt(
                columns,
                lora_dim,
                bias=False,
                has_fp16_weights=bnb_8bit_config.trainable,
                threshold=bnb_8bit_config.threshold,
                device=self.linear.weight.device,
            ).to(dtype=dtype)
        else:
            self.lora_right = nn.Linear(lora_dim, rows, bias=False).to(dtype=dtype,
                                                                       device=self.linear.weight.device)
            self.lora_left = nn.Linear(columns, lora_dim, bias=False).to(dtype=dtype,
                                                                         device=self.linear.weight.device)

        self.lora_scaling = lora_scaling / lora_dim

        if lora_dropout > 0:
            self.lora_dropout = nn.Dropout(lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        # disable the original weight gradient
        self.linear.weight.requires_grad = False
        # fuse LoRA to the original weight
        self.fuse_lora = False

        self.squashed = False

    def eval(self):
        if not self.squashed:
            self.lora_dropout.eval()

    def train(self, mode=True):
        if not self.squashed:
            self.lora_dropout.train(mode)

    def fuse_lora_weight(self):
        if not self.squashed and not self.fuse_lora:
            self.linear.weight.data += self.lora_scaling * torch.matmul(self.lora_right.weight,
                                                                        self.lora_left.weight)
        self.fuse_lora = True

    def unfuse_lora_weight(self):
        if not self.squashed and self.fuse_lora:
            self.linear.weight.data -= self.lora_scaling * torch.matmul(self.lora_right.weight,
                                                                        self.lora_left.weight)
        self.fuse_lora = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        if self.squashed or self.fuse_lora:
            return y
        x = self.lora_dropout(x).to(dtype=self.lora_left.weight.dtype)
        x = self.lora_right(self.lora_left(x)) * self.lora_scaling
        return y + x.to(y)


def is_lora_model(model: nn.Module) -> bool:
    return len([name for name, module in model.named_modules() if isinstance(module, LinearLoRA)]) > 0


def convert_linear_layer_to_lora(model: nn.Module, lora_keys_to_replace: List[str], lora_module_kwargs: dict,
                                 lora_exclude_module_names: List) -> nn.Module:
    if not isinstance(lora_keys_to_replace, list):
        lora_keys_to_replace = [lora_keys_to_replace]
    replace_name = []
    for lora_key_to_replace in lora_keys_to_replace:
        for name, module in model.named_modules():
            if lora_key_to_replace not in name:
                continue
            if any(x in name for x in lora_exclude_module_names):
                continue
            if isinstance(module, (bnb.nn.Linear8bitLt, nn.Linear)):
                replace_name.append(name)
            elif 'linear' in module.__class__.__name__.lower():
                logger.warning(
                    f"Found a linear-like layer {name} that is not `nn.Linear` or `bnb.nn.Linear8bitLt`. "
                    f"Class {module.__class__.__name__}. This layer will not be converted to LoRA.")

    for name in replace_name:
        module: nn.Linear = deepspeed.compression.helper.recursive_getattr(model, name)
        tmp = LinearLoRA(module, **lora_module_kwargs)
        deepspeed.compression.helper.recursive_setattr(model, name, tmp)
    return model


def delete_all_lora_layers(model: nn.Module) -> nn.Module:
    for name in [name for name, module in model.named_modules() if isinstance(module, LinearLoRA)]:
        module: LinearLoRA = deepspeed.compression.helper.recursive_getattr(model, name)
        deepspeed.compression.helper.recursive_setattr(model, name, module.linear)
    return model


def fuse_all_lora_layers(model: nn.Module) -> nn.Module:
    for name in [name for name, module in model.named_modules() if isinstance(module, LinearLoRA)]:
        module: LinearLoRA = deepspeed.compression.helper.recursive_getattr(model, name)
        module.fuse_lora_weight()
    return model


def unfuse_all_lora_layers(model: nn.Module) -> nn.Module:
    for name in [name for name, module in model.named_modules() if isinstance(module, LinearLoRA)]:
        module: LinearLoRA = deepspeed.compression.helper.recursive_getattr(model, name)
        module.unfuse_lora_weight()
    return model


def only_optimize_lora_parameters(model: nn.Module, additional_module_names_to_opt: List[str]) -> nn.Module:
    for name, param in model.named_parameters():
        requires_grad = "lora_right" in name or "lora_left" in name
        for x in additional_module_names_to_opt:
            requires_grad |= x in name
        param.requires_grad = requires_grad
    logger.debug(f"Parameter names to be optimized: "
                 f"{list(n for n, p in model.named_parameters() if p.requires_grad)}.")
    return model


def get_lora_state_dict(model: nn.Module) -> List[Dict[str, torch.Tensor]]:
    lora_sds = []
    for name in [name for name, module in model.named_modules() if isinstance(module, LinearLoRA)]:
        module: LinearLoRA = deepspeed.compression.helper.recursive_getattr(model, name)
        lora_sds.append(module.state_dict())
    return lora_sds


def lora_wrap_fn(
    lora_module_kwargs: dict,
    lora_keys_to_replace: List[str],
    lora_exclude_module_names: Optional[List[str]] = None,
    additional_module_names_to_opt: Optional[List[str]] = None,
    load_lora_path: Optional[str] = None,
    lora_op_after_creation: Optional[Literal['squash', 'fuse']] = None,
):

    if additional_module_names_to_opt is None:
        additional_module_names_to_opt = []
    if lora_exclude_module_names is None:
        lora_exclude_module_names = []

    def lora_wrap_fn_(model: api.model.Model) -> api.model.Model:
        model.module = convert_linear_layer_to_lora(
            model.module,
            lora_keys_to_replace,
            lora_module_kwargs=lora_module_kwargs,
            lora_exclude_module_names=lora_exclude_module_names,
        )
        model.module = only_optimize_lora_parameters(model.module, additional_module_names_to_opt)

        if load_lora_path is not None:
            logger.info(f"Loading LoRA from {load_lora_path}")
            lora_sds = load_from_disk(load_lora_path)
            lora_names = [
                name for name, module in model.module.named_modules() if isinstance(module, LinearLoRA)
            ]
            assert len(lora_names) == len(lora_sds) > 0, (len(lora_sds), len(lora_names))
            for name, sd in zip(lora_names, lora_sds):
                m: LinearLoRA = deepspeed.compression.helper.recursive_getattr(model.module, name)
                m.to(model.device)
                m.load_state_dict(sd)

        if lora_op_after_creation is None:
            pass
        elif lora_op_after_creation == 'squash':
            model.module = delete_all_lora_layers(fuse_all_lora_layers(model.module))
        elif lora_op_after_creation == 'fuse':
            model.module = fuse_all_lora_layers(model.module)
        else:
            raise NotImplementedError(f"Unknown lora_op_after_creation: {lora_op_after_creation}")

        return model

    return lora_wrap_fn_


api.model.register_wrapper("lora", lora_wrap_fn)