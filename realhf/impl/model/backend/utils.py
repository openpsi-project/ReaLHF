import dataclasses
import math
from typing import *

import torch
import torch.distributed as dist

try:
    from megatron.core.distributed.distributed_data_parallel import (
        DistributedDataParallel as MegatronDDP,
    )
    from megatron.core.optimizer.clip_grads import clip_grad_norm_fp32, count_zeros_fp32
    from megatron.core.optimizer.distrib_optimizer import (
        DistributedOptimizer as MegatronDistOptim,
    )
    from megatron.core.optimizer.distrib_optimizer import (
        MixedPrecisionOptimizer as MegatronMixedPrecOptim,
    )
except (ModuleNotFoundError, ImportError):
    # importing megatron.core in CPU container will fail due to the requirement of apex
    # Here class types must be defined for type hinting
    class MegatronDDP:
        pass

    class MegatronDistOptim:
        pass


from realhf.base import constants, logging

if TYPE_CHECKING:
    from realhf.impl.model.nn.real_llm_api import ReaLModel, ReaLModelBlock

logger = logging.getLogger("Model Backend Utils")


@dataclasses.dataclass
class MegatronEngine:
    ddp: MegatronDDP
    optim: MegatronDistOptim
    lr_scheduler: Any

    def zero_grad(self, set_to_none=True):
        self.ddp.zero_grad_buffer()
        self.optim.zero_grad(set_to_none=set_to_none)


# Adopted from Megatron-LM/megatron/training/optimizer_param_scheduler.py
class OptimizerParamScheduler(object):
    """Anneals learning rate and weight decay.

    Adopted from Megatron-LM. This class is not included in megatron.core,
    so we have to copy-paste it here.
    """

    def __init__(
        self,
        optimizer,
        init_lr,
        max_lr,
        min_lr,
        lr_warmup_steps,
        lr_decay_steps,
        lr_decay_style,
        start_wd,
        end_wd,
        wd_incr_steps,
        wd_incr_style,
        use_checkpoint_opt_param_scheduler=True,
        override_opt_param_scheduler=False,
    ):

        # Class values.
        self.optimizer = optimizer

        self.init_lr = init_lr
        self.max_lr = float(max_lr)
        self.min_lr = min_lr
        assert self.min_lr >= 0.0
        assert self.max_lr >= self.min_lr
        assert self.init_lr <= self.max_lr

        self.lr_warmup_steps = lr_warmup_steps
        self.num_steps = 0
        self.lr_decay_steps = lr_decay_steps
        assert self.lr_decay_steps > 0
        assert self.lr_warmup_steps < self.lr_decay_steps

        self.lr_decay_style = lr_decay_style

        self.start_wd = start_wd
        self.end_wd = end_wd
        assert self.start_wd >= 0.0
        assert self.end_wd >= self.start_wd
        self.wd_incr_steps = wd_incr_steps
        self.wd_incr_style = wd_incr_style

        self.override_opt_param_scheduler = override_opt_param_scheduler
        self.use_checkpoint_opt_param_scheduler = use_checkpoint_opt_param_scheduler
        if self.override_opt_param_scheduler:
            assert not self.use_checkpoint_opt_param_scheduler, (
                "both override and " "use-checkpoint are set."
            )

        # Set the learning rate
        self.step(0)
        self.log_rank_0("> learning rate decay style: {}".format(self.lr_decay_style))

    def log_rank_0(self, msg):
        if constants.parallelism_rank() == 0:
            logger.info(msg)

    def get_wd(self):
        """Weight decay incr functions"""
        if self.num_steps > self.wd_incr_steps:
            return self.end_wd

        if self.wd_incr_style == "constant":
            assert self.start_wd == self.end_wd
            return self.end_wd

        incr_ratio = float(self.num_steps) / float(self.wd_incr_steps)
        assert incr_ratio >= 0.0
        assert incr_ratio <= 1.0
        delta_wd = self.end_wd - self.start_wd

        if self.wd_incr_style == "linear":
            coeff = incr_ratio
        elif self.wd_incr_style == "cosine":
            coeff = 0.5 * (math.cos(math.pi * (1 - incr_ratio)) + 1.0)
        else:
            raise Exception(
                "{} weight decay increment style is not supported.".format(
                    self.wd_incr_style
                )
            )

        return self.start_wd + coeff * delta_wd

    def get_lr(self, param_group):
        """Learning rate decay functions from:
        https://openreview.net/pdf?id=BJYwwY9ll pg. 4"""

        max_lr = param_group.get("max_lr", self.max_lr)
        min_lr = param_group.get("min_lr", self.min_lr)

        # Use linear warmup for the initial part.
        if self.lr_warmup_steps > 0 and self.num_steps <= self.lr_warmup_steps:
            return self.init_lr + (
                (max_lr - self.init_lr)
                * float(self.num_steps)
                / float(self.lr_warmup_steps)
            )

        # If the learning rate is constant, just return the initial value.
        if self.lr_decay_style == "constant":
            return max_lr

        # For any steps larger than `self.lr_decay_steps`, use `min_lr`.
        if self.num_steps > self.lr_decay_steps:
            return min_lr

        # If we are done with the warmup period, use the decay style.
        if self.lr_decay_style == "inverse-square-root":
            warmup_steps = max(self.lr_warmup_steps, 1)
            num_steps = max(self.num_steps, 1)
            lr = max_lr * warmup_steps**0.5 / (num_steps**0.5)
            return max(min_lr, lr)

        num_steps_ = self.num_steps - self.lr_warmup_steps
        decay_steps_ = self.lr_decay_steps - self.lr_warmup_steps
        decay_ratio = float(num_steps_) / float(decay_steps_)
        assert decay_ratio >= 0.0
        assert decay_ratio <= 1.0
        delta_lr = max_lr - min_lr

        if self.lr_decay_style == "linear":
            coeff = 1.0 - decay_ratio
        elif self.lr_decay_style == "cosine":
            coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        else:
            raise Exception(
                "{} decay style is not supported.".format(self.lr_decay_style)
            )

        return min_lr + coeff * delta_lr

    def step_absolute(self, num_steps):
        """Set lr for all parameters groups."""
        if num_steps is None:
            self.num_steps += 1
        else:
            self.num_steps = num_steps
        new_wd = self.get_wd()
        for param_group in self.optimizer.param_groups:
            new_lr = self.get_lr(param_group)
            param_group["lr"] = new_lr * param_group.get("lr_mult", 1.0)
            param_group["weight_decay"] = new_wd * param_group.get("wd_mult", 1.0)

    def step(self, increment):
        """Set lr for all parameters groups."""
        self.num_steps += increment
        new_wd = self.get_wd()
        for param_group in self.optimizer.param_groups:
            new_lr = self.get_lr(param_group)
            param_group["lr"] = new_lr * param_group.get("lr_mult", 1.0)
            param_group["weight_decay"] = new_wd * param_group.get("wd_mult", 1.0)

    def state_dict(self):
        state_dict = {
            "max_lr": self.max_lr,
            "lr_warmup_steps": self.lr_warmup_steps,
            "num_steps": self.num_steps,
            "lr_decay_style": self.lr_decay_style,
            "lr_decay_steps": self.lr_decay_steps,
            "min_lr": self.min_lr,
            "start_wd": self.start_wd,
            "end_wd": self.end_wd,
            "wd_incr_style": self.wd_incr_style,
            "wd_incr_steps": self.wd_incr_steps,
        }
        return state_dict

    def _check_and_set(self, cls_value, sd_value, name):
        """Auxiliary function for checking the values in the checkpoint and
        setting them."""
        if self.override_opt_param_scheduler:
            self.log_rank_0(" > overriding {} value to {}".format(name, cls_value))
            return cls_value

        if not self.use_checkpoint_opt_param_scheduler:
            assert cls_value == sd_value, (
                f"OptimizerParamScheduler: class input value {cls_value} and checkpoint"
                f"value {sd_value} for {name} do not match"
            )
        self.log_rank_0(" > using checkpoint value {} for {}".format(sd_value, name))
        return sd_value

    def load_state_dict(self, sd):

        if "start_lr" in sd:
            max_lr_ = sd["start_lr"]
        else:
            max_lr_ = sd["max_lr"]
        self.max_lr = self._check_and_set(self.max_lr, max_lr_, "learning rate")

        self.min_lr = self._check_and_set(
            self.min_lr, sd["min_lr"], "minimum learning rate"
        )

        if "warmup_iter" in sd:
            lr_warmup_steps_ = sd["warmup_iter"]
        elif "warmup_steps" in sd:
            lr_warmup_steps_ = sd["warmup_steps"]
        else:
            lr_warmup_steps_ = sd["lr_warmup_steps"]
        self.lr_warmup_steps = self._check_and_set(
            self.lr_warmup_steps, lr_warmup_steps_, "warmup iterations"
        )

        if "end_iter" in sd:
            lr_decay_steps_ = sd["end_iter"]
        elif "decay_steps" in sd:
            lr_decay_steps_ = sd["decay_steps"]
        else:
            lr_decay_steps_ = sd["lr_decay_steps"]
        self.lr_decay_steps = self._check_and_set(
            self.lr_decay_steps, lr_decay_steps_, "total number of iterations"
        )

        if "decay_style" in sd:
            lr_decay_style_ = sd["decay_style"]
        else:
            lr_decay_style_ = sd["lr_decay_style"]
        self.lr_decay_style = self._check_and_set(
            self.lr_decay_style, lr_decay_style_, "learning rate decay style"
        )

        if "num_iters" in sd:
            num_steps = sd["num_iters"]
        else:
            num_steps = sd["num_steps"]
        self.step(increment=num_steps)

        if "start_wd" in sd:
            self.start_wd = self._check_and_set(
                self.start_wd, sd["start_wd"], "start weight decay"
            )
            self.end_wd = self._check_and_set(
                self.end_wd, sd["end_wd"], "end weight decay"
            )
            self.wd_incr_steps = self._check_and_set(
                self.wd_incr_steps,
                sd["wd_incr_steps"],
                "total number of weight decay iterations",
            )
            self.wd_incr_style = self._check_and_set(
                self.wd_incr_style,
                sd["wd_incr_style"],
                "weight decay incr style",
            )


@torch.no_grad()
def _step_megatron_distrib_optimizer_internal(optim: MegatronDistOptim):
    # NOTE: patching this function to use the correct model parallel group

    optim._copy_model_grads_to_main_grads()

    # Do unscale, check for inf, and update grad scaler only for
    # the case that grad scaler is provided.
    if optim.grad_scaler:

        def _unscale_main_grads_and_check_for_nan(optim: MegatronDistOptim):

            # Collect main grads.
            main_grads = optim._collect_main_grad_data_for_unscaling()

            # Reset found inf.
            optim.found_inf.fill_(0.0)

            # Unscale and set found inf/nan
            torch._amp_foreach_non_finite_check_and_unscale_(
                main_grads, optim.found_inf, optim.grad_scaler.inv_scale
            )

            # Update across all model parallel instances.
            dist.all_reduce(
                optim.found_inf,
                op=dist.ReduceOp.MAX,
                group=constants.parallelism_group(),
            )

            # Check for nan.
            found_inf_flag = optim.found_inf.item() > 0

            return found_inf_flag

        # Unscale and check for inf/nan.
        found_inf_flag = _unscale_main_grads_and_check_for_nan(optim)

        # We are done with scaling gradients
        # so we can update the loss scale.
        optim.grad_scaler.update(found_inf_flag)

        # If we found inf/nan, skip the update.
        if found_inf_flag:
            optim.update_successful, grad_norm, num_zeros_in_grad = (
                False,
                None,
                None,
            )
            return optim.update_successful, grad_norm, num_zeros_in_grad

    def clip_grad_norm(optim: MegatronDistOptim, clip_grad: float) -> float:
        """Compute grad norm."""
        params = optim.get_parameters()
        grads_for_norm = optim.get_main_grads_for_grad_norm()
        return clip_grad_norm_fp32(
            params,
            grads_for_norm,
            clip_grad,
            model_parallel_group=constants.parallelism_group(),
        )

    def count_zeros(optim: MegatronDistOptim) -> float:
        """Count number of zeros in model's gradients."""
        params = optim.get_parameters()
        return count_zeros_fp32(
            params,
            model_parallel_group=constants.parallelism_group(),
        )

    # Clip the main gradients.
    grad_norm = None
    if optim.config.clip_grad > 0.0:
        grad_norm = clip_grad_norm(optim, optim.config.clip_grad)

    # Count the zeros in the grads.
    num_zeros_in_grad = None
    if optim.config.log_num_zeros_in_grad:
        num_zeros_in_grad = count_zeros(optim)

    # Step the optimizer.
    optim.optimizer.step()

    # Update params from main params.
    optim._copy_main_params_to_model_params()
    # Successful update.
    optim.update_successful, grad_norm, num_zeros_in_grad = (
        True,
        grad_norm,
        num_zeros_in_grad,
    )
    return optim.update_successful, grad_norm, num_zeros_in_grad


def step_megatron_distrb_optimizer(optim: MegatronDistOptim):

    optim.update_successful, grad_norm, num_zeros_in_grad = (
        _step_megatron_distrib_optimizer_internal(optim)
    )

    # If not overlapping all-gather for parameters, launch synchronous all-gather
    # communication calls here. If overlapping all-gather for parameters, the following
    # call to _gather_all_model_params is a no-op: the first all-gather is launched
    # asynchronously in the next optimizer.zero_grad() call and subsequent all-gathers
    # are launched in the forward pre-hook.
    optim._reset_metadata_and_sync_gather_all_model_params(force_sync=False)

    return optim.update_successful, grad_norm, num_zeros_in_grad


def _flatten_dense_tensors(tensors):
    """Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.

    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.

    Args:
        tensors (Iterable[Tensor]): dense tensors to flatten.

    Returns:
        A contiguous 1D buffer containing input tensors.
    """
    return torch._C._nn.flatten_dense_tensors(tensors)


def _unflatten_dense_tensors(flat, tensors):
    """View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by _flatten_dense_tensors.

    Args:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
          unflatten flat.

    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    return torch._C._nn.unflatten_dense_tensors(flat, tensors)


def megatron_all_reduce_layernorm_grads(engine: MegatronEngine):
    if not (
        constants.sequence_parallel() and constants.model_parallel_world_size() > 1
    ):
        return
    real_model: ReaLModel = engine.ddp.module
    grads = []
    for i in range(real_model.layer_idx_start, real_model.layer_idx_end):
        if i == 0:
            continue
        elif i == real_model.config.n_layers + 1:
            continue
        else:
            assert 0 < i < real_model.config.n_layers + 1
            layer: ReaLModelBlock = real_model.layers[i - real_model.layer_idx_start]
            grads.append(layer.attn.c_attn.ln.weight.main_grad)
            if getattr(layer.attn.c_attn.ln, "bias", None) is not None:
                grads.append(layer.attn.c_attn.ln.bias.main_grad)
            grads.append(layer.mlp.ln.weight.main_grad)
            if getattr(layer.mlp.ln, "bias", None) is not None:
                grads.append(layer.mlp.ln.bias.main_grad)
            if i == real_model.config.n_layers:
                grads.append(layer.ln_f.weight.main_grad)
                if getattr(layer.ln_f, "bias", None) is not None:
                    grads.append(layer.ln_f.bias.main_grad)

    assert all(x is not None for x in grads)
    coalesced = _flatten_dense_tensors(grads)
    dist.all_reduce(coalesced, group=constants.model_parallel_group())
    for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
        buf.copy_(synced)


def megatron_all_reduce_word_embedding_grads(engine: MegatronEngine):
    real_model: ReaLModel = engine.ddp.module
    if not real_model.config.tied_embedding or real_model.config.is_critic:
        return
    pp_size = constants.pipe_parallel_world_size()
    pp_rank = constants.pipe_parallel_rank()
    if pp_size == 1:
        return
    if pp_rank not in [0, pp_size - 1]:
        return

    if pp_rank == 0:
        grad = real_model.layers[0].wte.weight.main_grad
    else:
        grad = real_model.layers[-1].weight.main_grad

    dist.all_reduce(grad, group=constants.grid().embedding_proc_group)


def finalize_grads_megatron(engine: MegatronEngine):
    engine.ddp.finish_grad_sync()
    megatron_all_reduce_layernorm_grads(engine)
    megatron_all_reduce_word_embedding_grads(engine)
