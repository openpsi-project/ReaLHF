from typing import *
import dataclasses
import functools
import math

from deepspeed.runtime import zero
from deepspeed.runtime.bf16_optimizer import BF16_Optimizer
from deepspeed.runtime.config import DeepSpeedConfig
from deepspeed.runtime.engine import (
    DeepSpeedEngine,
    DeepSpeedOptimizerCallable,
    DeepSpeedSchedulerCallable,
)
import deepspeed
import torch
import torch.distributed as dist
import transformers

from realhf.impl.model.backend.pipe_runner import PipelineRunner
from realhf.impl.model.nn.real_llm_api import ReaLModel
from realhf.impl.model.nn.real_llm_generate import GenerationConfig
import realhf.api.core.model_api as model_api
import realhf.base.constants as constants
import realhf.base.logging as logging

logger = logging.getLogger("DeepSpeed Backend")


class ReaLDeepSpeedEngine:

    def __init__(
        self,
        module: ReaLModel,
        ds_engine: DeepSpeedEngine,
    ):
        self.module = module

        self.device = module.device
        self.dtype = module.dtype

        self.ds_engine = ds_engine

        if constants.pipe_parallel_world_size() > 1:
            self.pipe_runner = PipelineRunner(module)

            assert (
                self.ds_engine.zero_optimization_stage() < 2
            ), "ZeRO-2 and ZeRO-3 are incompatible with pipeline parallelism"
            if self.ds_engine.bfloat16_enabled():
                assert isinstance(self.ds_engine.optimizer, BF16_Optimizer)

            model_parameters = filter(
                lambda p: p.requires_grad, self.module.parameters()
            )
            num_params = sum([p.numel() for p in model_parameters])
            unique_params = num_params

            params_tensor = torch.LongTensor(
                data=[num_params, unique_params]
            ).to(self.device)
            dist.all_reduce(
                params_tensor, group=constants.grid().get_model_parallel_group()
            )
            params_tensor = params_tensor.tolist()
            total_params = params_tensor[0]
            unique_params = params_tensor[1]

            if constants.parallelism_rank() == 0:
                logger.info(
                    f"CONFIG: default_train_mbs={self.pipe_runner.default_train_mbs} "
                    f"default_inf_mbs={self.pipe_runner.default_inf_mbs} "
                    f"num_layers(this stage)={self.module.num_layers} "
                    f"pp_size={constants.pipe_parallel_world_size()} "
                    f"dp_size={constants.data_parallel_world_size()} "
                    f"mp_size={constants.model_parallel_world_size()} "
                    f"bf16={self.ds_engine.bfloat16_enabled()} "
                )
            if constants.data_parallel_rank() == 0:
                logger.info(
                    f"rank={constants.parallelism_rank()} "
                    f"stage={constants.pipe_parallel_rank()} "
                    f"layers={self.module.num_layers} "
                    f"[{self.module.layer_idx_start}, {self.module.layer_idx_end}) "
                    f"stage_params={num_params} ({num_params/1e6:0.3f}M) "
                    f"total_params={total_params} ({total_params/1e6:0.3f}M) "
                    f"unique_params={unique_params} ({unique_params/1e6:0.3f}M)"
                )

    def train(self, mode: bool = True):
        self.ds_engine.train(mode)
        self.module.train(mode)
        return self

    def eval(self):
        self.ds_engine.eval()
        self.module.eval()
        return self

    def train_batch(
        self,
        seqlens_cpu: List[int],
        packed_input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        loss_fn: Callable,
        version_steps: int,
        input_lens_for_partition: Optional[torch.Tensor] = None,
        num_micro_batches: Optional[int] = None,
        **loss_fn_kwargs,
    ):
        if constants.pipe_parallel_world_size() > 1:
            return self.pipe_runner.train_batch(
                engine=self.ds_engine,
                seqlens_cpu=seqlens_cpu,
                packed_input_ids=packed_input_ids,
                cu_seqlens=cu_seqlens,
                loss_fn=loss_fn,
                version_steps=version_steps,
                input_lens_for_partition=input_lens_for_partition,
                num_micro_batches=num_micro_batches,
                **loss_fn_kwargs,
            )
        else:
            # FIXME: num_micro_batches is ignored here
            max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))
            model_output = self.ds_engine(
                packed_input_ids=packed_input_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            ).logits
            loss, stat = loss_fn(
                model_output, packed_input_ids, cu_seqlens, **loss_fn_kwargs
            )
            self.ds_engine.backward(loss)
            lr_kwargs = (
                {"epoch": version_steps} if version_steps is not None else None
            )
            self.ds_engine.step(lr_kwargs=lr_kwargs)
            return stat

    @torch.no_grad()
    def eval_batch(
        self,
        seqlens_cpu: List[int],
        packed_input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        loss_fn: Callable,
        input_lens_for_partition: Optional[torch.Tensor] = None,
        num_micro_batches: Optional[int] = None,
        **loss_fn_kwargs,
    ):
        if constants.pipe_parallel_world_size() > 1:
            return self.pipe_runner.eval_batch(
                seqlens_cpu=seqlens_cpu,
                packed_input_ids=packed_input_ids,
                cu_seqlens=cu_seqlens,
                loss_fn=loss_fn,
                input_lens_for_partition=input_lens_for_partition,
                num_micro_batches=num_micro_batches,
                **loss_fn_kwargs,
            )
        else:
            max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))
            model_output = self.ds_engine(
                packed_input_ids=packed_input_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            ).logits
            _, stat = loss_fn(
                model_output, packed_input_ids, cu_seqlens, **loss_fn_kwargs
            )
            return stat

    def forward(
        self,
        seqlens_cpu: List[int],
        packed_input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        num_micro_batches: Optional[int] = None,
    ):
        if constants.pipe_parallel_world_size() > 1:
            return self.pipe_runner.forward(
                packed_input_ids=packed_input_ids,
                cu_seqlens=cu_seqlens,
                seqlens_cpu=seqlens_cpu,
                num_micro_batches=num_micro_batches,
            )
        else:
            max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))
            return self.ds_engine(
                packed_input_ids=packed_input_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            ).logits

    @torch.no_grad()
    def generate(
        self,
        seqlens_cpu: List[int],
        packed_input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        tokenizer: transformers.PreTrainedTokenizerFast,
        gconfig: GenerationConfig = dataclasses.field(
            default_factory=GenerationConfig
        ),
        num_micro_batches: Optional[int] = None,
    ):
        if constants.pipe_parallel_world_size() > 1:
            return self.pipe_runner.generate(
                seqlens_cpu=seqlens_cpu,
                num_micro_batches=num_micro_batches,
                tokenizer=tokenizer,
                packed_input_ids=packed_input_ids,
                cu_seqlens=cu_seqlens,
                gconfig=gconfig,
            )
        else:
            max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))
            res = self.module.generate(
                tokenizer=tokenizer,
                packed_input_ids=packed_input_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                gconfig=gconfig,
            )
            return res.sequences, res.scores, res.logits_mask


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


def get_eval_ds_config(
    offload=False,
    stage=0,
    enable_fp16: bool = True,
    enable_bf16: bool = False,
    **kwargs,
):
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
    no_decay_name_list: List[str] = ["bias", "ln.weight", "ln_f.weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (
                    not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad
                )
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (
                    any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad
                )
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def deepspeed_initialize(
    model: torch.nn.Module,
    config: Dict,
    optimizer: Optional[
        Union[torch.optim.Optimizer, DeepSpeedOptimizerCallable]
    ] = None,
    model_parameters: Optional[torch.nn.Module] = None,
    lr_scheduler: Optional[
        Union[torch.optim.lr_scheduler._LRScheduler, DeepSpeedSchedulerCallable]
    ] = None,
    mpu=None,
) -> Tuple[DeepSpeedEngine, torch.optim.Optimizer, Any, Any]:
    """A simple wrapper around deepspeed.initialize."""
    if mpu is None:
        mpu = constants.grid()
    config_class = DeepSpeedConfig(config, mpu)
    logger.info(
        f"DeepSpeedEngine Config: train_batch_size={config_class.train_batch_size}, "
        f"train_micro_batch_size_per_gpu={config_class.train_micro_batch_size_per_gpu}, "
        f"gradient_accumulation_steps={config_class.gradient_accumulation_steps}"
    )

    # Disable zero.Init context if it's currently enabled
    zero.partition_parameters.shutdown_init_context()

    from deepspeed import comm as dist

    deepspeed.dist = dist

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

    runner = ReaLDeepSpeedEngine(
        module=model,
        ds_engine=engine,
    )
    return_items = [
        runner,
        engine.optimizer,
        engine.training_dataloader,
        engine.lr_scheduler,
    ]
    return tuple(return_items)


@dataclasses.dataclass
class DeepspeedBackend(model_api.ModelBackend):
    optimizer_name: str = "adam"
    optimizer_config: dict = dataclasses.field(
        default_factory=lambda: dict(
            lr=1e-5, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-5
        )
    )
    lr_scheduler_type: str = "cosine"
    warmup_steps_proportion: float = 0.0
    min_lr_ratio: float = 0.0  # will be used for linear and cosine schedule
    offload_param: bool = False
    offload_optimizer_state: bool = False
    enable_fp16: bool = True
    enable_bf16: bool = False
    zero_stage: int = 2
    # addtional deepspeed args
    additional_ds_config: Dict = dataclasses.field(default_factory=dict)

    def _initialize(self, model: model_api.Model, spec: model_api.FinetuneSpec):
        deepspeed.init_distributed(auto_mpi_discovery=False)
        module = model.module
        weight_decay = self.optimizer_config.get("weight_decay", 0.0)
        if self.optimizer_name == "adam":
            if not self.offload_param and not self.offload_optimizer_state:
                optim_cls = deepspeed.ops.adam.FusedAdam
            else:
                optim_cls = deepspeed.ops.adam.DeepSpeedCPUAdam
            optimizer = optim_cls(
                get_optimizer_grouped_parameters(module, weight_decay),
                **self.optimizer_config,
            )
        else:
            raise NotImplementedError(
                f"Unsupported optimizer: {self.optimizer_name}."
            )

        ds_config = get_train_ds_config(
            offload_param=self.offload_param,
            offload_optimizer_state=self.offload_optimizer_state,
            stage=self.zero_stage,
            enable_bf16=self.enable_bf16,
            enable_fp16=self.enable_fp16,
            **self.additional_ds_config,
        )

        # NOTE: Just a fake batch size to make DeepSpeed happy.
        ds_config["train_batch_size"] = constants.data_parallel_world_size()

        def warmup_then_cosine_anneal(
            step, warmup_steps_proportion, total_steps, min_lr_ratio
        ):
            warmup_steps = max(5, int(total_steps * warmup_steps_proportion))
            cosine_steps = total_steps - warmup_steps
            if step < warmup_steps:
                return 1.0 / warmup_steps * step
            return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (
                1 + math.cos((step - warmup_steps) / cosine_steps * math.pi)
            )

        def warmup_then_linear_anneal(
            step, warmup_steps_proportion, total_steps, min_lr_ratio
        ):
            warmup_steps = max(5, int(total_steps * warmup_steps_proportion))
            linear_steps = total_steps - warmup_steps
            if step < warmup_steps:
                return 1.0 / warmup_steps * step
            return 1.0 - (1.0 - min_lr_ratio) / linear_steps * (
                step - warmup_steps
            )

        def warmup_then_constant_anneal(
            step, warmup_steps_proportion, total_steps, min_lr_ratio
        ):
            warmup_steps = max(5, int(total_steps * warmup_steps_proportion))
            if step < warmup_steps:
                return 1.0 / warmup_steps * step
            return 1.0

        if self.lr_scheduler_type == "cosine":
            lr_scheduler_fn = warmup_then_cosine_anneal
        elif self.lr_scheduler_type == "linear":
            lr_scheduler_fn = warmup_then_linear_anneal
        elif self.lr_scheduler_type == "constant":
            lr_scheduler_fn = warmup_then_constant_anneal
        else:
            raise NotImplementedError(
                f"Unknown lr_scheduler_type {self.lr_scheduler_type}."
            )

        lr_lambda = functools.partial(
            lr_scheduler_fn,
            warmup_steps_proportion=self.warmup_steps_proportion,
            total_steps=spec.total_train_steps,
            min_lr_ratio=self.min_lr_ratio,
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        module, *_ = deepspeed_initialize(
            model=module,
            optimizer=optimizer,
            config=ds_config,
            lr_scheduler=lr_scheduler,
        )

        model.module = module
        return model


model_api.register_backend("deepspeed", DeepspeedBackend)
