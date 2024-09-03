import collections
import dataclasses
import functools
import math
from typing import *

import deepspeed
import torch
import torch.distributed as dist
import transformers
from deepspeed.runtime import zero
from deepspeed.runtime.bf16_optimizer import BF16_Optimizer
from deepspeed.runtime.config import DeepSpeedConfig
from deepspeed.runtime.engine import (
    MEMORY_OPT_ALLREDUCE_SIZE,
    DeepSpeedEngine,
    DeepSpeedOptimizerCallable,
    DeepSpeedSchedulerCallable,
)
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

import realhf.api.core.model_api as model_api
import realhf.base.constants as constants
import realhf.base.logging as logging
from realhf.api.core.data_api import SequenceSample
from realhf.base.datapack import flat2d
from realhf.base.monitor import CUDATimeMarkType, cuda_tmark, cuda_tmarked
from realhf.impl.model.backend.inference import PipelinableInferenceEngine
from realhf.impl.model.backend.pipe_runner import PipeTrainInstrSet
from realhf.impl.model.nn.real_llm_api import ReaLModel
from realhf.impl.model.parallelism.pipeline_parallel.tensor_storage import TensorBuffer

logger = logging.getLogger("DeepSpeed Backend")


@dataclasses.dataclass
class PipeTrainSetForDeepSpeed(PipeTrainInstrSet):
    ds_engine: DeepSpeedEngine

    def __post_init__(self):
        self.ds_engine.pipeline_parallelism = True

    @cuda_tmark("bwd", CUDATimeMarkType.backward)
    def _exec_backward_pass(
        self,
        module: ReaLModel,
        tensor_buffer: TensorBuffer,
        stage_id: int,
        micro_batch_id: int,
        step_id: int,
    ):
        assert self.ds_engine is not None
        assert self.ds_engine.optimizer is not None, (
            "must provide optimizer during " "init in order to use backward"
        )
        # The last stage just runs backward on the loss using DeepSpeed's typical
        # mechanisms.
        output_x = tensor_buffer.get("batch_output_x", micro_batch_id, remove=True)

        # We schedule the all-reduces, so disable it in super().backward()
        self.ds_engine.enable_backward_allreduce = False
        self.ds_engine.set_gradient_accumulation_boundary(False)

        is_last_stage = constants.is_last_pipe_stage()
        if is_last_stage:
            loss = tensor_buffer.get("losses", micro_batch_id, remove=True)
            self.ds_engine.backward(loss)
            tensor_buffer.put("losses", micro_batch_id, loss.detach().clone())
            return False, None

        if self.ds_engine.bfloat16_enabled() and not is_last_stage:
            # manually call because we don't call optimizer.backward()
            self.ds_engine.optimizer.clear_lp_grads()

        if not is_last_stage and self.ds_engine.zero_optimization():
            # manually call because we don't call optimizer.backward()
            self.ds_engine.optimizer.micro_step_id += 1

            if self.ds_engine.optimizer.contiguous_gradients:
                self.ds_engine.optimizer.ipg_buffer = []
                buf_0 = torch.empty(
                    int(self.ds_engine.optimizer.reduce_bucket_size),
                    dtype=module.dtype,
                    device=module.device,
                )
                self.ds_engine.optimizer.ipg_buffer.append(buf_0)

                # Use double buffers to avoid data access conflict when overlap_comm is enabled.
                if self.ds_engine.optimizer.overlap_comm:
                    buf_1 = torch.empty(
                        int(self.ds_engine.optimizer.reduce_bucket_size),
                        dtype=module.dtype,
                        device=module.device,
                    )
                    self.ds_engine.optimizer.ipg_buffer.append(buf_1)
                self.ds_engine.optimizer.ipg_index = 0

        grad = tensor_buffer.get("grad", micro_batch_id, remove=True)

        output_tensor = output_x.pp_output
        torch.autograd.backward(tensors=output_tensor, grad_tensors=grad)

        if not is_last_stage and self.ds_engine.zero_optimization():
            # manually call because we don't call optimizer.backward()
            # Only for Stage 1, Mode 2
            if self.ds_engine.optimizer.use_grad_accum_attribute:
                self.ds_engine.optimizer.fill_grad_accum_attribute()

        if self.ds_engine.bfloat16_enabled() and not is_last_stage:
            # manually call because we don't call optimizer.backward()
            self.ds_engine.optimizer.update_hp_grads(clear_lp_grads=False)

    def _exec_reduce_grads(
        self,
        module: ReaLModel,
        tensor_buffer: TensorBuffer,
        stage_id: int,
        micro_batch_id: int,
        step_id: int,
    ):

        self.ds_engine.set_gradient_accumulation_boundary(True)
        if self.ds_engine.bfloat16_enabled():
            if self.ds_engine.zero_optimization_stage() < ZeroStageEnum.gradients:
                # Make our own list of gradients from the optimizer's FP32 grads
                self.ds_engine.buffered_allreduce_fallback(
                    grads=self.ds_engine.optimizer.get_grads_for_reduction(),
                    elements_per_buffer=MEMORY_OPT_ALLREDUCE_SIZE,
                )
            else:
                raise NotImplementedError("PP+BF16 only work for ZeRO Stage 1")
        else:
            self.ds_engine.allreduce_gradients(bucket_size=MEMORY_OPT_ALLREDUCE_SIZE)

    def _exec_optimizer_step(
        self,
        module: ReaLModel,
        tensor_buffer: TensorBuffer,
        stage_id: int,
        micro_batch_id: int,
        step_id: int,
    ):
        self.ds_engine.set_gradient_accumulation_boundary(True)
        version_steps = tensor_buffer.get("version_steps", 0)
        lr_kwargs = {"epoch": version_steps}
        self.ds_engine._take_model_step(lr_kwargs=lr_kwargs)

        # sync loss scale across pipeline stages
        if not self.ds_engine.bfloat16_enabled():
            loss_scale = self.ds_engine.optimizer.loss_scale
            total_scale_cuda = torch.FloatTensor([float(loss_scale)]).to(module.device)
            dist.all_reduce(
                total_scale_cuda,
                op=dist.ReduceOp.MIN,
                group=constants.grid().get_model_parallel_group(),
            )
            # all_loss_scale = total_scale_cuda[0].item()
            logger.info(
                f"loss scale: {total_scale_cuda}, "
                f"group: {dist.get_process_group_ranks(self.ds_engine.mpu.get_model_parallel_group())}"
            )
            self.ds_engine.optimizer.loss_scaler.cur_scale = min(
                total_scale_cuda[0].item(), 8192
            )


class ReaLDeepSpeedEngine(model_api.PipelinableEngine):

    def __init__(
        self,
        module: ReaLModel,
        ds_engine: DeepSpeedEngine,
    ):
        self.module = module

        self.inf_engine = PipelinableInferenceEngine(module)
        if constants.pipe_parallel_world_size() > 1:
            self.pipe_runner = self.inf_engine.pipe_runner

        self.device = module.device
        self.dtype = module.dtype

        self.ds_engine = ds_engine

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
        input_: SequenceSample,
        loss_fn: Callable,
        version_steps: int,
        num_micro_batches: Optional[int] = None,
    ):
        if num_micro_batches is None:
            num_micro_batches = 1
        if constants.pipe_parallel_world_size() > 1:
            # Fusing the minibatched forward-backward in a pipeline training schedule.
            instr_set = PipeTrainSetForDeepSpeed(self.ds_engine)
            # NOTE: When training with pipeline parallel, num micro batches should be
            # larger than 2 x num_pipeline_stages to avoid idle time.
            return self.pipe_runner.train_batch(
                instr_set=instr_set,
                input_=input_,
                loss_fn=loss_fn,
                version_steps=version_steps,
                n_pp_mbs=self.pipe_runner.default_train_mbs * num_micro_batches,
            )
        else:
            self.ds_engine._config.gradient_accumulation_steps = num_micro_batches
            self.ds_engine.set_gradient_accumulation_boundary(False)
            if isinstance(
                self.ds_engine.optimizer,
                (DeepSpeedZeroOptimizer, DeepSpeedZeroOptimizer_Stage3),
            ):
                self.ds_engine.optimizer.gradient_accumulation_steps = num_micro_batches

            stat = collections.defaultdict(int)
            for i, mb_input in enumerate(input_.split(num_micro_batches)):
                if i == num_micro_batches - 1:
                    self.ds_engine.set_gradient_accumulation_boundary(True)
                input_lens = torch.tensor(
                    flat2d(mb_input.seqlens["packed_input_ids"]),
                    dtype=torch.int32,
                    device="cuda",
                )
                max_seqlen = int(max(input_lens))
                cu_seqlens = torch.nn.functional.pad(input_lens.cumsum(0), (1, 0)).int()
                model_output = self.ds_engine(
                    packed_input_ids=mb_input.data["packed_input_ids"],
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                ).logits
                loss, _stat = loss_fn(model_output, mb_input)
                self.ds_engine.backward(loss)
                for k, v in _stat.items():
                    stat[k] += v
            lr_kwargs = {"epoch": version_steps} if version_steps is not None else None
            self.ds_engine.step(lr_kwargs=lr_kwargs)
            return stat

    @torch.no_grad()
    def forward(
        self,
        input_: SequenceSample,
        num_micro_batches: Optional[int] = None,
        post_hook: Callable[[torch.Tensor, SequenceSample], Any] | None = None,
        aggregate_fn: Callable[[List[Any]], Any] = torch.cat,
    ):
        return self.inf_engine.forward(
            input_, num_micro_batches, post_hook=post_hook, aggregate_fn=aggregate_fn
        )

    @torch.no_grad()
    def generate(
        self,
        input_: SequenceSample,
        tokenizer: transformers.PreTrainedTokenizerFast,
        gconfig: model_api.GenerationHyperparameters = dataclasses.field(
            default_factory=model_api.GenerationHyperparameters
        ),
        num_micro_batches: Optional[int] = None,
    ):
        return self.inf_engine.generate(input_, tokenizer, gconfig, num_micro_batches)


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
                if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
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
            raise NotImplementedError(f"Unsupported optimizer: {self.optimizer_name}.")

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
            return 1.0 - (1.0 - min_lr_ratio) / linear_steps * (step - warmup_steps)

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
