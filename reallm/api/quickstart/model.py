from typing import *
import dataclasses
import json
import os

from reallm.api.core.config import Model, ModelWrapper, ModelFamily
from reallm.api.core.model_api import SUPPORTED_MODELS
import reallm.base.logging as logging

logger = logging.getLogger("Quickstart Model Config")


@dataclasses.dataclass
class ParallelismConfig:
    """Model parallelism configuration.

    Args:
        model_parallel_size (int): Tensor model parallelism size.
        pipeline_parallel_size (int): Pipeline parallelism size.
        data_parallel_size (int): Data parallelism size.
        use_sequence_parallel (bool): Whether to use sequence parallelism combined with model parallelism.
        partition_method (str): Partition method for modules using pipeline parallel.
                                Support "uniform", "parameters" and "parameters_balanced".
    """

    model_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    use_sequence_parallel: bool = False

    def __post_init__(self):
        if self.pipeline_parallel_size < 1 or self.data_parallel_size < 1 or self.model_parallel_size < 1:
            raise ValueError("pp_size, mp_size and dp_size must be positive integers.")
        if self.use_sequence_parallel and self.model_parallel_size <= 1:
            logger.warning("Sequence parallelism requires model parallelism.")
            self.use_sequence_parallel = False


@dataclasses.dataclass
class LoRAConfig:
    """LoRA configuration.

    Args:
        dim (int): LoRA dimension.
        scaling (float): LoRA scaling factor.
    """

    dim: int = 32
    scaling: float = 32.0


@dataclasses.dataclass
class OptimizerConfig:
    """Optimizer configuration.

    Args:
        type (str): Optimizer type. Currently only adam optimizer is supported.
        lr (float): Learning rate.
        weight_decay (float): Weight decay.
        beta1 (float): Adam beta1.
        beta2 (float): Adam beta2.
        eps (float): Adam epsilon in the denominator.
            If pipeline parallelism is used, this should be at most 1.
        min_lr_ratio (float): Minimum learning rate ratio after learning rate annealing.
            Should be in the interval of [0.0, 1.0].
        lr_scheduler_type (str): Learning rate scheduler type.
        warmup_steps_proportion (float): Proportion of total training steps to warm up.
            Should be in the interval of [0.0, 1.0].
    """

    type: str = dataclasses.field(
        metadata={"choices": ["adam", "empty"]},
        default="empty",
    )
    lr: float = 1e-6
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-5
    min_lr_ratio: float = 0.0
    lr_scheduler_type: str = dataclasses.field(
        metadata={"choices": ["linear", "cosine", "constant"]},
        default="constant",
    )
    warmup_steps_proportion: float = 0.0
    offload: bool = False

    def __post_init__(self):
        if self.min_lr_ratio < 0.0 or self.min_lr_ratio > 1.0:
            raise ValueError(f"Invalid min_lr_ratio: {self.min_lr_ratio}")
        if self.warmup_steps_proportion < 0.0 or self.warmup_steps_proportion > 1.0:
            raise ValueError(f"Invalid warmup_steps_proportion: {self.warmup_steps_proportion}")


@dataclasses.dataclass
class ModelTrainEvalConfig:
    """Model configuration.

    We use customized model class, i.e., impl.nn.model.real_model, instead of HuggingFace's.
    This class enables 3D parallelism and flash-attention for better scalibility.
    The model uses no-pad flash-attn to save GPU memory, i.e., input sequences are packed into
    a single 1D tensor. The price is that we need to convert each HuggingFace model of interest
    to this customized model manually.

    If you find that the model of your interest is not supported, please reach out @fuwei for help.

    Args:
        type (str): Model type. Please check SUPPORTED_MODELS.
        path (str): Model checkpoint path, the directory instead of the file.
        base_model_path (str): HuggingFace model checkpoint path. Used for loading tokenizer and HuggingFace config.
        tokenizer_path (str): Tokenizer path.
        lora (bool): Whether to use LoRA.
        gradient_checkpointing (bool): Whether to use gradient checkpointing of MLP inside each block.
        enable_fp16 (bool): Whether to use fp16.
        enable_bf16 (bool): Whether to use bf16. Mutual exclusive with fp16.
        offload (bool): Whether to offload model to CPU.
        parallel (ParallelismConfig): Parallelism configuration.
        zero_stage (int): Stage of DeepSpeed ZeRO optimization. Should be one of 0, 1, 2, 3.
        optimizer (OptimizerConfig): Optimizer configuration.
    """

    type: ModelFamily = dataclasses.field(default=ModelFamily("llama", 7, False))
    lora: Optional[LoRAConfig] = None
    gradient_checkpointing: bool = False
    enable_fp16: bool = True
    enable_bf16: bool = False
    offload: bool = False
    parallel: ParallelismConfig = dataclasses.field(default_factory=ParallelismConfig)
    zero_stage: int = dataclasses.field(
        metadata={"choices": [0, 1, 2, 3]},
        default=2,
    )
    optimizer: Optional[OptimizerConfig] = dataclasses.field(default_factory=OptimizerConfig)

    def __post_init__(self):
        if self.enable_bf16 and self.enable_fp16:
            raise ValueError("enable_bf16 and enable_fp16 cannot be both True.")
        if self.enable_bf16 and (self.parallel.model_parallel_size > 1
                                 or self.parallel.pipeline_parallel_size > 1):
            raise ValueError("enable_bf16 cannot be used with model parallelism or pipeline parallelism.")
        if self.parallel.pipeline_parallel_size > 1 and self.lora is not None:
            raise ValueError("Use LoRA with pipeline parallel is not supported.")
        if self.parallel.pipeline_parallel_size > 1 and self.zero_stage > 1:
            logger.warning(f"ZeRO stage should be at most 1 when pipeline parallelism is used. "
                           f"Force to set it to 1. (original {self.zero_stage})")
            self.zero_stage = 1


def get_real_model_config(
    model_path: str,
    hf_model_family: str,
    is_critic: bool,
    init_critic_from_actor: bool,
    dtype: Optional[str] = None,
    # LoRA config
    lora: Optional[LoRAConfig] = None,
    is_sft_lora: bool = False,
    sft_lora_path: Optional[str] = None,
    is_rew_lora: bool = False,
    rew_lora_path: Optional[str] = None,
) -> Model:
    """Make a configuration to build model.
    """
    model = Model(
        "real_model",
        args=dict(
            model_path=model_path,
            is_critic=is_critic,
            init_critic_from_actor=init_critic_from_actor,
            dtype=dtype,
            hf_model_family=hf_model_family,
        ),
    )
    if is_sft_lora:
        model.wrappers += [
            ModelWrapper(
                "lora",
                args=dict(
                    lora_module_kwargs=dict(
                        lora_dim=lora.dim,
                        lora_scaling=lora.scaling,
                    ),
                    lora_keys_to_replace=["c_attn.linear", "c_proj."],
                    load_lora_path=sft_lora_path,
                    lora_op_after_creation="squash",
                ),
            ),
        ]
    if is_rew_lora:
        model.wrappers += [
            ModelWrapper(
                "lora",
                args=dict(
                    lora_module_kwargs=dict(
                        lora_dim=lora.dim,
                        lora_scaling=lora.scaling,
                    ),
                    lora_keys_to_replace=["c_attn.linear", "c_proj."],
                    load_lora_path=rew_lora_path,
                    lora_op_after_creation="squash",
                ),
            ),
        ]
    if lora is not None:
        model.wrappers += [
            ModelWrapper(
                "lora",
                args=dict(
                    lora_module_kwargs=dict(
                        lora_dim=lora.dim,
                        lora_scaling=lora.scaling,
                    ),
                    lora_keys_to_replace=["c_attn.linear", "c_proj."],
                    additional_module_names_to_opt=["v_head"],
                ),
            ),
        ]
    return model
