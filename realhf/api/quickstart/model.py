# Importing dataclasses is intended.
# Hydra does not recognize pydantic.Field,
# but it does recognize dataclasses.field.
import dataclasses
from typing import *

from pydantic import dataclasses as pdclasses
from pydantic import field_validator, model_validator
from typing_extensions import Self

import realhf.base.logging as logging
from realhf.api.core.config import (
    ModelAbstraction,
    ModelFamily,
    ModelWrapperAbstraction,
)

logger = logging.getLogger("Quickstart Model Config")


@pdclasses.dataclass
class ParallelismConfig:
    """Model 3D parallelism configuration.

    :param model_parallel_size: Tensor-model parallelism size.
    :type model_parallel_size: int
    :param pipeline_parallel_size: The number of pipeline parallelism stages.
    :type pipeline_parallel_size: int
    :param data_parallel_size: Data parallelism size for ZeRO optimization.
    :type data_parallel_size: int
    :param use_sequence_parallel: Whether to use sequence parallelism
        in Megatron combined with tensor-model parallelism.
    :type use_sequence_parallel: bool
    """

    model_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    use_sequence_parallel: bool = False

    @model_validator(mode="after")
    def _validate_parallelism(self) -> Self:
        if (
            self.pipeline_parallel_size < 1
            or self.data_parallel_size < 1
            or self.model_parallel_size < 1
        ):
            raise ValueError("pp_size, mp_size and dp_size must be positive integers.")
        return self

    @model_validator(mode="after")
    def _validata_sequence_parallel(self) -> Self:
        if self.use_sequence_parallel and self.model_parallel_size <= 1:
            logger.warning("Sequence parallelism requires model parallelism.")
            self.use_sequence_parallel = False
        return self

    def __str__(self):
        return (
            f"Parallel(mp={self.model_parallel_size},"
            f"pp={self.pipeline_parallel_size},"
            f"dp={self.data_parallel_size})"
        )


@pdclasses.dataclass
class LoRAConfig:
    dim: int = 32
    scaling: float = 32.0


@pdclasses.dataclass
class OptimizerConfig:
    """Optimizer configuration.

    For models that will not be trained, its type should be "empty".

    :param type: Optimizer type. Currently only adam and empty optimizer are supported.
    :type type: str
    :param lr: Learning rate.
    :type lr: float
    :param weight_decay: Weight decay.
    :type weight_decay: float
    :param beta1: Adam beta1.
    :type beta1: float
    :param beta2: Adam beta2.
    :type beta2: float
    :param eps: Adam epsilon in the denominator.
    :type eps: float
    :param min_lr_ratio: Minimum learning rate ratio after learning rate annealing.
        Should be in the interval of [0.0, 1.0].
    :type min_lr_ratio: float
    :param lr_scheduler_type: Learning rate scheduler type.
        One of "linear", "cosine", "constant".
    :type lr_scheduler_type: str
    :param warmup_steps_proportion: Proportion of total training steps to warm up.
        Should be in the interval of [0.0, 1.0].
    :type warmup_steps_proportion: float
    :param offload: Whether to offload optimizer to CPU.
        Only valid for the deepspeed backend.
    :type offload: bool
    """

    type: str = dataclasses.field(
        metadata={"choices": ["adam", "empty"]},
        default="empty",
    )
    lr: float = 1e-5
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-5
    min_lr_ratio: float = 0.0
    lr_scheduler_type: str = dataclasses.field(
        metadata={"choices": ["linear", "cosine", "constant"]},
        default="cosine",
    )
    warmup_steps_proportion: float = 0.02
    offload: bool = False

    @field_validator("min_lr_ratio")
    @classmethod
    def _validate_min_lr_ratio(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Invalid min_lr_ratio: {value}")
        return value

    @field_validator("warmup_steps_proportion")
    @classmethod
    def _validate_warmup_steps_proportion(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Invalid warmup_steps_proportion: {value}")
        return value


@pdclasses.dataclass
class ModelTrainEvalConfig:
    """
    Model (or LLM) runtime configuration in ReaL.

    We use a customized model class instead of HuggingFace's.
    Our customized model has the following highlights:

    1. Support 3D parallelism and sequence parallelism.

    2. Support flash attention for both training and generation.

    3. Input sequences are packed into a single 1D tensor to save GPU memory and improve efficiency.

    The price is that we need to convert each HuggingFace model
    of interest to this customized model manually.
    Implemented models can be found in the ``realhf/api/from_hf/`` directory.

    :param type: Model family type, e.g., llama, qwen2, etc.
    :type type: ModelFamily
    :param backend: Backend for training.
        Currently only "megatron" and "deepspeed" are supported.
        For offloading parameters or optimizer states, use "deepspeed".
        For parameter reallocation, use "megatron".
    :type backend: str
    :param path: Path or identifier of the HuggingFace checkpoint.
    :type path: str
    :param lora: Whether to use LoRA.
    :type lora: LoraConfig
    :param gradient_checkpointing: Whether to use gradient checkpointing.
    :type gradient_checkpointing: bool
    :param enable_fp16: Whether to use fp16.
    :type enable_fp16: bool
    :param enable_bf16: Whether to use bf16. Mutual exclusive with fp16.
    :type enable_bf16: bool
    :param offload: Whether to offload model parameters to CPU.
        Only valid for the deepspeed backend.
    :type offload: bool
    :param parallel: Parallelism configuration.
    :type parallel: ParallelismConfig
    :param zero_stage: Stage of ZeRO optimization.
        Should be one of 0, 1, 2, 3.
    :type zero_stage: int
    :param optimizer: Optimizer configuration.
    :type optimizer: OptimizerConfig
    :param init_critic_from_actor: Whether to initialize a
        critic/reward model from a saved LM checkpoint.
    :type init_critic_from_actor: bool
    """

    type: ModelFamily = dataclasses.field(default=ModelFamily("llama", 7, False))
    backend: str = dataclasses.field(
        default="megatron", metadata={"choices": ["megatron", "deepspeed"]}
    )
    path: str = ""
    lora: Optional[LoRAConfig] = None
    gradient_checkpointing: bool = True
    enable_fp16: bool = True
    enable_bf16: bool = False
    offload: bool = False
    zero_stage: int = dataclasses.field(
        metadata={"choices": [0, 1, 2, 3]},
        default=2,
    )
    optimizer: Optional[OptimizerConfig] = dataclasses.field(
        default_factory=OptimizerConfig
    )
    init_critic_from_actor: bool = False

    @model_validator(mode="after")
    def _validate_fp16_bf16(self) -> Self:
        if self.enable_bf16 and self.enable_fp16:
            raise ValueError("enable_bf16 and enable_fp16 cannot be both True.")
        return self

    @model_validator(mode="after")
    def _validate_offload(self) -> Self:
        if (self.offload or self.optimizer.offload) and self.backend != "deepspeed":
            raise ValueError("offload is only valid for the deepspeed backend.")
        return self

    @model_validator(mode="after")
    def _validate_megatron_zero_stage(self) -> Self:
        if self.backend == "megatron" and self.zero_stage in [1, 3]:
            raise ValueError("The Megatron backend only supports zero stage 0 or 2.")
        return self


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
) -> ModelAbstraction:
    """Make a configuration to build model."""
    model = ModelAbstraction(
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
            ModelWrapperAbstraction(
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
            ModelWrapperAbstraction(
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
            ModelWrapperAbstraction(
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
