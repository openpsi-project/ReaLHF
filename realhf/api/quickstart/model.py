import dataclasses
from typing import *

import realhf.base.logging as logging
from realhf.api.core.config import (
    ModelAbstraction,
    ModelFamily,
    ModelWrapperAbstraction,
)

logger = logging.getLogger("Quickstart Model Config")


@dataclasses.dataclass(unsafe_hash=True)
class ParallelismConfig:
    """Configuration for 3D parallelism.

    :param model_parallel_size: Size of tensor-model parallelism.
    :type model_parallel_size: int
    :param pipeline_parallel_size: Number of pipeline parallelism
        stages.
    :type pipeline_parallel_size: int
    :param data_parallel_size: Data parallelism size for ZeRO
        optimization.
    :type data_parallel_size: int
    :param use_sequence_parallel: Whether to use sequence parallelism in
        Megatron in combination with tensor-model parallelism.
    :type use_sequence_parallel: bool
    """

    model_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    use_sequence_parallel: bool = False

    def __str__(self):
        return (
            f"Parallel(mp={self.model_parallel_size},"
            f"pp={self.pipeline_parallel_size},"
            f"dp={self.data_parallel_size})"
        )


def parallelism_eq(this, other):
    # NOTE: We write this function because
    # 1) we don't want to compare sequence_parallelism (it's irrelevant to parameter reallocation)
    # 2) implementing this function as a method of ParallelismConfig would cause a OmegaConf bug
    return (
        (this.model_parallel_size == other.model_parallel_size)
        and (this.pipeline_parallel_size == other.pipeline_parallel_size)
        and (this.data_parallel_size == other.data_parallel_size)
    )


@dataclasses.dataclass
class LoRAConfig:
    dim: int = 32
    scaling: float = 32.0


@dataclasses.dataclass
class OptimizerConfig:
    """Configuration for the optimizer.

    For models that will not be trained, the optimizer type should be
    set to "empty".

    :param type: Type of optimizer. Currently, only "adam" and "empty"
        optimizers are supported.
    :type type: str
    :param lr: Learning rate.
    :type lr: float
    :param weight_decay: Weight decay.
    :type weight_decay: float
    :param beta1: Adam beta1 parameter.
    :type beta1: float
    :param beta2: Adam beta2 parameter.
    :type beta2: float
    :param eps: Adam epsilon parameter in the denominator.
    :type eps: float
    :param min_lr_ratio: Minimum learning rate ratio after learning rate
        annealing. Should be in the interval [0.0, 1.0].
    :type min_lr_ratio: float
    :param lr_scheduler_type: Type of learning rate scheduler. One of
        "linear", "cosine", or "constant".
    :type lr_scheduler_type: str
    :param warmup_steps_proportion: Proportion of total training steps
        allocated for warming up. Should be in the interval [0.0, 1.0].
    :type warmup_steps_proportion: float
    :param offload: Whether to offload the optimizer to CPU. Only valid
        for the DeepSpeed backend.
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


@dataclasses.dataclass
class ModelTrainEvalConfig:
    """Runtime configuration for models (or LLMs) in ReaL.

    We use a customized model class instead of HuggingFace's. This customized model has
    the following highlights:

    1. Support for 3D parallelism and sequence parallelism.

    2. Support for flash attention during both training and generation.

    3. Input sequences are packed into a single 1D tensor to save GPU memory and improve efficiency.

    Consequently, each HuggingFace model of interest needs to be manually converted to this
    customized model. Implemented models can be found in the ``realhf/api/from_hf/`` directory.

    :param type: Model family type, e.g., llama, qwen2, etc.
    :type type: ModelFamily
    :param backend: Backend for training. Currently, only "megatron" and "deepspeed" are supported.
        Use "deepspeed" for offloading parameters or optimizer states, and "megatron" for
        parameter reallocation.
    :type backend: str
    :param path: Path of the HuggingFace checkpoint.
    :type path: str
    :param lora: Whether to use LoRA (Low-Rank Adaptation).
    :type lora: Optional[LoRAConfig]
    :param gradient_checkpointing: Whether to use gradient checkpointing to save memory.
    :type gradient_checkpointing: bool
    :param enable_fp16: Whether to use fp16 precision.
    :type enable_fp16: bool
    :param enable_bf16: Whether to use bf16 precision. Mutually exclusive with fp16.
    :type enable_bf16: bool
    :param offload: Whether to offload model parameters to CPU. Only valid for the DeepSpeed backend.
    :type offload: bool
    :param parallel: Configuration for parallelism.
    :type parallel: ParallelismConfig
    :param zero_stage: Stage of ZeRO optimization. Should be one of 0, 1, 2, or 3.
    :type zero_stage: int
    :param optimizer: Configuration for the optimizer.
    :type optimizer: Optional[OptimizerConfig]
    :param init_critic_from_actor: Whether to initialize a critic/reward model from a saved LM checkpoint.
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
