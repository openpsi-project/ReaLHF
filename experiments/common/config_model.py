from typing import *
import dataclasses
import json

from api.config import *

SUPPORTED_MODELS = ["starcoder", "llama", "gpt2", "deepseek", "codellama"]


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
    partition_method: Optional[str] = "parameters_balanced"

    def __post_init__(self):
        if self.pipeline_parallel_size < 1 or self.data_parallel_size < 1 or self.model_parallel_size < 1:
            raise ValueError("pp_size, mp_size and dp_size must be positive integers.")
        if self.use_sequence_parallel and self.model_parallel_size <= 1:
            raise ValueError("Sequence parallelism requires model parallelism.")


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
        zero_stage (int): Stage of DeepSpeed ZeRO optimization. Should be one of 0, 1, 2, 3.
            If pipeline parallelism is used, this should be at most 1.
        min_lr_ratio (float): Minimum learning rate ratio after learning rate annealing.
            Should be in the interval of [0.0, 1.0].
        lr_scheduler_type (str): Learning rate scheduler type.
        warmup_steps_proportion (float): Proportion of total training steps to warm up.
            Should be in the interval of [0.0, 1.0].
    """

    type: str = dataclasses.field(
        metadata={"choices": ["adam"]},
        default="adam",
    )
    lr: float = 1e-6
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-5
    zero_stage: int = dataclasses.field(
        metadata={"choices": [0, 1, 2, 3]},
        default=2,
    )
    min_lr_ratio: float = 0.0
    lr_scheduler_type: str = dataclasses.field(
        metadata={"choices": ["linear", "cosine"]},
        default="cosine",
    )
    warmup_steps_proportion: float = 0.02
    offload: bool = False

    def __post_init__(self):
        if self.min_lr_ratio < 0.0 or self.min_lr_ratio > 1.0:
            raise ValueError(f"Invalid min_lr_ratio: {self.min_lr_ratio}")
        if self.warmup_steps_proportion < 0.0 or self.warmup_steps_proportion > 1.0:
            raise ValueError(f"Invalid warmup_steps_proportion: {self.warmup_steps_proportion}")


@dataclasses.dataclass
class ModelConfig:
    """Model configuration.

    We use customized model class, i.e., impl.nn.model.flash_mqat, instead of HuggingFace's.
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
        optimizer (OptimizerConfig): Optimizer configuration.
        enable_async_p2p (bool): Whether to use async p2p, only effective when using pipeline parallelism.
    """

    type: str = dataclasses.field(
        metadata={"choices": SUPPORTED_MODELS},
        default="llama",
    )
    path: Optional[str] = None
    base_model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    lora: Optional[LoRAConfig] = None
    gradient_checkpointing: bool = False
    enable_fp16: bool = True
    enable_bf16: bool = False
    offload: bool = False
    parallel: ParallelismConfig = dataclasses.field(default_factory=ParallelismConfig)
    optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
    enable_async_p2p: bool = False

    def __post_init__(self):
        if self.enable_bf16 and self.enable_fp16:
            raise ValueError("enable_bf16 and enable_fp16 cannot be both True.")
        if self.enable_bf16 and (self.parallel.model_parallel_size > 1
                                 or self.parallel.pipeline_parallel_size > 1):
            raise ValueError("enable_bf16 cannot be used with model parallelism or pipeline parallelism.")
        if self.parallel.pipeline_parallel_size > 1 and self.lora is not None:
            raise ValueError("Use LoRA with pipeline parallel is not supported.")
        if self.offload and not self.optimizer.zero_stage != 3:
            raise ValueError("offload model is only supported when zero stage=3.")


def get_flash_mqat_model_config(
    from_type: str,
    model_path: str,
    hf_model_type: str,
    tokenizer_path: str,
    use_pipe: bool,
    dtype: Optional[str] = None,
    # model parallelism optimization
    sequence_parallel: bool = False,
    gradient_accumulation_fusion: bool = False,
    # pipeline partition method
    partition_method: Optional[str] = "parameters_balanced",
    # LoRA config
    lora: Optional[LoRAConfig] = None,
    is_sft_lora: bool = False,
    sft_lora_path: Optional[str] = None,
    is_rew_lora: bool = False,
    rew_lora_path: Optional[str] = None,
):
    """Make a configuration to build model.

    Possible values of `from_type`:
        > hf_as_actor: build actor (decoder-only LLM) from huggingface models
        > hf_as_critic: build critic (transformer that outputs values instead of logits) from huggingface models
        > actor_as_critic: build critic from actor, replace the head with a new one, whether using pipeline depends on `use_pipe`
        > random_actor: build a randomly initialized actor, whether using pipeline depends on `use_pipe`
        > random_critic build a randomly initialized critic, whether using pipeline depends on `use_pipe`
        > self: build a actor/critic from itself, whether using pipeline depends on `use_pipe`
            Note that it may not be built successfully if `use_pipe` is not consistent with the saved checkpoint
    """
    if gradient_accumulation_fusion:
        raise RuntimeError("gradient_accumulation_fusion is not supported yet")
    if (lora is not None or is_sft_lora or is_rew_lora) and use_pipe:
        raise NotImplementedError("LORA is not supported in pipeline model")

    if use_pipe:
        pipe_init_from_scratch = from_type == "random_actor" or from_type == "random_critic"
        pipe_init_critic_from_actor = from_type == "actor_as_critic"
        with open(os.path.join(model_path, "flash_mqat_config.json"), "r") as f:
            original_is_critic = json.load(f)["is_critic"]
        is_critic = original_is_critic or from_type == "actor_as_critic"
        from_type = "empty_critic" if is_critic else "empty_actor"

    model = Model(
        "flash_mqat",
        args=dict(
            model_path=model_path,
            from_type=from_type,
            dtype=dtype,
            hf_model_type=hf_model_type,
            tokenizer_path=tokenizer_path,
            sequence_parallel=sequence_parallel,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
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
    if use_pipe:
        model.wrappers += [
            ModelWrapper(
                "pipe_flash_mqat",
                args=dict(
                    model_path=model_path,
                    partition_method=partition_method,
                    init_from_scratch=pipe_init_from_scratch,
                    init_critic_from_actor=pipe_init_critic_from_actor,
                ),
            )
        ]
    return model
