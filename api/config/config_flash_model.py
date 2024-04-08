from typing import *
import dataclasses
import json
import os

import transformers

from api.config.config_base import Model, ModelWrapper, SUPPORTED_MODELS
import base.logging as logging

logger = logging.getLogger("Flash Model Config")


@dataclasses.dataclass
class FlashMQATConfig:
    n_layers: int
    n_kv_heads: int
    head_dim: int
    hidden_dim: int
    intermediate_dim: int  # for mlp, usually 4*h
    vocab_size: int
    n_positions: Optional[int] = None
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    activation_function: str = "gelu"
    scale_attn_by_inverse_layer_idx: bool = True
    # llama does not use attention bias and uses special MLP/LayerNorm layers
    use_attention_bias: bool = True
    layer_norm_type: Optional[str] = None
    mlp_type: Optional[str] = None
    # rotary embedding
    apply_rotary: bool = False
    rotary_base: float = 10000.0
    rotary_interleaved: bool = False
    rotary_scaling: Optional[float] = None
    rotary_scaling_type: Optional[str] = None
    # parallelism optimization
    sequence_parallel: bool = False
    gradient_accumulation_fusion: bool = False

    is_critic: bool = False

    # only used for debugging, True for GPT2
    fixed_abs_position_ids: bool = False

    # remained for compatibility, not used any more
    ckpt_attn: bool = False
    ckpt_mlp: bool = False


def convert_config_starcoder(starcoder_config: transformers.GPTBigCodeConfig) -> FlashMQATConfig:
    return FlashMQATConfig(
        n_layers=starcoder_config.n_layer,
        n_kv_heads=1,
        attn_pdrop=starcoder_config.attn_pdrop,
        embd_pdrop=starcoder_config.embd_pdrop,
        layer_norm_epsilon=starcoder_config.layer_norm_epsilon,
        hidden_dim=starcoder_config.n_embd,
        head_dim=starcoder_config.n_embd // starcoder_config.n_head,
        intermediate_dim=starcoder_config.n_inner,
        n_positions=starcoder_config.n_positions,
        resid_pdrop=starcoder_config.resid_pdrop,
        vocab_size=starcoder_config.vocab_size,
    )


def gpt2_config_converter(gpt2config: transformers.GPT2Config) -> FlashMQATConfig:
    return FlashMQATConfig(
        n_layers=gpt2config.n_layer,
        n_kv_heads=gpt2config.n_head,
        attn_pdrop=gpt2config.attn_pdrop,
        embd_pdrop=gpt2config.embd_pdrop,
        layer_norm_epsilon=gpt2config.layer_norm_epsilon,
        hidden_dim=gpt2config.n_embd,
        head_dim=gpt2config.n_embd // gpt2config.n_head,
        intermediate_dim=gpt2config.n_inner if gpt2config.n_inner is not None else 4 * gpt2config.n_embd,
        n_positions=gpt2config.n_positions,
        resid_pdrop=gpt2config.resid_pdrop,
        vocab_size=gpt2config.vocab_size,
        activation_function=gpt2config.activation_function,
        scale_attn_by_inverse_layer_idx=False,
        fixed_abs_position_ids=True,
    )


def convert_config_llama(hf_config: transformers.LlamaConfig) -> FlashMQATConfig:
    return FlashMQATConfig(
        n_layers=hf_config.num_hidden_layers,
        n_kv_heads=hf_config.num_key_value_heads,
        hidden_dim=hf_config.hidden_size,
        head_dim=hf_config.hidden_size // hf_config.num_attention_heads,
        intermediate_dim=hf_config.intermediate_size,
        vocab_size=hf_config.vocab_size,
        n_positions=hf_config.max_position_embeddings,
        embd_pdrop=0.0,
        attn_pdrop=hf_config.attention_dropout if hasattr(hf_config, "attention_dropout") else 0.1,
        layer_norm_epsilon=hf_config.rms_norm_eps,
        activation_function=hf_config.hidden_act,
        use_attention_bias=hf_config.attention_bias,
        scale_attn_by_inverse_layer_idx=False,
        layer_norm_type="rms",
        mlp_type="llama",
        apply_rotary=True,
        rotary_base=hf_config.rope_theta,
        rotary_interleaved=False,
        rotary_scaling=None if hf_config.rope_scaling is None else hf_config.rope_scaling["factor"],
        rotary_scaling_type=None if hf_config.rope_scaling is None else hf_config.rope_scaling["type"],
    )


FLASH_MODEL_CONFIG_CONVERTER: Dict[str, Callable[[Any], FlashMQATConfig]] = {
    "starcoder": convert_config_starcoder,
    "gpt2": gpt2_config_converter,
    "llama": convert_config_llama,
    "codellama": convert_config_llama,
    "deepseek": convert_config_llama,
}


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
        zero_stage (int): Stage of DeepSpeed ZeRO optimization. Should be one of 0, 1, 2, 3.
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
    zero_stage: int = dataclasses.field(
        metadata={"choices": [0, 1, 2, 3]},
        default=2,
    )
    optimizer: Optional[OptimizerConfig] = dataclasses.field(default_factory=OptimizerConfig)
    enable_async_p2p: bool = False

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


def get_flash_mqat_model_config(
    from_type: str,
    model_path: str,
    hf_model_type: str,
    tokenizer_path: str,
    dtype: Optional[str] = None,
    # model parallelism optimization
    sequence_parallel: bool = False,
    gradient_accumulation_fusion: bool = False,
    # LoRA config
    lora: Optional[LoRAConfig] = None,
    is_sft_lora: bool = False,
    sft_lora_path: Optional[str] = None,
    is_rew_lora: bool = False,
    rew_lora_path: Optional[str] = None,
) -> Model:
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

    if os.path.exists(os.path.join(model_path, "flash_mqat_config.json")):
        # This is a saved model from a previous run
        with open(os.path.join(model_path, "flash_mqat_config.json"), "r") as f:
            original_is_critic = json.load(f)["is_critic"]
        # correct from_type if necessary
        if from_type == "hf_as_critic":
            from_type = "self" if original_is_critic else "actor_as_critic"
        if from_type == "hf_as_actor":
            from_type = "self"
    else:
        # This is a checkpoint from HuggingFace.
        if from_type == "actor_as_critic":
            from_type = "hf_as_critic"
        if from_type == "self":
            from_type = "hf_as_actor"

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
    return model
