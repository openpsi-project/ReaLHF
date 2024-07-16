import abc
import dataclasses
import keyword
from typing import *

import torch
import torch.utils.data
import transformers

import realhf.base.logging as logging
from realhf.api.core.config import (
    ModelAbstraction,
    ModelBackendAbstraction,
    ModelInterfaceAbstraction,
    ModelName,
    ModelWrapperAbstraction,
)
from realhf.api.core.data_api import SequenceSample, load_hf_tokenizer

logger = logging.getLogger("model_api")


@dataclasses.dataclass
class GenerationHyperparameters:
    """Generation hyperparameters.

    We implement a customized generation function instead of
    using HuggingFace's to support pipelined generation.
    As a result, advanced generation techniques like
    diversity-promoting sampling or repeatition penalty
    are not supported during PPO training.
    However, we don't find it to be a problem in practice.
    Increasing the sampling temperature and enabling
    top-k/top-p sampling can produce good models.

    :param max_new_tokens: The maximum number of new tokens to generate.
    :type max_new_tokens: int
    :param min_new_tokens: The minimum number of new tokens to generate.
    :type min_new_tokens: int
    :param greedy: Whether to use greedy decoding.
    :type greedy: bool
    :param top_k: The number of highest probability tokens to keep.
    :type top_k: int
    :param top_p: The cumulative probability of the highest probability tokens to keep.
    :type top_p: float
    :param temperature: The temperature of the sampling process.
    :type temperature: float
    :param use_cuda_graph: Whether to use CUDA graph to reduce kernel launch overhead
        during generation. Recommended for pure generation.
    :type use_cuda_graph: bool
    """

    max_new_tokens: int = 256
    min_new_tokens: int = 256
    greedy: bool = False
    top_p: float = 0.9
    top_k: int = 200
    temperature: float = 1.0
    use_cuda_graph: bool = False

    def __post_init__(self):
        if self.temperature == 0.0:
            self.greedy = True
            self.temperature = 1.0
        if self.top_p <= 0.0 or self.top_p > 1:
            raise ValueError("top_p must be in (0.0, 1.0]")
        if self.top_k <= 0:
            raise ValueError("top_k must be a positive integer.")


@dataclasses.dataclass
class ReaLModelConfig:
    """Configuration for ReaLModel.

    :param n_layers: Number of transformer blocks.
    :type n_layers: int
    :param n_kv_heads: Number of key-value attention heads.
    :type n_kv_heads: int
    :param n_q_heads: Number of query attention heads.
    :type n_q_heads: int
    :param head_dim: Dimension of each attention head.
        If None, defaults to hidden_dim // n_q_heads.
        If given, the query layer will have shape (hidden_dim, head_dim * n_q_heads).
    :type head_dim: int
    :param hidden_dim: Hidden dimension of the transformer block.
    :type hidden_dim: int
    :param intermediate_dim: Dimension of the intermediate layer in the MLP.
    :type intermediate_dim: int
    :param vocab_size: Vocabulary size.
    :type vocab_size: int
    :param n_positions: Maximum context length. Can be None for
        rotary embedding, where the context length is decided during runtime.
    :type n_positions: Optional[int]
    :param embd_pdrop: Dropout probability for the embedding layer.
    :type embd_pdrop: float
    :param resid_pdrop: Dropout probability for the residual connections.
    :type resid_pdrop: float
    :param attn_pdrop: Dropout probability for the attention weights.
    :type attn_pdrop: float
    :param layer_norm_epsilon: Epsilon for layer normalization.
    :type layer_norm_epsilon: float
    :param activation_function: Activation function for the MLP.
    :type activation_function: str
    :param scale_attn_by_inverse_layer_idx: Whether to scale the attention weights
        by the inverse of the layer index.
    :type scale_attn_by_inverse_layer_idx: bool
    :param use_attention_bias: Whether to use bias for QKV layers.
    :type use_attention_bias: bool
    :param use_attn_proj_bias: Whether to use bias for the attention projection layer.
    :type use_attn_proj_bias: bool
    :param layer_norm_type: Type of layer normalization, can be None, "rms", or "gemma".
    :type layer_norm_type: Optional[str]
    :param mlp_type: Type of the MLP. Can be None, "llama", or "moe".
    :type mlp_type: Optional[str]
    :param apply_rotary: Whether to apply rotary embedding.
    :type apply_rotary: bool
    :param rotary_base: Exponential base for the rotary embedding.
    :type rotary_base: float
    :param rotary_interleaved: Whether to use interleaved rotary embedding.
    :type rotary_interleaved: bool
    :param rotary_scaling: Scaling factor for the rotary embedding.
    :type rotary_scaling: Optional[float]
    :param rotary_scaling_type: Type of scaling for the rotary embedding.
    :type rotary_scaling_type: Optional[str]
    :param normalize_embed: Whether to normalize the embeddings
        before transformer blocks. Used by Gemma.
    :type normalize_embed: bool
    :param abs_position_embedding_offset: Offset for the absolute position embedding.
        Used by OPT, but OPT is currently not supported.
    :type abs_position_embedding_offset: int
    :param do_layernorm_before: Whether to apply layer normalization before the attention
        rather than after. Used by OPT, but OPT is currently not supported.
    :type do_layernorm_before: bool
    :param tied_embedding: Whether to share the embeddings and output weights.
        Used by models like GPT-2 and Gemma.
    :type tied_embedding: bool
    :param sliding_window: Sliding window size for the attention.
        Currently a placeholder and not supported.
    :type sliding_window: Optional[int]
    :param num_experts: Number of experts in the mixture of experts.
        Only effective when mlp_type="moe".
    :type num_experts: int
    :param moe_top_k: The number of experts to route per-token, can be also
        interpreted as the `top-k` routing parameter.
    :type moe_top_k: int
    :param routing_type: The load balancing type for the MoE router.
        Can be "aux_loss", "sinkhorn", or "none".
    :type routing_type: str
    :param aux_loss_coeff: The coefficient for the auxiliary loss.
        Only effective when routing_type="aux_loss".
    :type aux_loss_coeff: float
    :param capacity_factor: The capacity factor of each expert.
        An expert will drop tokens if the number of tokens exceeds capacity_factor * (num_tokens / num_experts).
        Drop nothing when capacity_factor is None.
    :type capacity_factor: float
    :param pad_to_capacity: Whether to pad the input to the capacity of the expert.
    :type pad_to_capacity: bool
    :param token_drop_policy: The token drop policy for the MoE. Can be either "prob" or "position".
        If "prob", the tokens with the lowest probabilities will be dropped.
        If "position", tokens at the end of each batch will be dropped.
    :type token_drop_policy: str
    :param z_loss_coeff: The coefficient for the z-loss.
    :type z_loss_coeff: float
    :param input_jitter_eps: The input jitter noise for the router.
    :type input_jitter_eps: float
    :param is_critic: Whether the model is a critic model.
    :type is_critic: bool
    :param gradient_accumulation_fusion: Whether to fuse
        gradient accumulation in Megatron.
        Currently not supported.
    :type gradient_accumulation_fusion: bool
    :param use_grouped_gemm: Whether to use grouped GEMM for MoE.
    :type use_grouped_gemm: bool
    """

    ### Architectural configurations. ###
    n_layers: int
    n_kv_heads: int
    n_q_heads: int
    hidden_dim: int
    intermediate_dim: int  # for mlp, usually 4*h
    vocab_size: int
    head_dim: Optional[int] = None
    n_positions: Optional[int] = None
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    activation_function: str = "gelu"
    scale_attn_by_inverse_layer_idx: bool = True
    scale_attn_weights: bool = True
    # llama does not use attention bias and uses special MLP/LayerNorm layers
    use_attention_bias: bool = True
    use_attn_proj_bias: bool = True
    layer_norm_type: Optional[str] = None
    mlp_type: Optional[str] = None
    # rotary embedding
    apply_rotary: bool = False
    rotary_base: float = 10000.0
    rotary_interleaved: bool = False
    rotary_scaling: Optional[float] = None
    rotary_scaling_type: Optional[str] = None
    # for gemma
    normalize_embed: bool = False
    # for opt, it's 2
    abs_position_embedding_offset: int = 0
    do_layernorm_before: bool = True
    # Tied embedding
    tied_embedding: bool = False
    sliding_window: Optional[int] = None
    # For MoE
    num_experts: int = 8
    moe_top_k: int = 2
    routing_type: str = "aux_loss"  # "sinkhorn" "none"
    aux_loss_coeff: float = 1e-3
    capacity_factor: float = None
    pad_to_capacity: bool = False
    token_drop_policy: str = "probs"  # "position"
    z_loss_coeff: float = 0.0
    input_jitter_eps: float = 0.0
    use_grouped_gemm: bool = False

    # Whether it is a critic/reward model that outputs scores.
    is_critic: bool = False

    ### Running configurations. ###
    gradient_accumulation_fusion: bool = False

    def __post_init__(self):
        if self.is_critic and self.tied_embedding:
            raise ValueError("Critic model cannot share embeddings and output weights.")
        if self.head_dim is None:
            self.head_dim = self.hidden_dim // self.n_q_heads


@dataclasses.dataclass
class ModelVersion:
    epoch: int = 0
    epoch_step: int = 0
    global_step: int = 0


@dataclasses.dataclass
class FinetuneSpec:
    total_train_epochs: int
    total_train_steps: int
    steps_per_epoch: int


class PipelinableEngine(abc.ABC):
    @abc.abstractmethod
    def train_batch(
        self,
        input_: SequenceSample,
        loss_fn: Callable,
        version_steps: int,
        num_micro_batches: Optional[int] = None,
    ):
        raise NotImplementedError()

    @abc.abstractmethod
    def eval_batch(
        self,
        input_: SequenceSample,
        loss_fn: Callable,
        num_micro_batches: Optional[int] = None,
    ):
        raise NotImplementedError()

    @abc.abstractmethod
    def forward(
        self,
        input_: SequenceSample,
        num_micro_batches: Optional[int] = None,
    ):
        raise NotImplementedError()

    @abc.abstractmethod
    def generate(
        self,
        input_: SequenceSample,
        tokenizer: transformers.PreTrainedTokenizerFast,
        gconfig: GenerationHyperparameters = dataclasses.field(
            default_factory=GenerationHyperparameters
        ),
        num_micro_batches: Optional[int] = None,
    ):
        raise NotImplementedError()


@dataclasses.dataclass
class Model:
    name: ModelName
    module: PipelinableEngine | torch.nn.Module
    tokenizer: transformers.PreTrainedTokenizerFast
    device: Union[str, torch.device]
    dtype: Optional[torch.dtype] = None
    version: ModelVersion = dataclasses.field(default_factory=ModelVersion)
    ft_spec: FinetuneSpec = None  # will be initialized by the backend

    def __post_init__(self):
        try:
            self.module = self.module.to(self.device)
        except ValueError as e:
            # 4-bit and 8-bit model may fail here
            logger.warning(
                f"Failed to move model to device {self.device} because {e}. Abort to device."
            )

    def inc_version(self):
        self.version.global_step += 1
        self.version.epoch_step += 1
        if self.version.epoch_step >= self.ft_spec.steps_per_epoch:
            self.version.epoch_step = 0
            self.version.epoch += 1


class ModelBackend(abc.ABC):

    @abc.abstractmethod
    def _initialize(self, model: Model, spec: FinetuneSpec) -> Model:
        raise NotImplementedError()

    def initialize(self, model: Model, spec: FinetuneSpec) -> Model:
        model.ft_spec = spec
        return self._initialize(model, spec)


class NullBackend(ModelBackend):

    def _initialize(self, model: Model, spec: FinetuneSpec) -> Model:
        return model


def null_model(name: ModelName, device: Union[str, torch.device]) -> Model:
    return Model(name, torch.nn.Identity(), None, device)


def tokenizer_only_model(
    name: ModelName, device: Union[str, torch.device], tokenizer_path: str
) -> Model:
    return Model(name, torch.nn.Identity(), load_hf_tokenizer(tokenizer_path), device)


class ModelInterface(abc.ABC):
    """Interface for model training, evaluation, inference and generation."""

    def save(self, model: Model, save_dir: str):
        pass

    def evaluate(
        self, model: Model, eval_dataloader: torch.utils.data.DataLoader
    ) -> Dict:
        return {}

    def inference(self, model: Model, data: SequenceSample) -> SequenceSample:
        raise NotImplementedError()

    def generate(self, model: Model, data: SequenceSample) -> SequenceSample:
        raise NotImplementedError()

    def train_step(self, model: Model, data: SequenceSample) -> Dict:
        raise NotImplementedError()


ALL_MODEL_CLASSES = {}
ALL_INTERFACE_CLASSES = {}
ALL_BACKEND_CLASSES = {}
ALL_WRAPPER_CLASSES = {}


def register_model(name, model_cls):
    assert name not in ALL_MODEL_CLASSES
    ALL_MODEL_CLASSES[name] = model_cls


def register_interface(name, cls_):
    assert name not in ALL_INTERFACE_CLASSES
    assert issubclass(cls_, ModelInterface)
    ALL_INTERFACE_CLASSES[name] = cls_


def register_backend(name, cls_):
    assert name not in ALL_BACKEND_CLASSES
    assert issubclass(cls_, ModelBackend)
    ALL_BACKEND_CLASSES[name] = cls_


def register_wrapper(name, cls_):
    assert name not in ALL_WRAPPER_CLASSES
    ALL_WRAPPER_CLASSES[name] = cls_


def make_model_wrapper(
    cfg: ModelWrapperAbstraction,
) -> Callable[[Model], Model]:
    cls_ = ALL_WRAPPER_CLASSES[cfg.type_]
    return cls_(**cfg.args)


def make_model(
    cfg: ModelAbstraction, name: ModelName, device: Union[str, torch.device]
) -> Model:
    model_cls = ALL_MODEL_CLASSES[cfg.type_]
    model = model_cls(**cfg.args, name=name, device=device)
    assert isinstance(model, Model)
    for w in cfg.wrappers:
        model = make_model_wrapper(w)(model)
        assert isinstance(model, Model)
    return model


def make_interface(cfg: ModelInterfaceAbstraction) -> ModelInterface:
    cls_ = ALL_INTERFACE_CLASSES[cfg.type_]
    return cls_(**cfg.args)


def make_backend(cfg: ModelBackendAbstraction) -> ModelBackend:
    cls_ = ALL_BACKEND_CLASSES[cfg.type_]
    return cls_(**cfg.args)


register_backend("null", NullBackend)
register_model("null", null_model)
register_model("tokenizer", tokenizer_only_model)

SUPPORTED_MODELS = []
HF_MODEL_FAMILY_REGISTRY = {}


def is_valid_function_name(name):
    if not name.isidentifier():
        return False
    if keyword.iskeyword(name):
        return False
    return True


def register_hf_family(
    name: str,
    hf_cls_name: str,
    config_from_hf_converter: Callable[
        [transformers.PretrainedConfig], ReaLModelConfig
    ],
    config_to_hf_converter: Callable[[ReaLModelConfig], transformers.PretrainedConfig],
    sd_from_hf_converter: Callable[[Dict, ReaLModelConfig], Dict],
    sd_to_hf_converter: Callable[[Dict, ReaLModelConfig], Dict],
    embedding_param_names: Callable[[ReaLModelConfig], List[str]],
    tblock_param_names: Callable[[ReaLModelConfig, int], List[str]],
    head_param_names: Callable[[ReaLModelConfig], List[str]],
    real_config_maker: Optional[Callable] = None,
):
    if name in SUPPORTED_MODELS:
        raise ValueError(f"Model {name} is already registered.")
    if not is_valid_function_name(name):
        raise ValueError(f"Model name {name} is not a valid function name in Python.")
    SUPPORTED_MODELS.append(name)
    HF_MODEL_FAMILY_REGISTRY[name] = dict(
        name=name,
        hf_cls_name=hf_cls_name,
        config_from_hf_converter=config_from_hf_converter,
        config_to_hf_converter=config_to_hf_converter,
        sd_from_hf_converter=sd_from_hf_converter,
        sd_to_hf_converter=sd_to_hf_converter,
        embedding_param_names=embedding_param_names,
        tblock_param_names=tblock_param_names,
        head_param_names=head_param_names,
        real_config_maker=real_config_maker,
    )
