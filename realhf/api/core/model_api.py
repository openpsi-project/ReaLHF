from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Union
import abc
import copy
import dataclasses
import keyword
import os

import torch
import torch.utils.data
import transformers

from realhf.api.core import dfg, system_api
from realhf.api.core.config import ModelFamily, ModelName
from realhf.base.namedarray import NamedArray
import realhf.base.logging as logging

logger = logging.getLogger("model")


@dataclasses.dataclass
class ReaLModelConfig:
    """Configuration for ReaLModel.

    :param n_layers: Number of transformer blocks.
    :type n_layers: int
    :param n_kv_heads: Number of key-value attention heads.
    :type n_kv_heads: int
    :param head_dim: Dimension of each attention head.
        The number of query heads is hidden_dim // head_dim.
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
    :param use_attention_bias: Whether to use attention bias.
    :type use_attention_bias: bool
    :param layer_norm_type: Type of layer normalization. Either None or "rms".
    :type layer_norm_type: Optional[str]
    :param mlp_type: Type of the MLP. Either None or "llama".
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
    :param is_critic: Whether the model is a critic model.
    :type is_critic: bool
    :param gradient_accumulation_fusion: Whether to fuse
        gradient accumulation in Megatron.
        Currently not supported.
    :type gradient_accumulation_fusion: bool
    :param share_embeddings_and_output_weights: Whether to share
        the embeddings and output weights.
        Currently not supported.
    :type share_embeddings_and_output_weights: bool
    """

    ### Architectural configurations. ###
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
    # Whether it is a critic/reward model that outputs scores.
    is_critic: bool = False

    ### Running configurations. ###
    gradient_accumulation_fusion: bool = False

    # Placeholder, not implemented
    share_embeddings_and_output_weights: bool = False

    def __post_init__(self):
        assert not self.share_embeddings_and_output_weights


def load_hf_tokenizer(
    model_name_or_path: str,
    fast_tokenizer=True,
    padding_side: Optional[str] = None,
) -> transformers.PreTrainedTokenizerFast:
    kwargs = {}
    if padding_side is not None:
        kwargs["padding_side"] = padding_side
    if os.path.exists(model_name_or_path):
        # Locally tokenizer loading has some issue, so we need to force download
        model_json = os.path.join(model_name_or_path, "config.json")
        if "codet5" in model_name_or_path:
            tokenizer = transformers.RobertaTokenizer.from_pretrained(
                model_name_or_path, fast_tokenizer=fast_tokenizer, **kwargs
            )
        if os.path.exists(model_json):
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name_or_path, fast_tokenizer=fast_tokenizer, **kwargs
            )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path, fast_tokenizer=fast_tokenizer, **kwargs
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


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


@dataclasses.dataclass
class Model:
    name: ModelName
    module: torch.nn.Module
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
    return Model(
        name, torch.nn.Identity(), load_hf_tokenizer(tokenizer_path), device
    )


class ModelInterface(abc.ABC):
    """Interface for model training, evaluation, inference and generation."""

    def save(self, model: Model, save_dir: str):
        pass

    def evaluate(
        self, model: Model, eval_dataloader: torch.utils.data.DataLoader
    ) -> Dict:
        return {}

    def inference(self, model: Model, data: NamedArray) -> NamedArray:
        raise NotImplementedError()

    def generate(self, model: Model, data: NamedArray) -> NamedArray:
        raise NotImplementedError()

    def train_step(self, model: Model, data: NamedArray) -> Dict:
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
    cfg: system_api.ModelWrapper,
) -> Callable[[Model], Model]:
    cls_ = ALL_WRAPPER_CLASSES[cfg.type_]
    return cls_(**cfg.args)


def make_model(
    cfg: system_api.Model, name: ModelName, device: Union[str, torch.device]
) -> Model:
    logger.debug(f"making model {cfg.type_} on {device}")
    model_cls = ALL_MODEL_CLASSES[cfg.type_]
    model = model_cls(**cfg.args, name=name, device=device)
    assert isinstance(model, Model)
    for w in cfg.wrappers:
        model = make_model_wrapper(w)(model)
        assert isinstance(model, Model)
    return model


def make_interface(cfg: dfg.ModelInterface) -> ModelInterface:
    cls_ = ALL_INTERFACE_CLASSES[cfg.type_]
    return cls_(**cfg.args)


def make_backend(cfg: system_api.ModelBackend) -> ModelBackend:
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
    config_to_hf_converter: Callable[
        [ReaLModelConfig], transformers.PretrainedConfig
    ],
    sd_from_hf_converter: Callable[[Dict, ReaLModelConfig], Dict],
    sd_to_hf_converter: Callable[[Dict, ReaLModelConfig], Dict],
    embedding_param_names: Callable[[ReaLModelConfig], List[str]],
    tblock_param_names: Callable[[ReaLModelConfig, int], List[str]],
    head_param_names: Callable[[ReaLModelConfig], List[str]],
):
    if name in SUPPORTED_MODELS:
        raise ValueError(f"Model {name} is already registered.")
    if not is_valid_function_name(name):
        raise ValueError(
            f"Model name {name} is not a valid function name in Python."
        )
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
    )
