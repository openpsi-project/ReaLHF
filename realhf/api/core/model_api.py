import abc
import dataclasses
import keyword
from typing import *

import torch
import torch.utils.data
import transformers
from packaging.version import Version

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

    We implement a customized generation function instead of using
    HuggingFace's to support pipelined generation. As a result, advanced
    generation techniques like diversity-promoting sampling or
    repetition penalty are not supported during PPO training. However,
    we do not find this to be a problem in practice. Increasing the
    sampling temperature and enabling top-k/top-p sampling can produce
    effective models.

    :param max_new_tokens: The maximum number of new tokens to generate.
    :type max_new_tokens: int
    :param min_new_tokens: The minimum number of new tokens to generate.
    :type min_new_tokens: int
    :param greedy: Whether to use greedy decoding.
    :type greedy: bool
    :param top_k: The number of highest probability tokens to keep.
    :type top_k: int
    :param top_p: The cumulative probability of the highest probability
        tokens to keep.
    :type top_p: float
    :param temperature: The temperature of the sampling process.
    :type temperature: float
    :param use_cuda_graph: Whether to use CUDA graph to reduce kernel
        launch overhead during generation.
    :type use_cuda_graph: bool
    :param force_cudagraph_recapture: Whether to capture the CUDA graph
        every time `generate` is called, even if the graph has been captured
        before. This will introduce minor overhead but will release the
        kvcache when not running generation.
    :type force_cudagraph_recapture: bool
    :param force_no_logits_mask: Whether to omit the logits mask. The logits
        mask is produced when using top-k or top-p sampling, marking tokens
        that are filtered out. This mask is used by the reference model and
        the actor model during training to align inferred logits with those
        during generation and produce accurate KLs. Using the logits mask with
        top-k/top-p sampling greatly improves the stability of PPO training
        by narrowing the action space. However, this benefit comes at the cost
        of additional GPU memory usage. If this option is set to True, the
        logits mask will be omitted to save GPU memory, which may lead to a
        decrease in learning performance.
    :type force_no_logits_mask: bool
    """

    max_new_tokens: int = 256
    min_new_tokens: int = 256
    greedy: bool = False
    top_p: float = 0.9
    top_k: int = 200
    temperature: float = 1.0
    use_cuda_graph: bool = False
    force_cudagraph_recapture: bool = True
    force_no_logits_mask: bool = False

    def __post_init__(self):
        if self.temperature == 0.0:
            self.greedy = True
            self.temperature = 1.0
        if self.top_p <= 0.0 or self.top_p > 1:
            raise ValueError("top_p must be in (0.0, 1.0].")
        if self.top_k <= 0:
            raise ValueError("top_k must be a positive integer.")

        if self.use_cuda_graph and Version(
            Version(torch.__version__).base_version
        ) < Version("2.3.0"):
            raise ValueError(
                f"To use CUDAGraph, ReaL's PyTorch version should be at least 2.3.0."
            )


@dataclasses.dataclass
class ReaLMoEConfig:
    """Configuration for MoE models.

    :param num_experts: The number of experts in the mixture of experts.
    :type num_experts: int
    :param top_k: The number of experts to route per token, also
        interpreted as the `top-k` routing parameter.
    :type top_k: int
    :param routing_type: The load balancing type for the MoE router. Can
        be "aux_loss", "sinkhorn", or "none".
    :type routing_type: str
    :param aux_loss_coeff: The coefficient for the auxiliary loss.
        Effective only when routing_type="aux_loss".
    :type aux_loss_coeff: float
    :param capacity_factor: The capacity factor of each expert. An
        expert will drop tokens if the number of tokens exceeds
        capacity_factor * (num_tokens / num_experts). No tokens will be
        dropped if capacity_factor is None.
    :type capacity_factor: float or None
    :param pad_to_capacity: Whether to pad the input to the capacity of
        the expert.
    :type pad_to_capacity: bool
    :param token_drop_policy: The token drop policy for the MoE. Can be
        either "prob" or "position". If "prob", the tokens with the
        lowest probabilities will be dropped. If "position", tokens at
        the end of each batch will be dropped.
    :type token_drop_policy: str
    :param z_loss_coeff: The coefficient for the z-loss.
    :type z_loss_coeff: float
    :param input_jitter_eps: The input jitter noise for the router.
    :type input_jitter_eps: float
    """

    num_experts: int = 8
    top_k: int = 2
    routing_type: str = "aux_loss"
    aux_loss_coeff: float = 1e-3
    capacity_factor: float = None
    pad_to_capacity: bool = False
    token_drop_policy: str = "probs"
    z_loss_coeff: float = 0.0
    input_jitter_eps: float = 0.0
    use_grouped_gemm: bool = False


@dataclasses.dataclass
class ReaLModelConfig:
    """Configuration for the ReaLModel.

    :param n_layers: The number of transformer blocks.
    :type n_layers: int
    :param n_kv_heads: The number of key-value attention heads.
    :type n_kv_heads: int
    :param n_q_heads: The number of query attention heads.
    :type n_q_heads: int
    :param head_dim: The dimension of each attention head.
        If None, it defaults to hidden_dim // n_q_heads.
        If specified, the query layer will have the shape
        (hidden_dim, head_dim * n_q_heads).
    :type head_dim: int or None
    :param hidden_dim: The hidden dimension of the transformer block.
    :type hidden_dim: int
    :param intermediate_dim: The dimension of the intermediate layer in the MLP.
    :type intermediate_dim: int
    :param vocab_size: The vocabulary size.
    :type vocab_size: int
    :param n_positions: The maximum context length. Can be None for
        rotary embedding, where the context length is determined during runtime.
    :type n_positions: Optional[int]
    :param embd_pdrop: The dropout probability for the embedding layer.
    :type embd_pdrop: float
    :param resid_pdrop: The dropout probability for the residual connections.
    :type resid_pdrop: float
    :param attn_pdrop: The dropout probability for the attention weights.
    :type attn_pdrop: float
    :param layer_norm_epsilon: The epsilon value for layer normalization.
    :type layer_norm_epsilon: float
    :param activation_function: The activation function for the MLP.
    :type activation_function: str
    :param scale_attn_by_inverse_layer_idx: Whether to scale the attention weights
        by the inverse of the layer index.
    :type scale_attn_by_inverse_layer_idx: bool
    :param use_attention_bias: Whether to use bias for QKV layers.
    :type use_attention_bias: bool
    :param use_attn_proj_bias: Whether to use bias for the attention projection layer.
    :type use_attn_proj_bias: bool
    :param layer_norm_type: The type of layer normalization. Can be None, "rms", or "gemma".
    :type layer_norm_type: Optional[str]
    :param mlp_type: The type of the MLP. Can be None, "llama", or "moe".
    :type mlp_type: Optional[str]
    :param apply_rotary: Whether to apply rotary embedding.
    :type apply_rotary: bool
    :param rotary_base: The exponential base for the rotary embedding.
    :type rotary_base: float
    :param rotary_interleaved: Whether to use interleaved rotary embedding.
    :type rotary_interleaved: bool
    :param rotary_scaling: The scaling factor for the rotary embedding.
    :type rotary_scaling: Optional[float]
    :param rotary_scaling_type: The type of scaling for the rotary embedding.
    :type rotary_scaling_type: Optional[str]
    :param normalize_embed: Whether to normalize the embeddings
        before passing them through the transformer blocks. Used by Gemma.
    :type normalize_embed: bool
    :param abs_position_embedding_offset: The offset for the absolute position embedding.
        Used by OPT, but OPT is currently not supported.
    :type abs_position_embedding_offset: int
    :param do_layernorm_before: Whether to apply layer normalization before the attention
        rather than after. Used by OPT, but OPT is currently not supported.
    :type do_layernorm_before: bool
    :param tied_embedding: Whether to share the embeddings and output weights.
        Used by models like GPT-2 and Gemma.
    :type tied_embedding: bool
    :param sliding_window: The sliding window size for the attention.
        Currently a placeholder and not supported.
    :type sliding_window: Optional[int]
    :param moe: Configuration for MoE models, only effective when mlp_type="moe".
    :type moe: Optional[ReaLMoEConfig]
    :param is_critic: Whether the model is a critic model.
    :type is_critic: bool
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
    # MoE Config
    moe: Optional[ReaLMoEConfig] = None

    # Whether it is a critic/reward model that outputs scores.
    is_critic: bool = False

    def __post_init__(self):
        if self.is_critic and self.tied_embedding:
            raise ValueError("Critic model cannot share embeddings and output weights.")
        if self.head_dim is None:
            self.head_dim = self.hidden_dim // self.n_q_heads


@dataclasses.dataclass
class ModelVersion:
    """A version counter.

    :param epoch: The current epoch.
    :type epoch: int
    :param epoch_step: The current step within the current epoch. A
        "step" refers to a traversal of the dataflow graph (DFG), which
        may include multiple model update steps depending on the
        interface (e.g., PPO mini-batched updates).
    :type epoch_step: int
    :param global_step: The total number of steps since the start of the
        experiment.
    :type global_step: int
    """

    epoch: int = 0
    epoch_step: int = 0
    global_step: int = 0


@dataclasses.dataclass
class FinetuneSpec:
    """The specification for the fine-tuning task.

    :param total_train_epochs: The total number of epochs for training.
    :type total_train_epochs: int
    :param total_train_steps: The total number of steps for training.
    :type total_train_steps: int
    :param steps_per_epoch: The number of steps per epoch.
    :type steps_per_epoch: int
    """

    total_train_epochs: int
    total_train_steps: int
    steps_per_epoch: int


class PipelinableEngine(abc.ABC):
    """Defines the signature for modules after backend initialization.

    Modules with this signature will be passed to :class:`ModelInterface`
    for model function call execution.

    See `inference.py
    <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/backend/inference.py>`_,
    `deepspeed.py
    <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/backend/deepspeed.py>`_,
    and `megatron.py
    <https://github.com/openpsi-project/ReaLHF/tree/main/realhf/impl/model/backend/megatron.py>`_
    for concrete implementations.
    """

    def train_batch(
        self,
        input_: SequenceSample,
        loss_fn: Callable[[torch.Tensor, SequenceSample], Tuple[torch.Tensor, Dict]],
        version_steps: int,
        num_micro_batches: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict] | None:
        """Update the model with a batch of data and a loss function.

        :param input_: The input data. It should contain at least the key ``packed_input_ids``,
            which includes the concatenated token sequences. It should also include any other
            entries required to compute the loss.
        :type input_: SequenceSample
        :param loss_fn: The loss function. It takes the output of the forward pass and the
            input data, returning the loss and a dictionary of statistics.
        :type loss_fn: Callable[[torch.Tensor, SequenceSample], Tuple[torch.Tensor, Dict]]
        :param version_steps: The global step counter for this experiment,
            used by the backend to determine the learning rate schedule.
        :type version_steps: int
        :param num_micro_batches: The number of micro-batches to split the batch into.
            Gradients will be accumulated across micro-batches, and only one update will
            occur. For pipelined training, micro-batches are processed together by the engine,
            which automatically schedules the forward and backward passes. For non-pipelined
            training, forward and backward passes are executed iteratively over mini-batches
            to accumulate gradients. If None, the batch will not be split.
        :type num_micro_batches: Optional[int]
        :return: The aggregated scalar loss and a dictionary of statistics from the last pipeline
            stage. Returns None otherwise.
        :rtype: Tuple[torch.Tensor, Dict]
        """
        raise NotImplementedError()

    @torch.no_grad()
    def eval_batch(
        self,
        input_: SequenceSample,
        loss_fn: Callable[[torch.Tensor, SequenceSample], Tuple[torch.Tensor, Dict]],
        num_micro_batches: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict] | None:
        """Evaluate the model using the forward pass and loss function.

        This method wraps :meth:`forward` with a customized ``post_hook`` and ``aggregate_fn``.

        :param input_: The input data. It should contain at least the key ``packed_input_ids``,
            which includes the concatenated token sequences. It should also include any other
            entries required to compute the loss.
        :type input_: SequenceSample
        :param loss_fn: The loss function. It takes the output of the forward pass and the
            input data, returning the loss and a dictionary of statistics.
        :type loss_fn: Callable[[torch.Tensor, SequenceSample], Tuple[torch.Tensor, Dict]]
        :param num_micro_batches: The number of micro-batches to split the batch into.
            **This argument is retained for compatibility, although it should not be used**,
            since different batch sizes can be directly set in the dataloader, and batch size
            during evaluation does not impact algorithmic performance like it does during training.
        :type num_micro_batches: Optional[int]
        :return: The aggregated scalar loss and a dictionary of statistics from the last pipeline
            stage. Returns None otherwise.
        :rtype: Tuple[torch.Tensor, Dict]
        """

        def agg(xs: List[Tuple[torch.Tensor, Dict]]):
            losses, stats = zip(*xs)
            return sum(losses), {k: sum(s[k] for s in stats) for k in stats[0].keys()}

        return self.forward(
            input_,
            post_hook=loss_fn,
            aggregate_fn=agg,
            num_micro_batches=num_micro_batches,
        )

    def forward(
        self,
        input_: SequenceSample,
        num_micro_batches: Optional[int] = None,
        post_hook: Callable[[torch.Tensor, SequenceSample], Any] | None = None,
        aggregate_fn: Callable[[List[Any]], Any] = torch.cat,
    ) -> Any | None:
        """Run the forward pass or inference on the model. Note that it is
        gradient-free.

        To train the model, use :meth:`train_batch` instead.

        :param input_: The input data. It should contain at least the key ``packed_input_ids``,
            which includes the concatenated token sequences.
        :type input_: SequenceSample
        :param num_micro_batches: The number of micro-batches to split the batch into.
            Regardless of pipelining, mini-batches will be fed into the module one-by-one.
            This approach helps reduce GPU memory usage of hidden states. If None, the batch
            will not be split.
        :type num_micro_batches: Optional[int]
        :param post_hook: A function to apply to the output after the forward pass.
            It takes the output tensor and the input data, returning an arbitrary result.
            With a post_hook, we can process the output in mini-batches,
            reducing memory usage for operations such as gathering log-probabilities.
            If None, this function just returns the output tensor.
        :type post_hook: Callable[[torch.Tensor, SequenceSample], Any] | None
        :param aggregate_fn: A function to aggregate the results of the post_hook.
        :type aggregate_fn: Callable[[List[Any]], Any]
        :return: The aggregated result of the post_hook from the last pipeline stage. Returns None otherwise.
            The output before post_hook is a concatenated tensor along the batch-sequence dimension, similar to
            ``packed_input_ids``. For example, if we have 3 sequences with lengths [2, 3, 4],
            and the vocabulary size is 1000, ``packed_input_ids`` should have shape [9],
            and the logits should have shape [9, 1000].
        :rtype: Any | None
        """
        raise NotImplementedError()

    def generate(
        self,
        input_: SequenceSample,
        tokenizer: transformers.PreTrainedTokenizerFast,
        gconfig: GenerationHyperparameters = dataclasses.field(
            default_factory=GenerationHyperparameters
        ),
        num_micro_batches: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
        """Generate outputs from the model.

        :param input_: The input data. It should contain at least the key ``packed_input_ids``,
            which includes the concatenated prompts.
        :type input_: SequenceSample
        :param tokenizer: The tokenizer for the model.
        :type tokenizer: transformers.PreTrainedTokenizerFast
        :param gconfig: The generation hyperparameters.
        :type gconfig: GenerationHyperparameters
        :param num_micro_batches: The number of micro-batches to split the batch into.
            Regardless of pipelining, mini-batches will be processed one-by-one by the module.
            This approach helps reduce GPU memory usage for hidden states and KV-caches.
            If None, the batch will not be split.
        :type num_micro_batches: Optional[int]
        :return: For the last pipeline stage, returns the generated tokens, log probabilities, and optionally the logits mask.
            See :class:`GenerationHyperparameters` for more details about the logits mask.
            Returns None for other stages.
            The outputs are stacked tensors along the batch dimension. For example,
            if we have 3 prompts with lengths [2, 3, 4], a maximum generated length of 5,
            and a vocabulary size of 1000, ``packed_input_ids`` should have shape [9],
            generated tokens and log probabilities should have shape [3, 5],
            and the logits should have shape [3, 5, 1000].
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None
        """
        raise NotImplementedError()


@dataclasses.dataclass
class Model:
    """A collection consisting of a neural network, a tokenizer, and metadata
    with a unique name.

    :param name: The unique name of the model.
    :type name: ModelName
    :param module: The neural network module. Its parameters may be
        sharded by tensor or pipeline parallelism.
    :type module: PipelinableEngine | torch.nn.Module
    :param tokenizer: The tokenizer associated with the model.
    :type tokenizer: transformers.PreTrainedTokenizerFast
    :param device: The device on which to run the model.
    :type device: Union[str, torch.device]
    :param dtype: The data type of the model. Defaults to torch.float16
        if None.
    :type dtype: Optional[torch.dtype]
    :param version: The version of the model.
    :type version: ModelVersion
    :param ft_spec: The fine-tuning specification for the model.
        Generally not used.
    :type ft_spec: FinetuneSpec
    """

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
    """A backend that wraps :class:`Model` to provide additional
    functionalities such as pipelined model function calls and ZeRO
    optimization.

    Current backend implementations include inference, DeepSpeed, and Megatron.
    The inference backend provides only inference and generation APIs,
    while the DeepSpeed and Megatron backends also support training.

    The backend offers two main functionalities:

    1. Pipelined generation, inference, and training, implemented in ReaL.

    2. ZeRO optimization, implemented in DeepSpeed and Megatron.

    After initialization, the ``module`` attribute in :class:`Model`
    will have the same signature as :class:`PipelinableEngine`.
    See ``realhf/impl/model/backend`` for concrete implementations.
    """

    @abc.abstractmethod
    def _initialize(self, model: Model, spec: FinetuneSpec) -> Model:
        raise NotImplementedError()

    def initialize(self, model: Model, spec: FinetuneSpec) -> Model:
        """Initialize the model with the backend to support pipelining and
        distributed optimization."""
        model.ft_spec = spec
        return self._initialize(model, spec)

    def destroy(self, model: Model):
        """Destroy the backend and release GPU memory."""
        pass


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
    """An interface for model training, evaluation, inference, and generation.

    This interface is designed to follow the dependency injection pattern.
    We pass the model to the interface and call its methods, ensuring that model APIs
    and algorithms are fully decoupled. For example, REINFORCE and PPO can exhibit
    different behaviors during training. Separate interfaces can be written for these
    algorithms while using the same model that provides basic forward-backward-update
    functionality (i.e., :class:`PipelinableEngine`).

    During runtime, the master worker requests model workers to execute a specific
    interface type (e.g., generate) on a specific model. The model worker locates
    the corresponding model, passes it into the requested interface, performs the
    computation, and returns the result.

    Users can easily create new interfaces to support customized usage.
    See :doc:`customization` for more details.
    """

    def save(self, model: Model, save_dir: str):
        pass

    def evaluate(
        self,
        model: Model,
        eval_dataloader: torch.utils.data.DataLoader,
    ) -> Dict:
        # NOTE: No n_mbs here because the batch size can be configured in the dataloader.
        return {}

    def inference(
        self, model: Model, data: SequenceSample, n_mbs: Optional[int] = None
    ) -> SequenceSample:
        raise NotImplementedError()

    def generate(
        self, model: Model, data: SequenceSample, n_mbs: Optional[int] = None
    ) -> SequenceSample:
        raise NotImplementedError()

    def train_step(
        self, model: Model, data: SequenceSample, n_mbs: Optional[int] = None
    ) -> Dict:
        raise NotImplementedError()

    # Mock methods for creating data and profiling an individual MFC.
    def _mock_generate(self, model: Model, data: SequenceSample):
        return data

    def _mock_inference(self, model: Model, data: SequenceSample):
        return data

    def _mock_train_step(self, model: Model, data: SequenceSample):
        return data

    def mock(
        self,
        type_: str,
        model: Model,
        data: SequenceSample,
    ) -> SequenceSample:
        if type_ == "generate":
            return self._mock_generate(model, data)
        elif type_ == "inference":
            return self._mock_inference(model, data)
        elif type_ == "train_step":
            return self._mock_train_step(model, data)
        else:
            raise ValueError(f"Unsupported interface type {type_}")


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
