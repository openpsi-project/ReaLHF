import dataclasses

from omegaconf import MISSING

from realhf.api.core.dfg import MFCDef, ModelInterface, ModelInterfaceType
from realhf.api.core.system_api import *
from realhf.api.quickstart.dataset import PromptOnlyDatasetConfig
from realhf.api.quickstart.device_mesh import AllocationConfig
from realhf.api.quickstart.entrypoint import register_quickstart_exp
from realhf.api.quickstart.model import ModelTrainEvalConfig
from realhf.experiments.common.common import CommonExperimentConfig


@dataclasses.dataclass
class GenerationHyperparameters:
    """Generation hyperparameters.

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
    :param use_cuda_graph: Whether to use CUDA graph.
    :type use_cuda_graph: bool
    """

    max_new_tokens: int = 256
    min_new_tokens: int = 256
    greedy: bool = False
    top_p: float = 0.9
    top_k: int = 200
    temperature: float = 1.0
    use_cuda_graph: bool = False


@dataclasses.dataclass
class GenerationConfig(CommonExperimentConfig):
    """Generation experiment configuration. Used for testing only.

    It is a subclass of :class:`CommonExperimentConfig`,
    so all CLI options in the base class are available.

    :param model: Model runtime configuration.
    :type model: ModelTrainEvalConfig
    :param gen_params: Generation hyperparameters.
    :type gen_params: GenerationHyperparameters
    :param dataset: Dataset configuration
    :type dataset: PromptOnlyDatasetConfig
    :param allocation: Device allocation and parallelism configuration.
    :type allocation: AllocationConfig
    """

    model: ModelTrainEvalConfig = dataclasses.field(
        default_factory=ModelTrainEvalConfig
    )
    gen: GenerationHyperparameters = dataclasses.field(
        default_factory=GenerationHyperparameters
    )
    dataset: PromptOnlyDatasetConfig = dataclasses.field(
        default_factory=PromptOnlyDatasetConfig
    )
    allocation: AllocationConfig = dataclasses.field(default_factory=AllocationConfig)

    def __post_init__(self):
        if self.gen.use_cuda_graph:
            os.environ["USE_CUDA_GRAPH"] = "1"

    @property
    def models(self):
        return {
            "default": self.model,
        }

    @property
    def rpcs(self):
        generation_kwargs = {
            "max_new_tokens": self.gen.max_new_tokens,
            "min_new_tokens": self.gen.min_new_tokens,
            "greedy": self.gen.greedy,
            "top_k": self.gen.top_k,
            "top_p": self.gen.top_p,
            "temperature": self.gen.temperature,
        }
        interface = ModelInterface(
            "generation", args={"generation_config": generation_kwargs}
        )
        gen = MFCDef(
            model_name=ModelName("default", 0),
            interface_type=ModelInterfaceType.GENERATE,
            model_type=self.model,
            model_path=self.model,
            interface_impl=interface,
            input_data=["packed_prompts"],
            balanced_dp=True,
            log_return_value=True,
            min_n_seqs=self.dataset.train_bs_n_seqs,
            max_n_seqs=self.dataset.train_bs_n_seqs,
        )
        return {"default": gen}

    @property
    def allocations(self):
        return {"default": self.allocation}

    @property
    def max_prompt_len(self):
        return self.dataset.max_prompt_len

    @property
    def datasets(self):
        return [
            Dataset(
                "prompt",
                args=dict(
                    dataset_path=self.dataset.path,
                    max_length=self.dataset.max_prompt_len,
                    pad_to_max_length=self.dataset.pad_to_max_length,
                ),
            )
        ]

    @property
    def allocations(self):
        return {"default": self.allocation}

    @property
    def tokenizer_name_or_path(self):
        return self.model.path

    @property
    def exp_ctrl(self):
        return ExperimentSaveEvalControl(
            total_train_epochs=1,
            save_frequency_steps=None,
            eval_frequency_epochs=None,
        )


register_quickstart_exp("gen", GenerationConfig)
