import dataclasses

from realhf.api.core.config import (
    DatasetAbstraction,
    ModelInterfaceAbstraction,
    ModelInterfaceType,
    ModelName,
)
from realhf.api.core.dfg import MFCDef
from realhf.api.core.model_api import GenerationHyperparameters
from realhf.api.quickstart.dataset import PromptOnlyDatasetConfig
from realhf.api.quickstart.device_mesh import AllocationConfig
from realhf.api.quickstart.entrypoint import register_quickstart_exp
from realhf.api.quickstart.model import ModelTrainEvalConfig
from realhf.experiments.common.common import CommonExperimentConfig


@dataclasses.dataclass
class GenerationConfig(CommonExperimentConfig):
    """Generation experiment configuration.

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

    @property
    def models(self):
        return {
            "default": self.model,
        }

    @property
    def rpcs(self):
        interface = ModelInterfaceAbstraction(
            "generation", args={"generation_config": self.gen}
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
            n_seqs=self.dataset.train_bs_n_seqs,
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
            DatasetAbstraction(
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


register_quickstart_exp("gen", GenerationConfig)
