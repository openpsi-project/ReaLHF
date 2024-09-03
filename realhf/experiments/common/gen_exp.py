import dataclasses

from omegaconf import OmegaConf

from realhf.api.core.config import (
    DatasetAbstraction,
    ModelInterfaceAbstraction,
    ModelInterfaceType,
    ModelName,
)
from realhf.api.core.data_api import DatasetUtility, load_hf_tokenizer
from realhf.api.core.dfg import MFCDef
from realhf.api.core.model_api import GenerationHyperparameters
from realhf.api.quickstart.dataset import PromptOnlyDatasetConfig
from realhf.api.quickstart.device_mesh import MFCConfig
from realhf.api.quickstart.entrypoint import register_quickstart_exp
from realhf.api.quickstart.model import ModelTrainEvalConfig
from realhf.experiments.common.common import CommonExperimentConfig


@dataclasses.dataclass
class GenerationConfig(CommonExperimentConfig):
    """Configuration for generation experiments.

    This class is a subclass of :class:`CommonExperimentConfig`,
    so all CLI options from the base class are available.

    :param model: Runtime configuration for the model.
    :type model: ModelTrainEvalConfig
    :param gen_params: Hyperparameters for generation.
    :type gen_params: GenerationHyperparameters
    :param dataset: Configuration for the dataset.
    :type dataset: PromptOnlyDatasetConfig
    :param allocation: Configuration for device allocation and parallelism.
    :type allocation: MFCConfig
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
    allocation: MFCConfig = dataclasses.field(default_factory=MFCConfig)

    output_file: str = "output.jsonl"

    def __post_init__(self):
        from realhf.impl.dataset.prompt_dataset import PromptDataset

        util = DatasetUtility(
            seed=0,
            dp_rank=0,
            world_size=1,
            tokenizer=load_hf_tokenizer(self.model.path),
        )
        d = PromptDataset(
            util, max_length=self.dataset.max_prompt_len, dataset_path=self.dataset.path
        )
        if len(d) % self.dataset.train_bs_n_seqs != 0:
            raise ValueError(
                f"The size of the dataset must be a multiple of batch size for generation. "
                f"Otherwise the final batch will be dropped. Please pad your dataset size with random prompts. "
                f"Current dataset size: {len(d)}, batch size: {self.dataset.train_bs_n_seqs}."
            )
        if self.output_file is not None:
            if not self.output_file.endswith(".jsonl"):
                raise ValueError("Output path must end with .jsonl")
            if "/" in self.output_file:
                raise ValueError(
                    "Output path must not contain '/'. It should be a simple "
                    "filename that will be saved to the logging directory."
                )

    @property
    def models(self):
        return {
            "default": self.model,
        }

    @property
    def rpcs(self):
        # NOTE: to_container converts the object to a dict
        # It is used for unifying the profiling API, which requires to
        # pass external interface configurations in the launch command.
        # Customized dataclass objects will not work in that case.
        interface = ModelInterfaceAbstraction(
            "generation",
            args={
                "generation_config": OmegaConf.to_container(self.gen, resolve=True),
                "output_file": self.output_file,
            },
        )
        gen = MFCDef(
            name="gen",
            n_mbs=self.allocation.n_mbs,
            model_name=ModelName("default", 0),
            interface_type=ModelInterfaceType.GENERATE,
            model_type=self.model,
            model_path=self.model,
            interface_impl=interface,
            input_keys=["packed_prompts"],
            balanced_dp=True,
            log_return_value=True,
            n_seqs=self.dataset.train_bs_n_seqs,
        )
        return {"gen": gen}

    @property
    def allocations(self):
        return {"gen": self.allocation}

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
                ),
            )
        ]

    @property
    def tokenizer_name_or_path(self):
        return self.model.path


register_quickstart_exp("gen", GenerationConfig)
