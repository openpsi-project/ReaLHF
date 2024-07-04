import dataclasses

from omegaconf import MISSING

from realhf.api.core.dfg import MFCDef, ModelFamily, ModelInterface, ModelInterfaceType
from realhf.api.core.system_api import *
from realhf.api.quickstart.dataset import PromptAnswerDatasetConfig
from realhf.api.quickstart.device_mesh import AllocationConfig
from realhf.api.quickstart.entrypoint import register_quickstart_exp
from realhf.api.quickstart.model import (
    ModelTrainEvalConfig,
    OptimizerConfig,
    get_real_model_config,
)
from realhf.experiments.common.common import CommonExperimentConfig


@dataclasses.dataclass
class SFTConfig(CommonExperimentConfig):
    """SFT experiment configuration.

    It is a subclass of :class:`CommonExperimentConfig`,
    so all CLI options in the base class are available.

    :param model: Model runtime configuration.
    :type model: ModelTrainEvalConfig
    :param allocation: Device allocation and parallelism configuration.
    :type allocation: AllocationConfig
    :param dataset: Dataset configuration
    :type dataset: PromptAnswerDatasetConfig
    """

    model: ModelTrainEvalConfig = dataclasses.field(
        default_factory=ModelTrainEvalConfig
    )
    allocation: AllocationConfig = dataclasses.field(default_factory=AllocationConfig)
    dataset: PromptAnswerDatasetConfig = dataclasses.field(
        default_factory=PromptAnswerDatasetConfig
    )

    @property
    def models(self):
        return {
            "default": self.model,
        }

    @property
    def rpcs(self):
        interface = ModelInterface("sft")
        rpc = MFCDef(
            model_name=ModelName("default", 0),
            interface_type=ModelInterfaceType.TRAIN_STEP,
            interface_impl=interface,
            model_type=self.model.type,
            model_path=self.model.path,
            input_data=["packed_input_ids", "prompt_mask"],
            log_return_value=True,
            min_n_seqs=self.dataset.train_bs_n_seqs,
            max_n_seqs=self.dataset.train_bs_n_seqs,
        )
        return {"default": rpc}

    @property
    def allocations(self):
        return {"default": self.allocation}

    @property
    def datasets(self):
        return [
            Dataset(
                "prompt_answer",
                args=dict(
                    max_length=self.dataset.max_seqlen,
                    dataset_path=self.dataset.train_path,
                ),
            )
        ]

    @property
    def allocations(self):
        return {"default": self.allocation}

    @property
    def eval_datasets(self):
        return [
            Dataset(
                "prompt_answer",
                args=dict(
                    max_length=self.dataset.max_seqlen,
                    dataset_path=self.dataset.valid_path,
                ),
            )
        ]

    @property
    def eval_dataloader(self):
        return DataLoader(
            "packed_eval", args=dict(batch_size=self.dataset.valid_bs_n_seqs)
        )

    @property
    def tokenizer_name_or_path(self):
        return self.model.path


register_quickstart_exp("sft", SFTConfig)
