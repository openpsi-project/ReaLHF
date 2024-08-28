import dataclasses

from realhf.api.core.config import (
    DataLoaderAbstraction,
    DatasetAbstraction,
    ModelInterfaceAbstraction,
    ModelInterfaceType,
    ModelName,
)
from realhf.api.core.dfg import MFCDef
from realhf.api.quickstart.dataset import PromptAnswerDatasetConfig
from realhf.api.quickstart.device_mesh import MFCConfig
from realhf.api.quickstart.entrypoint import register_quickstart_exp
from realhf.api.quickstart.model import ModelTrainEvalConfig
from realhf.experiments.common.common import CommonExperimentConfig


@dataclasses.dataclass
class SFTConfig(CommonExperimentConfig):
    """Configuration for SFT experiments.

    This class is a subclass of :class:`CommonExperimentConfig`,
    so all CLI options from the base class are available.

    :param model: Configuration for model runtime.
    :type model: ModelTrainEvalConfig
    :param allocation: Configuration for device allocation and parallelism.
    :type allocation: MFCConfig
    :param dataset: Configuration for the dataset.
    :type dataset: PromptAnswerDatasetConfig
    """

    model: ModelTrainEvalConfig = dataclasses.field(
        default_factory=ModelTrainEvalConfig
    )
    allocation: MFCConfig = dataclasses.field(default_factory=MFCConfig)
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
        rpc = MFCDef(
            n_seqs=self.dataset.train_bs_n_seqs,
            name="trainDefault",
            n_mbs=self.allocation.n_mbs,
            interface_type=ModelInterfaceType.TRAIN_STEP,
            interface_impl=ModelInterfaceAbstraction("sft"),
            model_name="default",
            input_keys=["packed_input_ids", "prompt_mask"],
            log_return_value=True,
            model_type=self.model.type,
            model_path=self.model.path,
        )
        return {"trainDefault": rpc}

    @property
    def allocations(self):
        return {"trainDefault": self.allocation}

    @property
    def datasets(self):
        return [
            DatasetAbstraction(
                "prompt_answer",
                args=dict(
                    max_length=self.dataset.max_seqlen,
                    dataset_path=self.dataset.train_path,
                    pad_to_max_length=self.dataset.pad_to_max_length,
                ),
            )
        ]

    @property
    def eval_datasets(self):
        return [
            DatasetAbstraction(
                "prompt_answer",
                args=dict(
                    max_length=self.dataset.max_seqlen,
                    dataset_path=self.dataset.valid_path,
                ),
            )
        ]

    @property
    def eval_dataloader(self):
        return DataLoaderAbstraction(
            "packed_eval", args=dict(batch_size=self.dataset.valid_bs_n_seqs)
        )

    @property
    def tokenizer_name_or_path(self):
        return self.model.path


register_quickstart_exp("sft", SFTConfig)
