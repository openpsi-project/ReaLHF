import dataclasses

from omegaconf import MISSING

from reallm.api.core.dfg import ModelFamily, ModelInterface, ModelInterfaceType, ModelRPC
from reallm.api.core.system_api import *
from reallm.api.quickstart.dataset import PromptAnswerDatasetConfig
from reallm.api.quickstart.model import ModelTrainEvalConfig
from reallm.experiments.common.common import CommonExperimentConfig


@dataclasses.dataclass
class SFTConfig(CommonExperimentConfig):
    seed: int = 1
    total_train_epochs: int = 1
    save_freq_steps: Optional[int] = 50
    eval_freq_epochs: Optional[int] = 1
    model: ModelTrainEvalConfig = dataclasses.field(default_factory=ModelTrainEvalConfig)
    dataset: PromptAnswerDatasetConfig = dataclasses.field(default_factory=PromptAnswerDatasetConfig)

    @property
    def models(self):
        return {
            "default": self.model,
        }

    @property
    def rpcs(self):
        interface = ModelInterface("sft")
        rpc = ModelRPC(
            model_name=ModelName("default", 0),
            interface_type=ModelInterfaceType.TRAIN_STEP,
            interface_impl=interface,
            model_type=self.model.type,
            model_path=self.model.path,
            input_data=["packed_input_ids", "cu_seqlens", "prompt_mask"],
            log_return_value=True,
            min_n_seqs=self.dataset.train_tokens_per_batch // self.dataset.max_seqlen,
            max_n_seqs=self.dataset.train_tokens_per_batch // self.dataset.max_seqlen,
        )
        return {rpc.name: rpc}

    @property
    def datasets(self):
        return [
            Dataset(
                "packed_prompt_answer",
                args=dict(
                    n_tokens_per_batch=self.dataset.train_tokens_per_batch,
                    min_seqs_per_batch=self.dataset.train_tokens_per_batch // self.dataset.max_seqlen,
                    max_length=self.dataset.max_seqlen,
                    dataset_path=self.dataset.train_path,
                ),
            )
        ]

    @property
    def dataloader(self):
        return DataLoader("iterable_dataset_loader")

    @property
    def eval_datasets(self):
        return [
            Dataset(
                "packed_prompt_answer",
                args=dict(
                    n_tokens_per_batch=self.dataset.valid_tokens_per_batch,
                    min_seqs_per_batch=self.dataset.valid_tokens_per_batch // self.dataset.max_seqlen,
                    max_length=self.dataset.max_seqlen,
                    dataset_path=self.dataset.valid_path,
                ),
            )
        ]

    @property
    def eval_dataloader(self):
        return DataLoader("iterable_dataset_loader")

    @property
    def tokenizer_name_or_path(self):
        return self.model.path

    @property
    def exp_ctrl(self):
        return ExperimentSaveEvalControl(
            total_train_epochs=self.total_train_epochs,
            save_frequency_steps=self.save_freq_steps,
            eval_frequency_epochs=self.eval_freq_epochs,
        )
