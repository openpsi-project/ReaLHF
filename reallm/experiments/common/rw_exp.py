import dataclasses

from reallm.api.core.dfg import ModelInterface, ModelInterfaceType, ModelRPC
from reallm.api.core.system_api import *
from reallm.api.quickstart.dataset import PairedComparisonDatasetConfig
from reallm.api.quickstart.model import ModelTrainEvalConfig


@dataclasses.dataclass
class RWConfig(Experiment):
    total_train_epochs: int = 1
    save_freq_steps: Optional[int] = 20
    eval_freq_epochs: Optional[int] = 1
    is_sft_lora: bool = False
    sft_lora_path: Optional[str] = None
    model: ModelTrainEvalConfig = dataclasses.field(default_factory=ModelTrainEvalConfig)
    dataset: PairedComparisonDatasetConfig = dataclasses.field(default_factory=PairedComparisonDatasetConfig)

    def __post_init__(self):
        assert not self.is_sft_lora and self.sft_lora_path is None, "LoRA is not supported for now."

    @property
    def models(self):
        return {
            "default": self.model,
        }

    @property
    def rpcs(self):
        interface = ModelInterface("paired_rw")
        rpc = ModelRPC(
            model_name=ModelName("default", 0),
            interface_type=ModelInterfaceType.TRAIN_STEP,
            interface_impl=interface,
            model_type=self.model.type,
            input_data=["packed_input_ids", "input_lens", "group_factor", "pos_input_lens"],
            log_return_value=True,
            min_n_seqs=self.dataset.train_tokens_per_batch // self.dataset.max_seqlen,
            max_n_seqs=self.dataset.train_tokens_per_batch // self.dataset.max_seqlen,
        )
        return {rpc.name: rpc}

    @property
    def datasets(self):
        return [
            Dataset(
                "packed_rw_pair",
                args=dict(
                    n_tokens_per_batch=self.dataset.train_tokens_per_batch,
                    max_length=self.dataset.max_seqlen,
                    max_pairs_per_prompt=self.dataset.max_pairs_per_prompt,
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
                "packed_rw_pair",
                args=dict(
                    n_tokens_per_batch=self.dataset.valid_tokens_per_batch,
                    max_length=self.dataset.max_seqlen,
                    max_pairs_per_prompt=self.dataset.max_pairs_per_prompt,
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
