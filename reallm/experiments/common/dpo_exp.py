import dataclasses

from reallm.api.core.dfg import ModelInterface, ModelInterfaceType, ModelRPC
from reallm.api.core.system_api import *
from reallm.api.quickstart.dataset import PairedComparisonDatasetConfig
from reallm.api.quickstart.model import ModelTrainEvalConfig
from reallm.experiments.common.common import CommonExperimentConfig
import reallm.base.logging as logging

logger = logging.getLogger("DPO Experiment")


@dataclasses.dataclass
class DPOConfig(CommonExperimentConfig):
    total_train_epochs: int = 1
    save_freq_steps: Optional[int] = 20
    is_sft_lora: bool = False
    sft_lora_path: Optional[str] = None

    actor: ModelTrainEvalConfig = dataclasses.field(default_factory=ModelTrainEvalConfig)
    ref: ModelTrainEvalConfig = dataclasses.field(default_factory=ModelTrainEvalConfig)

    dataset: PairedComparisonDatasetConfig = dataclasses.field(default_factory=PairedComparisonDatasetConfig)
    beta: float = 0.1

    def __post_init__(self):
        assert not self.is_sft_lora and self.sft_lora_path is None, "LoRA is not supported for now."

    @property
    def models(self):
        return {
            "actor": self.actor,
            "ref": self.ref,
        }

    @property
    def rpcs(self):
        interface = ModelInterface("dpo", args=dict(beta=self.beta, enable_save=True))
        ref_interface = ModelInterface("dpo", args=dict(beta=self.beta, enable_save=False))
        ref_inf = ModelRPC(
            model_name=ModelName("ref", 0),
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=ref_interface,
            model_type=self.ref.type,
            model_path=self.ref.path,
            input_data=["packed_input_ids", "input_lens", "pos_input_lens", "prompt_lens"],
            output_data=["seqlogp"],
            min_n_seqs=self.dataset.train_tokens_per_batch // self.dataset.max_seqlen,
            max_n_seqs=self.dataset.train_tokens_per_batch // self.dataset.max_seqlen,
        )
        dpo = ModelRPC(
            model_name=ModelName("actor", 0),
            interface_type=ModelInterfaceType.TRAIN_STEP,
            interface_impl=interface,
            model_type=self.actor.type,
            model_path=self.actor.path,
            input_data=[
                "packed_input_ids",
                "input_lens",
                "pos_input_lens",
                "seqlogp",
                "prompt_lens",
            ],
            log_return_value=True,
            min_n_seqs=self.dataset.train_tokens_per_batch // self.dataset.max_seqlen,
            max_n_seqs=self.dataset.train_tokens_per_batch // self.dataset.max_seqlen,
        )
        rpcs = [dpo, ref_inf]
        return {rpc.name: rpc for rpc in rpcs}

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
    def dataloader(self) -> DataLoader:
        return DataLoader("iterable_dataset_loader")

    @property
    def tokenizer_name_or_path(self):
        return self.actor.path

    @property
    def exp_ctrl(self):
        return ExperimentSaveEvalControl(
            total_train_epochs=self.total_train_epochs,
            save_frequency_steps=self.save_freq_steps,
        )
