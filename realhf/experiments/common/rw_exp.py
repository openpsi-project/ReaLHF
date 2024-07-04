import dataclasses
from typing import List

from realhf.api.core.dfg import MFCDef, ModelInterface, ModelInterfaceType
from realhf.api.core.system_api import *
from realhf.api.quickstart.dataset import PairedComparisonDatasetConfig
from realhf.api.quickstart.device_mesh import AllocationConfig
from realhf.api.quickstart.entrypoint import register_quickstart_exp
from realhf.api.quickstart.model import (
    ModelTrainEvalConfig,
    OptimizerConfig,
    get_real_model_config,
)
from realhf.experiments.common.common import CommonExperimentConfig


@dataclasses.dataclass
class RWConfig(CommonExperimentConfig):
    """Pairwise reward modeling experiment configuration.

    It is a subclass of :class:`CommonExperimentConfig`,
    so all CLI options in the base class are available.

    :param is_sft_lora: Whether LoRA was used for SFT.
        If so, the saved SFT model should only contain LoRA parameters.
        Since LoRA is currently not supported for SFT,
        this option is not used for now.
    :type is_sft_lora: bool
    :param sft_lora_path: Path to the LoRA model for SFT.
        Since LoRA is currently not supported for SFT,
        this option is not used for now.
    :param model: Model runtime configuration.
    :type model: ModelTrainEvalConfig
    :param allocation: Device allocation and parallelism configuration.
    :type allocation: AllocationConfig
    :param dataset: Dataset configuration.
    :type dataset: PairedComparisonDatasetConfig
    """

    is_sft_lora: bool = False
    sft_lora_path: Optional[str] = None
    model: ModelTrainEvalConfig = dataclasses.field(
        default_factory=ModelTrainEvalConfig
    )
    allocation: AllocationConfig = dataclasses.field(default_factory=AllocationConfig)

    dataset: PairedComparisonDatasetConfig = dataclasses.field(
        default_factory=PairedComparisonDatasetConfig
    )

    def __post_init__(self):
        assert (
            not self.is_sft_lora and self.sft_lora_path is None
        ), "LoRA is not supported for now."
        self.model.init_critic_from_actor = True

    @property
    def models(self):
        return {
            "default": self.model,
        }

    @property
    def rpcs(self):
        interface = ModelInterface("paired_rw")
        rpc = MFCDef(
            model_name=ModelName("default", 0),
            interface_type=ModelInterfaceType.TRAIN_STEP,
            interface_impl=interface,
            model_type=self.model.type,
            model_path=self.model.path,
            input_data=["packed_input_ids", "group_factor", "pos_input_lens"],
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
                "rw_pair",
                args=dict(
                    max_length=self.dataset.max_seqlen,
                    max_pairs_per_prompt=self.dataset.max_pairs_per_prompt,
                    dataset_path=self.dataset.train_path,
                ),
            )
        ]

    @property
    def eval_datasets(self):
        return [
            Dataset(
                "rw_pair",
                args=dict(
                    max_length=self.dataset.max_seqlen,
                    max_pairs_per_prompt=self.dataset.max_pairs_per_prompt,
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


register_quickstart_exp("rw", RWConfig)
