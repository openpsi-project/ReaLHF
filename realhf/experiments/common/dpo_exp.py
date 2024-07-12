import dataclasses
from typing import *

import realhf.base.logging as logging
from realhf.api.core.config import (
    DatasetAbstraction,
    ModelInterfaceAbstraction,
    ModelInterfaceType,
    ModelName,
)
from realhf.api.core.dfg import MFCDef
from realhf.api.quickstart.dataset import PairedComparisonDatasetConfig
from realhf.api.quickstart.device_mesh import AllocationConfig
from realhf.api.quickstart.entrypoint import register_quickstart_exp
from realhf.api.quickstart.model import ModelTrainEvalConfig
from realhf.experiments.common.common import CommonExperimentConfig

logger = logging.getLogger("DPO Experiment")


@dataclasses.dataclass
class DPOConfig(CommonExperimentConfig):
    """Direct Preference Optimization (DPO) experiment configuration.

    It is a subclass of :class:`CommonExperimentConfig`,
    so all CLI options in the base class are available.

    We don't implement runtime evaluation for DPO.

    :param is_sft_lora: Whether LoRA was used for SFT.
        If so, the saved SFT model should only contain LoRA parameters.
        Since LoRA is currently not supported for SFT,
        this option is not used for now.
    :type is_sft_lora: bool
    :param sft_lora_path: Path to the LoRA model for SFT.
        Since LoRA is currently not supported for SFT,
        this option is not used for now.
    :param actor: Runtime configuration of the primary LLM.
    :type actor: ModelTrainEvalConfig
    :param ref: Runtime configuration of the reference LLM.
        This model is only used for inference to provide KL regularization.
        In ReaL, this model is automatically offloaded to CPU.
        As a result, training DPO is as efficient as training a single LLM.
    :type ref: ModelTrainEvalConfig
    :param actor_train: Device allocation and parallelism configuration for
        running training on the primary LLM.
    :type actor_train: AllocationConfig
    :param ref_inf: Device allocation and parallelism configuration for
        running inference on the reference LLM.
        This can be different from the training allocation.
        A large data parallel degree with additional pipelining
        can be faster for inference.
    :type ref_inf: AllocationConfig
    :param dataset: Dataset configuration. The same for reward modeling.
    :type dataset: PairedComparisonDatasetConfig
    :param beta: KL regularization coefficient.
    :type beta: float
    """

    is_sft_lora: bool = False
    sft_lora_path: Optional[str] = None

    actor: ModelTrainEvalConfig = dataclasses.field(
        default_factory=ModelTrainEvalConfig
    )
    ref: ModelTrainEvalConfig = dataclasses.field(default_factory=ModelTrainEvalConfig)

    actor_train: AllocationConfig = dataclasses.field(default_factory=AllocationConfig)
    ref_inf: AllocationConfig = dataclasses.field(default_factory=AllocationConfig)

    dataset: PairedComparisonDatasetConfig = dataclasses.field(
        default_factory=PairedComparisonDatasetConfig
    )
    beta: float = 0.1

    def __post_init__(self):
        assert (
            not self.is_sft_lora and self.sft_lora_path is None
        ), "LoRA is not supported for now."

    @property
    def models(self):
        return {
            "actor": self.actor,
            "ref": self.ref,
        }

    @property
    def rpcs(self):
        interface = ModelInterfaceAbstraction(
            "dpo", args=dict(beta=self.beta, enable_save=True)
        )
        ref_interface = ModelInterfaceAbstraction(
            "dpo", args=dict(beta=self.beta, enable_save=False)
        )
        ref_inf = MFCDef(
            name="ref_inf",
            model_name=ModelName("ref", 0),
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=ref_interface,
            model_type=self.ref.type,
            model_path=self.ref.path,
            input_keys=[
                "packed_input_ids",
                "prompt_lens",
            ],
            output_keys=["seqlogp"],
            n_seqs=self.dataset.train_bs_n_seqs,
        )
        dpo = MFCDef(
            name="actor_train",
            model_name=ModelName("actor", 0),
            interface_type=ModelInterfaceType.TRAIN_STEP,
            interface_impl=interface,
            model_type=self.actor.type,
            model_path=self.actor.path,
            input_keys=[
                "packed_input_ids",
                "seqlogp",
                "prompt_lens",
            ],
            log_return_value=True,
            n_seqs=self.dataset.train_bs_n_seqs,
        )
        return {
            "actor_train": dpo,
            "ref_inf": ref_inf,
        }

    @property
    def allocations(self):
        return {
            "actor_train": self.actor_train,
            "ref_inf": self.ref_inf,
        }

    @property
    def datasets(self):
        return [
            DatasetAbstraction(
                "rw_pair",
                args=dict(
                    max_length=self.dataset.max_seqlen,
                    max_pairs_per_prompt=self.dataset.max_pairs_per_prompt,
                    dataset_path=self.dataset.train_path,
                ),
            )
        ]

    @property
    def tokenizer_name_or_path(self):
        return self.actor.path


register_quickstart_exp("dpo", DPOConfig)
