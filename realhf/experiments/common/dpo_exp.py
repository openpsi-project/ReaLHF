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
from realhf.api.quickstart.device_mesh import MFCConfig
from realhf.api.quickstart.entrypoint import register_quickstart_exp
from realhf.api.quickstart.model import ModelTrainEvalConfig
from realhf.experiments.common.common import CommonExperimentConfig

logger = logging.getLogger("DPO Experiment")


@dataclasses.dataclass
class DPOConfig(CommonExperimentConfig):
    """Configuration for Direct Preference Optimization (DPO) experiments.

    This class is a subclass of :class:`CommonExperimentConfig`,
    so all CLI options from the base class are available.

    Note that runtime evaluation is not implemented for DPO.

    :param is_sft_lora: Whether LoRA was used for SFT.
        If LoRA was used, the saved SFT model should only contain LoRA parameters.
        Since LoRA is currently not supported for SFT, this option is not utilized at present.
    :type is_sft_lora: bool
    :param sft_lora_path: Path to the LoRA model for SFT.
        Since LoRA is currently not supported for SFT, this option is not utilized at present.
    :type sft_lora_path: str or None
    :param actor: Runtime configuration for the primary LLM.
    :type actor: ModelTrainEvalConfig
    :param ref: Runtime configuration for the reference LLM.
        This model is used only for inference to provide KL regularization.
        In ReaL, this model is automatically offloaded to CPU,
        making DPO training as efficient as training a single LLM.
    :type ref: ModelTrainEvalConfig
    :param actor_train: Device allocation and parallelism configuration for training on the primary LLM.
    :type actor_train: MFCConfig
    :param ref_inf: Device allocation and parallelism configuration for inference on the reference LLM.
        This configuration can differ from the training allocation.
        A larger data parallel degree with additional pipelining can improve inference performance.
    :type ref_inf: MFCConfig
    :param dataset: Configuration for the dataset, which is the same as for reward modeling.
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

    actor_train: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    ref_inf: MFCConfig = dataclasses.field(default_factory=MFCConfig)

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
            n_mbs=self.ref_inf.n_mbs,
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
            n_mbs=self.actor_train.n_mbs,
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
