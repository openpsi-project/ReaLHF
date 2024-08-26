import copy
import dataclasses
from typing import *

from omegaconf import OmegaConf

import realhf.api.core.model_api as model_api
import realhf.base.logging as logging
from examples.new_algorithms.reinforce.reinforce_interface import ReinforceInterface
from realhf.api.core.config import (
    DatasetAbstraction,
    ModelInterfaceAbstraction,
    ModelInterfaceType,
)
from realhf.api.core.dfg import MFCDef
from realhf.api.core.model_api import GenerationHyperparameters
from realhf.api.quickstart.dataset import PromptOnlyDatasetConfig
from realhf.api.quickstart.device_mesh import DeviceMesh, MFCConfig, RPCAllocation
from realhf.api.quickstart.entrypoint import register_quickstart_exp
from realhf.api.quickstart.model import ModelTrainEvalConfig, ParallelismConfig
from realhf.experiments.common.common import CommonExperimentConfig

logger = logging.getLogger("REINFORCE exp", "colored")


@dataclasses.dataclass
class ReinforceConfig(CommonExperimentConfig):
    """The REINFORCE or ReMax (https://arxiv.org/abs/2310.10505) algorithm."""

    actor: ModelTrainEvalConfig = dataclasses.field(
        default_factory=ModelTrainEvalConfig
    )
    rew: ModelTrainEvalConfig = dataclasses.field(default_factory=ModelTrainEvalConfig)

    # for manual allocation only
    actor_train: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    greedy_gen: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    sample_gen: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    greedy_rew_inf: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    sample_rew_inf: MFCConfig = dataclasses.field(default_factory=MFCConfig)

    dataset: PromptOnlyDatasetConfig = dataclasses.field(
        default_factory=PromptOnlyDatasetConfig
    )

    gen: GenerationHyperparameters = dataclasses.field(
        default_factory=GenerationHyperparameters
    )

    reward_output_scaling: float = 1.0
    reward_output_bias: float = 0.0
    discount: float = 0.99
    adv_norm: bool = True

    @property
    def models(self) -> Dict[str, ModelTrainEvalConfig]:
        # role to config
        return {
            "actor": self.actor,
            "reward": self.rew,
        }

    @property
    def rpcs(self):
        # interfaces
        actor_sample_interface = ModelInterfaceAbstraction(
            "reinforce",
            args=dict(
                # NOTE: to_container converts the object to a dict
                # It is used for unifying the profiling API, which requires to
                # pass external interface configurations in the launch command.
                # Customized dataclass objects will not work in that case.
                generation_config=OmegaConf.to_container(self.gen, resolve=True),
                discount=self.discount,
                adv_norm=self.adv_norm,
            ),
        )
        actor_greedy_interface = copy.deepcopy(actor_sample_interface)
        actor_greedy_interface.args["force_greedy"] = True

        rw_interface = ModelInterfaceAbstraction(
            "paired_rw",
            args=dict(
                output_scaling=self.reward_output_scaling,
                output_bias=self.reward_output_bias,
                enable_save=False,
            ),
        )

        sample_gen = MFCDef(
            name="sample_gen",
            model_name="actor",
            n_mbs=self.sample_gen.n_mbs,
            interface_type=ModelInterfaceType.GENERATE,
            model_type=self.actor.type,
            model_path=self.actor.path,
            interface_impl=actor_sample_interface,
            input_keys=["packed_prompts"],
            input_key_remap=dict(packed_prompts="packed_input_ids"),
            output_keys=[
                "packed_input_ids",
                "prompt_mask",
            ],
            balanced_dp=True,
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        greedy_gen = MFCDef(
            name="greedy_gen",
            model_name="actor",
            n_mbs=self.greedy_gen.n_mbs,
            interface_type=ModelInterfaceType.GENERATE,
            model_type=self.actor.type,
            model_path=self.actor.path,
            interface_impl=actor_greedy_interface,
            input_keys=["packed_prompts"],
            input_key_remap=dict(packed_prompts="packed_input_ids"),
            output_keys=[
                "greedy_packed_input_ids",
            ],
            output_key_remap=dict(packed_input_ids="greedy_packed_input_ids"),
            balanced_dp=True,
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        sample_rw = MFCDef(
            name="sample_rw",
            model_name="reward",
            n_mbs=self.sample_rew_inf.n_mbs,
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=rw_interface,
            model_type=self.rew.type,
            model_path=self.rew.path,
            input_keys=["packed_input_ids"],
            output_keys=["rewards"],
            n_seqs=self.dataset.train_bs_n_seqs,
        )
        greedy_rw = MFCDef(
            name="greedy_rw",
            model_name="reward",
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=rw_interface,
            n_mbs=self.greedy_rew_inf.n_mbs,
            model_type=self.rew.type,
            model_path=self.rew.path,
            input_keys=["greedy_packed_input_ids"],
            input_key_remap={"greedy_packed_input_ids": "packed_input_ids"},
            output_keys=["greedy_rewards"],
            output_key_remap={"rewards": "greedy_rewards"},
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        actor_train = MFCDef(
            name="actor_train",
            model_name="actor",
            n_mbs=self.actor_train.n_mbs,
            interface_type=ModelInterfaceType.TRAIN_STEP,
            model_type=self.actor.type,
            model_path=self.actor.path,
            interface_impl=actor_sample_interface,
            input_keys=[
                "packed_input_ids",
                "rewards",
                "greedy_rewards",
                "prompt_mask",
            ],
            log_return_value=True,
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        return dict(
            sample_gen=sample_gen,
            greedy_gen=greedy_gen,
            sample_rw=sample_rw,
            greedy_rw=greedy_rw,
            actor_train=actor_train,
        )

    @property
    def allocations(self):
        return dict(
            sample_gen=self.sample_gen,
            greedy_gen=self.greedy_gen,
            sample_rw=self.sample_rew_inf,
            greedy_rw=self.greedy_rew_inf,
            actor_train=self.actor_train,
        )

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
    def tokenizer_name_or_path(self) -> str:
        return self.actor.path

    @property
    def max_prompt_len(self):
        return self.dataset.max_prompt_len


model_api.register_interface("reinforce", ReinforceInterface)
register_quickstart_exp("reinforce", ReinforceConfig)

if __name__ == "__main__":

    from realhf.apps.quickstart import main

    main()
