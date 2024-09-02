import copy
import dataclasses
import math
from typing import *

import numpy as np

import realhf.base.logging as logging
from examples.new_algorithms.grpo.grpo_interface import GRPOInterface
from realhf.api.core.config import (
    DatasetAbstraction,
    ModelInterfaceAbstraction,
    ModelInterfaceType,
)
from realhf.api.core.dfg import MFCDef
from realhf.api.core.model_api import register_interface
from realhf.api.quickstart.dataset import PromptOnlyDatasetConfig
from realhf.api.quickstart.device_mesh import DeviceMesh, MFCConfig, RPCAllocation
from realhf.api.quickstart.entrypoint import register_quickstart_exp
from realhf.api.quickstart.model import ModelTrainEvalConfig, ParallelismConfig
from realhf.experiments.common.common import CommonExperimentConfig
from realhf.experiments.common.ppo_exp import PPOHyperparameters

logger = logging.getLogger("GRPO exp", "colored")


@dataclasses.dataclass
class GRPOConfig(CommonExperimentConfig):
    """The GRPO algorithm proposed in https://arxiv.org/abs/2402.03300."""

    actor: ModelTrainEvalConfig = dataclasses.field(
        default_factory=ModelTrainEvalConfig
    )
    ref: ModelTrainEvalConfig = dataclasses.field(default_factory=ModelTrainEvalConfig)
    rew: ModelTrainEvalConfig = dataclasses.field(default_factory=ModelTrainEvalConfig)

    # for manual allocation only
    actor_train: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    actor_gen: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    rew_inf: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    ref_inf: MFCConfig = dataclasses.field(default_factory=MFCConfig)

    dataset: PromptOnlyDatasetConfig = dataclasses.field(
        default_factory=PromptOnlyDatasetConfig
    )

    ppo: PPOHyperparameters = dataclasses.field(default_factory=PPOHyperparameters)
    group_size: int = 4

    def __post_init__(self):

        self.ppo_kwargs = dict(
            n_minibatches=self.ppo.ppo_n_minibatches,
            kl_ctl=self.ppo.kl_ctl,
            discount=self.ppo.discount,
            eps_clip=self.ppo.eps_clip,
            max_reward_clip=self.ppo.max_reward_clip,
            adaptive_kl_ctl=self.ppo.use_adaptive_kl_ctl,
        )

    @property
    def models(self) -> Dict[str, ModelTrainEvalConfig]:
        # role to config
        return {
            "actor": self.actor,
            "ref": self.ref,
            "reward": self.rew,
        }

    @property
    def rpcs(self):
        # interfaces
        actor_interface = ModelInterfaceAbstraction(
            "grpo",
            args={
                **copy.deepcopy(self.ppo_kwargs),
                "generation_config": self.ppo.gen,
                "early_stop_imp_ratio": self.ppo.early_stop_imp_ratio,
                "adv_norm": self.ppo.adv_norm,
                "group_size": self.group_size,
            },
        )
        ref_interface = copy.deepcopy(actor_interface)
        ref_interface.args["enable_save"] = False

        rw_interface = ModelInterfaceAbstraction(
            "paired_rw",
            args=dict(
                enable_save=False,
                output_scaling=self.ppo.reward_output_scaling,
                output_bias=self.ppo.reward_output_bias,
            ),
        )

        rpcs = {}
        rollout = MFCDef(
            name=f"actor_gen",
            model_name="actor",
            n_mbs=self.actor_gen.n_mbs,
            interface_type=ModelInterfaceType.GENERATE,
            model_type=self.actor.type,
            model_path=self.actor.path,
            interface_impl=actor_interface,
            input_keys=["packed_prompts"],
            input_key_remap={"packed_prompts": "packed_input_ids"},
            output_keys=[
                f"packed_input_ids",
                f"packed_logprobs",
                f"prompt_mask",
                f"packed_logits_mask",
            ],
            balanced_dp=True,
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        inf_reward = MFCDef(
            name=f"rew_inf",
            model_name="reward",
            n_mbs=self.rew_inf.n_mbs,
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=rw_interface,
            model_type=self.rew.type,
            model_path=self.rew.path,
            input_keys=[f"packed_input_ids"],
            output_keys=[f"rewards"],
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        inf_ref_inputs = [f"packed_input_ids"]
        if not self.ppo.gen.force_no_logits_mask:
            inf_ref_inputs.append(
                f"packed_logits_mask",
            )
        inf_ref_logits = MFCDef(
            name=f"ref_inf",
            model_name="ref",
            n_mbs=self.ref_inf.n_mbs,
            interface_type=ModelInterfaceType.INFERENCE,
            model_type=self.ref.type,
            model_path=self.ref.path,
            interface_impl=ref_interface,
            input_keys=inf_ref_inputs,
            output_keys=[f"packed_ref_logprobs"],
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        rpcs[f"actor_gen"] = rollout
        rpcs[f"rew_inf"] = inf_reward
        rpcs[f"ref_inf"] = inf_ref_logits

        train_actor_inputs = [
            f"packed_input_ids",
            f"packed_logprobs",
            f"packed_ref_logprobs",
            f"rewards",
            f"prompt_mask",
            f"packed_logits_mask",
        ]
        if self.ppo.gen.force_no_logits_mask:
            train_actor_inputs.remove(f"packed_logits_mask")
        train_actor = MFCDef(
            name="actor_train",
            model_name="actor",
            n_mbs=self.actor_train.n_mbs,
            interface_type=ModelInterfaceType.TRAIN_STEP,
            model_type=self.actor.type,
            model_path=self.actor.path,
            interface_impl=actor_interface,
            input_keys=train_actor_inputs,
            log_return_value=True,
            n_seqs=self.dataset.train_bs_n_seqs,
        )
        rpcs["actor_train"] = train_actor
        return rpcs

    @property
    def allocations(self):
        allocations = {}
        allocations[f"actor_gen"] = self.actor_gen
        allocations[f"rew_inf"] = self.rew_inf
        allocations[f"ref_inf"] = self.ref_inf
        allocations["actor_train"] = self.actor_train
        return allocations

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

    def _heuristic_rpc_allocation(self):

        actor_gen_pp_size = max(2, self.n_nodes)
        actor_gen_dp_size = (self.n_nodes * self.n_gpus_per_node) // actor_gen_pp_size
        inf_mp_size = min(2, self.n_gpus_per_node // 2)
        inf_pp_size = max(2, self.n_nodes)
        inf_dp_size = (
            (self.n_nodes * self.n_gpus_per_node) // inf_pp_size // inf_mp_size
        )

        allocs = []
        actor_gen = RPCAllocation(
            rpc=self.rpcs[f"actor_gen"],
            device_mesh=self.global_device_mesh,
            parallel=ParallelismConfig(
                data_parallel_size=actor_gen_dp_size,
                pipeline_parallel_size=actor_gen_pp_size,
                model_parallel_size=1,
            ),
        )
        ref_inf = RPCAllocation(
            rpc=self.rpcs[f"ref_inf"],
            device_mesh=self.global_device_mesh,
            parallel=ParallelismConfig(
                data_parallel_size=inf_dp_size,
                pipeline_parallel_size=inf_pp_size,
                model_parallel_size=inf_mp_size,
                use_sequence_parallel=True,
            ),
        )
        rew_inf = RPCAllocation(
            rpc=self.rpcs[f"rew_inf"],
            device_mesh=self.global_device_mesh,
            parallel=ParallelismConfig(
                data_parallel_size=inf_dp_size,
                pipeline_parallel_size=inf_pp_size,
                model_parallel_size=inf_mp_size,
                use_sequence_parallel=True,
            ),
        )
        allocs.extend([actor_gen, rew_inf, ref_inf])

        actor_train = RPCAllocation(
            rpc=self.rpcs["actor_train"],
            device_mesh=self.global_device_mesh,
            parallel=ParallelismConfig(
                data_parallel_size=self.n_gpus_per_node // min(4, self.n_gpus_per_node),
                pipeline_parallel_size=self.n_nodes,
                model_parallel_size=min(4, self.n_gpus_per_node),
            ),
        )

        return allocs + [actor_train]


register_quickstart_exp("grpo", GRPOConfig)
register_interface("grpo", GRPOInterface)

if __name__ == "__main__":

    from realhf.apps.quickstart import main

    main()
