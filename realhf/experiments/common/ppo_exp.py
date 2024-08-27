import copy
import dataclasses
import math
from typing import *

import numpy as np

import realhf.base.logging as logging
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

logger = logging.getLogger("PPO exp", "colored")


@dataclasses.dataclass
class PPOHyperparameters:
    """Configuration of PPO hyperparameters.

    :param gen: Generation hyperparameters.
    :type gen: GenerationHyperparameters
    :param ppo_n_minibatches: Number of minibatches in each PPO update.
    :type ppo_n_minibatches: int
    :param kl_ctl: Coefficient of KL divergence rewards.
    :type kl_ctl: float
    :param discount: Discount factor.
    :type discount: float
    :param gae_lambda: Lambda factor in GAE.
    :type gae_lambda: float
    :param eps_clip: PPO actor probability ratio clipping factor.
    :type eps_clip: float
    :param value_eps_clip: PPO value clipping factor.
    :type value_eps_clip: float
    :param max_reward_clip: Maximum reward value.
    :type max_reward_clip: float
    :param reward_output_scaling: Scaling factor of the reward model output.
    :type reward_output_scaling: float
    :param reward_output_bias: Bias of the reward model output.
        The number outputed by the reward model will be
        CLIP((x - bias) * scaling, -max_reward_clip, max_reward_clip).
    :type reward_output_bias: float
    :param early_stop_imp_ratio: PPO update will be early stopped if importance ratio
        exceeds this maximum value.
    :type early_stop_imp_ratio: float
    :param use_adaptive_kl_ctl: Whether to use adaptive KL divergence coefficient.
    :type use_adaptive_kl_ctl: bool
    :param adv_norm: Whether to use advantage normalization.
    :type adv_norm: bool
    :param value_norm: Whether to denormalize valued and normalize return predictions.
    :type value_norm: bool
    :param value_norm_type: Type of value normalization.
        Either exponential moving average ("exp") or moving average ("ma").
    :type value_norm_type: str
    :param value_norm_beta: Exponential decay factor
        in exponential moving average.
    :type value_norm_beta: float
    :param value_norm_eps: Epsilon factor in the
        denominator of exponential moving average.
    :type value_norm_eps: float
    """

    gen: GenerationHyperparameters = dataclasses.field(
        default_factory=GenerationHyperparameters
    )
    ppo_n_minibatches: int = 4
    kl_ctl: float = 0.1
    discount: float = 1.0
    gae_lambda: float = 1.0
    eps_clip: float = 0.2
    value_eps_clip: float = 0.2
    max_reward_clip: float = 20.0
    reward_output_scaling: float = 1.0
    reward_output_bias: float = 0.0
    early_stop_imp_ratio: float = 5.0
    use_adaptive_kl_ctl: bool = False
    adv_norm: bool = True
    value_norm: bool = True
    value_norm_type: str = dataclasses.field(
        metadata={"choices": ["exp", "ma"]}, default="exp"
    )
    value_norm_beta: float = 0.99995
    value_norm_eps: float = 1e-5


@dataclasses.dataclass
class PPOConfig(CommonExperimentConfig):
    """PPO experiment configuration.

    It is a subclass of :class:`CommonExperimentConfig`,
    so all CLI options in the base class are available.

    We don't implement runtime evaluation for PPO.

    We identify that the RLHF process is composed of four
    distinct models with independent parameters and six
    *model function calls* upon these models.

    The four models are\:

    - Actor\: The primary LLM that generates text.
    - Critic\: The value function that estimates the value of a state.
    - Ref\: The reference LLM that provides KL regularization.
    - Rew\: The reward model that provides reward signals.

    The four model function calls and their dependencies are\:

    - Rollout\: Generate text from the actor model.
    - InfReward\: Infer rewards from the reward model given generated text.
    - InfRef\: Infer log probabilities from the reference model given generated text.
    - InfValues\: Infer values from the critic model given generated text.
    - TrainActor\: Train the actor model given generated text, rewards, values, and reference log probabilities.
    - TrainCritic\: Train the critic model given generated text, rewards, values, and reference log probabilities.

    This class resolves these dependencies under the hood.
    What the users should specify are the runtime configurations
    of models and allocations of *each model function call*.

    :param is_sft_lora: Whether LoRA was used for SFT.
        If so, the saved SFT model should only contain LoRA parameters.
        Since LoRA is currently not supported for SFT,
        this option is not used for now.
    :type is_sft_lora: bool
    :param sft_lora_path: Path to the LoRA model for SFT.
        Since LoRA is currently not supported for SFT,
        this option is not used for now.
    :param is_rw_lora: Whether LoRA was used for reward modeling.
        If so, the saved reward model should only contain LoRA parameters
        and the new reward head.
        Since LoRA is currently not supported for reward modeling,
        this option is not used for now.
    :type is_rw_lora: bool
    :param rw_lora_path: Path to the LoRA model for reward modeling.
        Since LoRA is currently not supported for reward modeling,
        this option is not used for now.
    :type rw_lora_path: str
    :param rew_head_path: Path to the new reward head for reward modeling.
        Since LoRA is currently not supported for reward modeling,
        this option is not used for now.
    :type rw_head_path: str
    :param actor: Runtime configuration of the primary LLM.
    :type actor: ModelTrainEvalConfig
    :param critic: Runtime configuration of the critic model of PPO.
    :type critic: ModelTrainEvalConfig
    :param ref: Runtime configuration of the reference LLM.
    :type ref: ModelTrainEvalConfig
    :param rew: Runtime configuration of the reward LLM.
    :type rew: ModelTrainEvalConfig
    :param actor_train: :class:`MFCConfig` for TrainActor.
    :type actor_train: MFCConfig
    :param critic_train: :class:`MFCConfig` for TrainCritic.
    :type critic_train: MFCConfig
    :param actor_gen: :class:`MFCConfig` for Rollout.
    :type actor_gen: MFCConfig
    :param critic_inf: :class:`MFCConfig` for InfValues.
    :type critic_inf: MFCConfig
    :param rew_inf: :class:`MFCConfig` for InfReward.
    :type rew_inf: MFCConfig
    :param ref_inf: :class:`MFCConfig` for InfRef.
    :type ref_inf: MFCConfig
    :param dataset: Dataset configuration.
    :type dataset: PromptOnlyDatasetConfig
    :param ppo: Configuration for the PPO algorithm.
    :type ppo: PPOHyperparameters
    :param actor_train_n_mbs: Number of minibatches for TrainActor.
    :type actor_train_n_mbs: int
    :param critic_train_n_mbs: Number of minibatches for TrainCritic.
    :type critic_train_n_mbs: int
    :param actor_gen_n_mbs: Number of minibatches for Rollout.
    :type actor_gen_n_mbs: int
    :param critic_inf_n_mbs: Number of minibatches for InfValues.
    :type critic_inf_n_mbs: int
    :param rew_inf_n_mbs: Number of minibatches for InfReward.
    :type rew_inf_n_mbs: int
    :param ref_inf_n_mbs: Number of minibatches for InfRef.
    :type ref_inf_n_mbs: int
    """

    is_sft_lora: bool = False
    sft_lora_path: Optional[str] = None
    is_rew_lora: bool = False
    rew_lora_path: Optional[str] = None
    rew_head_path: Optional[str] = None

    actor: ModelTrainEvalConfig = dataclasses.field(
        default_factory=ModelTrainEvalConfig
    )
    critic: ModelTrainEvalConfig = dataclasses.field(
        default_factory=ModelTrainEvalConfig
    )
    ref: ModelTrainEvalConfig = dataclasses.field(default_factory=ModelTrainEvalConfig)
    rew: ModelTrainEvalConfig = dataclasses.field(default_factory=ModelTrainEvalConfig)

    # for manual allocation only
    actor_train: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    critic_train: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    actor_gen: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    critic_inf: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    rew_inf: MFCConfig = dataclasses.field(default_factory=MFCConfig)
    ref_inf: MFCConfig = dataclasses.field(default_factory=MFCConfig)

    dataset: PromptOnlyDatasetConfig = dataclasses.field(
        default_factory=PromptOnlyDatasetConfig
    )

    ppo: PPOHyperparameters = dataclasses.field(default_factory=PPOHyperparameters)

    def __post_init__(self):
        if self.is_sft_lora or self.sft_lora_path is not None:
            raise NotImplementedError("SFT LoRA is not supported yet.")
        if self.is_rew_lora or self.rew_lora_path is not None:
            raise NotImplementedError("Rew LoRA is not supported yet.")

        self.ppo_kwargs = dict(
            n_minibatches=self.ppo.ppo_n_minibatches,
            kl_ctl=self.ppo.kl_ctl,
            discount=self.ppo.discount,
            gae_lambda=self.ppo.gae_lambda,
            eps_clip=self.ppo.eps_clip,
            value_eps_clip=self.ppo.value_eps_clip,
            max_reward_clip=self.ppo.max_reward_clip,
            adaptive_kl_ctl=self.ppo.use_adaptive_kl_ctl,
            value_norm=self.ppo.value_norm,
            value_norm_type=self.ppo.value_norm_type,
            value_norm_beta=self.ppo.value_norm_beta,
            value_norm_eps=self.ppo.value_norm_eps,
        )

        if self.ppo.gen.use_cuda_graph and (
            self.actor_train.parallel != self.actor_gen.parallel
        ):
            raise ValueError(
                "CUDA graph cannot be used with parameter reallocation "
                "because CUDA graph requires pinned parameter memory. "
                "Either set use_cuda_graph=False or set identical parallel "
                "strategies for actor_train and actor_gen."
            )

    @property
    def models(self) -> Dict[str, ModelTrainEvalConfig]:
        # role to config
        return {
            "actor": self.actor,
            "critic": self.critic,
            "ref": self.ref,
            "reward": self.rew,
        }

    @property
    def rpcs(self):
        # interfaces
        actor_interface = ModelInterfaceAbstraction(
            "ppo_actor",
            args={
                **copy.deepcopy(self.ppo_kwargs),
                "generation_config": self.ppo.gen,
                "early_stop_imp_ratio": self.ppo.early_stop_imp_ratio,
                "adv_norm": self.ppo.adv_norm,
            },
        )
        ref_interface = copy.deepcopy(actor_interface)
        ref_interface.args["enable_save"] = False

        critic_interface = ModelInterfaceAbstraction(
            "ppo_critic",
            args=copy.deepcopy(self.ppo_kwargs),
        )
        critic_interface.args.pop("eps_clip")
        rw_interface = ModelInterfaceAbstraction(
            "paired_rw",
            args=dict(
                enable_save=False,
                output_scaling=self.ppo.reward_output_scaling,
                output_bias=self.ppo.reward_output_bias,
            ),
        )
        rollout = MFCDef(
            name="actor_gen",
            model_name="actor",
            n_mbs=self.actor_gen.n_mbs,
            interface_type=ModelInterfaceType.GENERATE,
            model_type=self.actor.type,
            model_path=self.actor.path,
            interface_impl=actor_interface,
            input_keys=["packed_prompts"],
            output_keys=[
                "seq_no_eos_mask",
                "packed_input_ids",
                "packed_logprobs",
                "prompt_mask",
                "packed_logits_mask",
            ],
            balanced_dp=True,
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        inf_reward = MFCDef(
            name="rew_inf",
            model_name="reward",
            n_mbs=self.rew_inf.n_mbs,
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=rw_interface,
            model_type=self.rew.type,
            model_path=self.rew.path,
            input_keys=["packed_input_ids"],
            output_keys=["rewards"],
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        inf_ref_inputs = ["packed_input_ids"]
        if not self.ppo.gen.force_no_logits_mask:
            inf_ref_inputs.append(
                "packed_logits_mask",
            )
        inf_ref_logits = MFCDef(
            name="ref_inf",
            model_name="ref",
            n_mbs=self.ref_inf.n_mbs,
            interface_type=ModelInterfaceType.INFERENCE,
            model_type=self.ref.type,
            model_path=self.ref.path,
            interface_impl=ref_interface,
            input_keys=inf_ref_inputs,
            output_keys=["packed_ref_logprobs"],
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        inf_values = MFCDef(
            name="critic_inf",
            model_name="critic",
            n_mbs=self.critic_inf.n_mbs,
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=critic_interface,
            model_type=self.critic.type,
            model_path=self.critic.path,
            input_keys=["packed_input_ids", "seq_no_eos_mask"],
            output_keys=["values"],
            n_seqs=self.dataset.train_bs_n_seqs,
        )

        train_actor_inputs = [
            "packed_input_ids",
            "packed_logprobs",
            "packed_ref_logprobs",
            "rewards",
            "values",
            "prompt_mask",
            "seq_no_eos_mask",
            "packed_logits_mask",
        ]
        if self.ppo.gen.force_no_logits_mask:
            train_actor_inputs.remove("packed_logits_mask")
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

        train_critic = MFCDef(
            name="critic_train",
            model_name="critic",
            n_mbs=self.critic_train.n_mbs,
            interface_type=ModelInterfaceType.TRAIN_STEP,
            interface_impl=critic_interface,
            model_type=self.critic.type,
            model_path=self.critic.path,
            input_keys=[
                "packed_input_ids",
                "packed_logprobs",
                "packed_ref_logprobs",
                "rewards",
                "values",
                "prompt_mask",
                "seq_no_eos_mask",
            ],
            log_return_value=True,
            n_seqs=self.dataset.train_bs_n_seqs,
        )
        return {
            "actor_gen": rollout,
            "actor_train": train_actor,
            "critic_inf": inf_values,
            "critic_train": train_critic,
            "ref_inf": inf_ref_logits,
            "rew_inf": inf_reward,
        }

    @property
    def allocations(self):
        return {
            "actor_gen": self.actor_gen,
            "actor_train": self.actor_train,
            "critic_inf": self.critic_inf,
            "critic_train": self.critic_train,
            "ref_inf": self.ref_inf,
            "rew_inf": self.rew_inf,
        }

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
    def search_kwargs(self):
        return {
            "num_gen_tokens": self.ppo.gen.max_new_tokens,
            "n_ppo_minibatches": self.ppo.ppo_n_minibatches,
            "seq_len": self.dataset.max_prompt_len,
        }

    @property
    def max_prompt_len(self):
        return self.dataset.max_prompt_len

    def _heuristic_rpc_allocation(self):
        """Heurisitc RPC allocation for PPO experiments."""

        assert self.n_gpus_per_node == 8

        actor_size = self.actor.type.size
        critic_size = self.critic.type.size

        # level 1
        actor_gen_pp_size = max(2, self.n_nodes)
        actor_gen_dp_size = (self.n_nodes * 8) // actor_gen_pp_size
        actor_gen = RPCAllocation(
            rpc=self.rpcs["actor_gen"],
            device_mesh=DeviceMesh(
                n_nodes=self.n_nodes,
                n_gpus_per_node=8,
                mapping=np.ones((self.n_nodes, 8), dtype=np.int32),
                global_mesh_name=self.nodelist,
            ),
            parallel=ParallelismConfig(
                data_parallel_size=actor_gen_dp_size,
                pipeline_parallel_size=actor_gen_pp_size,
                model_parallel_size=1,
            ),
        )
        # level 2
        if self.n_nodes == 1:
            assert actor_size <= 16
            assert critic_size <= 16
            actor_train = RPCAllocation(
                rpc=self.rpcs["actor_train"],
                device_mesh=DeviceMesh(
                    n_nodes=1,
                    n_gpus_per_node=8,
                    mapping=np.array([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=np.int32),
                    global_mesh_name=self.nodelist,
                ),
                parallel=ParallelismConfig(
                    data_parallel_size=2 if actor_size <= 7 else 1,
                    pipeline_parallel_size=1,
                    model_parallel_size=2 if actor_size <= 7 else 4,
                    use_sequence_parallel=True,
                ),
            )
            critic_train = RPCAllocation(
                rpc=self.rpcs["critic_train"],
                device_mesh=DeviceMesh(
                    n_nodes=1,
                    n_gpus_per_node=8,
                    mapping=np.array([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=np.int32),
                    global_mesh_name=self.nodelist,
                ),
                parallel=ParallelismConfig(
                    data_parallel_size=2 if critic_size <= 7 else 1,
                    pipeline_parallel_size=1,
                    model_parallel_size=2 if critic_size <= 7 else 4,
                    use_sequence_parallel=True,
                ),
            )
        else:
            actor_train_n_nodes = min(
                math.ceil(self.n_nodes * actor_size / (actor_size + critic_size)),
                self.n_nodes - 1,
            )
            critic_train_n_nodes = self.n_nodes - actor_train_n_nodes

            actor_train_mapping = np.zeros((self.n_nodes, 8), dtype=np.int32)
            actor_train_mapping[:actor_train_n_nodes, :] = 1
            actor_train = RPCAllocation(
                rpc=self.rpcs["actor_train"],
                device_mesh=DeviceMesh(
                    n_nodes=self.n_nodes,
                    n_gpus_per_node=8,
                    mapping=actor_train_mapping,
                    global_mesh_name=self.nodelist,
                ),
                parallel=ParallelismConfig(
                    data_parallel_size=2,
                    pipeline_parallel_size=actor_train_n_nodes,
                    model_parallel_size=4,
                    use_sequence_parallel=True,
                ),
            )

            critic_train_mapping = np.zeros((self.n_nodes, 8), dtype=np.int32)
            critic_train_mapping[actor_train_n_nodes:, :] = 1
            critic_train = RPCAllocation(
                rpc=self.rpcs["critic_train"],
                device_mesh=DeviceMesh(
                    n_nodes=self.n_nodes,
                    n_gpus_per_node=8,
                    mapping=critic_train_mapping,
                    global_mesh_name=self.nodelist,
                ),
                parallel=ParallelismConfig(
                    data_parallel_size=2,
                    pipeline_parallel_size=critic_train_n_nodes,
                    model_parallel_size=4,
                    use_sequence_parallel=True,
                ),
            )
        # level 3
        ref_inf = RPCAllocation(
            rpc=self.rpcs["ref_inf"],
            device_mesh=DeviceMesh(
                n_nodes=self.n_nodes,
                n_gpus_per_node=8,
                mapping=np.ones((self.n_nodes, 8), dtype=np.int32),
                global_mesh_name=self.nodelist,
            ),
            parallel=ParallelismConfig(
                data_parallel_size=2,
                pipeline_parallel_size=self.n_nodes,
                model_parallel_size=4,
                use_sequence_parallel=True,
            ),
        )
        # level 4
        if self.n_nodes == 1:
            rew_inf = RPCAllocation(
                rpc=self.rpcs["rew_inf"],
                device_mesh=DeviceMesh(
                    n_nodes=1,
                    n_gpus_per_node=8,
                    mapping=np.array([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=np.int32),
                    global_mesh_name=self.nodelist,
                ),
                parallel=ParallelismConfig(
                    data_parallel_size=2,
                    pipeline_parallel_size=1,
                    model_parallel_size=2,
                    use_sequence_parallel=True,
                ),
            )
            critic_inf = RPCAllocation(
                rpc=self.rpcs["critic_inf"],
                device_mesh=DeviceMesh(
                    n_nodes=1,
                    n_gpus_per_node=8,
                    mapping=np.array([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=np.int32),
                    global_mesh_name=self.nodelist,
                ),
                parallel=ParallelismConfig(
                    data_parallel_size=2,
                    pipeline_parallel_size=1,
                    model_parallel_size=2,
                    use_sequence_parallel=True,
                ),
            )
        else:
            rew_inf_n_nodes = math.ceil(self.n_nodes / 2)
            rew_inf_mapping = np.zeros((self.n_nodes, 8), dtype=np.int32)
            rew_inf_mapping[:rew_inf_n_nodes, :] = 1
            rew_inf = RPCAllocation(
                rpc=self.rpcs["rew_inf"],
                device_mesh=DeviceMesh(
                    n_nodes=self.n_nodes,
                    n_gpus_per_node=8,
                    mapping=rew_inf_mapping,
                    global_mesh_name=self.nodelist,
                ),
                parallel=ParallelismConfig(
                    data_parallel_size=2,
                    pipeline_parallel_size=rew_inf_n_nodes,
                    model_parallel_size=4,
                    use_sequence_parallel=True,
                ),
            )

            critic_inf_n_nodes = self.n_nodes - rew_inf_n_nodes
            critic_inf_mapping = np.zeros((self.n_nodes, 8), dtype=np.int32)
            critic_inf_mapping[rew_inf_n_nodes:, :] = 1
            critic_inf = RPCAllocation(
                rpc=self.rpcs["critic_inf"],
                device_mesh=DeviceMesh(
                    n_nodes=self.n_nodes,
                    n_gpus_per_node=8,
                    mapping=critic_inf_mapping,
                    global_mesh_name=self.nodelist,
                ),
                parallel=ParallelismConfig(
                    data_parallel_size=2,
                    pipeline_parallel_size=critic_inf_n_nodes,
                    model_parallel_size=4,
                    use_sequence_parallel=True,
                ),
            )
        return [
            actor_gen,
            actor_train,
            ref_inf,
            rew_inf,
            critic_inf,
            critic_train,
        ]


register_quickstart_exp("ppo", PPOConfig)
