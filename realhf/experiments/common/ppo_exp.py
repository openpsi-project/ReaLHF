import torch

from realhf.api.core.dfg import (
    MFCDef,
    ModelFamily,
    ModelInterface,
    ModelInterfaceType,
)
from realhf.api.core.system_api import *
from realhf.api.quickstart.dataset import PromptOnlyDatasetConfig
from realhf.api.quickstart.device_mesh import (
    AllocationConfig,
    DeviceMesh,
    RPCAllocation,
)
from realhf.api.quickstart.entrypoint import register_quickstart_exp
from realhf.api.quickstart.model import (
    get_real_model_config,
    ModelTrainEvalConfig,
    ParallelismConfig,
)
from realhf.base.topology import PipeModelDataParallelTopology
from realhf.experiments.common.common import CommonExperimentConfig
from realhf.experiments.common.utils import *
import realhf.base.logging as logging

logger = logging.getLogger("PPO exp", "colored")


@dataclasses.dataclass
class PPOHyperparameters:
    """Configuration of PPO hyperparameters.

    We implement a customized generation function instead of
    using HuggingFace's to support pipelined generation.
    As a result, advanced generation techniques like
    diversity-promoting sampling or repeatition penalty
    are not supported during PPO training.
    However, we don't find it to be a problem in practice.
    Increasing the sampling temperature and enabling
    top-k/top-p sampling can produce good models.

    :param max_new_tokens: Maximum number of new tokens
        to generate.
    :type max_new_tokens: int
    :param min_new_tokens: Minimum number of new tokens
        to generate.
    :type min_new_tokens: int
    :param greedy: Whether to use greedy decoding.
        PPO may not work if set to True.
    :type greedy: bool
    :param top_p: Tokens will be sampled from a
        vocabulary subset with probability summation
        larger than p.
    :type top_p: float
    :param top_k: Tokens will be sampled from a
        vocabulary subset with top-k probabilities.
    :type top_k: int
    :param temperature: Sampling temperature.
    :type temperature: float
    :param force_no_logits_mask: Whether to omit logits mask.
        The logits mask will be produced when using top-k or top-p sampling,
        where it is used to mark tokens that are filtered out.
        This mask will be used by the reference model and the actor model
        during training in order to align inferred logits with that during
        generation and produce accurate KLs.
        Logits mask with top-k/top-p sampling will largely improve the
        stability of PPO training because it narrows the action space.
        However, this benefit does not come for free.
        The logits mask will occupy a large amount of additional GPU memory.
        If this option is set to True, logits mask will be forcely omitted to
        save GPU memory, but the learning performance may also drop.
    :type force_no_logits_mask: bool
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

    max_new_tokens: int = 256
    min_new_tokens: int = 256
    greedy: bool = False
    top_p: float = 0.9
    top_k: int = 200
    temperature: float = 1.0
    force_no_logits_mask: bool = False
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

    :param total_train_epochs: Total number of training epochs
        (i.e., the number of times the training dataset is iterated).
    :type total_train_epochs: int
    :param save_freq_steps: Save the model every this number of steps.
        "step" is a PPO training step, probabily composed of multiple
        model updates if ppo_n_minibatch > 1.
        If None, the model will not be saved during training.
        The directory to save the model will be automatically resolved
        and prompted in the terminal when the experiment starts.
    :type save_freq_steps: Optional[int]
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
    :param actor_train: :class:`AllocationConfig` for TrainActor.
    :type actor_train: AllocationConfig
    :param critic_train: :class:`AllocationConfig` for TrainCritic.
    :type critic_train: AllocationConfig
    :param actor_gen: :class:`AllocationConfig` for Rollout.
    :type actor_gen: AllocationConfig
    :param critic_inf: :class:`AllocationConfig` for InfValues.
    :type critic_inf: AllocationConfig
    :param rew_inf: :class:`AllocationConfig` for InfReward.
    :type rew_inf: AllocationConfig
    :param ref_inf: :class:`AllocationConfig` for InfRef.
    :type ref_inf: AllocationConfig
    :param dataset: Dataset configuration.
    :type dataset: PromptOnlyDatasetConfig
    :param ppo: Configuration for the PPO algorithm.
    :type ppo: PPOHyperparameters
    """

    total_train_epochs: int = 1
    save_freq_steps: Optional[int] = 20

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
    ref: ModelTrainEvalConfig = dataclasses.field(
        default_factory=ModelTrainEvalConfig
    )
    rew: ModelTrainEvalConfig = dataclasses.field(
        default_factory=ModelTrainEvalConfig
    )

    # for manual allocation only
    actor_train: AllocationConfig = dataclasses.field(
        default_factory=AllocationConfig
    )
    critic_train: AllocationConfig = dataclasses.field(
        default_factory=AllocationConfig
    )
    actor_gen: AllocationConfig = dataclasses.field(
        default_factory=AllocationConfig
    )
    critic_inf: AllocationConfig = dataclasses.field(
        default_factory=AllocationConfig
    )
    rew_inf: AllocationConfig = dataclasses.field(
        default_factory=AllocationConfig
    )
    ref_inf: AllocationConfig = dataclasses.field(
        default_factory=AllocationConfig
    )

    dataset: PromptOnlyDatasetConfig = dataclasses.field(
        default_factory=PromptOnlyDatasetConfig
    )

    ppo: PPOHyperparameters = dataclasses.field(
        default_factory=PPOHyperparameters
    )

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

        self.generation_kwargs = dict(
            max_new_tokens=self.ppo.max_new_tokens,
            min_new_tokens=self.ppo.min_new_tokens,
            greedy=self.ppo.greedy,
            top_p=self.ppo.top_p,
            top_k=self.ppo.top_k,
            temperature=self.ppo.temperature,
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
        actor_interface = ModelInterface(
            "ppo_actor",
            args={
                **copy.deepcopy(self.ppo_kwargs),
                "generation_config": self.generation_kwargs,
                "early_stop_imp_ratio": self.ppo.early_stop_imp_ratio,
                "force_no_logits_mask": self.ppo.force_no_logits_mask,
                "adv_norm": self.ppo.adv_norm,
            },
        )
        ref_interface = copy.deepcopy(actor_interface)
        ref_interface.args["enable_save"] = False

        critic_interface = ModelInterface(
            "ppo_critic",
            args=copy.deepcopy(self.ppo_kwargs),
        )
        rw_interface = ModelInterface(
            "paired_rw",
            args=dict(
                enable_save=False,
                output_scaling=self.ppo.reward_output_scaling,
                output_bias=self.ppo.reward_output_bias,
            ),
        )
        rollout = MFCDef(
            model_name=ModelName("actor", 0),
            interface_type=ModelInterfaceType.GENERATE,
            model_type=self.actor.type,
            model_path=self.actor.path,
            interface_impl=actor_interface,
            input_data=["packed_prompts"],
            output_data=[
                "seq_no_eos_mask",
                "packed_seq",
                "cu_seqlens",
                "packed_logprobs",
                "prompt_mask",
                "packed_logits_mask",
            ],
            balanced_dp=True,
            min_n_seqs=self.dataset.train_bs_n_seqs,
            max_n_seqs=self.dataset.train_bs_n_seqs,
        )

        inf_reward = MFCDef(
            model_name=ModelName("reward", 0),
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=rw_interface,
            model_type=self.rew.type,
            model_path=self.rew.path,
            input_data=["packed_seq", "cu_seqlens"],
            input_key_remap={"packed_seq": "packed_input_ids"},
            output_data=["scores"],
            output_key_remap={"scores": "rewards"},
            min_n_seqs=self.dataset.train_bs_n_seqs,
            max_n_seqs=self.dataset.train_bs_n_seqs,
        )

        inf_ref_inputs = ["packed_seq", "cu_seqlens"]
        if not self.ppo.force_no_logits_mask:
            inf_ref_inputs.append(
                "packed_logits_mask",
            )
        inf_ref_logits = MFCDef(
            model_name=ModelName("ref", 0),
            interface_type=ModelInterfaceType.INFERENCE,
            model_type=self.ref.type,
            model_path=self.ref.path,
            interface_impl=ref_interface,
            input_data=inf_ref_inputs,
            output_data=["logprobs"],
            output_key_remap={"logprobs": "packed_ref_logprobs"},
            min_n_seqs=self.dataset.train_bs_n_seqs,
            max_n_seqs=self.dataset.train_bs_n_seqs,
        )

        inf_values = MFCDef(
            model_name=ModelName("critic", 0),
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=critic_interface,
            model_type=self.critic.type,
            model_path=self.critic.path,
            input_data=["packed_seq", "cu_seqlens", "seq_no_eos_mask"],
            output_data=["scores"],
            output_key_remap={"scores": "values"},
            min_n_seqs=self.dataset.train_bs_n_seqs,
            max_n_seqs=self.dataset.train_bs_n_seqs,
        )

        train_actor_inputs = [
            "packed_seq",
            "cu_seqlens",
            "packed_logprobs",
            "packed_ref_logprobs",
            "rewards",
            "values",
            "prompt_mask",
            "seq_no_eos_mask",
            "packed_logits_mask",
        ]
        if self.ppo.force_no_logits_mask:
            train_actor_inputs.remove("packed_logits_mask")
        train_actor = MFCDef(
            model_name=ModelName("actor", 0),
            interface_type=ModelInterfaceType.TRAIN_STEP,
            model_type=self.actor.type,
            model_path=self.actor.path,
            interface_impl=actor_interface,
            input_data=train_actor_inputs,
            log_return_value=True,
            # pre_hooks=[SyncParamHook(source=ModelName("actor", 0))],
            # post_hooks=[SyncParamHook(target=ModelName("actor", 0))],
            min_n_seqs=self.dataset.train_bs_n_seqs,
            max_n_seqs=self.dataset.train_bs_n_seqs,
        )

        train_critic = MFCDef(
            model_name=ModelName("critic", 0),
            interface_type=ModelInterfaceType.TRAIN_STEP,
            interface_impl=critic_interface,
            model_type=self.critic.type,
            model_path=self.critic.path,
            input_data=[
                "packed_seq",
                "cu_seqlens",
                "packed_logprobs",
                "packed_ref_logprobs",
                "rewards",
                "values",
                "prompt_mask",
                "seq_no_eos_mask",
            ],
            log_return_value=True,
            # pre_hooks=[SyncParamHook(source=ModelName("critic", 0))],
            # post_hooks=[SyncParamHook(target=ModelName("critic", 0))],
            min_n_seqs=self.dataset.train_bs_n_seqs,
            max_n_seqs=self.dataset.train_bs_n_seqs,
        )
        # rpcs = [rollout, inf_reward, inf_ref_logits, inf_values, train_actor, train_critic]
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
            Dataset(
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
    def exp_ctrl(self) -> ExperimentSaveEvalControl:
        return ExperimentSaveEvalControl(
            total_train_epochs=self.total_train_epochs,
            save_frequency_steps=self.save_freq_steps,
        )

    @property
    def search_kwargs(self):
        return {
            "num_gen_tokens": self.ppo.max_new_tokens,
            "n_ppo_minibatches": self.ppo.ppo_n_minibatches,
            "seq_len": self.dataset.max_prompt_len,
        }

    @property
    def max_prompt_len(self):
        return self.dataset.max_prompt_len

    def _heuristic_rpc_allocation(self):
        """Heurisitc RPC allocation for PPO experiments."""
        import math

        import numpy as np

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
                    mapping=np.array(
                        [[1, 1, 1, 1, 0, 0, 0, 0]], dtype=np.int32
                    ),
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
                    mapping=np.array(
                        [[0, 0, 0, 0, 1, 1, 1, 1]], dtype=np.int32
                    ),
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
                math.ceil(
                    self.n_nodes * actor_size / (actor_size + critic_size)
                ),
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
                    mapping=np.array(
                        [[1, 1, 1, 1, 0, 0, 0, 0]], dtype=np.int32
                    ),
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
                    mapping=np.array(
                        [[0, 0, 0, 0, 1, 1, 1, 1]], dtype=np.int32
                    ),
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
