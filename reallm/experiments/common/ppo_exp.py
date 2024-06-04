from reallm.api.core.dfg import ModelFamily, ModelInterface, ModelInterfaceType, ModelRPC
from reallm.api.core.system_api import *
from reallm.api.quickstart.dataset import PromptOnlyDatasetConfig
from reallm.api.quickstart.model import ModelTrainEvalConfig, ParallelismConfig
from reallm.experiments.common.common import CommonExperimentConfig
from reallm.experiments.common.utils import *
import reallm.base.logging as logging

logger = logging.getLogger("PPO exp", "colored")


@dataclasses.dataclass
class PPOHyperparameters:
    """Configuration of PPO hyperparameters.

    Args:
        max_new_tokens (int): Maximum number of new tokens to generate in each iteration.
        min_new_tokens (int): Minimum number of new tokens to generate in each iteration.
        greedy (bool): Whether to use greedy decoding. PPO may not work if set to True.
        top_p (float): Top-p sampling ratio.
        top_k (float): Top-k sampling ratio.
        temperature (float): Sampling temperature.
        ppo_n_minibatches (int): Number of minibatches in each PPO update.
        kl_ctl (float): Coefficient of KL divergence rewards.
        discount (float): Discount factor.
        gae_lambda (float): Lambda factor in GAE.
        eps_clip (float): PPO clipping factor.
        value_eps_clip (float): PPO value clipping factor.
        max_reward_clip (float): Maximum reward value.
        reward_output_scaling (float): Scaling factor of the reward model output.
        reward_output_bias (float): Bias of the reward model output.
            The number outputed by the reward model will be
            CLIP((x - bias) * scaling, -max_reward_clip, max_reward_clip).
        early_stop_imp_ratio (float): PPO update will be early stopped if importance ratio
            exceeds this maximum value.
        use_adaptive_kl_ctl (bool): Whether to use adaptive KL divergence coefficient.
        adv_norm (bool): Whether use advantage normalization.
        value_norm (bool): Whether to denormalize valued and normalize return predictions.
        value_norm_type (str): Type of value normalization. Either exponential moving average or moving average.
        value_norm_beta (float): Exponential decay factor in exponential moving average.
        value_norm_eps (float): Epsilon factor in the denominator of exponential moving average.
    """

    max_new_tokens: int = 256
    min_new_tokens: int = 256
    greedy: bool = False
    top_p: float = 0.9
    top_k: int = 200
    temperature: float = 1.0
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
    value_norm_type: str = dataclasses.field(metadata={"choices": ["exp", "ma"]}, default="exp")
    value_norm_beta: float = 0.99995
    value_norm_eps: float = 1e-5


@dataclasses.dataclass
class PPOConfig(CommonExperimentConfig):
    total_train_epochs: int = 1
    save_freq_steps: Optional[int] = 20

    is_sft_lora: bool = False
    sft_lora_path: Optional[str] = None
    is_rew_lora: bool = False
    rew_lora_path: Optional[str] = None
    rew_head_path: Optional[str] = None

    actor: ModelTrainEvalConfig = dataclasses.field(default_factory=ModelTrainEvalConfig)
    critic: ModelTrainEvalConfig = dataclasses.field(default_factory=ModelTrainEvalConfig)
    ref: ModelTrainEvalConfig = dataclasses.field(default_factory=ModelTrainEvalConfig)
    rew: ModelTrainEvalConfig = dataclasses.field(default_factory=ModelTrainEvalConfig)

    actor_gen_parallel: ParallelismConfig = dataclasses.field(default_factory=ParallelismConfig)
    critic_inf_parallel: ParallelismConfig = dataclasses.field(default_factory=ParallelismConfig)

    dataset: PromptOnlyDatasetConfig = dataclasses.field(default_factory=PromptOnlyDatasetConfig)

    ppo: PPOHyperparameters = dataclasses.field(default_factory=PPOHyperparameters)

    global_train_bs: int = 512
    global_gen_bs: int = 512

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
                "force_no_logits_mask": True,
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

        print(self.actor, self.actor.path)
        rollout = ModelRPC(
            model_name=ModelName("actor", 0),
            interface_type=ModelInterfaceType.GENERATE,
            model_type=self.actor.type,
            model_path=self.actor.path,
            interface_impl=actor_interface,
            input_data=["packed_prompts", "prompt_cu_seqlens"],
            output_data=[
                "seq_no_eos_mask",
                "packed_seq",
                "cu_seqlens",
                "packed_logprobs",
                "prompt_mask",
            ],
            balanced_dp=True,
            min_n_seqs=self.global_gen_bs,
            max_n_seqs=self.global_gen_bs,
        )

        inf_reward = ModelRPC(
            model_name=ModelName("reward", 0),
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=rw_interface,
            model_type=self.rew.type,
            model_path=self.rew.path,
            input_data=["packed_seq", "cu_seqlens"],
            input_key_remap={"packed_seq": "packed_input_ids"},
            output_data=["scores"],
            output_key_remap={"scores": "rewards"},
            # post_hooks=[OffloadHook()],
            min_n_seqs=self.global_gen_bs,
            max_n_seqs=self.global_gen_bs,
        )

        inf_ref_logits = ModelRPC(
            model_name=ModelName("ref", 0),
            interface_type=ModelInterfaceType.INFERENCE,
            model_type=self.ref.type,
            model_path=self.ref.path,
            interface_impl=ref_interface,
            input_data=[
                "packed_seq",
                "cu_seqlens",
            ],
            output_data=["logprobs"],
            output_key_remap={"logprobs": "packed_ref_logprobs"},
            # post_hooks=[OffloadHook()],
            min_n_seqs=self.global_gen_bs,
            max_n_seqs=self.global_gen_bs,
        )

        inf_values = ModelRPC(
            model_name=ModelName("critic", 0),
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=critic_interface,
            model_type=self.critic.type,
            model_path=self.critic.path,
            input_data=["packed_seq", "cu_seqlens", "seq_no_eos_mask"],
            output_data=["scores"],
            output_key_remap={"scores": "values"},
            min_n_seqs=self.global_gen_bs,
            max_n_seqs=self.global_gen_bs,
        )

        train_actor = ModelRPC(
            model_name=ModelName("actor", 0),
            interface_type=ModelInterfaceType.TRAIN_STEP,
            model_type=self.actor.type,
            model_path=self.actor.path,
            interface_impl=actor_interface,
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
            # pre_hooks=[SyncParamHook(source=ModelName("actor", 0))],
            # post_hooks=[SyncParamHook(target=ModelName("actor", 0))],
            min_n_seqs=self.global_train_bs,
            max_n_seqs=self.global_train_bs,
        )

        train_critic = ModelRPC(
            model_name=ModelName("critic", 0),
            interface_type=ModelInterfaceType.TRAIN_STEP,
            interface_impl=critic_interface,
            model_type=ModelFamily("llama", 7, True),
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
            min_n_seqs=self.global_train_bs,
            max_n_seqs=self.global_train_bs,
        )
        rpcs = [rollout, inf_reward, inf_ref_logits, inf_values, train_actor, train_critic]
        return {rpc.name: rpc for rpc in rpcs}

    @property
    def datasets(self):
        return [
            Dataset(
                "packed_prompt",
                args=dict(
                    dataset_path=self.dataset.path,
                    n_tokens_per_batch=self.dataset.n_tokens_per_batch,
                    max_length=self.dataset.max_prompt_len,
                ),
            )
        ]

    @property
    def dataloader(self) -> DataLoader:
        return DataLoader("iterable_dataset_loader")

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
