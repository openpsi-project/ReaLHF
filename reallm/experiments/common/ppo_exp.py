import functools

from omegaconf import MISSING

from reallm.api.core.dfg import (ModelFamily, ModelInterface, ModelInterfaceType, ModelRPC, OffloadHook,
                                 SyncParamHook)
from reallm.api.core.system_api import *
from reallm.api.quickstart.dataset import PromptOnlyDatasetConfig
from reallm.api.quickstart.model import get_real_model_config, ModelTrainEvalConfig, ParallelismConfig
from reallm.base.topology import PipeModelDataParallelTopology
import reallm.base.logging as logging

logger = logging.getLogger("PPO exp", "colored")


def get_topo(
    parallel: ParallelismConfig,
    gradient_checkpointing: bool,
    max_prompt_len: Optional[int] = None,
) -> PipeModelDataParallelTopology:
    return PipeModelDataParallelTopology(
        num_mp=parallel.model_parallel_size,
        num_pp=parallel.pipeline_parallel_size,
        num_dp=parallel.data_parallel_size,
        sequence_parallel=parallel.use_sequence_parallel,
        gradient_checkpointing=gradient_checkpointing,
        max_prompt_len=max_prompt_len,
    )


def get_world_size(parallel: ParallelismConfig) -> int:
    return (parallel.model_parallel_size * parallel.pipeline_parallel_size * parallel.data_parallel_size)


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
class PPOConfig(Experiment):
    experiment_name: str = MISSING
    trial_name: str = MISSING
    seed: int = 1
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

        self.world_size = get_world_size(self.actor_gen_parallel)

        self.critic_train_ws = get_world_size(self.critic.parallel)
        self.actor_train_ws = get_world_size(self.actor.parallel)
        train_ws = self.actor_train_ws + self.critic_train_ws

        self.ref_inf_ws = get_world_size(self.ref.parallel)
        self.rew_inf_ws = get_world_size(self.rew.parallel)
        self.critic_inf_ws = get_world_size(self.critic_inf_parallel)
        inf_ws = self.ref_inf_ws + self.rew_inf_ws + self.critic_inf_ws

        if (self.world_size != train_ws) or (self.world_size != inf_ws):
            raise ValueError(
                "World size of generate, inference, and training must be the same. "
                "This is the restriction of this heuristic PPO execution plan. "
                f"Current world sizes are: (actor_train={self.actor_train_ws}, critic_train={self.critic_train_ws}, "
                f"ref_inf={self.ref_inf_ws}, rew_inf={self.rew_inf_ws}, critic_inf={self.critic_inf_ws}, "
                f"actor_gen={self.world_size}).")

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            master_worker=TasksGroup(
                count=1,
                scheduling=Scheduling.master_worker_default(
                    cpu=4,
                    mem=20000,
                ),
            ),
            model_worker=TasksGroup(
                count=self.world_size,
                scheduling=Scheduling.model_worker_default(
                    cpu=4,
                    gpu=1,
                    gpu_type="tesla",
                    mem=100000,
                ),
            ),
        )

    def initial_setup(self) -> ExperimentConfig:
        dataset = Dataset(
            "prompt",
            args=dict(
                dataset_path=self.dataset.path,
                max_length=self.dataset.max_prompt_len,
                pad_to_max_length=self.dataset.pad_to_max_length,
            ),
        )

        generation_kwargs = dict(
            max_new_tokens=self.ppo.max_new_tokens,
            min_new_tokens=self.ppo.min_new_tokens,
            greedy=self.ppo.greedy,
            top_p=self.ppo.top_p,
            top_k=self.ppo.top_k,
            temperature=self.ppo.temperature,
        )

        def _make_model_config(cfg: ModelTrainEvalConfig):
            return get_real_model_config(
                model_path=cfg.path,
                hf_model_family=cfg.type._class,
                is_critic=cfg.type.is_critic,
                init_critic_from_actor=False,
                dtype="bf16" if cfg.enable_bf16 else "fp16",
                lora=cfg.lora,
            )

        actor_model = _make_model_config(self.actor)
        ref_model = _make_model_config(self.ref)
        critic_model = _make_model_config(self.critic)
        rw_model = _make_model_config(self.rew)

        def _make_train_backend_config(cfg: ModelTrainEvalConfig):
            parallel = cfg.parallel
            if parallel.pipeline_parallel_size > 1:
                engine_type = "pipe"
            else:
                engine_type = "deepspeed"
            return ModelBackend(
                "ds_train",
                args=dict(
                    optimizer_name="adam",
                    optimizer_config=dict(
                        lr=cfg.optimizer.lr,
                        weight_decay=cfg.optimizer.weight_decay,
                        eps=cfg.optimizer.eps,
                        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
                    ),
                    lr_scheduler_type=cfg.optimizer.lr_scheduler_type,
                    warmup_steps_proportion=cfg.optimizer.warmup_steps_proportion,
                    min_lr_ratio=cfg.optimizer.min_lr_ratio,
                    zero_stage=(cfg.zero_stage if parallel.pipeline_parallel_size == 1 else min(
                        cfg.zero_stage, 1)),
                    engine_type=engine_type,
                    offload_optimizer_state=cfg.optimizer.offload,
                    offload_param=cfg.offload,
                    enable_bf16=cfg.enable_bf16,
                    enable_fp16=cfg.enable_fp16,
                ),
            )

        actor_backend = _make_train_backend_config(self.actor)
        critic_backend = _make_train_backend_config(self.critic)

        inf_backend = ModelBackend("pipe_inference" if self.ref.parallel.pipeline_parallel_size >
                                   1 else "null")

        ppo_kwargs = dict(
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

        actor_interface = ModelInterface(
            "ppo_actor",
            args={
                **copy.deepcopy(ppo_kwargs),
                "generation_config": generation_kwargs,
                "early_stop_imp_ratio": self.ppo.early_stop_imp_ratio,
                "force_no_logits_mask": True,
                "adv_norm": self.ppo.adv_norm,
            },
        )
        ref_interface = copy.deepcopy(actor_interface)
        ref_interface.args["enable_save"] = False

        critic_interface = ModelInterface(
            "ppo_critic",
            args=copy.deepcopy(ppo_kwargs),
        )
        rw_interface = ModelInterface(
            "paired_rw",
            args=dict(
                enable_save=False,
                output_scaling=self.ppo.reward_output_scaling,
                output_bias=self.ppo.reward_output_bias,
            ),
        )

        gen_topo = get_topo(self.actor_gen_parallel, False, self.dataset.max_prompt_len)
        actor_train_topo = get_topo(self.actor.parallel, self.actor.gradient_checkpointing)
        critic_train_topo = get_topo(self.critic.parallel, self.critic.gradient_checkpointing)
        critic_inf_topo = get_topo(self.critic_inf_parallel, False)
        ref_topo = get_topo(self.ref.parallel, False)
        rw_topo = get_topo(self.rew.parallel, False)

        model_worker = []
        for i in range(self.world_size):
            shards = []
            shards += [
                StandaloneModelShard(
                    id=ModelShardID(
                        model_name=ModelName("actor", 0),
                        topo=gen_topo,
                        dp_rank=gen_topo.get_coord(i).data,
                        pp_rank=gen_topo.get_coord(i).pipe,
                        mp_rank=gen_topo.get_coord(i).model,
                    ),
                    model=actor_model,
                    backend=inf_backend,
                ),
            ]
            if i < self.actor_train_ws:
                shards += [
                    StandaloneModelShard(
                        id=ModelShardID(
                            model_name=ModelName("actor", 1),
                            topo=actor_train_topo,
                            dp_rank=actor_train_topo.get_coord(i).data,
                            pp_rank=actor_train_topo.get_coord(i).pipe,
                            mp_rank=actor_train_topo.get_coord(i).model,
                        ),
                        model=actor_model,
                        backend=actor_backend,
                    ),
                ]
            else:
                offset = self.actor_train_ws
                shards += [
                    StandaloneModelShard(
                        id=ModelShardID(
                            model_name=ModelName("critic", 1),
                            topo=critic_train_topo,
                            dp_rank=critic_train_topo.get_coord(i - offset).data,
                            pp_rank=critic_train_topo.get_coord(i - offset).pipe,
                            mp_rank=critic_train_topo.get_coord(i - offset).model,
                        ),
                        model=critic_model,
                        backend=critic_backend,
                    ),
                ]
            if i < self.rew_inf_ws:
                shards += [
                    StandaloneModelShard(
                        id=ModelShardID(
                            model_name=ModelName("reward", 0),
                            topo=rw_topo,
                            dp_rank=rw_topo.get_coord(i).data,
                            pp_rank=rw_topo.get_coord(i).pipe,
                            mp_rank=rw_topo.get_coord(i).model,
                        ),
                        model=rw_model,
                        backend=inf_backend,
                    ),
                ]
            elif i < self.rew_inf_ws + self.critic_inf_ws:
                offset = self.rew_inf_ws
                shards += [
                    StandaloneModelShard(
                        id=ModelShardID(
                            model_name=ModelName("critic", 0),
                            topo=critic_inf_topo,
                            dp_rank=critic_inf_topo.get_coord(i - offset).data,
                            pp_rank=critic_inf_topo.get_coord(i - offset).pipe,
                            mp_rank=critic_inf_topo.get_coord(i - offset).model,
                        ),
                        model=critic_model,
                        backend=inf_backend,
                    ),
                ]
            else:
                offset = self.rew_inf_ws + self.critic_inf_ws
                shards += [
                    StandaloneModelShard(
                        id=ModelShardID(
                            model_name=ModelName("ref", 0),
                            topo=ref_topo,
                            dp_rank=ref_topo.get_coord(i - offset).data,
                            pp_rank=ref_topo.get_coord(i - offset).pipe,
                            mp_rank=ref_topo.get_coord(i - offset).model,
                        ),
                        model=ref_model,
                        backend=inf_backend,
                    ),
                ]
            mw = ModelWorker(
                seed=self.seed,
                shards=shards,
                tokenizer_name_or_path=self.actor.path,
                datasets=[dataset],
                cuda_cache_cleanliness=True,
                cuda_cache_clear_freq=10,
            )
            model_worker.append(mw)

        rollout = ModelRPC(
            model_name=ModelName("actor", 0),
            interface_type=ModelInterfaceType.GENERATE,
            model_type=self.actor.type,
            interface_impl=actor_interface,
            input_data=["packed_prompts"],
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
            input_data=["packed_seq", "cu_seqlens"],
            input_key_remap={"packed_seq": "packed_input_ids"},
            output_data=["scores"],
            output_key_remap={"scores": "rewards"},
            post_hooks=[OffloadHook()],
            min_n_seqs=self.global_gen_bs,
            max_n_seqs=self.global_gen_bs,
        )

        inf_ref_logits = ModelRPC(
            model_name=ModelName("ref", 0),
            interface_type=ModelInterfaceType.INFERENCE,
            model_type=self.ref.type,
            interface_impl=ref_interface,
            input_data=[
                "packed_seq",
                "cu_seqlens",
            ],
            output_data=["logprobs"],
            output_key_remap={"logprobs": "packed_ref_logprobs"},
            post_hooks=[OffloadHook()],
            min_n_seqs=self.global_gen_bs,
            max_n_seqs=self.global_gen_bs,
        )

        inf_values = ModelRPC(
            model_name=ModelName("critic", 0),
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=critic_interface,
            model_type=self.critic.type,
            input_data=["packed_seq", "cu_seqlens", "seq_no_eos_mask"],
            output_data=["scores"],
            output_key_remap={"scores": "values"},
            min_n_seqs=self.global_gen_bs,
            max_n_seqs=self.global_gen_bs,
        )

        train_actor = ModelRPC(
            model_name=ModelName("actor", 1),
            interface_type=ModelInterfaceType.TRAIN_STEP,
            model_type=self.actor.type,
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
            pre_hooks=[SyncParamHook(source=ModelName("actor", 0))],
            post_hooks=[SyncParamHook(target=ModelName("actor", 0))],
            min_n_seqs=self.global_train_bs,
            max_n_seqs=self.global_train_bs,
        )

        train_critic = ModelRPC(
            model_name=ModelName("critic", 1),
            interface_type=ModelInterfaceType.TRAIN_STEP,
            interface_impl=critic_interface,
            model_type=ModelFamily("llama", 7, True),
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
            pre_hooks=[SyncParamHook(source=ModelName("critic", 0))],
            post_hooks=[SyncParamHook(target=ModelName("critic", 0))],
            min_n_seqs=self.global_train_bs,
            max_n_seqs=self.global_train_bs,
        )

        if actor_train_topo.get_dim("pipe") > 1:
            pp_nmbs = actor_train_topo.get_dim("pipe") * 2
            train_actor.min_n_seqs_per_dp = self.ppo.ppo_n_minibatches * pp_nmbs
        else:
            train_actor.min_n_seqs_per_dp = self.ppo.ppo_n_minibatches

        rollout.min_n_seqs_per_dp = self.global_gen_bs // gen_topo.get_dim("data")

        exp_ctrl = ExperimentSaveEvalControl(
            total_train_epochs=self.total_train_epochs,
            save_frequency_steps=self.save_freq_steps,
        )
        return ExperimentConfig(
            exp_ctrl=exp_ctrl,
            model_rpcs=[
                rollout,
                inf_ref_logits,
                inf_reward,
                inf_values,
                train_actor,
                train_critic,
            ],
            model_worker=model_worker,
        )
