import functools

from omegaconf import MISSING

from .config_dataset import PromptOnlyDatasetConfig
from .config_model import get_flash_mqat_model_config, ModelConfig
from api.config import *
from api.dfg import ModelInterfaceType, ModelRPC
from base.topology import PipeModelDataParallelTopology
import base.logging as logging

logger = logging.getLogger("PPO exp", "colored")

rollout = ModelRPC(
    "actor",
    ModelInterfaceType.GENERATE,
    input_data=["packed_prompts", "prompt_cu_seqlens"],
    output_data=[
        "seq_no_eos_mask",
        "packed_seq",
        "cu_seqlens",
        "packed_logprobs",
        "packed_logits_mask",
        "prompt_mask",
    ],
    dp_broker_type="packed",
    min_n_seqs=256,
    max_n_seqs=257,
    max_concurrent_calls=4,
)

inf_reward = ModelRPC(
    "reward",
    ModelInterfaceType.INFERENCE,
    input_data=["packed_seq", "cu_seqlens"],
    input_key_remap={"packed_seq": "packed_input_ids"},
    output_data=["scores"],
    output_key_remap={"scores": "rewards"},
    dp_broker_type="packed",
    min_n_seqs=256,
    max_concurrent_calls=1,
)

inf_ref_logits = ModelRPC(
    "ref",
    ModelInterfaceType.INFERENCE,
    input_data=[
        "packed_seq",
        "cu_seqlens",
        "packed_logits_mask",
    ],
    output_data=["logprobs"],
    output_key_remap={"logprobs": "packed_ref_logprobs"},
    dp_broker_type="packed",
    min_n_seqs=256,
    max_concurrent_calls=1,
)

inf_values = ModelRPC(
    "critic",
    ModelInterfaceType.INFERENCE,
    input_data=["packed_seq", "cu_seqlens", "seq_no_eos_mask"],
    output_data=["scores"],
    output_key_remap={"scores": "values"},
    dp_broker_type="packed",
    min_n_seqs=256,
    max_concurrent_calls=1,
)

train_actor = ModelRPC(
    "actor",
    ModelInterfaceType.TRAIN_STEP,
    input_data=[
        "packed_seq",
        "cu_seqlens",
        "packed_logprobs",
        "packed_ref_logprobs",
        "rewards",
        "values",
        "prompt_mask",
        "seq_no_eos_mask",
        "packed_logits_mask",
    ],
    log_return_value=True,
    dp_broker_type="packed",
    min_n_seqs=256,
    max_n_seqs=257,
    min_n_tokens=1,
    max_concurrent_calls=1,
)

train_critic = ModelRPC(
    "critic",
    ModelInterfaceType.TRAIN_STEP,
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
    dp_broker_type="packed",
    log_return_value=True,
    min_n_seqs=256,
    max_n_seqs=257,
    min_n_tokens=1,
    max_concurrent_calls=1,
)


@dataclasses.dataclass
class PPOHyperparmeters:
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
        actor_as_critic (bool): Whether to use actor as critic for critic and reward models.
        use_stream_pipe_engine (bool): Whether to use stream pipe engine for actor model.
    """

    max_new_tokens: int = 512
    min_new_tokens: int = 10
    greedy: bool = False
    top_p: float = 1.0
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
    adv_norm: bool = False
    value_norm: bool = False
    value_norm_type: str = dataclasses.field(metadata={"choices": ["exp", "ma"]}, default="exp")
    value_norm_beta: float = 0.99995
    value_norm_eps: float = 1e-5
    actor_as_critic: bool = False
    use_stream_pipe_engine: bool = False


@dataclasses.dataclass
class PPOConfig(Experiment):
    experiment_name: str = MISSING
    trial_name: str = MISSING
    trace: bool = False
    seed: int = 1
    total_train_epochs: int = 1
    save_freq_steps: Optional[int] = 20

    is_sft_lora: bool = False
    sft_lora_path: Optional[str] = None
    is_rew_lora: bool = False
    rew_lora_path: Optional[str] = None
    rew_head_path: Optional[str] = None

    actor: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    critic: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    ref: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    rew: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    dataset: PromptOnlyDatasetConfig = dataclasses.field(default_factory=PromptOnlyDatasetConfig)
    ppo: PPOHyperparmeters = dataclasses.field(default_factory=PPOHyperparmeters)

    def __post_init__(self):
        if self.is_sft_lora and (self.sft_lora_path is None or self.actor.type is None):
            raise ValueError("sft_lora_path and base_model_type must be specified when is_sft_lora is True.")
        if self.is_rew_lora and (self.rew_lora_path is None or self.rew.type is None
                                 or self.rew_head_path is None):
            raise ValueError(
                "rew_lora_path, rew_base_model_type and rew_head_path must be specified when is_rw_lora is True."
            )

        self.n_actors = int(self.actor.parallel.pipeline_parallel_size *
                            self.actor.parallel.model_parallel_size * self.actor.parallel.data_parallel_size)
        self.n_critics = int(self.critic.parallel.pipeline_parallel_size *
                             self.critic.parallel.model_parallel_size *
                             self.critic.parallel.data_parallel_size)
        self.n_rewards = int(self.rew.parallel.pipeline_parallel_size *
                             self.rew.parallel.model_parallel_size * self.rew.parallel.data_parallel_size)
        self.n_refs = int(self.ref.parallel.pipeline_parallel_size * self.ref.parallel.model_parallel_size *
                          self.ref.parallel.data_parallel_size)

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            data_worker=TasksGroup(
                count=1,
                scheduling=Scheduling.data_worker_default(
                    cpu=2,
                    mem=10000,
                ),
            ),
            master_worker=TasksGroup(
                count=1,
                scheduling=Scheduling.master_worker_default(
                    cpu=16,
                    mem=100000,
                    gpu=1,
                    gpu_type="tesla",
                ),
            ),
            model_worker=[
                TasksGroup(
                    count=self.n_actors,
                    scheduling=Scheduling.model_worker_default(
                        cpu=4,
                        gpu=1,
                        gpu_type="tesla",
                        mem=50000,
                    ),
                ),
                TasksGroup(
                    count=self.n_critics,
                    scheduling=Scheduling.model_worker_default(
                        cpu=4,
                        gpu=1,
                        gpu_type="tesla",
                        mem=100000,
                    ),
                ),
                TasksGroup(
                    count=self.n_rewards,
                    scheduling=Scheduling.model_worker_default(
                        cpu=4,
                        gpu=1,
                        gpu_type="tesla",
                        mem=100000,
                    ),
                ),
                TasksGroup(
                    count=self.n_refs,
                    scheduling=Scheduling.model_worker_default(
                        cpu=4,
                        gpu=1,
                        gpu_type="tesla",
                        mem=100000,
                    ),
                ),
            ],
        )

    def initial_setup(self) -> ExperimentConfig:
        dataset = Dataset(
            "packed_prompt",
            args=dict(
                dataset_path=self.dataset.path,
                n_tokens_per_batch=self.dataset.n_tokens_per_batch,
                max_length=self.dataset.max_prompt_len,
            ),
        )
        dataloader = DataLoader("iterable_dataset_loader")
        data_worker = [
            DataWorker(
                tokenizer_name_or_path=self.actor.base_model_path,
                datasets=[dataset],
                dataloader=dataloader,
                seed=self.seed,
            )
        ]

        generation_kwargs = dict(
            max_new_tokens=self.ppo.max_new_tokens,
            min_new_tokens=self.ppo.min_new_tokens,
            greedy=self.ppo.greedy,
            top_p=self.ppo.top_p,
            top_k=self.ppo.top_k,
            temperature=self.ppo.temperature,
        )

        def _make_model_config(cfg: ModelConfig, from_type: str):
            return get_flash_mqat_model_config(
                from_type=from_type,
                model_path=cfg.path,
                hf_model_type=cfg.type,
                tokenizer_path=cfg.base_model_path,
                use_pipe=(cfg.parallel.pipeline_parallel_size > 1),
                dtype="bf16" if cfg.enable_bf16 else "fp16",
                sequence_parallel=cfg.parallel.use_sequence_parallel,
                partition_method=cfg.parallel.partition_method,
                lora=cfg.lora,
            )

        actor_model = _make_model_config(self.actor, "self")
        ref_model = _make_model_config(self.ref, "self")
        critic_type = "self" if not self.ppo.actor_as_critic else "actor_as_critic"
        # critic_type = "random_critic"
        critic_model = _make_model_config(self.critic, critic_type)
        rw_model = _make_model_config(self.rew, critic_type)

        def _make_train_backend_config(cfg: ModelConfig, use_stream_pipe_engine: bool):
            if cfg.parallel.pipeline_parallel_size > 1:
                engine_type = "stream_pipe" if use_stream_pipe_engine else "pipe"
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
                    zero_stage=(cfg.optimizer.zero_stage if cfg.parallel.pipeline_parallel_size == 1 else min(
                        cfg.optimizer.zero_stage, 1)),
                    gradient_checkpointing=cfg.gradient_checkpointing,
                    num_pipeline_stages=cfg.parallel.pipeline_parallel_size,
                    engine_type=engine_type,
                    offload_optimizer_state=cfg.optimizer.offload,
                    offload_param=cfg.offload,
                    enable_bf16=cfg.enable_bf16,
                    enable_fp16=cfg.enable_fp16,
                    sequence_parallel=cfg.parallel.use_sequence_parallel,
                    enable_async_p2p_communication=cfg.enable_async_p2p,
                ),
            )

        actor_backend = _make_train_backend_config(self.actor, self.ppo.use_stream_pipe_engine)
        critic_backend = _make_train_backend_config(self.critic, False)

        def make_inf_backend(cfg: ModelConfig):
            return ModelBackend(
                "ds_inference",
                args=dict(
                    enable_fp16=(not cfg.enable_bf16),
                    zero_stage=3 if cfg.offload else 0,
                    offload=cfg.offload,
                    enable_bf16=cfg.enable_bf16,
                    engine_type="pipe" if cfg.parallel.pipeline_parallel_size > 1 else "deepspeed",
                    sequence_parallel=cfg.parallel.use_sequence_parallel,
                ),
            )

        ref_backend = make_inf_backend(self.ref)
        rw_backend = make_inf_backend(self.rew)

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
            "flash_actor",
            args={
                **copy.deepcopy(ppo_kwargs),
                "generation_config": generation_kwargs,
                "early_stop_imp_ratio": self.ppo.early_stop_imp_ratio,
                "force_no_logits_mask": False,
                "adv_norm": self.ppo.adv_norm,
            },
        )
        ref_interface = copy.deepcopy(actor_interface)
        ref_interface.args["enable_save"] = False

        if self.ppo.use_stream_pipe_engine:
            actor_interface = ModelInterface(
                "stream_pipe_ppo_actor",
                args={
                    **copy.deepcopy(ppo_kwargs),
                    "generation_config": generation_kwargs,
                    "early_stop_imp_ratio": self.ppo.early_stop_imp_ratio,
                    "force_no_logits_mask": False,
                    "adv_norm": self.ppo.adv_norm,
                },
            )

        critic_interface = ModelInterface(
            "flash_critic",
            args=copy.deepcopy(ppo_kwargs),
        )
        rw_interface = ModelInterface(
            "flash_paired_rw",
            args=dict(
                enable_save=False,
                output_scaling=self.ppo.reward_output_scaling,
                output_bias=self.ppo.reward_output_bias,
            ),
        )

        actor_topo = PipeModelDataParallelTopology(
            num_pp=self.actor.parallel.pipeline_parallel_size,
            num_mp=self.actor.parallel.model_parallel_size,
            num_dp=self.actor.parallel.data_parallel_size,
        )
        critic_topo = PipeModelDataParallelTopology(
            num_pp=self.critic.parallel.pipeline_parallel_size,
            num_mp=self.critic.parallel.model_parallel_size,
            num_dp=self.critic.parallel.data_parallel_size,
        )
        ref_topo = PipeModelDataParallelTopology(
            num_pp=self.ref.parallel.pipeline_parallel_size,
            num_mp=self.ref.parallel.model_parallel_size,
            num_dp=self.ref.parallel.data_parallel_size,
        )
        rw_topo = PipeModelDataParallelTopology(
            num_pp=self.rew.parallel.pipeline_parallel_size,
            num_mp=self.rew.parallel.model_parallel_size,
            num_dp=self.rew.parallel.data_parallel_size,
        )

        model_worker = ([
            ModelWorker(
                seed=self.seed,
                model=actor_model,
                backend=actor_backend,
                interface=actor_interface,
                model_name="actor",
                topo=actor_topo,
                dp_rank=actor_topo.get_coord(i).data,
                pp_rank=actor_topo.get_coord(i).pipe,
                mp_rank=actor_topo.get_coord(i).model,
                cuda_cache_cleanliness=True,
            ) for i in range(self.n_actors)
        ] + [
            ModelWorker(
                seed=self.seed,
                model=critic_model,
                backend=critic_backend,
                interface=critic_interface,
                model_name="critic",
                topo=critic_topo,
                dp_rank=critic_topo.get_coord(i).data,
                pp_rank=critic_topo.get_coord(i).pipe,
                mp_rank=critic_topo.get_coord(i).model,
                cuda_cache_cleanliness=True,
            ) for i in range(self.n_critics)
        ] + [
            ModelWorker(
                seed=self.seed,
                model=ref_model,
                backend=ref_backend,
                interface=ref_interface,
                model_name="ref",
                dp_rank=ref_topo.get_coord(i).data,
                pp_rank=ref_topo.get_coord(i).pipe,
                mp_rank=ref_topo.get_coord(i).model,
                topo=ref_topo,
                cuda_cache_cleanliness=True,
            ) for i in range(self.n_refs)
        ] + [
            ModelWorker(
                seed=self.seed,
                model=rw_model,
                backend=rw_backend,
                interface=rw_interface,
                model_name="reward",
                dp_rank=rw_topo.get_coord(i).data,
                pp_rank=rw_topo.get_coord(i).pipe,
                mp_rank=rw_topo.get_coord(i).model,
                topo=rw_topo,
                cuda_cache_cleanliness=True,
            ) for i in range(self.n_rewards)
        ])

        return ExperimentConfig(
            total_train_epochs=self.total_train_epochs,
            benchmark_steps=getattr(self, "benchmark_steps", None),
            save_frequency_steps=self.save_freq_steps,
            model_rpcs=[rollout, inf_ref_logits, inf_reward, inf_values, train_actor, train_critic],
            data_worker=data_worker,
            model_worker=model_worker,
        )
