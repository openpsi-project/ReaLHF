import functools

from api.config import *
from api.dfg import ModelInterfaceType, ModelRPC
from base.topology import PipeModelDataParallelTopology
from experiments.common.config_utils import get_flash_mqat_model_config
import base.logging as logging

logger = logging.getLogger("PPO exp", "colored")

rollout = ModelRPC(
    "actor",
    ModelInterfaceType.GENERATE,
    input_data=["prompts", "prompt_att_mask"],
    output_data=[
        "seq_no_eos_mask",
        "packed_seq",
        "cu_seqlens",
        "packed_logprobs",
        "packed_logits_mask",
        "prompt_mask",
    ],
)
inf_reward = ModelRPC(
    "reward",
    ModelInterfaceType.INFERENCE,
    input_data=["packed_seq", "cu_seqlens"],
    input_key_remap={"packed_seq": "packed_input_ids"},
    output_data=["scores"],
    output_key_remap={"scores": "rewards"},
    dp_broker_type="packed",
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
)

inf_values = ModelRPC(
    "critic",
    ModelInterfaceType.INFERENCE,
    input_data=["packed_seq", "cu_seqlens", "seq_no_eos_mask"],
    output_data=["scores"],
    output_key_remap={"scores": "values"},
    dp_broker_type="packed",
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
)


@dataclasses.dataclass
class PPOExperiment(Experiment):
    sft_model_path: Optional[str] = None
    rew_model_path: Optional[str] = None

    tokenizer_path: Optional[str] = None  # Since we use SFT model, we need to specify HF tokenizer path

    seed: int = 1
    total_train_epochs: int = 1
    save_freq_steps: int = 20

    is_sft_lora: bool = False
    is_sft_pipe: bool = False
    sft_base_model_type: Optional[str] = None
    sft_lora_path: Optional[str] = None

    is_rew_lora: bool = False
    is_rew_pipe: bool = False
    rew_base_model_type: Optional[str] = None
    rew_lora_path: Optional[str] = None
    rew_head_path: Optional[str] = None
    # actor model
    actor_dp_size: int = 1
    actor_mp_size: int = 1
    actor_pp_size: int = 1
    actor_use_lora: bool = False
    actor_lora_scaling: float = 32.0
    actor_lora_dim: int = 32
    actor_enable_fp16: bool = True
    actor_enable_bf16: bool = False
    actor_gradient_checkpointing: bool = True
    actor_num_pipeline_micro_batches: Optional[int] = None
    actor_use_sequence_parallel: bool = False
    # critic model
    critic_dp_size: int = 1
    critic_mp_size: int = 1
    critic_pp_size: int = 1
    critic_use_lora: bool = False
    critic_lora_scaling: float = 32.0
    critic_lora_dim: int = 32
    critic_enable_fp16: bool = True
    critic_enable_bf16: bool = False
    critic_gradient_checkpointing: bool = True
    critic_num_pipeline_micro_batches: Optional[int] = None
    critic_use_sequence_parallel: bool = False
    # ref model
    ref_dp_size: int = 1
    offload_ref: bool = False
    ref_enable_bf16: bool = False
    ref_mp_size: int = 1
    ref_pp_size: int = 1
    ref_num_pipeline_micro_batches: Optional[int] = None
    ref_use_sequence_parallel: bool = False
    # reward model
    rew_dp_size: int = 1  # Since reward model is usually not large, we disable PP and TP for it.
    offload_reward: bool = False
    rew_enable_bf16: bool = False
    # dataset
    max_prompt_len: int = 256
    batch_size: int = 256
    dataset_path: str = "/lustre/fw/datasets/imdb/rl/ppo_prompt.jsonl"
    # actor optimizer
    actor_lr: float = 9.65e-6
    actor_weight_decay: float = 0.0
    actor_adam_betas: tuple = (0.9, 0.95)
    actor_lr_scheduler_type: str = "linear"
    actor_warmup_proportion: float = 0.075
    actor_adam_eps: float = 1e-5
    actor_min_lr_ratio: float = 0.0
    actor_zero_stage: int = 2
    actor_partition_method: Optional[str] = "parameters"
    offload_actor_param: bool = False
    offload_actor_optimizer_state: bool = False
    # critic optimizer
    critic_lr: float = 5e-6
    critic_weight_decay: float = 0.0
    critic_adam_betas: tuple = (0.9, 0.95)
    critic_lr_scheduler_type: str = "linear"
    critic_warmup_proportion: float = 0.075
    critic_adam_eps: float = 1e-5
    critic_min_lr_ratio: float = 0.0
    critic_zero_stage: int = 2
    critic_partition_method: Optional[str] = "parameters"
    offload_critic_param: bool = False
    offload_critic_optimizer_states: bool = False
    # ppo
    rew_output_scaling: float = 1.0
    rew_output_bias: float = 0.0
    max_new_tokens: int = 512
    min_new_tokens: int = 10
    greedy: bool = False
    top_p: float = 1.0
    top_k: int = int(1e9)
    temperature: float = 1.0
    ppo_n_minibatches: int = 8
    kl_ctl: float = 0.1
    adv_norm: bool = False
    discount: float = 1.0
    gae_lambda: float = 1.0
    eps_clip: float = 0.2
    value_eps_clip: float = 0.2
    max_reward_clip: float = 20.0
    use_adaptive_kl_ctl: bool = False
    early_stop_imp_ratio: float = 5.0
    value_norm: bool = False
    value_norm_type: str = dataclasses.field(metadata={"choices": ["exp", "ma"]}, default="exp")
    value_norm_beta: float = 0.99995
    value_norm_eps: float = 1e-5
    # benchmark
    hybrid_engine: bool = False
    benchmark: bool = False

    def __post_init__(self):
        if self.actor_pp_size < 1 or self.actor_dp_size < 1 or self.actor_mp_size < 1:
            raise ValueError("pp_size, mp_size and dp_size must be positive integers.")
        if self.actor_pp_size > 1 and self.actor_use_lora:
            raise ValueError("Use LoRA with pipeline parallel is not supported.")
        if self.critic_pp_size < 1 or self.critic_dp_size < 1 or self.critic_mp_size < 1:
            raise ValueError("pp_size, mp_size and dp_size must be positive integers.")
        if self.critic_pp_size > 1 and self.critic_use_lora:
            raise ValueError("Use LoRA with pipeline parallel is not supported.")

        if self.is_sft_lora and (self.sft_lora_path is None or self.sft_base_model_type is None):
            raise ValueError("sft_lora_path and base_model_type must be specified when is_sft_lora is True.")
        if self.is_rew_lora and (self.rew_lora_path is None or self.rew_base_model_type is None
                                 or self.rew_head_path is None):
            raise ValueError(
                "rew_lora_path, rew_base_model_type and rew_head_path must be specified when is_rw_lora is True."
            )

        if self.actor_enable_bf16 and (self.actor_pp_size > 1 or self.actor_mp_size):
            raise ValueError("Use bf16 with pipeline parallel or model parallel is not supported.")
        if self.critic_enable_bf16 and (self.critic_pp_size > 1 or self.critic_mp_size):
            raise ValueError("Use bf16 with pipeline parallel or model parallel is not supported.")
        if self.ref_enable_bf16 and (self.ref_pp_size > 1 or self.ref_mp_size > 1):
            raise ValueError("Use bf16 with pipeline parallel or model parallel is not supported.")

        self.n_actors = int(self.actor_pp_size * self.actor_dp_size * self.actor_mp_size)
        self.n_critics = int(self.critic_pp_size * self.critic_dp_size * self.critic_mp_size)
        self.n_rewards = self.rew_dp_size
        self.n_refs = int(self.ref_pp_size * self.ref_dp_size * self.ref_mp_size)

        if self.benchmark:
            logger.warning("Benchmark mode is enabled. This will forcely set some experiment parameters. "
                           "Please check experiment configuration for details.")
            self.benchmark_steps = 30
            self.save_freq_steps = None
            self.is_sft_lora = self.is_sft_pipe = False
            self.is_rew_lora = self.is_rew_pipe = False
            self.tokenizer_path = self.sft_model_path
            logger.warning(f"Benchmark enabled. Please ensure sft_model_path={self.sft_model_path} and "
                           f"{self.rew_model_path} are a valid HF models (instead SFT or RW models). "
                           f"All models in the benchmark mode will initialize directly from HF models. "
                           "Also make sure the tokenizers are the same. We don't do any check here.")
            self.actor_use_lora = self.critic_use_lora = False
            self.dataset_path = "/lustre/fw/datasets/imdb/rl/ppo_prompt.jsonl"
            self.use_adaptive_kl_ctl = False
            self.early_stop_imp_ratio = 1e10
            self.top_k = 1000000
            self.top_p = 1.0

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
                    cpu=4,
                    mem=100000,
                ),
            ),
            model_worker=[
                TasksGroup(
                    count=self.n_actors,
                    scheduling=Scheduling.model_worker_default(
                        cpu=4,
                        gpu=1,
                        gpu_type="tesla",
                        mem=100000,
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
            "prompt",
            args=dict(
                dataset_path=self.dataset_path,
                max_prompt_len=self.max_prompt_len,
                pad_to_max_length=False,  # since we only have one dataloader, it's ok to use without padding
            ),
        )
        dataloader = DataLoader(
            "default",
            args=dict(
                shuffle=True,
                drop_last=True,
                batch_size=self.batch_size,
            ),
        )
        data_worker = [
            DataWorker(
                tokenizer_name_or_path=self.tokenizer_path,
                datasets=[dataset],
                dataloader=dataloader,
                seed=self.seed,
            )
        ]

        generation_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            min_new_tokens=self.min_new_tokens,
            greedy=self.greedy,
            top_p=self.top_p,
            top_k=self.top_k,
            temperature=self.temperature,
        )

        if self.benchmark or self.is_sft_lora:
            actor_model_type = ref_model_type = self.sft_base_model_type
        else:
            actor_model_type = "self"
            if self.is_sft_pipe and self.ref_mp_size == 1 and self.ref_pp_size == 1:
                ref_model_type = "pipe"
            else:
                ref_model_type = "self"

        actor_model = get_flash_mqat_model_config(
            model_path=self.sft_model_path,
            from_model_type=actor_model_type,
            tokenizer_path=self.tokenizer_path,
            pp_size=self.actor_pp_size,
            mp_size=self.actor_mp_size,
            dp_size=self.actor_dp_size,
            is_critic=False,
            use_lora=self.actor_use_lora,
            lora_dim=self.actor_lora_dim,
            lora_scaling=self.actor_lora_scaling,
            is_sft_lora=self.is_sft_lora,
            sft_lora_path=self.sft_lora_path,
            partition_method=self.actor_partition_method,
            sequence_parallel=self.actor_use_sequence_parallel,
            dtype="bf16" if self.actor_enable_bf16 else "fp16",
        )

        ref_model = get_flash_mqat_model_config(
            model_path=self.sft_model_path,
            from_model_type=ref_model_type,
            tokenizer_path=self.tokenizer_path,
            pp_size=self.ref_pp_size,
            mp_size=self.ref_mp_size,
            dp_size=self.ref_dp_size,
            is_critic=False,
            use_lora=False,
            lora_dim=self.actor_lora_dim,
            lora_scaling=self.actor_lora_scaling,
            is_sft_lora=self.is_sft_lora,
            sft_lora_path=self.sft_lora_path,
            sequence_parallel=self.ref_use_sequence_parallel,
            dtype="bf16" if self.ref_enable_bf16 else "fp16",
        )

        if self.benchmark:
            critic_from_type = rew_from_type = self.rew_base_model_type
        elif self.is_rew_lora:
            if self.is_sft_lora:
                critic_from_type = rew_from_type = self.rew_base_model_type
            else:
                critic_from_type = rew_from_type = "sft"
        else:
            if self.is_sft_pipe and self.critic_mp_size == 1 and self.critic_pp_size == 1:
                critic_from_type = "pipe"
            else:
                critic_from_type = "self"
            if self.is_sft_pipe:
                rew_from_type = "pipe"
            else:
                rew_from_type = "self"

        rw_model = get_flash_mqat_model_config(
            model_path=self.rew_model_path,
            from_model_type=rew_from_type,
            tokenizer_path=self.tokenizer_path,
            pp_size=1,
            mp_size=1,
            dp_size=self.rew_dp_size,
            is_critic=True,
            use_lora=False,
            lora_dim=self.critic_lora_dim,
            lora_scaling=self.critic_lora_scaling,
            is_sft_lora=self.is_sft_lora,
            sft_lora_path=self.sft_lora_path,
            is_rew_lora=self.is_rew_lora,
            rew_lora_path=self.rew_lora_path,
            v_head_path=self.rew_head_path,
            dtype="bf16" if self.rew_enable_bf16 else "fp16",
        )

        critic_model = get_flash_mqat_model_config(
            model_path=self.rew_model_path,
            from_model_type=critic_from_type,
            tokenizer_path=self.tokenizer_path,
            pp_size=self.critic_pp_size,
            mp_size=self.critic_mp_size,
            dp_size=self.critic_dp_size,
            is_critic=True,
            use_lora=self.critic_use_lora,
            lora_dim=self.critic_lora_dim,
            lora_scaling=self.critic_lora_scaling,
            is_sft_lora=self.is_sft_lora,
            sft_lora_path=self.sft_lora_path,
            is_rew_lora=self.is_rew_lora,
            rew_lora_path=self.rew_lora_path,
            v_head_path=self.rew_head_path,
            partition_method=self.critic_partition_method,
            sequence_parallel=self.critic_use_sequence_parallel,
            dtype="bf16" if self.critic_enable_bf16 else "fp16",
            # NOTE: critic is not scaled as rewards
        )

        # actor train backend
        actor_backend = ModelBackend(
            "ds_train",
            args=dict(optimizer_name="adam",
                      optimizer_config=dict(
                          lr=self.actor_lr,
                          weight_decay=self.actor_weight_decay,
                          eps=self.actor_adam_eps,
                          betas=self.actor_adam_betas,
                      ),
                      lr_scheduler_type=self.actor_lr_scheduler_type,
                      warmup_steps_proportion=self.actor_warmup_proportion,
                      min_lr_ratio=self.actor_min_lr_ratio,
                      zero_stage=min(1, self.actor_zero_stage) if self.actor_pp_size > 0 else 2,
                      gradient_checkpointing=self.actor_gradient_checkpointing,
                      num_pipeline_stages=self.actor_pp_size,
                      engine_type="pipe" if self.actor_pp_size > 1 else "deepspeed",
                      offload_param=self.offload_actor_param,
                      offload_optimizer_state=self.offload_actor_optimizer_state,
                      enable_hybrid_engine=self.hybrid_engine,
                      num_pipeline_micro_batches=self.actor_num_pipeline_micro_batches,
                      enable_fp16=self.actor_enable_fp16,
                      enable_bf16=self.actor_enable_bf16,
                      sequence_parallel=self.actor_use_sequence_parallel),
        )
        # critic train backend
        critic_backend = ModelBackend(
            "ds_train",
            args=dict(optimizer_name="adam",
                      optimizer_config=dict(
                          lr=self.critic_lr,
                          weight_decay=self.critic_weight_decay,
                          eps=self.critic_adam_eps,
                          betas=self.critic_adam_betas,
                      ),
                      lr_scheduler_type=self.critic_lr_scheduler_type,
                      warmup_steps_proportion=self.critic_warmup_proportion,
                      min_lr_ratio=self.critic_min_lr_ratio,
                      zero_stage=min(1, self.critic_zero_stage) if self.critic_pp_size > 0 else 2,
                      gradient_checkpointing=self.critic_gradient_checkpointing,
                      num_pipeline_stages=self.critic_pp_size,
                      engine_type="pipe" if self.critic_pp_size > 1 else "deepspeed",
                      offload_param=self.offload_critic_param,
                      offload_optimizer_state=self.offload_critic_optimizer_states,
                      num_pipeline_micro_batches=self.critic_num_pipeline_micro_batches,
                      enable_fp16=self.critic_enable_fp16,
                      enable_bf16=self.critic_enable_bf16,
                      sequence_parallel=self.critic_use_sequence_parallel),
        )

        ref_backend = ModelBackend(
            "ds_inference",
            args=dict(
                enable_fp16=(not self.ref_enable_bf16),
                zero_stage=3 if self.offload_ref else 0,
                offload=self.offload_ref,
                enable_bf16=self.ref_enable_bf16,
                engine_type="pipe" if self.ref_pp_size > 1 else "deepspeed",
                num_pipeline_micro_batches=self.ref_num_pipeline_micro_batches,
                sequence_parallel=self.ref_use_sequence_parallel,
            ),
        )
        rw_backend = ModelBackend(
            "ds_inference",
            args=dict(
                enable_fp16=(not self.rew_enable_bf16),
                zero_stage=3 if self.offload_reward else 0,
                offload=self.offload_reward,
                enable_bf16=self.rew_enable_bf16,
            ),
        )

        ppo_kwargs = dict(
            n_minibatches=self.ppo_n_minibatches,
            kl_ctl=self.kl_ctl,
            discount=self.discount,
            gae_lambda=self.gae_lambda,
            eps_clip=self.eps_clip,
            value_eps_clip=self.value_eps_clip,
            max_reward_clip=self.max_reward_clip,
            adaptive_kl_ctl=self.use_adaptive_kl_ctl,
            value_norm=self.value_norm,
            value_norm_type=self.value_norm_type,
            value_norm_beta=self.value_norm_beta,
            value_norm_eps=self.value_norm_eps,
        )

        actor_interface = ModelInterface(
            "flash_actor",
            args={
                **copy.deepcopy(ppo_kwargs),
                "generation_config": generation_kwargs,
                "early_stop_imp_ratio": self.early_stop_imp_ratio,
                "force_no_logits_mask": self.benchmark,  # For benchmark only
                "adv_norm": self.adv_norm,
            },
        )
        ref_interface = copy.deepcopy(actor_interface)
        ref_interface.args["enable_save"] = False

        critic_interface = ModelInterface(
            "flash_critic",
            args=copy.deepcopy(ppo_kwargs),
        )
        rw_interface = ModelInterface("flash_paired_rw",
                                      args=dict(enable_save=False,
                                                output_scaling=self.rew_output_scaling,
                                                output_bias=self.rew_output_bias))

        actor_topo = PipeModelDataParallelTopology(num_pp=self.actor_pp_size,
                                                   num_mp=self.actor_mp_size,
                                                   num_dp=self.actor_dp_size)
        critic_topo = PipeModelDataParallelTopology(num_pp=self.critic_pp_size,
                                                    num_mp=self.critic_mp_size,
                                                    num_dp=self.critic_dp_size)
        ref_topo = PipeModelDataParallelTopology(num_pp=self.ref_pp_size,
                                                 num_mp=self.ref_mp_size,
                                                 num_dp=self.ref_dp_size)
        rw_topo = PipeModelDataParallelTopology(num_pp=1, num_mp=1, num_dp=self.rew_dp_size)
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
                cuda_cache_cleanliness=(not self.benchmark),
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
                cuda_cache_cleanliness=(not self.benchmark),
            ) for i in range(self.n_critics)
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
                cuda_cache_cleanliness=(not self.benchmark),
            ) for i in range(self.n_rewards)
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
                cuda_cache_cleanliness=(not self.benchmark),
            ) for i in range(self.n_refs)
        ])

        return ExperimentConfig(
            total_train_epochs=self.total_train_epochs,
            benchmark_steps=getattr(self, "benchmark_steps", None),
            save_frequency_steps=self.save_freq_steps,
            model_rpcs=[rollout, inf_ref_logits, inf_reward, inf_values, train_actor, train_critic],
            data_worker=data_worker,
            model_worker=model_worker,
        )
