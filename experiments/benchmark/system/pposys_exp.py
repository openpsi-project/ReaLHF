import functools

from omegaconf import MISSING

from api.config.config_system import *
from api.config.dfg import ModelInterfaceType, ModelRPC
from base.topology import PipeModelDataParallelTopology
from experiments.common.config_dataset import PromptOnlyDatasetConfig
from experiments.common.config_model import get_flash_mqat_model_config, ModelConfig
from experiments.common.ppo_exp import PPOConfig, PPOHyperparmeters
import base.logging as logging

rollout = ModelRPC(
    "actor",
    ModelInterfaceType.GENERATE,
    input_data=["packed_prompts", "prompt_cu_seqlens"],
    output_data=[
        "seq_no_eos_mask",
        "packed_seq",
        "cu_seqlens",
        "packed_logprobs",
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
)

inf_ref_logits = ModelRPC(
    "ref",
    ModelInterfaceType.INFERENCE,
    input_data=[
        "packed_seq",
        "cu_seqlens",
    ],
    output_data=["logprobs"],
    output_key_remap={"logprobs": "packed_ref_logprobs"},
)

inf_values = ModelRPC(
    "critic",
    ModelInterfaceType.INFERENCE,
    input_data=["packed_seq", "cu_seqlens", "seq_no_eos_mask"],
    output_data=["scores"],
    output_key_remap={"scores": "values"},
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
    ],
    log_return_value=True,
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
    log_return_value=True,
)

ppo_benchmark_hyperparam = PPOHyperparmeters(
    max_new_tokens=-1,  # will be overwritten
    min_new_tokens=-1,  # will be overwritten
    greedy=True,
    top_p=0.9,
    top_k=2048,
    temperature=1.2,
    ppo_n_minibatches=4,
    early_stop_imp_ratio=1e5,
)

ppo_benchmark_dataset = PromptOnlyDatasetConfig(
    max_prompt_len=-1,  # will be overwritten
    batch_size=-1,  # will be overwritten
    path="/lustre/fw/datasets/antropic-hh/ppo_prompt_only.jsonl",
)


@dataclasses.dataclass
class PPOSysExperiment(Experiment):
    master_nodelist: str
    actor_nodelist: str
    critic_nodelist: str
    ref_nodelist: str
    rew_nodelist: str

    max_answer_len: int
    batch_size: Optional[int] = 512
    max_prompt_len: int = 256

    seed: int = 1
    benchmark_steps: int = 10

    actor: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    critic: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    ref: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    rew: ModelConfig = dataclasses.field(default_factory=ModelConfig)

    def __post_init__(self):
        global ppo_benchmark_dataset
        dataset = copy.deepcopy(ppo_benchmark_dataset)
        dataset.batch_size = self.batch_size
        dataset.max_prompt_len = self.max_prompt_len
        global ppo_benchmark_hyperparam
        hyperparam = copy.deepcopy(ppo_benchmark_hyperparam)
        hyperparam.max_new_tokens = self.max_answer_len
        hyperparam.min_new_tokens = self.max_answer_len
        self.base_config = PPOConfig(
            seed=self.seed,
            actor=self.actor,
            critic=self.critic,
            ref=self.ref,
            rew=self.rew,
            ppo=hyperparam,
            dataset=dataset,
        )

    def scheduling_setup(self) -> ExperimentScheduling:
        base_setup = self.base_config.scheduling_setup()
        exclude = "QH-com02,QH-com03,QH-com29,QH-com35"
        base_setup.master_worker.scheduling.nodelist = self.master_nodelist
        base_setup.master_worker.scheduling.exclude = exclude
        base_setup.model_worker[0].scheduling.nodelist = self.actor_nodelist
        base_setup.model_worker[1].scheduling.nodelist = self.critic_nodelist
        base_setup.model_worker[2].scheduling.nodelist = self.rew_nodelist
        base_setup.model_worker[3].scheduling.nodelist = self.ref_nodelist
        for s in base_setup.model_worker:
            s.scheduling.exclude = exclude
        return base_setup

    def initial_setup(self) -> ExperimentConfig:
        base_setup = self.base_config.initial_setup()

        def _make_model_config(cfg: ModelConfig, is_critic: bool):
            return get_flash_mqat_model_config(
                from_type="hf_as_actor" if not is_critic else "hf_as_critic",
                model_path=cfg.path,
                hf_model_type=cfg.type,
                tokenizer_path=cfg.base_model_path,
                use_pipe=(cfg.parallel.pipeline_parallel_size > 1),
                dtype="bf16" if cfg.enable_bf16 else "fp16",
                sequence_parallel=cfg.parallel.use_sequence_parallel,
                partition_method=cfg.parallel.partition_method,
                lora=cfg.lora,
            )

        actor_model = _make_model_config(self.actor, False)
        ref_model = _make_model_config(self.ref, False)
        critic_model = _make_model_config(self.critic, True)
        rw_model = _make_model_config(self.rew, True)

        offset = 0
        for m in base_setup.model_worker[offset:offset + self.base_config.n_actors]:
            m.model = actor_model
            m.interface.args["force_no_logits_mask"] = True
            m.backend.args["enable_hybrid_engine"] = self.actor.optimizer.use_hybrid_engine
            m.backend.args["max_out_tokens"] = ppo_benchmark_hyperparam.max_new_tokens
        offset += self.base_config.n_actors
        for m in base_setup.model_worker[offset:offset + self.base_config.n_critics]:
            m.model = critic_model
        offset += self.base_config.n_critics
        for m in base_setup.model_worker[offset:offset + self.base_config.n_refs]:
            m.model = ref_model
        offset += self.base_config.n_refs
        for m in base_setup.model_worker[offset:offset + self.base_config.n_rewards]:
            m.model = rw_model
        assert offset + self.base_config.n_rewards == len(base_setup.model_worker)

        global train_actor
        train_actor = copy.deepcopy(train_actor)
        if self.actor.parallel.pipeline_parallel_size > 1:
            pp_nmbs = (self.actor.parallel.pipe_mbs_config.train_step
                       if self.actor.parallel.pipe_mbs_config.train_step is not None else
                       self.actor.parallel.pipeline_parallel_size * 2)
            train_actor.min_n_seqs_per_dp = ppo_benchmark_hyperparam.ppo_n_minibatches * pp_nmbs
        else:
            train_actor.min_n_seqs_per_dp = ppo_benchmark_hyperparam.ppo_n_minibatches
        train_actor.min_n_seqs = self.batch_size
        train_actor.max_n_seqs = self.batch_size + 1

        global rollout
        rollout = copy.deepcopy(rollout)
        rollout.min_n_seqs = self.batch_size
        rollout.max_n_seqs = self.batch_size + 1
        rollout.max_concurrent_calls = 1

        global inf_ref_logits
        inf_ref_logits = copy.deepcopy(inf_ref_logits)
        inf_ref_logits.min_n_seqs = self.batch_size
        inf_ref_logits.max_n_seqs = self.batch_size + 1

        global inf_reward
        inf_reward = copy.deepcopy(inf_reward)
        inf_reward.min_n_seqs = self.batch_size
        inf_reward.max_n_seqs = self.batch_size + 1

        global inf_values
        inf_values = copy.deepcopy(inf_values)
        inf_values.min_n_seqs = self.batch_size
        inf_values.max_n_seqs = self.batch_size + 1

        global train_critic
        train_critic = copy.deepcopy(train_critic)
        train_critic.min_n_seqs = self.batch_size
        train_critic.max_n_seqs = self.batch_size + 1

        return ExperimentConfig(
            total_train_epochs=1,
            benchmark_steps=self.benchmark_steps,
            save_frequency_seconds=None,
            model_rpcs=[rollout, inf_ref_logits, inf_reward, inf_values, train_actor, train_critic],
            data_worker=base_setup.data_worker,
            model_worker=base_setup.model_worker,
        )
