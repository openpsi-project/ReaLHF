import functools

from omegaconf import MISSING

from experiments.common.config_dataset import PromptOnlyDatasetConfig
from experiments.common.config_model import get_flash_mqat_model_config, ModelConfig
from api.config import *
from api.dfg import ModelInterfaceType, ModelRPC
from base.topology import PipeModelDataParallelTopology
import base.logging as logging
from experiments.common.ppo_exp import PPOConfig, PPOHyperparmeters


rollout = ModelRPC(
    "actor",
    ModelInterfaceType.GENERATE,
    input_data=["prompts", "prompt_att_mask"],
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
    dp_broker_type="packed",
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

ppo_benchmark_hyperparam = PPOHyperparmeters(
    max_new_tokens=1024,
    min_new_tokens=10,
    greedy=False,
    top_p=0.9,
    top_k=2048,
    temperature=1.2,
    ppo_n_minibatches=4,
    early_stop_imp_ratio=1e5,
)

ppo_benchmark_dataset = PromptOnlyDatasetConfig(
    max_prompt_len=1024,
    batch_size=512,
    path="/lustre/fw/datasets/antropic-hh/ppo_prompt_only.jsonl",
)


@dataclasses.dataclass
class PPOSysExperiment(Experiment):
    master_nodelist: str
    actor_nodelist: str
    critic_nodelist: str
    ref_nodelist: str
    rew_nodelist: str

    seed: int = 1
    benchmark_steps: int = 20

    actor: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    critic: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    ref: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    rew: ModelConfig = dataclasses.field(default_factory=ModelConfig)

    def __post_init__(self):
        self.base_config = PPOConfig(
            seed=self.seed,
            actor=self.actor,
            critic=self.critic,
            ref=self.ref,
            rew=self.rew,
            ppo=ppo_benchmark_hyperparam,
            dataset=ppo_benchmark_dataset,
        )

    def scheduling_setup(self) -> ExperimentScheduling:
        base_setup = self.base_config.scheduling_setup()
        base_setup.data_worker.scheduling.nodelist = "QH-com47"
        base_setup.master_worker.scheduling.nodelist = self.master_nodelist
        base_setup.model_worker[0].scheduling.nodelist = self.actor_nodelist
        base_setup.model_worker[1].scheduling.nodelist = self.critic_nodelist
        base_setup.model_worker[2].scheduling.nodelist = self.ref_nodelist
        base_setup.model_worker[3].scheduling.nodelist = self.rew_nodelist
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
        for m in base_setup.model_worker[offset : offset + self.base_config.n_actors]:
            m.model = actor_model
            m.interface.args["force_no_logits_mask"] = True
            m.backend.args["num_inf_pipeline_mbs"] = self.actor.parallel.num_inf_pipeline_mbs
            m.backend.args["enable_hybrid_engine"] = self.actor.optimizer.use_hybrid_engine
            m.backend.args["max_out_tokens"] = ppo_benchmark_hyperparam.max_new_tokens
        offset += self.base_config.n_actors
        for m in base_setup.model_worker[offset : offset + self.base_config.n_critics]:
            m.model = critic_model
        offset += self.base_config.n_critics
        for m in base_setup.model_worker[offset : offset + self.base_config.n_rewards]:
            m.model = rw_model
        offset += self.base_config.n_rewards
        for m in base_setup.model_worker[offset : offset + self.base_config.n_refs]:
            m.model = ref_model
        assert offset + self.base_config.n_refs == len(base_setup.model_worker)

        return ExperimentConfig(
            total_train_epochs=1,
            benchmark_steps=self.benchmark_steps,
            save_frequency_seconds=None,
            model_rpcs=[rollout, inf_ref_logits, inf_reward, inf_values, train_actor, train_critic],
            data_worker=base_setup.data_worker,
            model_worker=base_setup.model_worker,
        )
