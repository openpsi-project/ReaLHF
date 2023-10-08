import functools
from experiments.chat_rlhf import ChatRLHFBenchmarkConfig, ChatRLHFBenchmarkExperiment
from api.config import register_experiment

resource_config = dict(
    n_actors=14,
    n_critics=1,
    n_rewards=1,
    n_refs=1,
    gpu_per_actor=1,
    gpu_per_critic=1,
    gpu_per_reward=0.25,
    gpu_per_ref=0.25,
)

zero_stage_option = dict(hybrid_engine=False, actor_zero_stage=2, critic_zero_stage=2)
offload_option = dict(offload_critic_param = False,
    offload_critic_optimizer_states = False,
    offload_ref = False,
    offload_reward = False
)

config = ChatRLHFBenchmarkConfig(
        actor_model_name="starcoder-6144-40",
        critic_model_name="starcoder-1536-10",
        **resource_config,
        **zero_stage_option,
        **offload_option,
)
register_class = functools.partial(ChatRLHFBenchmarkExperiment,
                                   config=config)
exp_name = f"starcoder-2x8-16b"
register_experiment(exp_name, register_class)