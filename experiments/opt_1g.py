import functools
import itertools
from experiments.chat_rlhf import ChatRLHFBenchmarkConfig, ChatRLHFBenchmarkExperiment
from api.config import register_experiment

spec_to_n_params = {
    (9216, 64): "66b",
    (7168, 48): "30b",
    (5120, 40): "13b",
    (4096, 32): "6.7b",
    (2560, 32): "2.7b",
    (2048, 24): "1.3b",
    (1024, 24): "350m",
    (768, 12): "125m",
}
n_params_to_spec = {v: k for k, v in spec_to_n_params.items()}

def n_params_to_dir_name(n_params):
    return f"opt-{n_params_to_spec[n_params][0]}-{n_params_to_spec[n_params][1]}"

# OPT 1 GPU benchmark
actor_params = ["13b", "6.7b", "2.7b", "1.3b", "350m", "125m"]
critic_params = ["1.3b", "350m", "125m"]

n_actor = n_critic = n_reward = n_ref = 1
gpu_per_actor = gpu_per_critic = gpu_per_reward = gpu_per_ref = 1

resource_config = dict(
    n_actors=1,
    n_critics=1,
    n_rewards=1,
    n_refs=1,
    gpu_per_actor=0.25,
    gpu_per_critic=0.25,
    gpu_per_reward=0.25,
    gpu_per_ref=0.25,
)

zero_stage_options = [
    dict(hybrid_engine=True, actor_zero_stage=0, critic_zero_stage=0),
    dict(hybrid_engine=False, actor_zero_stage=0, critic_zero_stage=0),
    dict(hybrid_engine=True, actor_zero_stage=2, critic_zero_stage=2),
    dict(hybrid_engine=True, actor_zero_stage=3, critic_zero_stage=2),
]

offload_options = [
    dict(offload_critic_param = False,
         offload_critic_optimizer_states = False,
         offload_ref = False,
         offload_reward = False),
    dict(offload_critic_param = False,
         offload_critic_optimizer_states = False,
         offload_ref = True,
         offload_reward = False),
    dict(offload_critic_param = False,
         offload_critic_optimizer_states = False,
         offload_ref = True,
         offload_reward = True),
    dict(offload_critic_param = True,
         offload_critic_optimizer_states = True,
         offload_ref = True,
         offload_reward = True),
]
batch_size = [2, 4, 8, 16, 32, 64, 128]
seq_len_options = [
    dict(max_prompt_length=256, max_answer_length=256),
    dict(max_prompt_length=512, max_answer_length=512),
]

# optimization option sweep
config_count = 0
for zero_stage_option, offload_option in itertools.product(zero_stage_options, offload_options):
    config = ChatRLHFBenchmarkConfig(
        actor_model_name=n_params_to_dir_name("1.3b"),
        critic_model_name=n_params_to_dir_name("350m"),
        **resource_config,
        **zero_stage_option,
        **offload_option,
    )
    register_class = functools.partial(ChatRLHFBenchmarkExperiment,
                                       config=config)
    exp_name = f"opt-1x1-optims-{config_count}"
    register_experiment(exp_name, register_class)
    config_count += 1

# parameter size sweep
config_count = 0
for actor_param, critic_param in itertools.product(actor_params, critic_params):
    config = ChatRLHFBenchmarkConfig(
        actor_model_name=n_params_to_dir_name(actor_param),
        critic_model_name=n_params_to_dir_name(critic_param),
        **resource_config,
    )
    register_class = functools.partial(ChatRLHFBenchmarkExperiment,
                                       config=config)
    exp_name = f"opt-1x1-params-{config_count}"
    register_experiment(exp_name, register_class)
    config_count += 1

# batch size + seq len sweep
config_count = 0
for batch_size_option, seq_len_option in itertools.product(batch_size, seq_len_options):
    config = ChatRLHFBenchmarkConfig(
        actor_model_name=n_params_to_dir_name("1.3b"),
        critic_model_name=n_params_to_dir_name("350m"),
        batch_size_per_device=batch_size_option,
        **resource_config,
        **seq_len_option,
    )
    register_class = functools.partial(ChatRLHFBenchmarkExperiment,
                                       config=config)
    exp_name = f"opt-1x1-batchseq-{config_count}"
    register_experiment(exp_name, register_class)
    config_count += 1





