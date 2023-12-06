import functools
import itertools

from api.config import register_experiment
from experiments.benchmark.system.rlhf import ChatRLHFBenchmarkConfig, get_exp_cls

spec_to_n_params = {
    (5120, 60): "19b",
    (6144, 40): "18.4b",
    (6144, 50): "23b",
    (6144, 60): "27.5b",
    (9216, 64): "66b",
    (7168, 48): "30b",
    (5120, 40): "13b",
    (5120, 50): "16b",
    (4096, 32): "6.7b",
    (2560, 32): "2.7b",
    (2048, 24): "1.3b",
    (1024, 24): "350m",
    (768, 12): "125m",
}
n_params_to_spec = {v: k for k, v in spec_to_n_params.items()}


def n_params_to_dir_name(n_params):
    return f"opt-{n_params_to_spec[n_params][0]}-{n_params_to_spec[n_params][1]}"


# OPT 16 GPU benchmark
# actor_params = ["66b", "30b", "13b", "6.7b", "2.7b", "1.3b", "350m", "125m"]
# critic_params = ["6.7b", "2.7b", "1.3b", "350m", "125m"]
actor_params = ["2.7b", "6.7b", "13b", "16b", "18.4b", "19b", "23b"]
critic_params = ["350m"]

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

resource_config_2 = dict(
    n_actors=15,
    n_critics=1,
    n_rewards=1,
    n_refs=1,
    gpu_per_actor=1,
    gpu_per_critic=0.25,
    gpu_per_reward=0.25,
    gpu_per_ref=0.25,
)

zero_stage_options = [
    dict(hybrid_engine=False, actor_zero_stage=2, critic_zero_stage=2),
    dict(hybrid_engine=False, actor_zero_stage=3, critic_zero_stage=2),
    dict(hybrid_engine=True, actor_zero_stage=2, critic_zero_stage=2),
    dict(hybrid_engine=True, actor_zero_stage=3, critic_zero_stage=2),
]

offload_options = [
    dict(offload_critic_param=False,
         offload_critic_optimizer_states=False,
         offload_ref=False,
         offload_reward=False),
    dict(offload_critic_param=False,
         offload_critic_optimizer_states=False,
         offload_ref=True,
         offload_reward=False),
    dict(offload_critic_param=False,
         offload_critic_optimizer_states=False,
         offload_ref=True,
         offload_reward=True),
    dict(offload_critic_param=True,
         offload_critic_optimizer_states=True,
         offload_ref=True,
         offload_reward=True),
]
batch_size = [2, 4, 8]
seq_len_options = [
    dict(max_prompt_length=256, max_answer_length=256),
    dict(max_prompt_length=512, max_answer_length=512),
]

# optimization option sweep
config_count = 0
for zero_stage_option, offload_option in itertools.product(zero_stage_options, offload_options):
    config = ChatRLHFBenchmarkConfig(
        actor_model_name=n_params_to_dir_name("13b"),
        critic_model_name=n_params_to_dir_name("1.3b"),
        **resource_config,
        **zero_stage_option,
        **offload_option,
    )
    register_class = get_exp_cls(config=config)
    exp_name = f"opt-2x8-optims-{config_count}"
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
    register_class = get_exp_cls(config=config)
    exp_name = f"opt-2x8-params-r3-{config_count}"
    register_experiment(exp_name, register_class)
    config_count += 1

# batch size + seq len sweep
config_count = 0
for batch_size_option, seq_len_option in itertools.product(batch_size, seq_len_options):
    config = ChatRLHFBenchmarkConfig(
        actor_model_name=n_params_to_dir_name("13b"),
        critic_model_name=n_params_to_dir_name("1.3b"),
        batch_size_per_device=batch_size_option,
        **resource_config,
        **seq_len_option,
    )
    register_class = get_exp_cls(config=config)
    exp_name = f"opt-2x8-batchseq-{config_count}"
    register_experiment(exp_name, register_class)
    config_count += 1

# offload ref
config_count = 0
for actor_param, critic_param in itertools.product(actor_params, critic_params):
    offload_option = offload_options[1]
    config = ChatRLHFBenchmarkConfig(
        actor_model_name=n_params_to_dir_name(actor_param),
        critic_model_name=n_params_to_dir_name(critic_param),
        **offload_option,
        **resource_config,
    )
    register_class = get_exp_cls(config=config)
    exp_name = f"opt-2x8-offloadref-r2-{config_count}"
    register_experiment(exp_name, register_class)
    config_count += 1

actor_params = ["66b", "30b"]
critic_params = ["1.3b"]

# offload ref + zero 3
config_count = 0
for actor_param, critic_param in itertools.product(actor_params, critic_params):
    offload_option = offload_options[1]
    zero_stage_option = zero_stage_options[3]
    config = ChatRLHFBenchmarkConfig(
        actor_model_name=n_params_to_dir_name(actor_param),
        critic_model_name=n_params_to_dir_name(critic_param),
        **offload_option,
        **resource_config,
    )
    register_class = get_exp_cls(config=config)
    exp_name = f"opt-2x8-zero3offloadref-r2-{config_count}"
    register_experiment(exp_name, register_class)
    config_count += 1

# 30b test

zero_stage_option = dict(hybrid_engine=False, actor_zero_stage=3, critic_zero_stage=2)
offload_option = dict(offload_critic_param=False,
                      offload_critic_optimizer_states=False,
                      offload_ref=True,
                      offload_reward=False)

config = ChatRLHFBenchmarkConfig(
    actor_model_name=n_params_to_dir_name("30b"),
    critic_model_name=n_params_to_dir_name("350m"),
    **zero_stage_option,
    **offload_option,
    **resource_config,
)
register_class = get_exp_cls(config=config)
exp_name = f"opt-2x8-30b-TEST"
register_experiment(exp_name, register_class)
