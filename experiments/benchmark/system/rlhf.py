import functools
import itertools
from api.config import *
import dataclasses
from experiments.common.ppo_exp import PPOExperiment
from api.config import register_experiment

EXPR_DEADLINE = None
EXPR_TIME_LIMIT = None


def customized_schedule(schedule: ExperimentScheduling):
    """Change the scheduling configuration of an existing experiment.

    We can specify nodelist/exclude/deadline/timelimit or bind multiple models into one GPU with this wrapper.

    Usage:
    register_experiment("my-experiment",
        functools.partial(
            @customized_schedule(schedule=ExperimentScheduling(...))(SFTExperiment),
            model_path=...,
        )
    )
    """

    def wrapper(exp_cls):
        class SchedulingPlugin:
            def scheduling_setup(self) -> ExperimentScheduling:
                return schedule

        class _WrappedExp(exp_cls, SchedulingPlugin):
            pass

        return _WrappedExp

    return wrapper


@dataclasses.dataclass
class ChatRLHFBenchmarkConfig:
    # resource
    n_actors: int = 2
    n_critics: int = 2
    n_rewards: int = 1
    n_refs: int = 1
    seed: int = 1
    gpu_per_actor: float = 1
    gpu_per_critic: float = 1
    gpu_per_reward: float = 1
    gpu_per_ref: float = 1
    # optimization options
    init_from_scratch: bool = False
    actor_model_name: str = "Llama-2-7b-hf"
    critic_model_name: str = "Llama-2-7b-hf"
    actor_zero_stage: int = 2
    critic_zero_stage: int = 2
    hybrid_engine: bool = False
    batch_size_per_device: int = 1
    max_prompt_length: int = 256
    max_answer_length: int = 256
    offload_actor_param: bool = False
    offload_actor_optimizer_state: bool = False
    offload_critic_param: bool = False
    offload_critic_optimizer_states: bool = False
    offload_ref: bool = False
    offload_reward: bool = False
    gradient_checkpointing: bool = False


def get_schduling_config(config: ChatRLHFBenchmarkConfig) -> ExperimentScheduling:
    # setup scheduling config
    # actor critic reward ref
    task_groups = []
    last_gpu_per_type = None
    last_n_types = 0
    for model_type in ["actor", "critic", "reward", "ref"]:
        n_types = getattr(config, f"n_{model_type}s")
        gpu_per_type = getattr(config, f"gpu_per_{model_type}")
        # print("last_gpu_per_type", last_gpu_per_type)
        # print("last_n_types", last_n_types)
        if gpu_per_type < 1:
            assert n_types == 1, "n_type must be 1 if gpu_per_type < 1"
        if last_gpu_per_type is None:
            last_gpu_per_type = gpu_per_type
            last_n_types = n_types
            continue
        if last_gpu_per_type != gpu_per_type:
            new_task_group = TasksGroup(
                count=last_n_types,
                scheduling=Scheduling.model_worker_default(
                    cpu=4,
                    gpu=last_gpu_per_type,
                    gpu_type="tesla",
                    mem=int(100000 * last_gpu_per_type),
                    nodelist='QH-com[44-45]',
                    deadline=EXPR_DEADLINE,
                    time_limit=EXPR_TIME_LIMIT,
                ),
            )
            task_groups.append(new_task_group)
            last_gpu_per_type = gpu_per_type
            last_n_types = n_types
        else:
            last_n_types += n_types

    new_task_group = TasksGroup(
        count=last_n_types,
        scheduling=Scheduling.model_worker_default(
            cpu=4,
            gpu=last_gpu_per_type,
            gpu_type="tesla",
            mem=100000,
            nodelist='QH-com[44-45]',
            deadline=EXPR_DEADLINE,
            time_limit=EXPR_TIME_LIMIT,
        ),
    )
    task_groups.append(new_task_group)
    return ExperimentScheduling(
        data_worker=TasksGroup(
            count=1,
            scheduling=Scheduling.data_worker_default(
                cpu=4,
                mem=10000,
                begin=None,
                nodelist='QH-com[44-45]',
                deadline=EXPR_DEADLINE,
                time_limit=EXPR_TIME_LIMIT,
            ),
        ),
        master_worker=TasksGroup(
            count=1,
            scheduling=Scheduling.master_worker_default(
                cpu=4,
                mem=10000,
                begin=None,
                nodelist='QH-com[44-45]',
                deadline=EXPR_DEADLINE,
                time_limit=EXPR_TIME_LIMIT,
            ),
        ),
        model_worker=task_groups,
    )


def get_exp_cls(config: ChatRLHFBenchmarkConfig):
    args = dict(
        benchmark=True,
        actor_dp_size=config.n_actors,
        actor_pp_size=1,
        critic_dp_size=config.n_critics,
        critic_pp_size=1,
        ref_dp_size=config.n_refs,
        rew_dp_size=config.n_rewards,
        seed=config.seed,
        sft_model_path=f"/lustre/public/pretrained_model_weights/{config.actor_model_name}",
        rew_model_path=f"/lustre/public/pretrained_model_weights/{config.critic_model_name}",
        actor_zero_stage=config.actor_zero_stage,
        critic_zero_stage=config.critic_zero_stage,
        hybrid_engine=config.hybrid_engine,
        batch_size=config.n_actors * config.batch_size_per_device,
        max_prompt_len=config.max_prompt_length,
        max_new_tokens=config.max_answer_length,
        offload_actor_param=config.offload_actor_param,
        offload_actor_optimizer_state=config.offload_actor_optimizer_state,
        offload_critic_param=config.offload_critic_param,
        offload_critic_optimizer_states=config.offload_critic_optimizer_states,
        offload_ref=config.offload_ref,
        offload_reward=config.offload_reward,
        actor_gradient_checkpointing=config.gradient_checkpointing,
        critic_gradient_checkpointing=config.gradient_checkpointing,
    )
    return functools.partial(customized_schedule(get_schduling_config(config))(PPOExperiment), **args)


register_experiment("sysb-llama-7b", get_exp_cls(ChatRLHFBenchmarkConfig()))
