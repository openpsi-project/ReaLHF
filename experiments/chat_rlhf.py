import os

import torch

from api.config import *
from api.ecs import Commands, DataQuery, MasterWorkerECS, ModelQuery, RawDataQuery

EXPR_DEADLINE = "now+8hours"
EXPR_TIME_LIMIT = "01:00:00"


def rollout(
    commands: Commands,
    model: ModelQuery['actor'],
    prompts: RawDataQuery['prompts'],
    prompt_att_mask: RawDataQuery['prompt_att_mask'],
):
    inputs = commands.build_model_inputs(
        prompts=prompts,
        prompt_att_mask=prompt_att_mask,
    )
    res = model.generate(inputs)
    commands.set_data('seq', res['seq'])
    commands.set_data('logp', res['logp'])
    commands.set_data('attention_mask', res['attention_mask'])
    # commands.set_data('logits_ignoring_mask', res['logits_ignoring_mask'])


def inference_reward(
    commands: Commands,
    model: ModelQuery['reward'],
    seq: DataQuery['seq'],
    attention_mask: DataQuery['attention_mask'],
    prompts: DataQuery['prompts'],
):
    inputs = commands.build_model_inputs(
        input_ids=seq,
        attention_mask=attention_mask,
        prompts=prompts,
    )
    res = model(inputs)
    commands.set_data('rewards', res['scores'])


def inference_ref_logits(
    commands: Commands,
    model: ModelQuery['ref'],
    seq: DataQuery['seq'],
    attention_mask: DataQuery['attention_mask'],
    # logits_ignoring_mask: DataQuery['logits_ignoring_mask'],
):
    inputs = commands.build_model_inputs(
        input_ids=seq,
        attention_mask=attention_mask,
        # logits_ignoring_mask=logits_ignoring_mask
    )
    res = model(inputs)
    commands.set_data('ref_logp', res['logp'])


def inference_values(
    commands: Commands,
    model: ModelQuery['critic'],
    prompts: DataQuery['prompts'],
    seq: DataQuery['seq'],
    attention_mask: DataQuery['attention_mask'],
):
    inputs = commands.build_model_inputs(
        input_ids=seq,
        prompts=prompts,
        attention_mask=attention_mask,
    )
    res = model(inputs)
    commands.set_data('values', res['scores'])


def train_actor(
    commands: Commands,
    model: ModelQuery['actor'],
    seq: DataQuery['seq'],
    rewards: DataQuery['rewards'],
    values: DataQuery['values'],
    logp: DataQuery['logp'],
    ref_logp: DataQuery['ref_logp'],
    prompts: DataQuery['prompts'],
    attention_mask: DataQuery['attention_mask'],
    # logits_ignoring_mask: DataQuery['logits_ignoring_mask'],
):
    data = commands.build_model_inputs(
        input_ids=seq,
        rewards=rewards,
        values=values,
        logp=logp,
        ref_logp=ref_logp,
        prompts=prompts,
        attention_mask=attention_mask,
        # logits_ignoring_mask=logits_ignoring_mask,
    )
    commands.log(model.train_step(data))


def train_critic(
    commands: Commands,
    model: ModelQuery['critic'],
    seq: DataQuery['seq'],
    rewards: DataQuery['rewards'],
    values: DataQuery['values'],
    logp: DataQuery['logp'],
    ref_logp: DataQuery['ref_logp'],
    prompts: DataQuery['prompts'],
    attention_mask: DataQuery['attention_mask'],
):
    data = commands.build_model_inputs(
        input_ids=seq,
        rewards=rewards,
        values=values,
        logp=logp,
        ref_logp=ref_logp,
        prompts=prompts,
        attention_mask=attention_mask,
    )
    commands.log(model.train_step(data))


@dataclasses.dataclass
class ChatRLHFBenchmarkConfig:
    # resource
    n_actors: int = 1
    n_critics: int = 1
    n_rewards: int = 1
    n_refs: int = 1
    seed: int = 1
    gpu_per_actor: float = 0.25
    gpu_per_critic: float = 0.25
    gpu_per_reward: float = 0.25
    gpu_per_ref: float = 0.25
    # optimization options
    actor_model_name: str = "opt-768-12"
    critic_model_name: str = "opt-768-12"
    actor_zero_stage: int = 2
    critic_zero_stage: int = 2
    hybrid_engine: bool = True
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


@dataclasses.dataclass
class ChatRLHFBenchmarkConfig:
    # resource
    n_actors: int = 1
    n_critics: int = 1
    n_rewards: int = 1
    n_refs: int = 1
    seed: int = 1
    gpu_per_actor: float = 0.25
    gpu_per_critic: float = 0.25
    gpu_per_reward: float = 0.25
    gpu_per_ref: float = 0.25
    # optimization options
    actor_model_name: str = "opt-125m"
    critic_model_name: str = "opt-125m"
    init_from_scratch: bool = False
    actor_zero_stage: int = 2
    critic_zero_stage: int = 2
    hybrid_engine: bool = True
    batch_size_per_device: int = 2
    max_prompt_length: int = 256
    max_answer_length: int = 256
    offload_actor_param: bool = False
    offload_actor_optimizer_state: bool = False
    offload_critic_param: bool = False
    offload_critic_optimizer_states: bool = False
    offload_ref: bool = False
    offload_reward: bool = False


class ChatRLHFBenchmarkExperiment(Experiment):

    def __init__(self, config: ChatRLHFBenchmarkConfig = ChatRLHFBenchmarkConfig()):
        self.config = config
        self.n_data_workers = config.n_actors
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
                        gpu_type='tesla',
                        mem=int(60000 * last_gpu_per_type),
                        # nodelist='YL-com02',
                        node_type="a100",
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
                gpu_type='tesla',
                mem=int(60000 * last_gpu_per_type),
                # nodelist='YL-com02',
                node_type="a100",
                deadline=EXPR_DEADLINE,
                time_limit=EXPR_TIME_LIMIT,
            ),
        )
        task_groups.append(new_task_group)
        # print(task_groups)
        self.model_worker_task_groups = task_groups
        self.actor_model_name = config.actor_model_name
        self.critic_model_name = config.critic_model_name
        self.seed = config.seed
        self.n_actors = config.n_actors
        self.n_critics = config.n_critics
        self.n_rewards = config.n_rewards
        self.n_refs = config.n_refs
        self.init_from_scratch = config.init_from_scratch

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(data_worker=TasksGroup(
            count=self.n_data_workers,
            scheduling=Scheduling.data_worker_default(
                cpu=2,
                mem=10000,
                begin=None,
                node_type="g1",
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
                                            node_type="g1",
                                            deadline=EXPR_DEADLINE,
                                            time_limit=EXPR_TIME_LIMIT,
                                        ),
                                    ),
                                    model_worker=self.model_worker_task_groups)

    def initial_setup(self) -> ExperimentConfig:
        # model_dir = "/lustre/meizy/base_models/cfgonly"
        model_dir = "/lustre/meizy/base_models"
        data_path = "/lustre/meizy/datasets/Dahoas/rm-static/data/data.jsonl"

        actor_path = os.path.join(model_dir, self.actor_model_name)
        critic_path = os.path.join(model_dir, self.critic_model_name)
        # rw_lora_head_path = \
        # "/data/aigc/llm/checkpoints/fw/wps-rw-pl-s1/20230822-3/default/epoch0step0/"

        self.lora_dim = 32
        self.lora_scaling = 32.0

        rw_output_scaling = 0.1
        rw_output_bias = 0.0

        mini_batch_size_per_device = self.config.batch_size_per_device
        batch_size_per_device = self.config.batch_size_per_device
        max_prompt_len = self.config.max_prompt_length
        max_answer_len = self.config.max_answer_length

        dataset = Dataset(
            'chat_prompt',
            args=dict(
                dataset_path=data_path,
                max_seq_len=max_prompt_len,
            ),
        )
        dataloader = DataLoader(
            'chat_rlhf',
            args=dict(
                max_token_len=max_prompt_len,
                shuffle=True,
                drop_last=True,
                batch_size=batch_size_per_device * self.n_actors // self.n_data_workers,
            ),
        )
        data_worker = [
            DataWorker(
                tokenizer_name_or_path=actor_path,
                datasets=[dataset],
                stream=RequestReplyStream(f"data{i}"),
                dataloader=dataloader,
                seed=self.seed,
            ) for i in range(self.n_data_workers)
        ]

        generation_kwargs = dict(
            max_new_tokens=max_answer_len,
            min_new_tokens=max_answer_len,
            # do_sample=False,
            # top_p=1.0,
            # top_k=1,
            # temperature=1.0,
            # num_beams=1,
            # num_beam_groups=1,
            # num_return_sequences=1,
        )
        actor_model = Model(
            "causal_lm",
            args=dict(
                model_name_or_path=actor_path,
                init_from_scratch=self.init_from_scratch,
                from_pretrained_kwargs=dict(torch_dtype=torch.float16),
                generation_kwargs=generation_kwargs,
                # quantization_kwargs=dict(load_in_8bit=True),
            ),
        )
        ref_model = Model(
            'causal_lm',
            args=dict(
                model_name_or_path=actor_path,
                init_from_scratch=self.init_from_scratch,
                from_pretrained_kwargs=dict(torch_dtype=torch.float16),
                generation_kwargs=generation_kwargs,
                # quantization_kwargs=dict(load_in_8bit=True),
            ),
        )
        rw_model = Model(
            "wps_reward",
            args=dict(
                model_name_or_path=critic_path,
                from_pretrained_kwargs=dict(torch_dtype=torch.float16),
                # quantization_kwargs=dict(load_in_8bit=True),
                output_bias=rw_output_bias,
                output_scaling=rw_output_scaling,
                load_v_head_path=None,
                init_from_scratch=True,
            ),
        )
        critic_model = copy.deepcopy(rw_model)
        # critic_model.args['lora_op_after_creation'] = None

        actor_backend = ModelBackend(
            'ds_train',
            args=dict(
                warmup_steps_proportion=0.0,
                min_lr_ratio=0.0,
                zero_stage=self.config.actor_zero_stage,
                offload_param=self.config.offload_actor_param,
                offload_optimizer_state=self.config.offload_actor_optimizer_state,
                enable_fp16=True,
                enable_hybrid_engine=self.config.hybrid_engine,
                gradient_checkpointing=self.config.gradient_checkpointing,
            ),
        )
        critic_backend = ModelBackend(
            'ds_train',
            args=dict(
                warmup_steps_proportion=0.0,
                min_lr_ratio=0.0,
                zero_stage=self.config.critic_zero_stage,
                offload_param=self.config.offload_critic_param,
                offload_optimizer_state=self.config.offload_critic_optimizer_states,
                enable_fp16=True,
                enable_hybrid_engine=False,
                gradient_checkpointing=False,
            ),
        )
        ref_backend = ModelBackend('ds_inference',
                                   args=dict(enable_fp16=True,
                                             zero_stage=3 if self.config.offload_ref else 0,
                                             offload=self.config.offload_ref))
        rw_backend = ModelBackend('ds_inference',
                                  args=dict(enable_fp16=True,
                                            zero_stage=3 if self.config.offload_reward else 0,
                                            offload=self.config.offload_reward))

        ppo_kwargs = dict(
            ppo_epochs=1,
            mini_batch_size=mini_batch_size_per_device,
            kl_ctl=0.1,
            discount=1.0,
            gae_lambda=1.0,
            eps_clip=0.2,
            value_eps_clip=0.2,
            max_reward_clip=20.0,
        )
        actor_interface = ref_interface = ModelInterface(
            'chat_actor',
            args=copy.deepcopy(ppo_kwargs),
        )
        critic_interface = ModelInterface(
            'chat_critic',
            args=copy.deepcopy(ppo_kwargs),
        )
        # critic_interface.args['mini_batch_size'] = mini_batch_size_per_device * self.n_actors // self.n_critics
        rw_interface = ModelInterface('chat_reward')

        actor_streams = [RequestReplyStream(f"actor{i}") for i in range(self.n_actors)]
        reward_streams = [RequestReplyStream(f"reward{i}") for i in range(self.n_rewards)]
        ref_model_streams = [RequestReplyStream(f"ref{i}") for i in range(self.n_refs)]
        critic_streams = [RequestReplyStream(f"critic{i}") for i in range(self.n_critics)]

        model_worker = [
            ModelWorker(seed=self.seed,
                        model=actor_model,
                        backend=actor_backend,
                        interface=actor_interface,
                        model_name='actor',
                        stream=actor_streams[i]) for i in range(self.n_actors)
        ] + [
            ModelWorker(seed=self.seed,
                        model=critic_model,
                        backend=critic_backend,
                        interface=critic_interface,
                        model_name='critic',
                        stream=critic_streams[i]) for i in range(self.n_critics)
        ] + [
            ModelWorker(seed=self.seed,
                        model=rw_model,
                        backend=rw_backend,
                        interface=rw_interface,
                        model_name='reward',
                        stream=reward_streams[i]) for i in range(self.n_rewards)
        ] + [
            ModelWorker(seed=self.seed,
                        model=ref_model,
                        backend=ref_backend,
                        interface=ref_interface,
                        model_name='ref',
                        stream=ref_model_streams[i]) for i in range(self.n_refs)
        ]

        ecs = MasterWorkerECS(model_worker).add_systems([
            rollout,
            inference_ref_logits,
            inference_reward,
            inference_values,
            train_actor,
            train_critic,
        ])

        return ExperimentConfig(total_train_epochs=8,
                                save_frequency_epochs=None,
                                save_frequency_seconds=None,
                                master_ecs=ecs,
                                data_worker=data_worker,
                                model_worker=model_worker,
                                benchmark_steps=30,
                                config=self.config)


register_experiment("chat-rlhf-benchmark", ChatRLHFBenchmarkExperiment)
