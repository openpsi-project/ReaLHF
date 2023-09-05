import torch

from api.config import *
from api.ecs import Commands, DataQuery, MasterWorkerECS, ModelQuery, RawDataQuery


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
    commands.set_data('logits_ignoring_mask', res['logits_ignoring_mask'])


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
    logits_ignoring_mask: DataQuery['logits_ignoring_mask'],
):
    inputs = commands.build_model_inputs(input_ids=seq,
                                         attention_mask=attention_mask,
                                         logits_ignoring_mask=logits_ignoring_mask)
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
    logits_ignoring_mask: DataQuery['logits_ignoring_mask'],
):
    data = commands.build_model_inputs(
        input_ids=seq,
        rewards=rewards,
        values=values,
        logp=logp,
        ref_logp=ref_logp,
        prompts=prompts,
        attention_mask=attention_mask,
        logits_ignoring_mask=logits_ignoring_mask,
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


class ChatRLHFBenchmarkExperiment(Experiment):

    def __init__(self, n_actors=4, n_critics=1, n_rewards=1, n_refs=1, seed=1):
        self.n_actors = n_actors
        self.n_rewards = n_rewards
        self.n_refs = n_refs
        self.n_critics = n_critics

        self.n_total = n_actors + n_rewards + n_refs + n_critics

        self.n_data_workers = n_actors

        self.seed = seed

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            data_worker=TasksGroup(
                count=self.n_data_workers,
                scheduling=Scheduling.data_worker_default(
                    cpu=2,
                    mem=10000,
                ),
            ),
            master_worker=TasksGroup(
                count=1,
                scheduling=Scheduling.master_worker_default(
                    cpu=4,
                    mem=10000,
                ),
            ),
            model_worker=TasksGroup(
                count=self.n_total,
                scheduling=Scheduling.model_worker_default(
                    cpu=4,
                    gpu=1,
                    gpu_type='tesla',
                    mem=60000,
                    nodelist='frl8a140',
                ),
            ),
        )

    def initial_setup(self) -> ExperimentConfig:
        actor_path = "/lustre/meizy/base_models/opt-6.7b"
        critic_path = "/lustre/meizy/base_models/opt-125m"
        # rw_lora_head_path = \
        # "/data/aigc/llm/checkpoints/fw/wps-rw-pl-s1/20230822-3/default/epoch0step0/"

        self.lora_dim = 32
        self.lora_scaling = 32.0

        rw_output_scaling = 0.1
        rw_output_bias = 0.0

        mini_batch_size_per_device = 1
        batch_size_per_device = 2
        max_prompt_len = 128
        max_answer_len = 512 - max_prompt_len

        dataset = Dataset(
            'chat_prompt',
            args=dict(
                dataset_path="/lustre/meizy/datasets/Dahoas/rm-static/data/data.jsonl",
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
            min_new_tokens=10,
            do_sample=True,
            top_p=1.0,
            top_k=1000,
            temperature=1.0,
            num_beams=1,
            num_beam_groups=1,
            num_return_sequences=1,
        )
        actor_model = Model(
            "causal_lm",
            args=dict(
                model_name_or_path=actor_path,
                init_from_scratch=False,
                from_pretrained_kwargs=dict(torch_dtype=torch.float16, use_cache=False),
                generation_kwargs=generation_kwargs,
                # quantization_kwargs=dict(load_in_8bit=True),
            ),
        )
        ref_model = Model(
            'causal_lm',
            args=dict(
                model_name_or_path=critic_path,
                init_from_scratch=False,
                from_pretrained_kwargs=dict(torch_dtype=torch.float16, use_cache=False),
                generation_kwargs=generation_kwargs,
                # quantization_kwargs=dict(load_in_8bit=True),
            ),
        )
        rw_model = Model(
            "wps_reward",
            args=dict(
                model_name_or_path=critic_path,
                from_pretrained_kwargs=dict(torch_dtype=torch.float16, use_cache=False),
                # quantization_kwargs=dict(load_in_8bit=True),
                output_bias=rw_output_bias,
                output_scaling=rw_output_scaling,
                load_v_head_path=None,
            ),
        )
        critic_model = copy.deepcopy(rw_model)
        # critic_model.args['lora_op_after_creation'] = None

        actor_backend = ModelBackend(
            'ds_train',
            args=dict(
                optimizer_name='adam',
                optimizer_config=dict(
                    lr=2.5e-4,
                    weight_decay=0.0,
                    eps=1e-5,
                    betas=(0.9, 0.95),
                ),
                lr_scheduler_type='linear',
                warmup_steps_proportion=0.075,
                min_lr_ratio=0.0,
                zero_stage=2,
                enable_fp16=False,
            ),
        )
        critic_backend = ModelBackend(
            'ds_train',
            args=dict(
                optimizer_name='adam',
                optimizer_config=dict(
                    lr=2.5e-4,
                    weight_decay=0.0,
                    eps=1e-5,
                    betas=(0.9, 0.95),
                ),
                lr_scheduler_type='linear',
                warmup_steps_proportion=0.075,
                min_lr_ratio=0.0,
                zero_stage=2,
                offload_param=False,
                offload_optimizer_state=False,
                enable_fp16=False,
            ),
        )
        ref_backend = rw_backend = ModelBackend('ds_inference', args=dict(enable_fp16=False))

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
        ] + [
            ModelWorker(seed=self.seed,
                        model=critic_model,
                        backend=critic_backend,
                        interface=critic_interface,
                        model_name='critic',
                        stream=critic_streams[i]) for i in range(self.n_critics)
        ]

        ecs = MasterWorkerECS(model_worker).add_systems([
            rollout,
            inference_ref_logits,
            inference_reward,
            inference_values,
            train_actor,
            train_critic,
        ])

        return ExperimentConfig(
            total_train_epochs=8,
            save_frequency_epochs=None,
            save_frequency_seconds=None,
            master_ecs=ecs,
            data_worker=data_worker,
            model_worker=model_worker,
        )


register_experiment("chat-rlhf-benchmark", ChatRLHFBenchmarkExperiment)
