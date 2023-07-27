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


class WpsRLHFExperiment(Experiment):

    def __init__(self, n_actors=8, n_critics=2, n_rewards=2, n_refs=4, n_data_workers=4, seed=1):
        self.n_actors = n_actors
        self.n_rewards = n_rewards
        self.n_refs = n_refs
        self.n_critics = n_critics

        self.n_total = n_actors + n_rewards + n_refs + n_critics

        self.n_data_workers = n_data_workers

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
                ),
            ),
        )

    def initial_setup(self) -> ExperimentConfig:
        actor_path = "/home/aigc/llm/checkpoints/starcoder-wps-best/"
        rw_paths = [
            '/home/aigc/llm/root/checkpoints/wps-rw-s1-wd0.0-starcoder-cm/20230725/default/epoch1step0/',
        ]
        critic_path = rw_paths[0]

        mini_batch_size_per_device = 4
        batch_size_per_device = 32
        max_prompt_len = max_answer_len = 256

        dataset = Dataset('excel_prompt',
                          args=dict(
                              dataset_path="/home/aigc/llm/datasets/prompts/train50000.jsonl",
                              tokenizer_name_or_path=actor_path,
                              max_seq_len=max_prompt_len,
                          ))
        dataloader = DataLoader('excel_rlhf',
                                args=dict(
                                    max_token_len=max_prompt_len,
                                    shuffle=True,
                                    drop_last=False,
                                    batch_size=batch_size_per_device * self.n_actors // self.n_data_workers,
                                ))
        data_worker = [
            DataWorker(
                datasets=[dataset],
                stream=RequestReplyStream(f"data{i}"),
                dataloader=dataloader,
                seed=self.seed,
            ) for i in range(self.n_data_workers)
        ]

        generation_kwargs = dict(max_new_tokens=max_answer_len,
                                 do_sample=True,
                                 top_p=0.95,
                                 top_k=100,
                                 temperature=1.0,
                                 num_beams=1,
                                 num_beam_groups=1,
                                 num_return_sequences=1)
        actor_model = Model(
            "causal_lm_lora",
            args=dict(
                model_name_or_path=actor_path,
                disable_dropout=True,
                generation_kwargs=generation_kwargs,
                lora_dim=32,
                lora_module_name="attn",
            ),
        )
        ref_model = Model(
            'causal_lm',
            args=dict(
                model_name_or_path=actor_path,
                disable_dropout=True,
                generation_kwargs=generation_kwargs,
            ),
        )
        rw_model = Model(
            "wps_reward",
            args=dict(
                model_name_or_path=rw_paths[0],
                load_state_dict=True,
                disable_dropout=True,
            ),
        )
        critic_model = Model(
            "wps_reward_lora",
            args=dict(
                model_name_or_path=critic_path,
                load_state_dict=True,
                disable_dropout=True,
                lora_dim=32,
                lora_module_name='attn',
                additional_module_names_to_opt=['v_head'],
            ),
        )

        actor_backend = ModelBackend(
            'ds_train',
            args=dict(
                optimizer_name='adam',
                optimizer_config=dict(lr=2e-3, betas=(0.9, 0.95)),
                warmup_steps_proportion=0.0,
                min_lr_ratio=0.1,
                zero_stage=2,
            ),
        )
        critic_backend = ModelBackend(
            'ds_train',
            args=dict(
                optimizer_name='adam',
                optimizer_config=dict(lr=1e-3, betas=(0.9, 0.95)),
                warmup_steps_proportion=0.0,
                min_lr_ratio=0.1,
                zero_stage=2,
            ),
        )
        ref_backend = rw_backend = ModelBackend('ds_inference')

        ppo_kwargs = dict(
            ppo_epochs=1,
            mini_batch_size=mini_batch_size_per_device,
            kl_ctl=0.1,
            discount=1.0,
            gae_lambda=0.95,
            eps_clip=0.2,
            value_eps_clip=0.2,
            max_reward_clip=5.0,
        )
        actor_interface = ref_interface = ModelInterface(
            'wps_actor',
            args=ppo_kwargs,
        )
        critic_interface = ModelInterface(
            'wps_critic',
            args=ppo_kwargs,
        )
        rw_interface = ModelInterface('wps_reward_unpaired')

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
            total_train_epochs=10,
            save_frequency_epochs=1,
            save_frequency_seconds=3600,
            master_ecs=ecs,
            data_worker=data_worker,
            model_worker=model_worker,
        )


register_experiment("wps-rlhf", WpsRLHFExperiment)
