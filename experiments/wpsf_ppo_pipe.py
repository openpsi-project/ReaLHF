import functools

from api.config import *
from api.dfg import ModelInterfaceType, ModelRPC
from base.topology import PipeModelDataParallelTopology

rollout = ModelRPC(
    "actor",
    ModelInterfaceType.GENERATE,
    input_data=["prompts", "prompt_att_mask"],
    output_data=[
        "seq_no_eos_mask", 'packed_seq', 'cu_seqlens', 'packed_logprobs', 'packed_logits_mask', 'prompt_mask'
    ],
)

inf_reward = ModelRPC("reward",
                      ModelInterfaceType.INFERENCE,
                      input_data=["packed_seq", "cu_seqlens"],
                      input_key_remap={'packed_seq': "packed_input_ids"},
                      output_data=["scores"],
                      output_key_remap={"scores": "rewards"},
                      dp_broker_type="packed")

inf_ref_logits = ModelRPC("ref",
                          ModelInterfaceType.INFERENCE,
                          input_data=["packed_seq", "cu_seqlens", "packed_logits_mask"],
                          output_data=["logprobs"],
                          output_key_remap={"logprobs": "packed_ref_logprobs"},
                          dp_broker_type="packed")

inf_values = ModelRPC("critic",
                      ModelInterfaceType.INFERENCE,
                      input_data=["packed_seq", "cu_seqlens", "seq_no_eos_mask"],
                      output_data=["scores"],
                      output_key_remap={"scores": "values"},
                      dp_broker_type="packed")

train_actor = ModelRPC("actor",
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
                           'packed_logits_mask',
                       ],
                       dp_broker_type="packed")

train_critic = ModelRPC("critic",
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
                        dp_broker_type="packed")


class PipeWpsfFlashPPOExperiment(Experiment):

    def __init__(self,
                 n_actors=8,
                 n_critics=1,
                 n_rewards=1,
                 n_refs=2,
                 seed=1,
                 num_actor_pipeline_stages=4,
                 benchmark_only=False,
                 model_type="starcoder"):
        if benchmark_only:
            n_actors = n_critics = n_rewards = n_refs = 1

        self.n_actors = n_actors
        self.n_rewards = n_rewards
        self.n_refs = n_refs
        self.n_critics = n_critics

        self.n_actor_num_pp = num_actor_pipeline_stages
        self.n_data_workers = self.dp_worldsize = self.n_actors // self.n_actor_num_pp

        self.n_total = n_actors + n_rewards + n_refs + n_critics

        self.n_data_workers = n_actors

        self.seed = seed
        self.model_type = model_type

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(data_worker=TasksGroup(
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
                                    model_worker=[
                                        TasksGroup(
                                            count=self.n_actors,
                                            scheduling=Scheduling.model_worker_default(
                                                cpu=4,
                                                gpu=1,
                                                gpu_type='tesla',
                                                mem=100000,
                                                nodelist='QH-com09',
                                            ),
                                        ),
                                        TasksGroup(
                                            count=self.n_critics + self.n_rewards + self.n_refs,
                                            scheduling=Scheduling.model_worker_default(
                                                cpu=4,
                                                gpu=1,
                                                gpu_type='tesla',
                                                mem=100000,
                                                nodelist='QH-com14',
                                            ),
                                        )
                                    ])

    def initial_setup(self) -> ExperimentConfig:
        if self.model_type == "starcoder":
            actor_path = "/lustre/meizy/models/pipe_pretrained/starcoder_4pp_3s"
            ref_path = "/lustre/fw/pretrained/starcoder"
            # actor_path = "/lustre/meizy/models/pipe_starcoder_4l_4pp_1s"
            # ref_path = "/lustre/meizy/models/starcoder_4l"
            critic_path = "/lustre/meizy/models/starcoder_4l"  # a 4 layer starcoder model only for testing purpose
        elif self.model_type == "llama":
            actor_path = "/home/meizy/models/llama-2-13b_4pp_3s"
            ref_path = "/lustre/fw/pretrained/llama-13b"
            critic_path = "/home/meizy/models/Llama-2-4l"

        # rw_lora_head_path = None

        # self.lora_dim = 32
        # self.lora_scaling = 32.0

        rw_output_scaling = 0.1
        rw_output_bias = 0.0

        batch_size_per_device = 2
        max_prompt_len = 1024
        max_answer_len = 512

        dataset = Dataset(
            'prompt',
            args=dict(
                dataset_path="/lustre/meizy/data/wps-formula-rw/dataset_train.jsonl",
                max_prompt_len=max_prompt_len,
                pad_to_max_length=True,
            ),
        )
        dataloader = DataLoader(
            'default',
            args=dict(
                shuffle=True,
                drop_last=True,
                batch_size=batch_size_per_device,
            ),
        )
        data_worker = [
            DataWorker(
                tokenizer_name_or_path=actor_path,
                datasets=[dataset],
                dataloader=dataloader,
                seed=self.seed,
            ) for i in range(self.n_data_workers)
        ]

        generation_kwargs = dict(
            max_new_tokens=max_answer_len,
            min_new_tokens=10,
            greedy=False,
            top_p=1.0,
            top_k=int(1e9),
            temperature=1.0,
        )

        model_class_name = "starcoder_flash_mqat_pipe" \
            if self.model_type == "starcoder" else "llama_flash_mqat_pipe"
        actor_model = Model(model_class_name,
                            args=dict(model_path=actor_path,
                                      num_pp=self.n_actor_num_pp,
                                      num_dp=self.dp_worldsize,
                                      load_from_full_ckpt=False))

        ref_model = Model(
            "flash_mqat_clm_hf",
            args=dict(
                model_path=ref_path,
                from_type="llama",
                tokenizer_path=ref_path,
            ),
        )
        rw_model = critic_model = Model(
            "flash_mqat_critic",
            args=dict(
                model_path=critic_path,
                from_type="llama",
                tokenizer_path=critic_path,
                output_bias=rw_output_bias,
                output_scaling=rw_output_scaling,
            ),
        )
        critic_model = copy.deepcopy(rw_model)

        # critic_model.wrappers.append(lora_wrapper(squash=False))

        actor_backend = ModelBackend(
            'ds_train',
            args=dict(optimizer_name='adam',
                      optimizer_config=dict(
                          lr=2.5e-4,
                          weight_decay=0.0,
                          eps=1e-5,
                          betas=(0.9, 0.95),
                      ),
                      lr_scheduler_type='linear',
                      warmup_steps_proportion=0.075,
                      min_lr_ratio=0.0,
                      zero_stage=1,
                      enable_fp16=True,
                      gradient_checkpointing=False,
                      engine_type="pipe",
                      num_pipeline_stages=self.n_actor_num_pp),
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
                enable_fp16=True,
            ),
        )
        ref_backend = rw_backend = ModelBackend('ds_inference', args=dict(enable_fp16=True))

        ppo_kwargs = dict(
            kl_ctl=0.1,
            discount=1.0,
            gae_lambda=1.0,
            eps_clip=0.2,
            value_eps_clip=0.2,
            max_reward_clip=20.0,
        )
        actor_interface = ModelInterface(
            'pipe_flash_actor',
            args={
                **copy.deepcopy(ppo_kwargs), "generation_config": generation_kwargs,
                "force_no_logits_mask": True
            },
        )
        ref_interface = ModelInterface(
            "flash_actor",
            args={
                **copy.deepcopy(ppo_kwargs),
                "force_no_logits_mask": True,
            },
        )
        critic_interface = ModelInterface(
            'flash_critic',
            args=copy.deepcopy(ppo_kwargs),
        )
        rw_interface = ModelInterface('flash_plrw')

        model_worker = []
        topo = PipeModelDataParallelTopology(self.n_actor_num_pp, 1, self.dp_worldsize)
        for i in range(self.n_actors):
            coord = topo.get_coord(i)
            mw = ModelWorker(
                seed=self.seed,
                model=actor_model,
                backend=actor_backend,
                interface=actor_interface,
                model_name='actor',
                topo=topo,
                dp_rank=coord.data,
                pp_rank=coord.pipe,
                mp_rank=coord.model,
                cuda_cache_cleanliness=True,
                cuda_cache_clear_freq=1,
            )
            model_worker.append(mw)

        model_worker += [
            ModelWorker(
                seed=self.seed,
                model=rw_model,
                backend=rw_backend,
                interface=rw_interface,
                model_name='reward',
                dp_rank=i,
                topo=PipeModelDataParallelTopology(1, 1, self.n_rewards),
            ) for i in range(self.n_rewards)
        ] + [
            ModelWorker(
                seed=self.seed,
                model=ref_model,
                backend=ref_backend,
                interface=ref_interface,
                model_name='ref',
                dp_rank=i,
                topo=PipeModelDataParallelTopology(1, 1, self.n_refs),
                cuda_cache_cleanliness=True,
                cuda_cache_clear_freq=1,
            ) for i in range(self.n_refs)
        ] + [
            ModelWorker(
                seed=self.seed,
                model=critic_model,
                backend=critic_backend,
                interface=critic_interface,
                model_name='critic',
                dp_rank=i,
                topo=PipeModelDataParallelTopology(1, 1, self.n_critics),
                cuda_cache_cleanliness=True,
                cuda_cache_clear_freq=1,
            ) for i in range(self.n_critics)
        ]

        return ExperimentConfig(
            total_train_epochs=1,
            save_frequency_epochs=1,
            save_frequency_seconds=None,
            model_rpcs=[rollout, inf_ref_logits, inf_reward, inf_values, train_actor, train_critic],
            data_worker=data_worker,
            model_worker=model_worker,
            benchmark_steps=20)


register_experiment("wpsf-flash-ppo-pipe-starcoder",
                    functools.partial(PipeWpsfFlashPPOExperiment, model_type="starcoder"))

register_experiment("wpsf-flash-ppo-pipe-llama",
                    functools.partial(PipeWpsfFlashPPOExperiment, model_type="llama"))
