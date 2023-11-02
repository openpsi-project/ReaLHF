import functools

import torch

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
inf_reward = ModelRPC(
    "reward",
    ModelInterfaceType.INFERENCE,
    input_data=["packed_seq", "cu_seqlens"],
    input_key_remap={'packed_seq': "packed_input_ids"},
    output_data=["scores"],
    output_key_remap={"scores": "rewards"},
    dp_broker_type='packed',
)

inf_ref_logits = ModelRPC(
    "ref",
    ModelInterfaceType.INFERENCE,
    input_data=["packed_seq", "cu_seqlens", "packed_logits_mask"],
    output_data=["logprobs"],
    output_key_remap={"logprobs": "packed_ref_logprobs"},
    dp_broker_type='packed',
)

inf_values = ModelRPC(
    "critic",
    ModelInterfaceType.INFERENCE,
    input_data=["packed_seq", "cu_seqlens", "seq_no_eos_mask"],
    output_data=["scores"],
    output_key_remap={"scores": "values"},
    dp_broker_type='packed',
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
        'packed_logits_mask',
    ],
    log_return_value=True,
    dp_broker_type='packed',
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
    dp_broker_type='packed',
    log_return_value=True,
)


class PackedPPOExperiment(Experiment):

    def __init__(
        self,
        n_actors=4,
        n_critics=4,
        seed=1,
        base_model: str = 'gpt2',
        train_dataset_path: str = "/lustre/fw/datasets/imdb/rl/ppo_prompt.jsonl",
    ):
        self.n_actors = n_actors
        self.n_critics = n_critics

        self.n_data_workers = 1

        self.seed = seed

        self.base_model = base_model
        self.train_dataset_path = train_dataset_path

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
            model_worker=[
                TasksGroup(
                    count=self.n_actors,
                    scheduling=Scheduling.model_worker_default(
                        cpu=4,
                        gpu=1,
                        mem=60000,
                        nodelist='frl8a138',
                    ),
                ),
                TasksGroup(
                    count=self.n_critics,
                    scheduling=Scheduling.model_worker_default(
                        cpu=4,
                        gpu=1,
                        mem=60000,
                        nodelist='frl8a139',
                    ),
                ),
                TasksGroup(
                    count=2,
                    scheduling=Scheduling.model_worker_default(
                        cpu=4,
                        gpu=0.5,
                        mem=30000,
                        nodelist='frl8a139',
                    ),
                )
            ],
        )

    def initial_setup(self) -> ExperimentConfig:
        if self.base_model == 'starcoder':
            base_model_path = "/data/aigc/public/starcoder-16bit"
        elif self.base_model == 'gpt2':
            base_model_path = "/lustre/fw/pretrained/gpt2-large/"
        else:
            raise NotImplementedError()
        sft_model_path = "/data/aigc/llm/checkpoints/fw/senti-sft-pos-s42/run20231031/default@pp_00-mp_00-dp_00/epoch8step0/"
        rw_model_path = "/data/aigc/llm/checkpoints/fw/flash-rw-paired-s42/run20231101/default@pp_00-mp_00-dp_00/epoch0step19/"

        rw_output_scaling = 1.0
        rw_output_bias = 0.0

        batch_size_per_device = 32
        max_prompt_len = 50
        max_answer_len = 512 - max_prompt_len

        dataset = Dataset(
            'prompt',
            args=dict(
                dataset_path=self.train_dataset_path,
                max_prompt_len=max_prompt_len,
                pad_to_max_length=False,
            ),
        )
        dataloader = DataLoader(
            'default',
            args=dict(
                shuffle=True,
                drop_last=True,
                batch_size=batch_size_per_device * self.n_actors // self.n_data_workers,
            ),
        )
        data_worker = [
            DataWorker(
                tokenizer_name_or_path=base_model_path,
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
            top_k=200,
            temperature=1.0,
            num_samples=1,
        )

        actor_model = ref_model = Model(
            "flash_mqat_clm_hf",
            args=dict(
                model_path=sft_model_path,
                from_type="self",
                tokenizer_path=base_model_path,
            ),
        )

        rw_model = critic_model = Model(
            "flash_mqat_critic",
            args=dict(
                model_path=rw_model_path,
                from_type='self',
                tokenizer_path=base_model_path,
                output_bias=rw_output_bias,
                output_scaling=rw_output_scaling,
            ),
        )

        actor_backend = ModelBackend(
            'ds_train',
            args=dict(
                optimizer_name='adam',
                optimizer_config=dict(
                    lr=9.65e-6,
                    weight_decay=0.0,
                    eps=1e-5,
                    betas=(0.9, 0.95),
                ),
                lr_scheduler_type='linear',
                warmup_steps_proportion=0.075,
                min_lr_ratio=0.0,
                zero_stage=2,
            ),
        )
        critic_backend = ModelBackend(
            'ds_train',
            args=dict(
                optimizer_name='adam',
                optimizer_config=dict(
                    lr=5e-6,
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
            n_minibatches=8,
            kl_ctl=0.1,
            discount=1.0,
            gae_lambda=1.0,
            eps_clip=0.2,
            value_eps_clip=0.2,
            max_reward_clip=20.0,
            adaptive_kl_ctl=False,
        )
        actor_interface = ModelInterface(
            'flash_actor',
            args={
                **copy.deepcopy(ppo_kwargs),
                "generation_config": generation_kwargs,
                "early_stop_imp_ratio": 5.0,
            },
        )
        ref_interface = copy.deepcopy(actor_interface)
        ref_interface.args['enable_save'] = False
        critic_interface = ModelInterface(
            'flash_critic',
            args=copy.deepcopy(ppo_kwargs),
        )
        rw_interface = ModelInterface('flash_paired_rw', args=dict(enable_save=False))

        model_worker = [
            ModelWorker(
                seed=self.seed,
                model=actor_model,
                backend=actor_backend,
                interface=actor_interface,
                model_name='actor',
                dp_rank=i,
                topo=PipeModelDataParallelTopology(1, 1, self.n_actors),
            ) for i in range(self.n_actors)
        ] + [
            ModelWorker(
                seed=self.seed,
                model=critic_model,
                backend=critic_backend,
                interface=critic_interface,
                model_name='critic',
                dp_rank=i,
                topo=PipeModelDataParallelTopology(1, 1, self.n_critics),
            ) for i in range(self.n_critics)
        ] + [
            ModelWorker(
                seed=self.seed,
                model=rw_model,
                backend=rw_backend,
                interface=rw_interface,
                model_name='reward',
                dp_rank=0,
                topo=PipeModelDataParallelTopology(1, 1, 1),
            )
        ] + [
            ModelWorker(
                seed=self.seed,
                model=ref_model,
                backend=ref_backend,
                interface=ref_interface,
                model_name='ref',
                dp_rank=0,
                topo=PipeModelDataParallelTopology(1, 1, 1),
            )
        ]

        return ExperimentConfig(
            total_train_epochs=8,
            save_frequency_epochs=1,
            save_frequency_steps=20,
            save_frequency_seconds=None,
            model_rpcs=[rollout, inf_ref_logits, inf_reward, inf_values, train_actor, train_critic],
            data_worker=data_worker,
            model_worker=model_worker,
        )


for s in range(1, 43):
    register_experiment(f"flash-ppo-s{s}", functools.partial(PackedPPOExperiment, seed=s))
