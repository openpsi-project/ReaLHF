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
)

inf_ref_logits = ModelRPC(
    "ref",
    ModelInterfaceType.INFERENCE,
    input_data=["packed_seq", "cu_seqlens", "packed_logits_mask"],
    output_data=["logprobs"],
    output_key_remap={"logprobs": "packed_ref_logprobs"},
)

inf_values = ModelRPC(
    "critic",
    ModelInterfaceType.INFERENCE,
    input_data=["packed_seq", "cu_seqlens", "seq_no_eos_mask"],
    output_data=["scores"],
    output_key_remap={"scores": "values"},
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
)


class WpsfFlashPPOExperiment(Experiment):

    def __init__(self, n_actors=1, n_critics=1, n_rewards=1, n_refs=1, seed=1, benchmark_only=False):
        if benchmark_only:
            n_actors = n_critics = n_rewards = n_refs = 1

        self.n_actors = n_actors
        self.n_rewards = n_rewards
        self.n_refs = n_refs
        self.n_critics = n_critics

        self.n_total = n_actors + n_rewards + n_refs + n_critics

        self.n_data_workers = n_actors

        self.seed = seed

        self.benchmark_only = benchmark_only

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
                    gpu_type='geforce',
                    mem=60000,
                    nodelist='frl8g134',
                ),
            ),
        )

    def initial_setup(self) -> ExperimentConfig:
        if self.benchmark_only:
            actor_path = f"{cluster_spec.fileroot}/checkpoints/1l-starcoder/"
            rw_lora_head_path = None
        else:
            actor_path = f"{cluster_spec.fileroot}/checkpoints/starcoder/"
            rw_lora_head_path = f"{cluster_spec.fileroot}/checkpoints/fw/wps-rw-pl-s1/20230822-3/default/epoch0step0/"

        self.lora_dim = 32
        self.lora_scaling = 32.0

        rw_output_scaling = 0.1
        rw_output_bias = 0.0

        mini_batch_size_per_device = 1
        batch_size_per_device = 2
        max_prompt_len = 2048
        max_answer_len = 2048

        dataset = Dataset(
            'wpsf_prompt',
            args=dict(
                dataset_path="/data/aigc/llm/datasets/wps-formula-rw/dataset_train.jsonl",
                max_prompt_len=max_prompt_len,
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

        def lora_wrapper(load_path=None, squash=True):
            return ModelWrapper(
                'lora',
                args=dict(
                    lora_module_kwargs=dict(
                        lora_dim=self.lora_dim,
                        lora_scaling=self.lora_scaling,
                    ),
                    lora_keys_to_replace=['c_attn.linear', 'c_proj.'],
                    load_lora_path=load_path,
                    lora_op_after_creation='squash' if squash else None,
                ),
            )

        actor_model = Model(
            "flash_mqat_clm_hf",
            args=dict(model_path=actor_path),
            wrappers=[
                # FIXME: lora path
                lora_wrapper(),
                lora_wrapper(squash=False),
            ],
        )
        ref_model = Model(
            'flash_mqat_clm_hf',
            args=dict(model_path=actor_path,),
            wrappers=[
                # FIXME: lora path
                lora_wrapper()
            ],
        )
        rw_model = Model(
            "flash_mqat_critic",
            args=dict(
                model_path=actor_path,
                from_type='starcoder',
                output_bias=rw_output_bias,
                output_scaling=rw_output_scaling,
                v_head_path=os.path.join(rw_lora_head_path, "rw_v_head.bin")
                if not self.benchmark_only else None,
            ),
            wrappers=[
                # FIXME: lora path
                lora_wrapper(),
                lora_wrapper(),
            ],
        )
        critic_model = copy.deepcopy(rw_model)
        critic_model.wrappers.append(lora_wrapper(squash=False))

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
                enable_fp16=True,
            ),
        )
        ref_backend = rw_backend = ModelBackend('ds_inference', args=dict(enable_fp16=True))

        ppo_kwargs = dict(
            mini_batch_size=mini_batch_size_per_device,
            kl_ctl=0.1,
            discount=1.0,
            gae_lambda=1.0,
            eps_clip=0.2,
            value_eps_clip=0.2,
            max_reward_clip=20.0,
        )
        actor_interface = ref_interface = ModelInterface(
            'flash_actor',
            args={
                **copy.deepcopy(ppo_kwargs), "generation_config": generation_kwargs
            },
        )
        critic_interface = ModelInterface(
            'flash_critic',
            args=copy.deepcopy(ppo_kwargs),
        )
        # critic_interface.args['mini_batch_size'] = mini_batch_size_per_device * self.n_actors // self.n_critics
        rw_interface = ModelInterface('flash_plrw')

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
            ) for i in range(self.n_critics)
        ]

        return ExperimentConfig(
            total_train_epochs=8 if not self.benchmark_only else 1,
            save_frequency_epochs=None,
            save_frequency_seconds=None,
            model_rpcs=[rollout, inf_ref_logits, inf_reward, inf_values, train_actor, train_critic],
            data_worker=data_worker,
            model_worker=model_worker,
        )


register_experiment("wpsf-flash-ppo", WpsfFlashPPOExperiment)
register_experiment("wpsf-flash-ppo-benchmark", functools.partial(WpsfFlashPPOExperiment,
                                                                  benchmark_only=True))
