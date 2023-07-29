import functools
import getpass
import itertools
import json
import math
import os
import random

from api.config import *
from api.ecs import Commands, DataQuery, MasterWorkerECS, ModelQuery, RawDataQuery


def train_rw(
    commands: Commands,
    model: ModelQuery['default'],
    input_ids: RawDataQuery['input_ids'],
    attention_mask: RawDataQuery['attention_mask'],
    correctness_labels: RawDataQuery['correctness_labels'],
):
    inputs = commands.build_model_inputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        correctness_labels=correctness_labels,
    )
    commands.log(model.train_step(inputs))


def sample_log_uniform(low, high):
    low = math.log(low)
    high = math.log(high)
    return math.exp(low + (high - low) * random.random())


class WpsRewardModelingExperiment(Experiment):

    def __init__(self, n_models=16, seed=1, weight_decay=0.0, lora_dim=32, lora_scaling=None, lr=None):
        self.n_models = self.n_data_workers = n_models
        self.seed = seed

        self.enable_sweep = (weight_decay is None) or (lora_dim is None) or (lora_scaling
                                                                             is None) or (lr is None)

        if weight_decay is None:
            wd_low = math.log(1e-6)
            wd_high = math.log(0.1)
            self.weight_decay = math.exp(wd_low + (wd_high - wd_low) * random.random())
        else:
            self.weight_decay = weight_decay

        if lora_dim is None:
            self.lora_dim = random.choice([8, 32, 128])
        else:
            self.lora_dim = lora_dim

        if lr is None or lora_scaling is None:
            prod = sample_log_uniform(1e-3, 1e-1)
            self.lora_lr = sample_log_uniform(1e-4, 2e-3)
            self.lora_scaling = prod / self.lora_lr
        else:
            self.lora_lr = lr
            self.lora_scaling = lora_scaling

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
                scheduling=Scheduling.master_worker_default(cpu=4, mem=10000),
            ),
            model_worker=TasksGroup(
                count=self.n_models,
                scheduling=Scheduling.model_worker_default(
                    cpu=4,
                    gpu=1,
                    gpu_type='tesla',
                    mem=60000,
                ),
            ),
        )

    def initial_setup(self) -> ExperimentConfig:
        root_dir = "/home"
        model_path = f"{root_dir}/aigc/llm/checkpoints/starcoder-wps-best/"
        train_batch_size_per_device = 2
        eval_batch_size_per_device = 12
        max_seq_len = 512

        # with open(f"{root_dir}/aigc/llm/datasets/rw-unpaired/train.jsonl", 'r') as f:
        #     data = [json.loads(ff) for ff in f]
        #     n_pos = len([d for d in data if d['correctness_label']])
        #     n_neg = len(data) - n_pos
        #     pos_weight = n_neg / n_pos
        # print(pos_weight)
        pos_weight = 2.0351339481774264

        dataset = Dataset(
            'excel_reward_modeling_unpaired',
            args=dict(
                dataset_path=f"{root_dir}/aigc/llm/datasets/rw-unpaired/train.jsonl",
                tokenizer_name_or_path=model_path,
                max_seq_len=max_seq_len,
            ),
        )
        dataloader = DataLoader(
            'default',
            args=dict(
                shuffle=True,
                drop_last=False,
                batch_size=train_batch_size_per_device * self.n_models // self.n_data_workers,
            ),
        )
        data_worker = [
            DataWorker(
                datasets=[dataset],
                stream=RequestReplyStream(f"data{i}"),
                dataloader=dataloader,
                seed=self.seed,
            ) for i in range(self.n_data_workers)
        ]

        eval_dataset = copy.deepcopy(dataset)
        eval_dataset.args['dataset_path'] = f"{root_dir}/aigc/llm/datasets/rw-unpaired/valid.jsonl"
        eval_dataloader = DataLoader("default_eval", args=dict(batch_size=eval_batch_size_per_device))

        backend = ModelBackend('ds_train',
                               args=dict(
                                   optimizer_name='adam',
                                   optimizer_config=dict(lr=1e-5,
                                                         weight_decay=self.weight_decay,
                                                         betas=(0.9, 0.95)),
                                   warmup_steps_proportion=0.0,
                                   min_lr_ratio=0.0,
                                   zero_stage=2,
                               ))

        rw_model = Model(
            "wps_reward_lora",
            args=dict(
                model_name_or_path=model_path,
                disable_dropout=True,
                load_state_dict=False,
                lora_dim=self.lora_dim,
                lora_module_name='attn',
                additional_module_names_to_opt=["v_head"],
                lora_scaling=self.lora_scaling,
            ),
        )
        backend.args['optimizer_config']['lr'] = self.lora_lr

        interface = ModelInterface('wps_reward_unpaired', args=dict(pos_weight=pos_weight))

        streams = [RequestReplyStream(f"model{i}") for i in range(self.n_models)]

        model_worker = [
            ModelWorker(
                seed=self.seed,
                model=rw_model,
                backend=backend,
                interface=interface,
                model_name='default',
                stream=streams[i],
                eval_datasets=[dataset],
                eval_dataloader=eval_dataloader,
            ) for i in range(self.n_models)
        ]

        ecs = MasterWorkerECS(model_worker).add_systems([train_rw])

        cfg = ExperimentConfig(
            total_train_epochs=1,
            save_frequency_steps=None,
            save_frequency_epochs=None,
            save_frequency_seconds=None,
            eval_frequency_epochs=1,
            master_ecs=ecs,
            data_worker=data_worker,
            model_worker=model_worker,
        )
        if not self.enable_sweep:
            cfg.save_frequency_epochs = 1
        return cfg


seeds = range(1, 6)
for s in seeds:
    exp_name = f"wps-rw-s{s}"
    register_experiment(
        exp_name,
        functools.partial(
            WpsRewardModelingExperiment,
            seed=s,
        ),
    )


def train_rw_contrastive(
    commands: Commands,
    model: ModelQuery['default'],
    prompts: RawDataQuery['prompts'],
    prompt_attention_mask: RawDataQuery['prompt_attention_mask'],
    responses: RawDataQuery['responses'],
    response_attention_mask: RawDataQuery['response_attention_mask'],
    labels: RawDataQuery['labels'],
):
    inputs = commands.build_model_inputs(
        prompts=prompts,
        prompt_attention_mask=prompt_attention_mask,
        responses=responses,
        response_attention_mask=response_attention_mask,
        labels=labels,
    )
    commands.log(model.train_step(inputs))


class WpsContrastiveRewardExperiment(WpsRewardModelingExperiment):

    def initial_setup(self) -> ExperimentConfig:
        self.weight_decay = 0.0
        self.lora_lr = 1e-3
        self.lora_scaling = 8.0
        self.lora_dim = 32.0

        root_dir = "/home"
        model_path = f"{root_dir}/aigc/llm/checkpoints/starcoder-wps-best/"
        train_batch_size_per_device = 2
        eval_batch_size_per_device = 12
        max_prompt_len = max_code_len = 256
        contrastive_dim = 5

        dataset = Dataset(
            'wps_reward_contrastive',
            args=dict(
                dataset_path=f"{root_dir}/aigc/llm/datasets/rw-contrastive/train.jsonl",
                tokenizer_name_or_path=model_path,
                max_prompt_len=max_prompt_len,
                max_code_len=max_code_len,
                contrastive_dim=contrastive_dim,
            ),
        )
        dataloader = DataLoader(
            'default',
            args=dict(
                shuffle=True,
                drop_last=False,
                batch_size=train_batch_size_per_device * self.n_models // self.n_data_workers,
            ),
        )
        data_worker = [
            DataWorker(
                datasets=[dataset],
                stream=RequestReplyStream(f"data{i}"),
                dataloader=dataloader,
                seed=self.seed,
            ) for i in range(self.n_data_workers)
        ]

        eval_dataset = copy.deepcopy(dataset)
        eval_dataset.args['dataset_path'] = f"{root_dir}/aigc/llm/datasets/rw-contrastive/valid.jsonl"
        eval_dataloader = DataLoader("default_eval", args=dict(batch_size=eval_batch_size_per_device))

        backend = ModelBackend('ds_train',
                               args=dict(
                                   optimizer_name='adam',
                                   optimizer_config=dict(lr=1e-5,
                                                         weight_decay=self.weight_decay,
                                                         betas=(0.9, 0.95)),
                                   warmup_steps_proportion=0.0,
                                   min_lr_ratio=0.0,
                                   zero_stage=2,
                               ))

        rw_model = Model(
            "lora_contrastive_reward",
            args=dict(
                model_name_or_path=model_path,
                disable_dropout=True,
                embed_dim=1024,
                load_state_dict=False,
                lora_dim=self.lora_dim,
                lora_module_name='attn',
                lora_scaling=self.lora_scaling,
            ),
        )
        backend.args['optimizer_config']['lr'] = self.lora_lr

        interface = ModelInterface('wps_contrastive_reward')

        streams = [RequestReplyStream(f"model{i}") for i in range(self.n_models)]

        model_worker = [
            ModelWorker(
                seed=self.seed,
                model=rw_model,
                backend=backend,
                interface=interface,
                model_name='default',
                stream=streams[i],
                eval_datasets=[dataset],
                eval_dataloader=eval_dataloader,
            ) for i in range(self.n_models)
        ]

        ecs = MasterWorkerECS(model_worker).add_systems([train_rw_contrastive])

        cfg = ExperimentConfig(
            total_train_epochs=1,
            save_frequency_steps=None,
            save_frequency_epochs=None,
            save_frequency_seconds=None,
            eval_frequency_epochs=1,
            master_ecs=ecs,
            data_worker=data_worker,
            model_worker=model_worker,
        )
        if not self.enable_sweep:
            cfg.save_frequency_epochs = 1
        return cfg


for s in range(1, 6):
    register_experiment(
        f"wps-contrastive-rw-s{s}",
        functools.partial(
            WpsContrastiveRewardExperiment,
            seed=s,
        ),
    )
