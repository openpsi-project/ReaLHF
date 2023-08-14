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

    def __init__(
        self,
        n_models=16,
        seed=1,
        weight_decay=0.0,
        lr=2.5e-4,
        lora_dim=32,
        lora_scaling=32,
        adam_betas=None,
        lr_scheduler_type=None,
        min_lr_ratio=None,
        warmup_steps_proportion=None,
        total_train_epochs=4,
    ):
        self.n_models = self.n_data_workers = n_models
        self.seed = seed

        self.enable_sweep = ((weight_decay is None) or (lora_dim is None) or (lora_scaling is None)
                             or (lr is None) or adam_betas is None or lr_scheduler_type is None
                             or min_lr_ratio is None or warmup_steps_proportion is None
                             or total_train_epochs is None)

        if weight_decay is None:
            wd_low = math.log(1e-6)
            wd_high = math.log(0.1)
            self.weight_decay = math.exp(wd_low + (wd_high - wd_low) * random.random())
        else:
            self.weight_decay = weight_decay

        if lora_dim is None:
            self.lora_dim = random.choice([8, 16, 32])
        else:
            self.lora_dim = lora_dim

        if lr is None:
            self.lora_lr = sample_log_uniform(7e-5, 7e-4)
        else:
            self.lora_lr = lr

        if lora_scaling is None:
            self.lora_scaling = sample_log_uniform(0.8, self.lora_dim * 8)
        else:
            self.lora_scaling = lora_scaling

        if adam_betas is None:
            self.adam_betas = random.choice([(0.9, 0.999), (0.9, 0.99), (0.9, 0.95)])
        else:
            self.adam_betas = adam_betas
        assert isinstance(self.adam_betas, tuple) and len(self.adam_betas) == 2

        if lr_scheduler_type is None:
            self.lr_scheduler_type = random.choice(['constant', 'cosine', 'linear'])
        else:
            self.lr_scheduler_type = lr_scheduler_type

        if warmup_steps_proportion is None:
            self.warmup_steps_proportion = random.random() * 0.1
        else:
            self.warmup_steps_proportion = warmup_steps_proportion

        if min_lr_ratio is None:
            self.min_lr_ratio = random.choice([0.0, 0.1])
        else:
            self.min_lr_ratio = min_lr_ratio

        if total_train_epochs is None:
            self.total_train_epochs = random.choice(list(range(1, 5)))
        else:
            self.total_train_epochs = total_train_epochs

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
        root_dir = "/data"
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
        pos_weight = 2.044542447629548

        dataset = Dataset(
            'excel_reward_modeling_unpaired',
            args=dict(
                dataset_path=f"{root_dir}/aigc/llm/datasets/rw-unpaired/train.jsonl",
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
                tokenizer_name_or_path=model_path,
                datasets=[dataset],
                stream=RequestReplyStream(f"data{i}"),
                dataloader=dataloader,
                seed=self.seed,
            ) for i in range(self.n_data_workers)
        ]

        eval_dataset = copy.deepcopy(dataset)
        eval_dataset.args['dataset_path'] = f"{root_dir}/aigc/llm/datasets/rw-unpaired/valid.jsonl"
        eval_dataloader = DataLoader("default_eval", args=dict(batch_size=eval_batch_size_per_device))

        backend = ModelBackend(
            'ds_train',
            args=dict(
                optimizer_name='adam',
                optimizer_config=dict(
                    lr=self.lora_lr,
                    weight_decay=self.weight_decay,
                    eps=1e-5,
                    betas=self.adam_betas,
                ),
                lr_scheduler_type=self.lr_scheduler_type,
                warmup_steps_proportion=self.warmup_steps_proportion,
                min_lr_ratio=self.min_lr_ratio,
                zero_stage=2,
            ),
        )

        rw_model = Model(
            "wps_reward_lora",
            args=dict(
                model_name_or_path=model_path,
                load_state_dict=False,
                lora_dim=self.lora_dim,
                lora_module_name='attn',
                additional_module_names_to_opt=["v_head"],
                lora_scaling=self.lora_scaling,
            ),
        )

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
            total_train_epochs=self.total_train_epochs,
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


def train_plackett_luce_rw(
    commands: Commands,
    model: ModelQuery['default'],
    input_ids: RawDataQuery['input_ids'],
    attention_mask: RawDataQuery['attention_mask'],
    labels: RawDataQuery['labels'],
):
    inputs = commands.build_model_inputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    commands.log(model.train_step(inputs))


class WpsPlackettLuceRewardExperiment(WpsRewardModelingExperiment):

    def initial_setup(self) -> ExperimentConfig:
        self.weight_decay = 0.0
        self.lora_lr = 2.5e-4
        self.lora_scaling = 32.0
        self.lora_dim = 32
        self.adam_betas = (0.9, 0.95)
        self.lr_scheduler_type = 'linear'
        self.warmup_steps_proportion = 0.075
        self.min_lr_ratio = 0.0
        self.total_train_epochs = 8

        root_dir = "/data"
        model_path = f"{root_dir}/aigc/llm/checkpoints/starcoder-wps-best/"
        train_batch_size_per_device = 1
        eval_batch_size_per_device = 12
        max_seq_len = 512
        contrastive_dim = 6

        dataset = Dataset(
            'wps_reward_plackett_luce',
            args=dict(
                dataset_path=f"{root_dir}/aigc/llm/datasets/rw-contrastive/train.jsonl",
                max_seq_len=max_seq_len,
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
                tokenizer_name_or_path=model_path,
                stream=RequestReplyStream(f"data{i}"),
                dataloader=dataloader,
                seed=self.seed,
            ) for i in range(self.n_data_workers)
        ]

        eval_dataset = copy.deepcopy(dataset)
        eval_dataset.args['dataset_path'] = f"{root_dir}/aigc/llm/datasets/rw-contrastive/valid.jsonl"
        eval_dataloader = DataLoader("default_eval", args=dict(batch_size=eval_batch_size_per_device))

        # TODO: regularization to prevent degeneration
        backend = ModelBackend(
            'ds_train',
            args=dict(
                optimizer_name='adam',
                optimizer_config=dict(
                    lr=self.lora_lr,
                    weight_decay=self.weight_decay,
                    eps=1e-5,
                    betas=self.adam_betas,
                ),
                lr_scheduler_type=self.lr_scheduler_type,
                warmup_steps_proportion=self.warmup_steps_proportion,
                min_lr_ratio=self.min_lr_ratio,
                zero_stage=2,
            ),
        )

        rw_model = Model(
            "wps_reward_lora",
            args=dict(
                model_name_or_path=model_path,
                load_state_dict=False,
                lora_dim=self.lora_dim,
                lora_module_name='attn',
                additional_module_names_to_opt=["v_head"],
                lora_scaling=self.lora_scaling,
            ),
        )

        interface = ModelInterface('wps_plackett_luce_reward')

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

        ecs = MasterWorkerECS(model_worker).add_systems([train_plackett_luce_rw])

        cfg = ExperimentConfig(
            total_train_epochs=self.total_train_epochs,
            save_frequency_steps=None,
            save_frequency_epochs=1,
            save_frequency_seconds=None,
            eval_frequency_epochs=1,
            master_ecs=ecs,
            data_worker=data_worker,
            model_worker=model_worker,
        )
        return cfg


for s in range(1, 6):
    register_experiment(
        f"wps-rw-pl-s{s}",
        functools.partial(
            WpsPlackettLuceRewardExperiment,
            seed=s,
        ),
    )
