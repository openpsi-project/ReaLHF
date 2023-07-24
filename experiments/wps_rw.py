import functools
import getpass
import itertools
import json
import os

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


class WpsRewardModelingExperiment(Experiment):

    def __init__(self,
                 n_data_workers=2,
                 n_models=4,
                 seed=1,
                 weight_decay=0.0,
                 remove_code_comments=True,
                 base_model='starcoder'):
        self.n_models = n_models
        self.n_data_workers = n_data_workers
        self.seed = seed
        self.weight_decay = weight_decay
        self.remove_code_comments = remove_code_comments

        self.base_model = base_model

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            data_worker=TasksGroup(
                count=self.n_data_workers,
                scheduling=Scheduling.data_worker_default(
                    cpu=4,
                    mem=10000,
                ),
            ),
            master_worker=TasksGroup(
                count=1,
                scheduling=Scheduling.master_worker_default(cpu=8, mem=50000),
            ),
            model_worker=TasksGroup(
                count=self.n_models,
                scheduling=Scheduling.model_worker_default(
                    cpu=8,
                    gpu=1,
                    gpu_type='tesla',
                    mem=60000,
                ),
            ),
        )

    def initial_setup(self) -> ExperimentConfig:
        root_dir = "/home"
        if self.base_model == 'starcoder':
            model_path = f"{root_dir}/aigc/llm/checkpoints/starcoder-wps-best/"
        else:
            model_path = f"{root_dir}/aigc/llm/checkpoints/codegen2b-wps/"
        train_batch_size_per_device = 6
        eval_batch_size_per_device = 12
        max_seq_len = 512

        # with open(f"{root_dir}/aigc/llm/fw/datasets/rw-unpaired/train.jsonl", 'r') as f:
        #     data = [json.loads(ff) for ff in f]
        #     n_pos = len([d for d in data if d['correctness_label']])
        #     n_neg = len(data) - n_pos
        #     pos_weight = n_neg / n_pos
        # print(pos_weight)
        pos_weight = 2.0380411018801925

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
        eval_dataset.args['dataset_path'] = f"{root_dir}/aigc/llm/fw/datasets/rw-unpaired/valid.jsonl"
        eval_dataloader = DataLoader("default_eval", args=dict(batch_size=eval_batch_size_per_device))

        if self.base_model == 'starcoder':
            rw_model = Model(
                "wps_reward_lora",
                args=dict(
                    model_name_or_path=model_path,
                    disable_dropout=True,
                    load_state_dict=False,
                    lora_dim=8,
                    lora_module_name='attn',
                ),
            )
        else:
            rw_model = Model(
                "wps_reward",
                args=dict(
                    model_name_or_path=model_path,
                    disable_dropout=True,
                    load_state_dict=False,
                ),
            )
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
        interface = ModelInterface('wps_reward_unpaired',
                                   args=dict(pos_weight=pos_weight,
                                             remove_code_comments=self.remove_code_comments))

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

        return ExperimentConfig(
            total_train_epochs=1,
            save_frequency_epochs=1,
            save_frequency_seconds=3600,
            eval_frequency_epochs=1,
            master_ecs=ecs,
            data_worker=data_worker,
            model_worker=model_worker,
        )


seed = range(1, 6)
weight_decay = [0.0, 0.1]
remove_code_comments = [True, False]
base_model = ['starcoder', 'codegen2b']
for s, wd, rcm, bm in itertools.product(seed, weight_decay, remove_code_comments, base_model):
    register_experiment(
        f"wps-rw-s{s}-wd{wd}-{bm}-cm" if not rcm else f"wps-rw-s{s}-wd{wd}-{bm}",
        functools.partial(
            WpsRewardModelingExperiment,
            seed=s,
            weight_decay=wd,
            remove_code_comments=rcm,
            base_model=bm,
        ),
    )
