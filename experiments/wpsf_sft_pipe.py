import functools
import math
import random

from api.config import *
from api.ecs import Commands, DataQuery, MasterWorkerECS, ModelQuery, RawDataQuery


def sft(
    commands: Commands,
    model: ModelQuery['default'],
    packed_input_ids: RawDataQuery['packed_input_ids'],
    cu_seqlens: RawDataQuery['cu_seqlens'],
    prompt_mask: RawDataQuery['prompt_mask'],
):
    inputs = commands.build_model_inputs(
        packed_input_ids=packed_input_ids,
        cu_seqlens=cu_seqlens,
        prompt_mask=prompt_mask,
    )
    commands.log(model.train_step(inputs))


class WpsFormulaSFTPipelineExperiment(Experiment):

    def __init__(self, n_models=8, num_pipeline_stages=4, seed=1, total_train_epochs=4):
        self.weight_decay = 0.05
        self.lora_lr = 2.5e-4
        self.lora_scaling = 32.0
        self.lora_dim = 32
        self.adam_betas = (0.9, 0.95)
        self.lr_scheduler_type = 'cosine'
        self.warmup_proportion = 0.02

        self.n_models = n_models
        self.num_pipeline_stages = num_pipeline_stages
        self.n_data_workers = self.dp_worldsize = self.n_models // self.num_pipeline_stages
        self.seed = seed

        self.total_train_epochs = total_train_epochs

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            data_worker=TasksGroup(
                count=self.n_data_workers,
                scheduling=Scheduling.data_worker_default(cpu=2, mem=10000, node_type="g1"),
            ),
            master_worker=TasksGroup(
                count=1,
                scheduling=Scheduling.master_worker_default(cpu=4, mem=10000, node_type="g1"),
            ),
            model_worker=TasksGroup(
                count=self.n_models,
                scheduling=Scheduling.model_worker_default(
                    cpu=4,
                    gpu=1,
                    gpu_type='tesla',
                    nodelist="frl8a138",
                    mem=60000,
                ),
            ),
        )

    def initial_setup(self) -> ExperimentConfig:
        # model_path = "/data/aigc/public/starcoder-16bit"
        model_path = "/lustre/meizy/backup_zy/model_saves/four_layers_starcoder"
        train_batch_size_per_device = 4
        eval_batch_size_per_device = 4
        max_seq_len = 2048

        dataset = Dataset(
            'wpsf_sft_packed',
            args=dict(
                n_tokens_per_batch=max_seq_len * train_batch_size_per_device,
                max_length=max_seq_len,
                max_n_seqs_per_batch=500,
                json_path="/data/aigc/llm/datasets/wps-formula-sft/dllm-train-0908-formula-psi.json",
            ),
        )
        dataloader = eval_dataloader = DataLoader('iterable_dataset_loader')
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
        eval_dataset.args[
            'dataset_path'] = "/data/aigc/llm/datasets/wps-formula-sft/dllm-valid-0908-formula-psi.json"
        eval_dataset.args['n_tokens_per_batch'] = max_seq_len * eval_batch_size_per_device

        backend = ModelBackend(
            'ds_train',
            args=dict(optimizer_name='adam',
                      optimizer_config=dict(
                          lr=self.lora_lr,
                          weight_decay=self.weight_decay,
                          eps=1e-5,
                          betas=self.adam_betas,
                      ),
                      lr_scheduler_type=self.lr_scheduler_type,
                      warmup_steps_proportion=self.warmup_proportion,
                      min_lr_ratio=0.0,
                      zero_stage=1,
                      enable_fp16=True,
                      gradient_checkpointing=False,
                      engine_type="pipe",
                      num_pipeline_stages=self.num_pipeline_stages),
        )

        model = Model("starcoder_flash_mqat_pipe",
                      args=dict(model_path=model_path,
                                num_pp=self.num_pipeline_stages,
                                num_dp=self.dp_worldsize))

        interface = ModelInterface('pipe_flash_sft')

        streams = [RequestReplyStream(f"model{i}") for i in range(self.n_models)]

        model_worker = [
            ModelWorker(
                seed=self.seed,
                model=model,
                backend=backend,
                interface=interface,
                model_name='default',
                stream=streams[i],
                eval_datasets=[dataset],
                eval_dataloader=eval_dataloader,
            ) for i in range(self.n_models)
        ]

        ecs = MasterWorkerECS(model_worker).add_systems([sft])

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


seeds = range(1, 6)
for s in seeds:
    exp_name = f"wpsf-sft-flash-pipe-s{s}"
    register_experiment(exp_name, functools.partial(WpsFormulaSFTPipelineExperiment, seed=s))
