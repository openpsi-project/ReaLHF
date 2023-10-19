import functools
import math
import random

from api.config import *
from api.ecs import Commands, MasterWorkerECS, ModelQuery, RawDataQuery
from base.cluster import spec as cluster_spec


def rw(
    commands: Commands,
    model: ModelQuery['default'],
    packed_input_ids: RawDataQuery['packed_input_ids'],
    cu_seqlens: RawDataQuery['cu_seqlens'],
    labels: RawDataQuery['labels'],
    n_contrastive_batches: RawDataQuery['n_contrastive_batches'],
    contrastive_dim: RawDataQuery['contrastive_dim'],
):
    inputs = commands.build_model_inputs(
        packed_input_ids=packed_input_ids,
        cu_seqlens=cu_seqlens,
        labels=labels,
        contrastive_dim=contrastive_dim,
        n_contrastive_batches=n_contrastive_batches,
    )
    commands.log(model.train_step(inputs))


class WpsFormulaPlackettLuceRewardModelingExperiment(Experiment):

    def __init__(self, n_models=1, seed=1, total_train_epochs=4, benchmark_only=False):
        self.weight_decay = 0.05
        self.lora_lr = 2.5e-4
        self.lora_scaling = 32.0
        self.lora_dim = 32
        self.adam_betas = (0.9, 0.95)
        self.lr_scheduler_type = 'cosine'
        self.warmup_proportion = 0.02

        self.n_models = self.n_data_workers = n_models
        assert self.n_models == self.n_data_workers
        self.seed = seed

        self.total_train_epochs = total_train_epochs
        self.benchmark_only = benchmark_only
        if benchmark_only:
            self.n_models = self.n_data_workers = 1
            self.total_train_epochs = 1

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
        if self.benchmark_only:
            model_path = f"{cluster_spec.fileroot}/checkpoints/1l-starcoder/"
        else:
            model_path = f"{cluster_spec.fileroot}/checkpoints/starcoder/"
        sft_lora_path = f"{cluster_spec.fileroot}/checkpoints/fw/wpsf-sft-flash-s1/test20230927/default/epoch0step0/lora.bin"

        train_batch_size_per_device = 5
        eval_batch_size_per_device = 2
        max_seq_len = 4096
        contrastive_dim = 6

        dataset = Dataset(
            'wpsf_plrw_packed',
            args=dict(
                n_tokens_per_batch=max_seq_len * train_batch_size_per_device,
                max_length=max_seq_len,
                max_n_seqs_per_batch=500,
                contrastive_dim=contrastive_dim,
                enforce_one_or_less_pos=True,
                json_path=f"{cluster_spec.fileroot}/datasets/wps-formula-rw/dataset_train.jsonl",
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
            'dataset_path'] = f"{cluster_spec.fileroot}/datasets/wps-formula-rw/dataset_val.jsonl"
        eval_dataset.args['n_tokens_per_batch'] = max_seq_len * eval_batch_size_per_device

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
                warmup_steps_proportion=self.warmup_proportion,
                min_lr_ratio=0.0,
                zero_stage=2,
                enable_fp16=True,
                gradient_checkpointing=True,
            ),
        )

        model = Model("flash_mqat_critic_lora",
                      args=dict(
                          model_path=model_path,
                          from_type='starcoder',
                          lora_module_kwargs=dict(
                              lora_dim=self.lora_dim,
                              lora_scaling=self.lora_scaling,
                          ),
                          lora_keys_to_replace=['c_attn.linear', 'c_proj.'],
                          load_lora_path=sft_lora_path if not self.benchmark_only else None,
                          lora_op_after_creation='squash_init',
                      ))

        interface = ModelInterface('flash_plrw')

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

        ecs = MasterWorkerECS(model_worker).add_systems([rw])

        cfg = ExperimentConfig(
            total_train_epochs=self.total_train_epochs,
            save_frequency_steps=None,
            save_frequency_epochs=1 if not self.benchmark_only else None,
            save_frequency_seconds=None,
            eval_frequency_epochs=None,
            master_ecs=ecs,
            data_worker=data_worker,
            model_worker=model_worker,
        )
        return cfg


seeds = range(1, 6)
for s in seeds:
    exp_name = f"wpsf-plrw-flash-s{s}"
    register_experiment(exp_name, functools.partial(WpsFormulaPlackettLuceRewardModelingExperiment, seed=s))
register_experiment("wpsf-plrw-flash-benchmark",
                    functools.partial(WpsFormulaPlackettLuceRewardModelingExperiment, benchmark_only=True))
