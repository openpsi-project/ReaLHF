import functools
import math
import random

from api.config import *
from api.dfg import ModelInterfaceType, ModelRPC
from base.topology import PipeModelDataParallelTopology

sft = ModelRPC(
    "default",
    ModelInterfaceType.TRAIN_STEP,
    input_data=['packed_input_ids', 'cu_seqlens', 'prompt_mask'],
    dp_broker_type='packed',
)


def sample_log_uniform(low, high):
    low = math.log(low)
    high = math.log(high)
    return math.exp(low + (high - low) * random.random())


class WpsFormulaSupervisedFinetuningExperiment(Experiment):

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
        if self.benchmark_only:
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
                    nodelist="frl8a138",
                    mem=60000,
                ),
            ),
        )

    def initial_setup(self) -> ExperimentConfig:
        if not self.benchmark_only:
            model_path = "/data/aigc/public/starcoder-16bit"
        else:
            model_path = "/data/aigc/llm/checkpoints/1l-starcoder"
        train_batch_size_per_device = 4
        eval_batch_size_per_device = 4
        max_seq_len = 4096

        dataset = Dataset(
            'wpsf_sft_packed',
            args=dict(
                n_tokens_per_batch=max_seq_len * train_batch_size_per_device,
                max_length=max_seq_len,
                json_path="/data/aigc/llm/datasets/wps-formula-sft/dllm-train-0908-formula-psi.json",
            ),
        )
        dataloader = eval_dataloader = DataLoader('iterable_dataset_loader')
        data_worker = [
            DataWorker(
                tokenizer_name_or_path=model_path,
                datasets=[dataset],
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

        model = Model("flash_mqat_clm_hf_lora",
                      args=dict(
                          model_path=model_path,
                          lora_module_kwargs=dict(
                              lora_dim=self.lora_dim,
                              lora_scaling=self.lora_scaling,
                          ),
                          lora_keys_to_replace=['c_attn.linear', 'c_proj.'],
                      ))

        interface = ModelInterface('flash_sft')

        model_worker = [
            ModelWorker(
                seed=self.seed,
                model=model,
                backend=backend,
                interface=interface,
                model_name='default',
                eval_datasets=[dataset],
                eval_dataloader=eval_dataloader,
                dp_rank=i,
                topo=PipeModelDataParallelTopology(1, 1, self.n_models),
            ) for i in range(self.n_models)
        ]

        cfg = ExperimentConfig(
            total_train_epochs=self.total_train_epochs,
            save_frequency_steps=None,
            save_frequency_epochs=None if self.benchmark_only else 1,
            save_frequency_seconds=None,
            eval_frequency_epochs=None if self.benchmark_only else 1,
            model_rpcs=[sft],
            data_worker=data_worker,
            model_worker=model_worker,
        )
        return cfg


seeds = range(1, 6)
for s in seeds:
    exp_name = f"wpsf-sft-flash-s{s}"
    register_experiment(exp_name, functools.partial(WpsFormulaSupervisedFinetuningExperiment, seed=s))
register_experiment("wpsf-sft-flash-benchmark",
                    functools.partial(WpsFormulaSupervisedFinetuningExperiment, benchmark_only=True))
