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
                scheduling=Scheduling.data_worker_default(cpu=2, mem=10000),
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
                    nodelist="QH-com08",
                    mem=100000,
                ),
            ),
        )

    def initial_setup(self) -> ExperimentConfig:
        model_path = "/lustre/meizy/models/pipe_pretrained/starcoder_4pp_3s"
        # if need to use ckpt saved in previous experiments, set ckpt_path
        ckpt_path = None
        train_batch_size_per_device = 8
        eval_batch_size_per_device = 8
        max_seq_len = 2048

        dataset = Dataset(
            'packed_prompt_answer',
            args=dict(
                n_tokens_per_batch=max_seq_len * train_batch_size_per_device,
                max_length=max_seq_len,
                dataset_path="/lustre/meizy/data/wps-formula/train.json",
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
        eval_dataset.args['dataset_path'] = "/lustre/meizy/data/wps-formula/valid.json"
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
                      args=dict(
                          model_path=model_path,
                          num_pp=self.num_pipeline_stages,
                          num_dp=self.dp_worldsize,
                          load_from_full_ckpt=False,
                          ckpt_path=ckpt_path,
                      ))

        interface = ModelInterface('pipe_flash_sft')

        model_worker = []
        assert self.dp_worldsize * self.num_pipeline_stages == self.n_models
        topo = PipeModelDataParallelTopology(self.num_pipeline_stages, 1, self.dp_worldsize)
        for i in range(self.n_models):
            coord = topo.get_coord(i)
            mw = ModelWorker(
                seed=self.seed,
                model=model,
                backend=backend,
                interface=interface,
                model_name='default',
                topo=topo,
                dp_rank=coord.data,
                pp_rank=coord.pipe,
                mp_rank=coord.model,
                eval_datasets=[dataset],
                eval_dataloader=eval_dataloader,
                cuda_cache_cleanliness=True,
                cuda_cache_clear_freq=1,
            )
            model_worker.append(mw)

        cfg = ExperimentConfig(
            total_train_epochs=self.total_train_epochs,
            save_frequency_steps=10,
            save_frequency_epochs=1,
            save_frequency_seconds=None,
            eval_frequency_epochs=1,
            model_rpcs=[sft],
            data_worker=data_worker,
            model_worker=model_worker,
            benchmark_steps=15,
        )
        return cfg


seeds = range(1, 6)
for s in seeds:
    exp_name = f"wpsf-sft-flash-pipe-s{s}"
    register_experiment(exp_name, functools.partial(WpsFormulaSFTPipelineExperiment, seed=s))
