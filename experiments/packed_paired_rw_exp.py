import functools
import math
import random

from api.config import *
from api.dfg import ModelInterfaceType, ModelRPC
from base.topology import PipeModelDataParallelTopology

rw_modeling = ModelRPC(
    "default",
    ModelInterfaceType.TRAIN_STEP,
    input_data=['packed_input_ids', 'input_lens', 'group_factor', 'pair_input_lens'],
    dp_broker_type='packed',
    log_return_value=True,
)


class PackedPairedRewardModelingExperiment(Experiment):

    def __init__(
        self,
        dp_size=8,
        seed=1,
        total_train_epochs=1,
        base_model='gpt2',
        train_dataset_path="/lustre/fw/datasets/imdb/rl/rm_paired-train.jsonl",
        valid_dataset_path="/lustre/fw/datasets/imdb/rl/rm_paired-valid.jsonl",
        train_tokens_per_batch: int = 65536,
        eval_tokens_per_batch: int = 131072,
        max_pairs_per_prompt: int = 2,
        use_lora: bool = False,
    ):
        self.use_lora = use_lora
        self.weight_decay = 0.05
        self.lr = 2.5e-4 if use_lora else 1e-5
        self.lora_scaling = 32.0
        self.lora_dim = 32
        self.adam_betas = (0.9, 0.95)
        self.lr_scheduler_type = 'cosine'
        self.warmup_proportion = 0.02

        self.dp_size = dp_size
        self.n_data_workers = 1
        self.seed = seed

        self.total_train_epochs = total_train_epochs
        self.base_model = base_model
        self.train_dataset_path = train_dataset_path
        self.valid_dataset_path = valid_dataset_path

        self.train_tokens_per_batch = train_tokens_per_batch
        self.eval_tokens_per_batch = eval_tokens_per_batch
        self.max_pairs_per_prompt = max_pairs_per_prompt

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
                scheduling=Scheduling.master_worker_default(cpu=4, mem=20000),
            ),
            model_worker=TasksGroup(
                count=self.dp_size,
                scheduling=Scheduling.model_worker_default(
                    cpu=4,
                    gpu=1,
                    gpu_type='tesla',
                    mem=60000,
                ),
            ),
        )

    def initial_setup(self) -> ExperimentConfig:
        if self.base_model == 'starcoder':
            base_model_path = "/data/aigc/public/starcoder-16bit"
        elif self.base_model == 'gpt2':
            base_model_path = "/lustre/fw/pretrained/gpt2-large/"
        else:
            raise NotImplementedError()
        sft_model_path = "/data/aigc/llm/checkpoints/fw/senti-sft-pos-s42/run20231031/default@pp_00-mp_00-dp_00/epoch8step0/"
        max_seq_len = 8192 if self.base_model == 'starcoder' else 512

        dataset = Dataset(
            'packed_rw_pair',
            args=dict(
                n_tokens_per_batch=self.train_tokens_per_batch // self.n_data_workers,
                max_length=max_seq_len,
                max_pairs_per_prompt=self.max_pairs_per_prompt,
                dataset_path=self.train_dataset_path,
            ),
        )
        dataloader = eval_dataloader = DataLoader('iterable_dataset_loader')
        data_worker = [
            DataWorker(
                tokenizer_name_or_path=base_model_path,
                datasets=[dataset],
                dataloader=dataloader,
                seed=self.seed,
            ) for i in range(self.n_data_workers)
        ]

        eval_dataset = copy.deepcopy(dataset)
        eval_dataset.args['dataset_path'] = self.valid_dataset_path
        eval_dataset.args['n_tokens_per_batch'] = self.eval_tokens_per_batch // self.n_data_workers

        backend = ModelBackend(
            'ds_train',
            args=dict(
                optimizer_name='adam',
                optimizer_config=dict(
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    eps=1e-5,
                    betas=self.adam_betas,
                ),
                lr_scheduler_type=self.lr_scheduler_type,
                warmup_steps_proportion=self.warmup_proportion,
                min_lr_ratio=0.0,
                zero_stage=2,
                enable_fp16=True,
                gradient_checkpointing=False,
            ),
        )

        model = Model("flash_mqat_critic",
                      args=dict(
                          model_path=sft_model_path,
                          from_type='sft',
                          tokenizer_path=base_model_path,
                      ))
        if self.use_lora:
            model.wrappers = [
                ModelWrapper(
                    'lora',
                    args=dict(
                        lora_module_kwargs=dict(
                            lora_dim=self.lora_dim,
                            lora_scaling=self.lora_scaling,
                        ),
                        lora_keys_to_replace=['c_attn.linear', 'c_proj.'],
                        additional_module_names_to_opt=['v_head'],
                    ),
                ),
            ]

        interface = ModelInterface('flash_paired_rw')

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
                topo=PipeModelDataParallelTopology(1, 1, self.dp_size),
                cuda_cache_clear_freq=60,
            ) for i in range(self.dp_size)
        ]

        cfg = ExperimentConfig(
            total_train_epochs=self.total_train_epochs,
            save_frequency_steps=20,
            save_frequency_epochs=None,
            save_frequency_seconds=None,
            eval_frequency_epochs=1,
            model_rpcs=[rw_modeling],
            data_worker=data_worker,
            model_worker=model_worker,
        )
        return cfg


seeds = list(range(1, 6)) + [42]
for s in seeds:
    exp_name = f"flash-rw-paired-s{s}"
    register_experiment(exp_name, functools.partial(PackedPairedRewardModelingExperiment, seed=s))
