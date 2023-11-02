import functools
import math
import random

from api.config import *
from api.dfg import ModelInterfaceType, ModelRPC
from base.topology import PipeModelDataParallelTopology

ref_inf = ModelRPC(
    "ref",
    ModelInterfaceType.INFERENCE,
    input_data=['pos_input_ids', 'pos_attention_mask', 'neg_input_ids', 'neg_attention_mask', 'prompt_lens'],
    output_data=['seqlogp'],
    output_key_remap={'seqlogp': 'ref_seqlogp'},
)
dpo = ModelRPC(
    'actor',
    ModelInterfaceType.TRAIN_STEP,
    input_data=[
        'pos_input_ids', 'pos_attention_mask', 'neg_input_ids', 'neg_attention_mask', 'prompt_lens',
        'ref_seqlogp'
    ],
    log_return_value=True,
)


class DPOExperiment(Experiment):

    def __init__(
        self,
        dp_size=6,
        seed=1,
        total_train_epochs=8,
        base_model='gpt2',
        dataset_path="/lustre/fw/datasets/imdb/rl/rm_paired-all.jsonl",
        train_batch_size_per_device: int = 2,
        eval_batch_size_per_device: int = 4,
        max_pairs_per_prompt: int = 2,
        use_lora: bool = False,
        beta: float = 0.1,
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
        self.dataset_path = dataset_path

        self.train_batch_size_per_device = train_batch_size_per_device
        self.eval_batch_size_per_device = eval_batch_size_per_device
        self.max_pairs_per_prompt = max_pairs_per_prompt

        self.beta = beta

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
                count=self.dp_size + 1,
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
            'rw_pair',
            args=dict(
                max_seq_len=max_seq_len,
                pad_to_max_length=True,
                max_pairs_per_prompt=self.max_pairs_per_prompt,
                dataset_path=self.dataset_path,
            ),
        )
        dataloader = DataLoader('concat',
                                args=dict(
                                    batch_size=self.train_batch_size_per_device * self.dp_size //
                                    self.n_data_workers,
                                    drop_last=True,
                                    shuffle=True,
                                ))
        data_worker = [
            DataWorker(
                tokenizer_name_or_path=base_model_path,
                datasets=[dataset],
                dataloader=dataloader,
                seed=self.seed,
            ) for i in range(self.n_data_workers)
        ]

        train_backend = ModelBackend(
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
        inf_backend = ModelBackend('ds_inference', args=dict(enable_fp16=True))

        # TODO: We should use SFT model here. Use base model for debugging.
        model = Model("causal_lm", args=dict(model_name_or_path=base_model_path))
        ref_model = copy.deepcopy(model)
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

        interface = ModelInterface('dpo', args=dict(beta=0.1, enable_save=True))
        ref_interface = ModelInterface('dpo', args=dict(beta=0.1, enable_save=False))

        model_worker = [
            ModelWorker(
                seed=self.seed,
                model=model,
                backend=train_backend,
                interface=interface,
                model_name='actor',
                dp_rank=i,
                topo=PipeModelDataParallelTopology(1, 1, self.dp_size),
                cuda_cache_clear_freq=60,
            ) for i in range(self.dp_size)
        ] + [
            ModelWorker(
                seed=self.seed,
                model=ref_model,
                backend=inf_backend,
                interface=ref_interface,
                model_name='ref',
                dp_rank=0,
                topo=PipeModelDataParallelTopology(1, 1, 1),
                cuda_cache_clear_freq=60,
            )
        ]

        cfg = ExperimentConfig(
            total_train_epochs=self.total_train_epochs,
            save_frequency_steps=20,
            save_frequency_epochs=None,
            save_frequency_seconds=None,
            eval_frequency_epochs=None,
            model_rpcs=[dpo, ref_inf],
            data_worker=data_worker,
            model_worker=model_worker,
        )
        return cfg


seeds = list(range(1, 6)) + [42]
for s in seeds:
    exp_name = f"dpo-s{s}"
    register_experiment(exp_name, functools.partial(DPOExperiment, seed=s))
