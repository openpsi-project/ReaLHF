import dataclasses
import functools
import math
import random

from api.config import *
from api.dfg import ModelInterfaceType, ModelRPC
from base.topology import PipeModelDataParallelTopology

sft = ModelRPC(
    "default",
    ModelInterfaceType.TRAIN_STEP,
    input_data=["packed_input_ids", "cu_seqlens", "prompt_mask"],
    dp_broker_type="packed",
    log_return_value=True,
)


@dataclasses.dataclass
class PackedSupervisedFinetuningExperiment(Experiment):
    seed: int = 1
    total_train_epochs: int = 1
    # model
    base_model: str = "gpt2"
    dp_size: int = 16
    use_lora: bool = False
    lora_scaling: float = 32.0
    lora_dim: int = 32
    enable_fp16: bool = True
    gradient_checkpointing: bool = True
    # dataset
    max_seqlen: int = 1024
    train_tokens_per_batch: int = 16384
    valid_tokens_per_batch: int = 16384
    train_dataset_path: str = "/lustre/meizy/data/wps-formula/train.json"
    valid_dataset_path: str = "/lustre/meizy/data/wps-formula/valid.json"
    # optimizer
    lr: float = 2.5e-4
    weight_decay: float = 0.05
    adam_betas: tuple = (0.9, 0.95)
    lr_scheduler_type: str = "cosine"
    warmup_proportion: float = 0.02
    adam_eps: float = 1e-5
    min_lr_ratio: float = 0.0
    zero_stage: int = 2

    def __post_init__(self):
        if self.base_model == "gpt2" and self.max_seqlen > 1024:
            raise ValueError("GPT2 only supports max seqlen of 1024")

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            data_worker=TasksGroup(
                count=1,
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
                    gpu_type="tesla",
                    mem=100000,
                ),
            ),
        )

    def initial_setup(self) -> ExperimentConfig:
        if self.base_model == "starcoder":
            model_path = "/data/aigc/public/starcoder-16bit"
        elif self.base_model == "gpt2":
            model_path = "/lustre/fw/pretrained/gpt2-large/"
        elif self.base_model == "llama":
            model_path = "/lustre/public/pretrained_model_weights/Llama-2-13b-hf"
        else:
            raise NotImplementedError()

        dataset = Dataset(
            "packed_prompt_answer",
            args=dict(
                n_tokens_per_batch=self.train_tokens_per_batch,
                max_length=self.max_seqlen,
                dataset_path=self.train_dataset_path,
            ),
        )
        dataloader = eval_dataloader = DataLoader("iterable_dataset_loader")
        data_worker = [
            DataWorker(
                tokenizer_name_or_path=model_path,
                datasets=[dataset],
                dataloader=dataloader,
                seed=self.seed,
            )
        ]

        eval_dataset = copy.deepcopy(dataset)
        eval_dataset.args["dataset_path"] = self.valid_dataset_path
        eval_dataset.args["n_tokens_per_batch"] = self.valid_tokens_per_batch

        backend = ModelBackend(
            "ds_train",
            args=dict(
                optimizer_name="adam",
                optimizer_config=dict(
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    eps=self.adam_eps,
                    betas=self.adam_betas,
                ),
                lr_scheduler_type=self.lr_scheduler_type,
                warmup_steps_proportion=self.warmup_proportion,
                min_lr_ratio=self.min_lr_ratio,
                zero_stage=self.zero_stage,
                enable_fp16=self.enable_fp16,
                gradient_checkpointing=self.gradient_checkpointing,
            ),
        )

        model = Model("flash_mqat_clm_hf", args=dict(model_path=model_path, from_type=self.base_model))
        if self.use_lora:
            model.wrappers = [
                ModelWrapper(
                    "lora",
                    args=dict(
                        lora_module_kwargs=dict(
                            lora_dim=self.lora_dim,
                            lora_scaling=self.lora_scaling,
                        ),
                        lora_keys_to_replace=["c_attn.linear", "c_proj."],
                    ),
                ),
            ]

        interface = ModelInterface("flash_sft")

        model_worker = [
            ModelWorker(
                seed=self.seed,
                model=model,
                backend=backend,
                interface=interface,
                model_name="default",
                eval_datasets=[dataset],
                eval_dataloader=eval_dataloader,
                cuda_cache_cleanliness=True,
                dp_rank=i,
                topo=PipeModelDataParallelTopology(1, 1, self.dp_size),
                cuda_cache_clear_freq=60,
            ) for i in range(self.dp_size)
        ]

        cfg = ExperimentConfig(
            total_train_epochs=self.total_train_epochs,
            save_frequency_steps=None,
            save_frequency_epochs=1,
            save_frequency_seconds=None,
            eval_frequency_epochs=1,
            model_rpcs=[sft],
            data_worker=data_worker,
            model_worker=model_worker,
        )
        return cfg


seeds = list(range(1, 6)) + [42]
for s in seeds:
    exp_name = f"wpsf-sft-flash-s{s}"
    register_experiment(exp_name, functools.partial(PackedSupervisedFinetuningExperiment, seed=s))
    exp_name = f"senti-sft-pos-neg-s{s}"
    register_experiment(
        exp_name,
        functools.partial(
            PackedSupervisedFinetuningExperiment,
            seed=s,
            base_model="gpt2",
            train_dataset_path="/lustre/fw/datasets/imdb/rl/sft_pos_neg-train.jsonl",
            valid_dataset_path="/lustre/fw/datasets/imdb/rl/sft_pos_neg-valid.jsonl",
            use_lora=False,
        ),
    )
    exp_name = f"senti-sft-pos-s{s}"
    register_experiment(
        exp_name,
        functools.partial(
            PackedSupervisedFinetuningExperiment,
            seed=s,
            base_model="gpt2",
            train_dataset_path="/lustre/fw/datasets/imdb/rl/sft_pos-train.jsonl",
            valid_dataset_path="/lustre/fw/datasets/imdb/rl/sft_pos-valid.jsonl",
            use_lora=False,
        ),
    )
