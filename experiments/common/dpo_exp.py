import dataclasses
import functools

from api.config import *
from api.dfg import ModelInterfaceType, ModelRPC
from base.topology import PipeModelDataParallelTopology
from experiments.common.config_utils import get_flash_mqat_model_config

ref_inf = ModelRPC(
    "ref",
    ModelInterfaceType.INFERENCE,
    input_data=["packed_input_ids", "input_lens", "pair_input_lens", "prompt_lens"],
    output_data=["seqlogp"],
    output_key_remap={"seqlogp": "pair_ref_seqlogp"},
    dp_broker_type="packed",
)
dpo = ModelRPC(
    "actor",
    ModelInterfaceType.TRAIN_STEP,
    input_data=["packed_input_ids", "input_lens", "pair_input_lens", "pair_ref_seqlogp", "prompt_lens"],
    dp_broker_type="packed",
    log_return_value=True,
)


@dataclasses.dataclass
class DPOExperiment(Experiment):
    model_path: str  # Must be SFT model path
    tokenizer_path: str  # Since we use SFT model, we need to specify HF tokenizer path

    seed: int = 1
    total_train_epochs: int = 1
    save_freq_steps: int = 20
    is_sft_pipe: bool = False
    is_sft_lora: bool = False
    base_model_type: Optional[str] = None
    sft_lora_path: Optional[str] = None
    # model
    dp_size: int = 1
    mp_size: int = 1
    pp_size: int = 1
    use_lora: bool = False
    lora_scaling: float = 32.0
    lora_dim: int = 32
    enable_fp16: bool = True
    enable_bf16: bool = False
    offload_optimizer: bool = False
    gradient_checkpointing: bool = True
    # dataset
    max_pairs_per_prompt: int = 2
    max_seqlen: int = 1024
    # NOTE: DPO does not support evaluation because we can't compute reference logp when training the actor.
    train_tokens_per_batch: int = 16384
    train_dataset_path: str = "/lustre/fw/datasets/imdb/rl/rm_paired-train.jsonl"
    # optimizer
    lr: float = 2.5e-4
    weight_decay: float = 0.05
    adam_betas: tuple = (0.9, 0.95)
    lr_scheduler_type: str = "cosine"
    warmup_proportion: float = 0.02
    adam_eps: float = 1e-5
    min_lr_ratio: float = 0.0
    zero_stage: int = 2
    # dpo
    beta: float = 0.1

    num_pipeline_micro_batches: Optional[int] = None
    use_sequence_parallel: bool = False
    partition_method: Optional[str] = "parameters"

    def __post_init__(self):
        if self.pp_size < 1 or self.dp_size < 1 or self.mp_size < 1:
            raise ValueError("pp_size, mp_size and dp_size must be positive integers.")
        if self.pp_size > 1 and self.use_lora:
            raise ValueError("Use LoRA with pipeline parallel is not supported.")
        if self.is_sft_lora and (self.sft_lora_path is None or self.base_model_type is None):
            raise ValueError("sft_lora_path and base_model_type must be specified when is_sft_lora is True.")
        if self.enable_bf16 and (self.pp_size > 1 or self.mp_size):
            raise ValueError("Use bf16 with pipeline parallel or model parallel is not supported.")

        self.n_actors = int(self.dp_size * self.pp_size * self.mp_size)

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
                count=self.n_actors + 1,
                scheduling=Scheduling.model_worker_default(
                    cpu=4,
                    gpu=1,
                    gpu_type="tesla",
                    mem=100000,
                ),
            ),
        )

    def initial_setup(self) -> ExperimentConfig:
        dataset = Dataset(
            "packed_rw_pair",
            args=dict(
                n_tokens_per_batch=self.train_tokens_per_batch,
                max_length=self.max_seqlen,
                max_pairs_per_prompt=self.max_pairs_per_prompt,
                dataset_path=self.train_dataset_path,
            ),
        )

        dataloader = DataLoader("iterable_dataset_loader")
        data_worker = [
            DataWorker(
                tokenizer_name_or_path=self.tokenizer_path,
                datasets=[dataset],
                dataloader=dataloader,
                seed=self.seed,
            )
        ]

        train_backend = ModelBackend(
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
                zero_stage=self.zero_stage if self.pp_size == 1 else min(self.zero_stage, 1),
                gradient_checkpointing=self.gradient_checkpointing,
                num_pipeline_stages=self.pp_size,
                engine_type="pipe" if self.pp_size > 1 else "deepspeed",
                num_pipeline_micro_batches=self.num_pipeline_micro_batches,
                enable_fp16=self.enable_fp16,
                enable_bf16=self.enable_bf16,
                offload_optimizer_state=self.offload_optimizer,
                sequence_parallel=self.use_sequence_parallel,
            ),
        )
        inf_backend = ModelBackend("ds_inference",
                                   args=dict(enable_fp16=(not self.enable_fp16),
                                             enable_bf16=self.enable_bf16))

        # We should merge pipeline model weights for the reference model to load.
        ref_model = get_flash_mqat_model_config(
            model_path=self.model_path,
            from_model_type=("pipe" if self.is_sft_pipe else "self")
            if not self.is_sft_lora else self.base_model_type,
            tokenizer_path=self.tokenizer_path,
            pp_size=1,
            mp_size=1,
            dp_size=1,
            is_critic=False,
            use_lora=False,
            partition_method=self.partition_method,
        )
        model = get_flash_mqat_model_config(
            model_path=self.model_path,
            from_model_type="self" if not self.is_sft_lora else self.base_model_type,
            tokenizer_path=self.tokenizer_path,
            pp_size=self.pp_size,
            mp_size=self.mp_size,
            dp_size=self.dp_size,
            is_critic=False,
            use_lora=self.use_lora,
            lora_dim=self.lora_dim,
            lora_scaling=self.lora_scaling,
            is_sft_lora=self.is_sft_lora,
            sft_lora_path=self.sft_lora_path,
            partition_method=self.partition_method,
            sequence_parallel=self.use_sequence_parallel,
        )

        interface = ModelInterface("flash_dpo", args=dict(beta=0.1, enable_save=True))
        ref_interface = ModelInterface("flash_dpo", args=dict(beta=0.1, enable_save=False))

        topo = PipeModelDataParallelTopology(self.pp_size, self.mp_size, self.dp_size)
        model_worker = []
        for i in range(self.pp_size * self.dp_size * self.mp_size):
            coord = topo.get_coord(i)
            mw = ModelWorker(
                seed=self.seed,
                model=model,
                backend=train_backend,
                interface=interface,
                model_name="actor",
                topo=topo,
                dp_rank=coord.data,
                pp_rank=coord.pipe,
                mp_rank=coord.model,
                cuda_cache_cleanliness=True,
                cuda_cache_clear_freq=1,
            )
            model_worker.append(mw)

        model_worker += [
            ModelWorker(
                seed=self.seed,
                model=ref_model,
                backend=inf_backend,
                interface=ref_interface,
                model_name="ref",
                topo=PipeModelDataParallelTopology(1, 1, 1),
                cuda_cache_cleanliness=True,
                cuda_cache_clear_freq=60,
            )
        ]

        cfg = ExperimentConfig(
            total_train_epochs=self.total_train_epochs,
            save_frequency_steps=self.save_freq_steps,
            model_rpcs=[dpo, ref_inf],
            data_worker=data_worker,
            model_worker=model_worker,
        )
        return cfg
