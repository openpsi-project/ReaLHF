import dataclasses
import functools

from omegaconf import MISSING

from reallm.api.config.config_dataset import PairedComparisonDatasetConfig
from reallm.api.quickstart.model import get_flash_mqat_model_config, ModelTrainEvalConfig, OptimizerConfig
from reallm.api.core.system import *
from reallm.api.core.dfg import ModelInterface, ModelInterfaceType, ModelRPC, ModelType
from reallm.base.topology import PipeModelDataParallelTopology
import reallm.base.logging as logging

logger = logging.getLogger("DPO Experiment")


@dataclasses.dataclass
class DPOConfig(Experiment):
    experiment_name: str = MISSING
    trial_name: str = MISSING
    trace: bool = False
    seed: int = 1
    total_train_epochs: int = 1
    save_freq_steps: Optional[int] = 20
    is_sft_lora: bool = False
    sft_lora_path: Optional[str] = None
    actor: ModelTrainEvalConfig = dataclasses.field(default_factory=ModelTrainEvalConfig)
    ref: ModelTrainEvalConfig = dataclasses.field(default_factory=ModelTrainEvalConfig)
    dataset: PairedComparisonDatasetConfig = dataclasses.field(default_factory=PairedComparisonDatasetConfig)
    beta: float = 0.1

    def __post_init__(self):
        if self.is_sft_lora and (self.sft_lora_path is None or self.actor.type is None):
            raise ValueError("sft_lora_path and base_model_type must be specified when is_sft_lora is True.")
        if self.actor.base_model_path != self.ref.base_model_path:
            raise ValueError("actor and ref must use the same base model.")
        if self.dataset.valid_path is not None:
            logger.warning(
                "DPO does not support validation because we can't compute reference logps during training.")
        self.n_actors = int(self.actor.parallel.pipeline_parallel_size *
                            self.actor.parallel.data_parallel_size * self.actor.parallel.model_parallel_size)
        self.n_refs = int(self.ref.parallel.pipeline_parallel_size * self.ref.parallel.data_parallel_size *
                          self.ref.parallel.model_parallel_size)

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
                count=self.n_actors + self.n_refs,
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
                n_tokens_per_batch=self.dataset.train_tokens_per_batch,
                max_length=self.dataset.max_seqlen,
                max_pairs_per_prompt=self.dataset.max_pairs_per_prompt,
                dataset_path=self.dataset.train_path,
            ),
        )
        dataloader = DataLoader("iterable_dataset_loader")
        data_worker = [
            DataWorker(
                tokenizer_name_or_path=self.actor.base_model_path,
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
                    lr=self.actor.optimizer.lr,
                    weight_decay=self.actor.optimizer.weight_decay,
                    eps=self.actor.optimizer.eps,
                    betas=(self.actor.optimizer.beta1, self.actor.optimizer.beta2),
                ),
                lr_scheduler_type=self.actor.optimizer.lr_scheduler_type,
                warmup_steps_proportion=self.actor.optimizer.warmup_steps_proportion,
                min_lr_ratio=self.actor.optimizer.min_lr_ratio,
                zero_stage=self.actor.optimizer.zero_stage if self.actor.parallel.pipeline_parallel_size == 1
                else min(self.actor.optimizer.zero_stage, 1),
                gradient_checkpointing=self.actor.gradient_checkpointing,
                num_pipeline_stages=self.actor.parallel.pipeline_parallel_size,
                engine_type="pipe" if self.actor.parallel.pipeline_parallel_size > 1 else "deepspeed",
                offload_optimizer_state=self.actor.optimizer.offload,
                enable_bf16=self.actor.enable_bf16,
                enable_fp16=self.actor.enable_fp16,
                sequence_parallel=self.actor.parallel.use_sequence_parallel,
            ),
        )
        inf_backend = ModelBackend(
            "ds_inference",
            args=dict(
                enable_fp16=(not self.ref.enable_bf16),
                zero_stage=3 if self.ref.offload else 0,
                offload=self.ref.offload,
                enable_bf16=self.ref.enable_bf16,
                engine_type="pipe" if self.ref.parallel.pipeline_parallel_size > 1 else "deepspeed",
                sequence_parallel=self.ref.parallel.use_sequence_parallel,
            ),
        )

        # We should merge pipeline model weights for the reference model to load.
        ref_model = get_flash_mqat_model_config(
            from_type="self",
            model_path=self.ref.path,
            hf_model_type=self.ref.type,
            tokenizer_path=self.ref.base_model_path,
            use_pipe=self.ref.parallel.pipeline_parallel_size > 1,
            dtype="bf16" if self.ref.enable_bf16 else "fp16",
            sequence_parallel=self.ref.parallel.use_sequence_parallel,
            partition_method=self.ref.parallel.partition_method,
        )
        model = get_flash_mqat_model_config(
            from_type="self",
            model_path=self.actor.path,
            hf_model_type=self.actor.type,
            tokenizer_path=self.actor.base_model_path,
            use_pipe=self.actor.parallel.pipeline_parallel_size > 1,
            dtype="bf16" if self.actor.enable_bf16 else "fp16",
            sequence_parallel=self.actor.parallel.use_sequence_parallel,
            partition_method=self.actor.parallel.partition_method,
            lora=self.actor.lora,
        )

        interface = ModelInterface("flash_dpo", args=dict(beta=self.beta, enable_save=True))
        ref_interface = ModelInterface("flash_dpo", args=dict(beta=self.beta, enable_save=False))

        actor_topo = PipeModelDataParallelTopology(
            self.actor.parallel.pipeline_parallel_size,
            self.actor.parallel.model_parallel_size,
            self.actor.parallel.data_parallel_size,
        )
        ref_topo = PipeModelDataParallelTopology(
            self.ref.parallel.pipeline_parallel_size,
            self.ref.parallel.model_parallel_size,
            self.ref.parallel.data_parallel_size,
        )
        model_worker = []
        for i in range(self.n_actors):
            coord = actor_topo.get_coord(i)
            mw = ModelWorker(
                seed=self.seed,
                model=model,
                backend=train_backend,
                interface=interface,
                model_name="actor",
                topo=actor_topo,
                dp_rank=coord.data,
                pp_rank=coord.pipe,
                mp_rank=coord.model,
                cuda_cache_cleanliness=True,
                cuda_cache_clear_freq=1,
            )
            model_worker.append(mw)
        for i in range(self.n_refs):
            coord = ref_topo.get_coord(i)
            mw = ModelWorker(
                seed=self.seed,
                model=ref_model,
                backend=inf_backend,
                interface=ref_interface,
                model_name="ref",
                topo=ref_topo,
                dp_rank=coord.data,
                pp_rank=coord.pipe,
                mp_rank=coord.model,
                cuda_cache_cleanliness=True,
                cuda_cache_clear_freq=1,
            )
            model_worker.append(mw)

        ref_inf = ModelRPC(
            model_name="ref",
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=ref_interface,
            model_type=ModelType("llama", 7, False),
            input_data=["packed_input_ids", "input_lens", "pair_input_lens", "prompt_lens"],
            output_data=["seqlogp"],
            output_key_remap={"seqlogp": "pair_ref_seqlogp"},
        )
        dpo = ModelRPC(
            model_name="actor",
            interface_type=ModelInterfaceType.TRAIN_STEP,
            interface_impl=interface,
            model_type=ModelType("llama", 7, False),
            input_data=[
                "packed_input_ids", "input_lens", "pair_input_lens", "pair_ref_seqlogp", "prompt_lens"
            ],
            log_return_value=True,
        )

        cfg = ExperimentConfig(
            total_train_epochs=self.total_train_epochs,
            save_frequency_steps=self.save_freq_steps,
            model_rpcs=[dpo, ref_inf],
            data_worker=data_worker,
            model_worker=model_worker,
        )
        return cfg
