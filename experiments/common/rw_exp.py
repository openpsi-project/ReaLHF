import dataclasses
import functools
import math
import random

from omegaconf import MISSING

from api.config.config_dataset import PairedComparisonDatasetConfig
from api.config.config_flash_model import get_flash_mqat_model_config, ModelTrainEvalConfig, OptimizerConfig
from api.config.config_system import *
from api.config.dfg import ModelInterface, ModelInterfaceType, ModelRPC
from base.topology import PipeModelDataParallelTopology


@dataclasses.dataclass
class RWConfig(Experiment):
    experiment_name: str = MISSING
    trial_name: str = MISSING
    trace: bool = False
    seed: int = 1
    total_train_epochs: int = 1
    save_freq_steps: Optional[int] = 20
    eval_freq_epochs: Optional[int] = 1
    is_sft_lora: bool = False
    sft_lora_path: Optional[str] = None
    model: ModelTrainEvalConfig = dataclasses.field(default_factory=ModelTrainEvalConfig)
    dataset: PairedComparisonDatasetConfig = dataclasses.field(default_factory=PairedComparisonDatasetConfig)

    def __post_init__(self):
        if self.is_sft_lora and (self.sft_lora_path is None or self.model.type is None):
            raise ValueError("sft_lora_path and base_model_type must be specified when is_sft_lora is True.")

        self.world_size = (self.model.parallel.pipeline_parallel_size *
                           self.model.parallel.model_parallel_size * self.model.parallel.data_parallel_size)

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
                scheduling=Scheduling.master_worker_default(
                    cpu=4,
                    mem=20000,
                ),
            ),
            model_worker=TasksGroup(
                count=self.world_size,
                scheduling=Scheduling.model_worker_default(
                    cpu=4,
                    gpu=1,
                    gpu_type="tesla",
                    mem=60000,
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
        dataloader = eval_dataloader = DataLoader("iterable_dataset_loader")
        data_worker = [
            DataWorker(
                tokenizer_name_or_path=self.model.base_model_path,
                datasets=[dataset],
                dataloader=dataloader,
                seed=self.seed,
            )
        ]

        eval_dataset = copy.deepcopy(dataset)
        eval_dataset.args["dataset_path"] = self.dataset.valid_path
        eval_dataset.args["n_tokens_per_batch"] = self.dataset.valid_tokens_per_batch

        backend = ModelBackend(
            "ds_train",
            args=dict(
                optimizer_name="adam",
                optimizer_config=dict(
                    lr=self.model.optimizer.lr,
                    weight_decay=self.model.optimizer.weight_decay,
                    eps=self.model.optimizer.eps,
                    betas=(self.model.optimizer.beta1, self.model.optimizer.beta2),
                ),
                lr_scheduler_type=self.model.optimizer.lr_scheduler_type,
                warmup_steps_proportion=self.model.optimizer.warmup_steps_proportion,
                min_lr_ratio=self.model.optimizer.min_lr_ratio,
                zero_stage=self.model.optimizer.zero_stage if self.model.parallel.pipeline_parallel_size == 1
                else min(self.model.optimizer.zero_stage, 1),
                gradient_checkpointing=self.model.gradient_checkpointing,
                num_pipeline_stages=self.model.parallel.pipeline_parallel_size,
                engine_type="pipe" if self.model.parallel.pipeline_parallel_size > 1 else "deepspeed",
                offload_optimizer_state=self.model.optimizer.offload,
                enable_bf16=self.model.enable_bf16,
                enable_fp16=self.model.enable_fp16,
                sequence_parallel=self.model.parallel.use_sequence_parallel,
            ),
        )

        model = get_flash_mqat_model_config(
            from_type="actor_as_critic",
            model_path=self.model.path,
            hf_model_type=self.model.type,
            tokenizer_path=self.model.base_model_path,
            use_pipe=self.model.parallel.pipeline_parallel_size > 1,
            dtype="bf16" if self.model.enable_bf16 else "fp16",
            sequence_parallel=self.model.parallel.use_sequence_parallel,
            partition_method=self.model.parallel.partition_method,
            lora=self.model.lora,
        )

        interface = ModelInterface("flash_paired_rw")

        # NOTE: The dims of the parallelism grid is [pipline, data, model]
        # i.e., model parallelism is scheduled as close as possible in a single node.
        # This may seem incorrect according the class name "PipeModelDataParallelTopology",
        # but after you inspect the code, you will find that the class name actually
        # should be "PipeDataModelParallelTopology". Thank you DeepSpeed!
        topo = PipeModelDataParallelTopology(
            self.model.parallel.pipeline_parallel_size,
            self.model.parallel.model_parallel_size,
            self.model.parallel.data_parallel_size,
        )
        model_worker = []
        for i in range(self.world_size):
            coord = topo.get_coord(i)
            mw = ModelWorker(
                seed=self.seed,
                model=model,
                backend=backend,
                interface=interface,
                model_name="default",
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

        rw_modeling = ModelRPC(
            "default",
            ModelInterfaceType.TRAIN_STEP,
            input_data=["packed_input_ids", "input_lens", "group_factor", "pair_input_lens"],
            log_return_value=True,
        )

        cfg = ExperimentConfig(
            total_train_epochs=self.total_train_epochs,
            save_frequency_steps=self.save_freq_steps,
            eval_frequency_epochs=self.eval_freq_epochs,
            model_rpcs=[rw_modeling],
            data_worker=data_worker,
            model_worker=model_worker,
        )
        return cfg
