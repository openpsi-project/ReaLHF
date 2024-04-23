import dataclasses

from omegaconf import MISSING

from reallm.api.config.config_dataset import PromptAnswerDatasetConfig
from reallm.api.quickstart.model import get_flash_mqat_model_config, ModelTrainEvalConfig, OptimizerConfig
from reallm.api.core.system import *
from reallm.api.core.dfg import ModelInterface, ModelInterfaceType, ModelRPC, ModelType
from reallm.base.topology import PipeModelDataParallelTopology


@dataclasses.dataclass
class SFTConfig(Experiment):
    _configuration_name: str = "Supervised-Finetuning"

    experiment_name: str = MISSING
    trial_name: str = MISSING
    trace: bool = False
    seed: int = 1
    total_train_epochs: int = 1
    save_freq_steps: Optional[int] = 50
    eval_freq_epochs: Optional[int] = 1
    model: ModelTrainEvalConfig = dataclasses.field(default_factory=ModelTrainEvalConfig)
    dataset: PromptAnswerDatasetConfig = dataclasses.field(default_factory=PromptAnswerDatasetConfig)

    def __post_init__(self):
        if self.model.type == "gpt2" and self.dataset.max_seqlen > 1024:
            raise ValueError("GPT2 only supports max seqlen of 1024")

        self.world_size = (self.model.parallel.pipeline_parallel_size *
                           self.model.parallel.data_parallel_size * self.model.parallel.model_parallel_size)

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
                count=self.world_size,
                scheduling=Scheduling.model_worker_default(
                    cpu=4,
                    gpu=1,
                    gpu_type="tesla",
                    mem=100000,
                ),
            ),
        )

    def initial_setup(self) -> ExperimentConfig:
        model_path = self.model.path

        dataset = Dataset(
            "packed_prompt_answer",
            args=dict(
                n_tokens_per_batch=self.dataset.train_tokens_per_batch,
                min_seqs_per_batch=self.dataset.train_tokens_per_batch // self.dataset.max_seqlen,
                max_length=self.dataset.max_seqlen,
                dataset_path=self.dataset.train_path,
            ),
        )
        dataloader = eval_dataloader = DataLoader("iterable_dataset_loader")
        data_worker = [
            DataWorker(
                tokenizer_name_or_path=self.model.path,
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
            from_type="hf_as_actor",
            model_path=model_path,
            hf_model_type=self.model.type,
            tokenizer_path=model_path,
            use_pipe=(self.model.parallel.pipeline_parallel_size > 1),
            dtype="bf16" if self.model.enable_bf16 else "fp16",
            sequence_parallel=self.model.parallel.use_sequence_parallel,
            partition_method=self.model.parallel.partition_method,
            lora=self.model.lora,
        )

        interface = ModelInterface("flash_sft")

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
                cuda_cache_cleanliness=False,
                cuda_cache_clear_freq=1,
            )
            model_worker.append(mw)

        sft = ModelRPC(
            model_name="default",
            interface_type=ModelInterfaceType.TRAIN_STEP,
            interface_impl=interface,
            model_type=ModelType("llama", 7, False),
            input_data=["packed_input_ids", "cu_seqlens", "prompt_mask"],
            log_return_value=True,
            min_n_tokens=100000,
            max_n_tokens=131072,
        )

        cfg = ExperimentConfig(
            total_train_epochs=self.total_train_epochs,
            save_frequency_steps=self.save_freq_steps,
            eval_frequency_epochs=self.eval_freq_epochs,
            model_rpcs=[sft],
            data_worker=data_worker,
            model_worker=model_worker,
        )
        return cfg
