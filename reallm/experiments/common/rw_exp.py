import dataclasses
import functools
import math
import random

from omegaconf import MISSING

from reallm.api.core.dfg import ModelInterface, ModelInterfaceType, ModelRPC
from reallm.api.core.system_api import *
from reallm.api.quickstart.dataset import PairedComparisonDatasetConfig
from reallm.api.quickstart.entrypoint import register_quickstart_exp
from reallm.api.quickstart.model import get_real_model_config, ModelTrainEvalConfig, OptimizerConfig
from reallm.base.topology import PipeModelDataParallelTopology


@dataclasses.dataclass
class RWConfig(Experiment):
    experiment_name: str = MISSING
    trial_name: str = MISSING
    seed: int = 1
    total_train_epochs: int = 1
    save_freq_steps: Optional[int] = 20
    eval_freq_epochs: Optional[int] = 1
    is_sft_lora: bool = False
    sft_lora_path: Optional[str] = None
    model: ModelTrainEvalConfig = dataclasses.field(default_factory=ModelTrainEvalConfig)
    dataset: PairedComparisonDatasetConfig = dataclasses.field(default_factory=PairedComparisonDatasetConfig)

    def __post_init__(self):
        assert (not self.is_sft_lora and self.sft_lora_path is None), "LoRA is not supported for now."

        self.world_size = (self.model.parallel.pipeline_parallel_size *
                           self.model.parallel.model_parallel_size * self.model.parallel.data_parallel_size)

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
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
                    mem=100000,
                ),
            ),
        )

    def initial_setup(self) -> ExperimentConfig:
        model_path = self.model.path

        dataset = Dataset(
            "rw_pair",
            args=dict(
                max_length=self.dataset.max_seqlen,
                max_pairs_per_prompt=self.dataset.max_pairs_per_prompt,
                dataset_path=self.dataset.train_path,
            ),
        )

        eval_dataset = copy.deepcopy(dataset)
        eval_dataset.args["dataset_path"] = self.dataset.valid_path
        eval_dataloader = DataLoader("packed_eval", args=dict(batch_size=self.dataset.valid_bs_n_seqs))

        backend = ModelBackend(
            "deepspeed",
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
                zero_stage=(self.model.zero_stage if self.model.parallel.pipeline_parallel_size == 1 else min(
                    self.model.zero_stage, 1)),
                offload_optimizer_state=self.model.optimizer.offload,
                enable_bf16=self.model.enable_bf16,
                enable_fp16=self.model.enable_fp16,
            ),
        )

        model = get_real_model_config(
            model_path=model_path,
            hf_model_family=self.model.type._class,
            is_critic=True,
            init_critic_from_actor=True,
            dtype="bf16" if self.model.enable_bf16 else "fp16",
            lora=self.model.lora,
        )

        interface = ModelInterface("paired_rw")

        # NOTE: The dims of the parallelism grid is [pipline, data, model]
        # i.e., model parallelism is scheduled as close as possible in a single node.
        # This may seem incorrect according the class name "PipeModelDataParallelTopology",
        # but after you inspect the code, you will find that the class name actually
        # should be "PipeDataModelParallelTopology". Thank you DeepSpeed!
        topo = PipeModelDataParallelTopology(
            self.model.parallel.pipeline_parallel_size,
            self.model.parallel.model_parallel_size,
            self.model.parallel.data_parallel_size,
            self.model.parallel.use_sequence_parallel,
            gradient_checkpointing=self.model.gradient_checkpointing,
        )
        model_worker = []
        for i in range(self.world_size):
            coord = topo.get_coord(i)
            mw = ModelWorker(
                seed=self.seed,
                shards=[
                    StandaloneModelShard(
                        id=ModelShardID(
                            ModelName("default", 0),
                            dp_rank=coord.data,
                            pp_rank=coord.pipe,
                            mp_rank=coord.model,
                            topo=topo,
                        ),
                        model=model,
                        backend=backend,
                        eval_datasets=[eval_dataset],
                        eval_dataloader=eval_dataloader,
                    )
                ],
                tokenizer_name_or_path=model_path,
                datasets=[dataset],
                cuda_cache_cleanliness=True,
                cuda_cache_clear_freq=10,
            )
            model_worker.append(mw)

        rw_modeling = ModelRPC(
            model_name=ModelName("default", 0),
            interface_type=ModelInterfaceType.TRAIN_STEP,
            interface_impl=interface,
            model_type=self.model.type,
            input_data=["packed_input_ids", "group_factor", "pos_input_lens"],
            log_return_value=True,
            min_n_seqs=self.dataset.train_bs_n_seqs,
            max_n_seqs=self.dataset.train_bs_n_seqs,
        )

        exp_ctrl = ExperimentSaveEvalControl(
            total_train_epochs=self.total_train_epochs,
            save_frequency_steps=self.save_freq_steps,
            eval_frequency_epochs=self.eval_freq_epochs,
        )

        cfg = ExperimentConfig(
            exp_ctrl=exp_ctrl,
            model_rpcs=[rw_modeling],
            model_worker=model_worker,
        )
        return cfg


register_quickstart_exp("rw", RWConfig)
