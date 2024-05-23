import dataclasses
import functools

from omegaconf import MISSING

from reallm.api.core.dfg import ModelFamily, ModelInterface, ModelInterfaceType, ModelRPC
from reallm.api.core.model_api import MODEL_FAMILY_TO_PATH
from reallm.api.core.system_api import *
from reallm.api.quickstart.dataset import PairedComparisonDatasetConfig
from reallm.api.quickstart.model import get_real_model_config, ModelTrainEvalConfig, OptimizerConfig
from reallm.base.topology import PipeModelDataParallelTopology
import reallm.base.logging as logging

logger = logging.getLogger("DPO Experiment")


@dataclasses.dataclass
class DPOConfig(Experiment):
    experiment_name: str = MISSING
    trial_name: str = MISSING
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
        assert not self.is_sft_lora and self.sft_lora_path is None, "LoRA is not supported for now."
        self.n_actors = int(self.actor.parallel.pipeline_parallel_size *
                            self.actor.parallel.data_parallel_size * self.actor.parallel.model_parallel_size)
        self.n_refs = int(self.ref.parallel.pipeline_parallel_size * self.ref.parallel.data_parallel_size *
                          self.ref.parallel.model_parallel_size)
        if self.n_actors != self.n_refs:
            raise ValueError(
                "Currelty, we restrict that the number of actors and references should be the same.")

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
                count=self.n_actors,
                scheduling=Scheduling.model_worker_default(
                    cpu=4,
                    gpu=1,
                    gpu_type="tesla",
                    mem=100000,
                ),
            ),
        )

    def initial_setup(self) -> ExperimentConfig:
        actor_path = MODEL_FAMILY_TO_PATH[ModelFamily(**self.actor.type)]
        ref_path = MODEL_FAMILY_TO_PATH[ModelFamily(**self.ref.type)]

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
                zero_stage=(self.actor.zero_stage if self.actor.parallel.pipeline_parallel_size == 1 else min(
                    self.actor.zero_stage, 1)),
                gradient_checkpointing=self.actor.gradient_checkpointing,
                engine_type="pipe" if self.actor.parallel.pipeline_parallel_size > 1 else "deepspeed",
                offload_optimizer_state=self.actor.optimizer.offload,
                enable_bf16=self.actor.enable_bf16,
                enable_fp16=self.actor.enable_fp16,
                sequence_parallel=self.actor.parallel.use_sequence_parallel,
            ),
        )
        inf_backend = ModelBackend("pipe_inference" if self.ref.parallel.pipeline_parallel_size >
                                   1 else "null")

        # We should merge pipeline model weights for the reference model to load.
        ref_model = get_real_model_config(
            model_path=ref_path,
            hf_model_family=self.ref.type._class,
            is_critic=False,
            init_critic_from_actor=False,
            dtype="bf16" if self.ref.enable_bf16 else "fp16",
        )
        model = get_real_model_config(
            model_path=actor_path,
            hf_model_family=self.actor.type._class,
            is_critic=False,
            init_critic_from_actor=False,
            dtype="bf16" if self.actor.enable_bf16 else "fp16",
            lora=self.actor.lora,
        )

        interface = ModelInterface("dpo", args=dict(beta=self.beta, enable_save=True))
        ref_interface = ModelInterface("dpo", args=dict(beta=self.beta, enable_save=False))

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
        # By default, we place one reference model and one actor model on each GPU.
        for i in range(self.n_actors):
            actor_coord = actor_topo.get_coord(i)
            ref_coord = ref_topo.get_coord(i)
            mw = ModelWorker(
                seed=self.seed,
                shards=[
                    StandaloneModelShard(
                        id=ModelShardID(
                            ModelName("actor", 0),
                            dp_rank=actor_coord.data,
                            pp_rank=actor_coord.pipe,
                            mp_rank=actor_coord.model,
                            topo=actor_topo,
                        ),
                        model=model,
                        backend=train_backend,
                    ),
                    StandaloneModelShard(
                        id=ModelShardID(
                            ModelName("ref", 0),
                            dp_rank=ref_coord.data,
                            pp_rank=ref_coord.pipe,
                            mp_rank=ref_coord.model,
                            topo=ref_topo,
                        ),
                        model=ref_model,
                        backend=inf_backend,
                    ),
                ],
                tokenizer_name_or_path=actor_path,
                datasets=[dataset],
                dataloader=dataloader,
                cuda_cache_cleanliness=True,
                cuda_cache_clear_freq=1,
            )
            model_worker.append(mw)

        ref_inf = ModelRPC(
            model_name=ModelName("ref", 0),
            interface_type=ModelInterfaceType.INFERENCE,
            interface_impl=ref_interface,
            model_type=self.ref.type,
            input_data=["packed_input_ids", "input_lens", "pos_input_lens", "prompt_lens"],
            output_data=["seqlogp"],
            min_n_seqs=self.dataset.train_tokens_per_batch // self.dataset.max_seqlen,
            max_n_seqs=self.dataset.train_tokens_per_batch // self.dataset.max_seqlen,
        )
        dpo = ModelRPC(
            model_name=ModelName("actor", 0),
            interface_type=ModelInterfaceType.TRAIN_STEP,
            interface_impl=interface,
            model_type=self.actor.type,
            input_data=[
                "packed_input_ids",
                "input_lens",
                "pos_input_lens",
                "seqlogp",
                "prompt_lens",
            ],
            log_return_value=True,
            min_n_seqs=self.dataset.train_tokens_per_batch // self.dataset.max_seqlen,
            max_n_seqs=self.dataset.train_tokens_per_batch // self.dataset.max_seqlen,
        )

        exp_ctrl = ExperimentSaveEvalControl(
            total_train_epochs=self.total_train_epochs,
            save_frequency_steps=self.save_freq_steps,
        )
        cfg = ExperimentConfig(
            exp_ctrl=exp_ctrl,
            model_rpcs=[dpo, ref_inf],
            model_worker=model_worker,
        )
        return cfg
