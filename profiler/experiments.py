import dataclasses

from api.config import *
from api.config import ExperimentScheduling
from base.topology import PipeModelDataParallelTopology
from experiments.common.config_model import get_flash_mqat_model_config


@dataclasses.dataclass
class ProfileExperiment(Experiment):
    seed: int = 1
    model_path: str = "/lustre/public/pretrained_model_weights/sharded/Llama-2-7b-hf_4pp_3s"
    model_type: str = "llama"
    num_dp: int = 1
    num_pp: int = 4
    num_mp: int = 1
    enable_bf16: bool = False
    use_sequence_parallel: bool = False
    use_gradient_checkpointing: bool = False

    def __post_init__(self):
        self.n_workers = self.num_dp * self.num_pp * self.num_mp

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(profile_worker=TasksGroup(count=self.n_workers,
                                                              scheduling=Scheduling.profile_worker_default(
                                                                  cpu=4,
                                                                  gpu=1,
                                                                  gpu_type="tesla",
                                                                  mem=100000,
                                                                  nodelist="QH-com49",
                                                              )))

    def initial_setup(self) -> ExperimentConfig:
        model = get_flash_mqat_model_config(
            from_type="self",
            model_path=self.model_path,
            hf_model_type=self.model_type,
            tokenizer_path=self.model_path,
            use_pipe=True,
            dtype="bf16" if self.enable_bf16 else "fp16",
            sequence_parallel=self.use_sequence_parallel,
            partition_method="parameters_balanced",
            lora=None,
        )

        backend = ModelBackend(
            type_="ds_train",
            args=dict(
                optimizer_name="adam",
                optimizer_config=dict(lr=1e-5, weight_decay=0.0, betas=(0.9, 0.95)),
                warmup_steps_proportion=0.0,
                min_lr_ratio=0.0,
                zero_stage=1,
                engine_type="profile",
                gradient_checkpointing=self.use_gradient_checkpointing,
                num_pipeline_stages=self.num_pp,
                enable_fp16=not self.enable_bf16,
                enable_bf16=self.enable_bf16,
                sequence_parallel=self.use_sequence_parallel,
                enable_async_p2p_communication=False,
            ),
        )

        interface = ModelInterface(type_="profile", args=dict())
        topo = PipeModelDataParallelTopology(
            num_pp=self.num_pp,
            num_mp=self.num_mp,
            num_dp=self.num_dp,
        )

        profile_workers = [
            ProfileWorker(
                seed=self.seed,
                model=model,
                backend=backend,
                interface=interface,
                model_name="actor",
                topo=topo,
                dp_rank=topo.get_coord(i).data,
                pp_rank=topo.get_coord(i).pipe,
                mp_rank=topo.get_coord(i).model,
                cuda_cache_cleanliness=True,
            ) for i in range(self.n_workers)
        ]

        return ExperimentConfig(total_train_epochs=1,
                                model_rpcs=[],
                                data_worker=[],
                                model_worker=[],
                                profile_worker=profile_workers)


register_experiment("profile", ProfileExperiment)
