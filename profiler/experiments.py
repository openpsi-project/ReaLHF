import dataclasses

from api.config import *
from api.config import ExperimentScheduling
from base.topology import PipeModelDataParallelTopology
from experiments.common.config_model import get_flash_mqat_model_config

NUM_GPUS_PER_NODE = 8


@dataclasses.dataclass
class ProfileExperiment(Experiment):
    seed: int = 1
    n_nodes: int = 4
    nodelist: str = "QH-com[40-43]"
    device_mesh_name: str = "QH-com[40-43]"

    def __post_init__(self):
        self.n_workers = self.n_nodes * NUM_GPUS_PER_NODE

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(profile_worker=TasksGroup(count=self.n_workers,
                                                              scheduling=Scheduling.profile_worker_default(
                                                                  cpu=4,
                                                                  gpu=1,
                                                                  gpu_type="tesla",
                                                                  mem=100000,
                                                                  nodelist=self.nodelist,
                                                              )))

    def initial_setup(self) -> ExperimentConfig:
        topo = PipeModelDataParallelTopology(
            num_pp=1,
            num_mp=1,
            num_dp=self.n_workers,
        )

        profile_workers = [
            ProfileWorker(
                seed=self.seed,
                model=None,
                backend=None,
                interface=None,
                model_name="profile",
                device_mesh_name=self.device_mesh_name,
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


register_experiment("profile_comm", ProfileExperiment)
