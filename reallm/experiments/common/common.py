from collections import defaultdict
from typing import *
import contextlib
import dataclasses
import functools

from omegaconf import MISSING

from reallm.api.core.config import Dataset
from reallm.api.core.dfg import ModelInterfaceType, ModelRPC
from reallm.api.core.system_api import *
from reallm.api.quickstart.device_mesh import make_device_mesh_from_name, RPCAllocation
from reallm.api.quickstart.model import ModelTrainEvalConfig
from reallm.experiments.common.utils import *
from reallm.search_engine.search import search_rpc_allocations


@dataclasses.dataclass
class CommonExperimentConfig(Experiment):
    """ Common config for quickstart experiments, which is parsed and utilized by apps/main.py
    
    Args:
        experiment_name (str): Name of the experiment
        trial_name (str): Name of the trial
        mode (str): Experiment launching mode: "slurm", "local", "ray", "local_ray"
        debug (bool): Whether to run in debug mode
        partition (str): Slurm partition to run the experiment, only effective when mode=="slurm"
        wandb_mode (str): Mode of wandb, "disabled", "online", "offline"
        image_name (Optional[str]): Name of the image used by controller and workers of ray cluster
        remote_reset (bool): Whether to reset name resolve repo remotely in computation nodes. 
        recover_mode (str): Recover mode, 
                            'auto': automatically recover the last failed run; 
                            'save': save recover states if any error occurs; 
                            'resume': resume from saved recover states and save states if fail again; 
                            'disabled': do nothing when error occurs. 
        recover_retries (int): Number of retries for recovery, only effective when recover_mode=="auto"
        ignore_worker_error (bool): Whether to ignore worker error, only effective when recover_mode=="disabled".
                                    When recover_mode!="disabled", ignore_worker_error is always False.
        allocation_mode (str): Mode of GPU resource/model parallel strategy allocation. 
                             'manual': manually allocate resources with experiment configs;
                             'search': allocate resources and configure parallel strategies with search.
                             'heuristic': allocate resources and configure parallel strategies 
                                          with heuristic strategy.
                             'pipe_data': allocate all models on all cluster nodes available 
                                          and configure parallel strategies with pipe+data parallelism.
                             'pipe_model': allocate all models on all cluster nodes available
                                           and configure parallel strategies with pipe+model parallelism. 
        n_nodes (int): Number of nodes to run the experiment, only effective when mode=="slurm"
        n_gpus_per_node (int): Number of GPUs per node, only effective when mode=="slurm"
        nodelist (Optional[str]): slurm nodelist, only effective when mode=="slurm"
    """
    experiment_name: str = MISSING
    trial_name: str = MISSING
    mode: str = "slurm"
    debug: bool = True
    partition: str = "dev"
    wandb_mode: str = "disabled"
    image_name: Optional[str] = None
    remote_reset: bool = False
    recover_mode: str = "disabled"
    recover_retries: int = 1
    ignore_worker_error: bool = False
    allocation_mode: str = "manual"
    n_nodes: int = 1
    n_gpus_per_node: int = 8
    nodelist: Optional[str] = None
    seed: int = 1

    @property
    def models(self) -> Dict[str, ModelTrainEvalConfig]:
        raise NotImplementedError(f"models is not implemented in {self.__class__}")

    @property
    def rpcs(self) -> Dict[str, ModelRPC]:
        raise NotImplementedError(f"rpcs is not implemented in {self.__class__}")

    @property
    def datasets(self) -> List[Dataset]:
        return []

    @property
    def dataloader(self) -> DataLoader:
        return None

    @property
    def eval_datasets(self) -> List[Dataset]:
        return []

    @property
    def eval_dataloader(self) -> DataLoader:
        return None

    @property
    def tokenizer_name_or_path(self) -> str:
        raise NotImplementedError(f"tokenizer_name_or_path is not implemented in {self.__class__}")

    @property
    def exp_ctrl(self) -> ExperimentSaveEvalControl:
        raise NotImplementedError(f"expr_ctrl is not implemented in {self.__class__}")

    @property
    def search_kwargs(self) -> Dict[str, Any]:
        return {}

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            master_worker=TasksGroup(
                count=1,
                scheduling=Scheduling.master_worker_default(cpu=4, mem=20000, nodelist=self.nodelist),
            ),
            model_worker=TasksGroup(
                count=self.n_nodes * self.n_gpus_per_node,
                scheduling=Scheduling.model_worker_default(cpu=4,
                                                           gpu=1,
                                                           gpu_type="tesla",
                                                           mem=100000,
                                                           nodelist=self.nodelist),
            ),
        )

    def initial_setup(self) -> ExperimentConfig:
        rpcs = self.rpcs
        model_worker = []

        if self.allocation_mode == "search":
            assert self.mode == "slurm"
            device_mesh = make_device_mesh_from_name(self.nodelist, self.nodelist)
            rpc_alloc_dict: Dict[str, RPCAllocation] = search_rpc_allocations(
                device_mesh=device_mesh,
                rpcs=rpcs,
                **self.search_kwargs,
            )
        else:
            raise NotImplementedError()

        shard_counter = defaultdict(lambda: 0)

        for i, j in zip(range(self.n_nodes), range(self.n_gpus_per_node)):
            mw = ModelWorker(
                seed=self.seed,
                shards=[],
                datasets=self.datasets,
                dataloader=self.dataloader,
                cuda_cache_cleanliness=False,
                cuda_cache_clear_freq=10,
                tokenizer_name_or_path=self.tokenizer_name_or_path,
            )
            for rpc_name, rpc in rpcs.items():
                rpc_alloc = rpc_alloc_dict[rpc_name]

                model_cfg = self.models[rpc.model_name.role]
                model_cfg.parallel = rpc_alloc.parallel

                model = make_model_config(model_cfg)
                mapping = rpc_alloc.device_mesh.mapping
                gradient_checkpointing = model_cfg.gradient_checkpointing\
                    and rpc.interface_type == ModelInterfaceType.TRAIN_STEP
                topo = get_topo(rpc_alloc.parallel, gradient_checkpointing=gradient_checkpointing)

                backend = make_train_backend_config(model_cfg)\
                    if rpc.interface_type == ModelInterfaceType.TRAIN_STEP\
                    else make_inf_backend_config(model_cfg)

                if mapping[i, j] and not any(rpc.model_name == s.id.model_name for s in mw.shards):
                    shard_idx = shard_counter[rpc.model_name]
                    mw.shards.append(
                        StandaloneModelShard(
                            id=ModelShardID(
                                model_name=rpc.model_name,
                                topo=topo,
                                dp_rank=topo.get_coord(shard_idx).data,
                                pp_rank=topo.get_coord(shard_idx).pipe,
                                mp_rank=topo.get_coord(shard_idx).model,
                            ),
                            model=model,
                            backend=backend,
                            eval_datasets=self.eval_datasets,
                            eval_dataloader=self.eval_dataloader,
                        ),)
                    shard_counter[rpc.model_name] += 1
            model_worker.append(mw)

        return ExperimentConfig(
            exp_ctrl=self.exp_ctrl,
            model_rpcs=rpcs,
            model_worker=model_worker,
        )
