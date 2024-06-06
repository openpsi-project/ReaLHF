from collections import defaultdict
from typing import *
import contextlib
import dataclasses
import functools

from omegaconf import MISSING
import numpy as np

from reallm.api.core.config import Dataset
from reallm.api.core.dfg import ModelInterfaceType, ModelRPC, OffloadHook, SyncParamHook
from reallm.api.core.system_api import *
from reallm.api.quickstart.device_mesh import (AllocationConfig, DeviceMesh, make_device_mesh_from_name,
                                               RPCAllocation)
from reallm.api.quickstart.model import ModelTrainEvalConfig
from reallm.experiments.common.utils import *
from reallm.search_engine.search import search_rpc_allocations
import reallm.base.logging as logging

logger = logging.getLogger("CommonExperimentConfig", "colored")


@dataclasses.dataclass
class CommonExperimentConfig(Experiment):
    """Common config for quickstart experiments, which is parsed and utilized by apps/main.py

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
        allocation_use_cache (bool): Whether to use cache in allocation search, only effective when
                                     allocation_mode=="search" and cache is available in the log dir of
                                     current experiment name and trial.
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
    allocation_mode: str = "pipe_model"
    allocation_use_cache: bool = False
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
    def eval_datasets(self) -> List[Dataset]:
        return None

    @property
    def eval_dataloader(self) -> DataLoader:
        return DataLoader("packed_eval", args=dict(batch_size=128))

    @property
    def tokenizer_name_or_path(self) -> str:
        raise NotImplementedError(f"tokenizer_name_or_path is not implemented in {self.__class__}")

    @property
    def exp_ctrl(self) -> ExperimentSaveEvalControl:
        raise NotImplementedError(f"expr_ctrl is not implemented in {self.__class__}")

    @property
    def max_prompt_len(self) -> int:
        return None

    @property
    def search_kwargs(self) -> Dict[str, Any]:
        return {}

    @property
    def allocations(self) -> Dict[str, AllocationConfig]:
        return {}

    def _heuristic_rpc_allocation(self) -> List[RPCAllocation]:
        raise NotImplementedError(f"_heuristic_rpc_allocation is not implemented in {self.__class__}")

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
        if self.allocation_mode == "manual" and self.nodelist is None:
            logger.warning("Warning: Nodelist is not set in manual allocation mode, "
                           "in this case you cannot specify device mesh for each model RPC. "
                           "All model RPC will be allocated on GPUs automatically "
                           f"allocated according to n_nodes {self.n_nodes} "
                           f"and n_gpus_per_node {self.n_gpus_per_node}.")

        rpcs = self.rpcs
        model_worker = []

        global_device_mesh = DeviceMesh(
            n_nodes=self.n_nodes,
            n_gpus_per_node=self.n_gpus_per_node,
            mapping=np.ones((self.n_nodes, self.n_gpus_per_node), dtype=np.int32),
            global_mesh_name=self.nodelist,
            name=self.nodelist,
        )
        if self.allocation_mode == "search":
            # assert self.mode == "slurm"
            # assumes gradient checkpointing for all training RPCs if one is enabled
            # for the simplicity of search configurations
            gradient_checkpointing = any(model.gradient_checkpointing for model in self.models.values())
            rpc_allocs: List[RPCAllocation] = search_rpc_allocations(
                device_mesh=global_device_mesh,
                rpcs=list(rpcs.values()),
                gradient_checkpointing=gradient_checkpointing,
                use_cache=self.allocation_use_cache,
                **self.search_kwargs,
            )
        elif (self.allocation_mode == "pipe_data" or self.allocation_mode == "pipe_model"):
            rpc_allocs: List[RPCAllocation] = [
                RPCAllocation(
                    rpc=rpc,
                    device_mesh=global_device_mesh,
                    parallel=ParallelismConfig(
                        data_parallel_size=(self.n_gpus_per_node
                                            if self.allocation_mode == "pipe_data" else 1),
                        pipeline_parallel_size=self.n_nodes,
                        model_parallel_size=(self.n_gpus_per_node
                                             if self.allocation_mode == "pipe_model" else 1),
                        use_sequence_parallel=(rpc.interface_type == ModelInterfaceType.TRAIN_STEP
                                               and self.allocation_mode == "pipe_model"),
                    ),
                ) for rpc in self.rpcs.values()
            ]
        elif self.allocation_mode == "manual":
            rpc_allocs: List[RPCAllocation] = [
                RPCAllocation(
                    rpc=rpc,
                    device_mesh=(make_device_mesh_from_name(
                        self.nodelist,
                        self.allocations[rpc_type].device_mesh,
                    ) if self.allocations[rpc_type].device_mesh is not None else global_device_mesh),
                    parallel=self.allocations[rpc_type].parallel,
                ) for rpc_type, rpc in self.rpcs.items()
            ]
        elif self.allocation_mode == "heuristic":
            rpc_allocs: List[RPCAllocation] = self._heuristic_rpc_allocation()
        else:
            raise NotImplementedError()

        shard_counter = defaultdict(lambda: 0)
        resolve_rpc_hooks(rpc_allocs)  # inplace modify ModelRPCs in rpc allocations
        import pprint

        pprint.pprint(rpc_allocs)

        model_name_to_rpc_allocs: Dict[ModelName, List[RPCAllocation]] = defaultdict(list)
        for rpc_alloc in rpc_allocs:
            model_name_to_rpc_allocs[rpc_alloc.rpc.model_name].append(rpc_alloc)

        for i, j in itertools.product(range(self.n_nodes), range(self.n_gpus_per_node)):
            mw = ModelWorker(
                seed=self.seed,
                shards=[],
                datasets=self.datasets,
                cuda_cache_cleanliness=False,
                cuda_cache_clear_freq=10,
                tokenizer_name_or_path=self.tokenizer_name_or_path,
            )
            # print(f"Setting up ModelWorker ({i},{j})")

            for model_name, model_rpc_allocs in model_name_to_rpc_allocs.items():
                rpcs = [rpc_alloc.rpc for rpc_alloc in model_rpc_allocs]
                rpc_alloc = model_rpc_allocs[0]
                model_cfg = self.models[model_name.role]
                model = make_model_config(model_cfg)
                mapping = rpc_alloc.device_mesh.mapping
                gradient_checkpointing = model_cfg.gradient_checkpointing and any(
                    rpc.interface_type == ModelInterfaceType.TRAIN_STEP for rpc in rpcs)

                topo = get_topo(
                    rpc_alloc.parallel,
                    gradient_checkpointing=gradient_checkpointing,
                    max_prompt_len=(self.max_prompt_len if any(
                        rpc.interface_type == ModelInterfaceType.GENERATE for rpc in rpcs) else None),
                )

                if any(rpc.interface_type == ModelInterfaceType.TRAIN_STEP for rpc in rpcs):
                    backend = make_train_backend_config(model_cfg, rpc_alloc.parallel)
                else:
                    backend = make_inf_backend_config(model_cfg, rpc_alloc.parallel)

                # print(f"model name {model_name}, device mesh name {rpc_alloc.device_mesh.name},"
                #       f"mapping {mapping}")
                if mapping[i, j]:
                    shard_idx = shard_counter[model_name]
                    # print(f"Setting up Shard {shard_idx}, "
                    #       f"(dp, pp, mp) = ({topo.get_coord(shard_idx).data}, "
                    #       f"{topo.get_coord(shard_idx).pipe}, {topo.get_coord(shard_idx).model})")
                    mw.shards.append(
                        StandaloneModelShard(
                            id=ModelShardID(
                                model_name=model_name,
                                topo=topo,
                                dp_rank=topo.get_coord(shard_idx).data,
                                pp_rank=topo.get_coord(shard_idx).pipe,
                                mp_rank=topo.get_coord(shard_idx).model,
                            ),
                            model=model,
                            backend=backend,
                            eval_datasets=self.eval_datasets,
                            eval_dataloader=self.eval_dataloader,
                        ))
                    shard_counter[model_name] += 1
            model_worker.append(mw)

        return ExperimentConfig(
            exp_ctrl=self.exp_ctrl,
            model_rpcs=[rpc_alloc.rpc for rpc_alloc in rpc_allocs],
            model_worker=model_worker,
        )
