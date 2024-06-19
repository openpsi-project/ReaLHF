from collections import defaultdict
from typing import *
import contextlib
import dataclasses
import functools

from omegaconf import MISSING
import numpy as np

from realhf.api.core.config import Dataset
from realhf.api.core.dfg import (
    MFCDef,
    ModelInterfaceType,
    OffloadHook,
    SyncParamHook,
)
from realhf.api.core.system_api import *
from realhf.api.quickstart.device_mesh import (
    AllocationConfig,
    DeviceMesh,
    make_device_mesh_from_name,
    RPCAllocation,
)
from realhf.api.quickstart.model import ModelTrainEvalConfig
from realhf.experiments.common.check import *
from realhf.experiments.common.utils import *
from realhf.search_engine.search import search_rpc_allocations
import realhf.base.logging as logging

logger = logging.getLogger("CommonExperimentConfig", "colored")


@dataclasses.dataclass
class CommonExperimentConfig(Experiment):
    """The common config for quickstart experiments.

    All members can be changed by the user in the command line,
    e.g.,

    .. code-block:: shell

        $ python3 -m realhf.apps.quickstart sft trial_name=my_trial seed=42 ...

    Recover mode is one of the followings\:

    - ``auto``\: automatically recover the last failed run.

    - ``save``\: save recover states if any error occurs.

    - ``resume``\: resume from saved recover states and save states if fail again.

    - ``disabled``\: do nothing but raise error when error occurs.

    Allocation mode is one of the followings\:

    - ``manual``\: manually allocate resources with experiment configs.

    - ``search``\: allocate resources and configure parallel strategies with search.

    - ``heuristic``\: allocate resources and configure parallel strategies with heuristic strategy.

    - ``pipe_data``\: allocate all models on all cluster nodes available and configure parallel strategies with pipe+data parallelism.

    - ``pipe_model``: allocate all models on all cluster nodes available and configure parallel strategies with pipe+model parallelism.

    :param experiment_name: Name of the experiment.
    :type experiment_name: str
    :param trial_name: Name of the trial.
    :type trial_name: str
    :param mode: Experiment launching mode.
        Currently only "local" and "slurm" are supported.
        The "local" mode implies ``n_nodes=1``.
    :type mode: str
    :param debug: Whether to run in the debug mode.
        The non-debug mode will disable all assertions.
    :type debug: bool
    :param partition: The slurm partition to run the experiment.
    :type partition: str
    :param wandb_mode: The mode of wandb. Currently the wandb logging is not supported.
    :type wandb_mode: str
    :param image_name: The name of the docker image used by the controller.
        Only used in the slurm mode.
    :type image_name: str or None
    :param recover_mode: The recover mode.
    :type recover_mode: str
    :param recover_retries: The number of retries for recovery.
        Only effective when recover_mode is "auto".
    :type recover_retries: int
    :param ignore_worker_error: Whether to ignore errors raised by
        workers during runtime. Please do not set it to be True unless
        the user is sure that some error is ignorable.
        Only effective when recover_mode is "disabled".
    :type ignore_worker_error: bool
    :param allocation_mode: Mode of GPU parallel strategy allocation.
    :type allocation_mode: str
    :param allocation_use_cache: Whether to use cache in allocation search.
        Only effective when allocation_mode=="search"
        and cache is available in the log dir of current experiment
        name and trial.
    :type allocation_use_cache: bool
    :param n_nodes: Number of nodes to run the experiment.
        Only effective when mode=="slurm".
    :type n_nodes: int
    :param n_gpus_per_node: Number of GPUs per node.
        Only effective when mode=="slurm".
    :type n_gpus_per_node: int
    :param nodelist: Slurm nodelist.
        Only effective when mode=="slurm".
    :type nodelist: str or None
    :param seed: Random seed.
    :type seed: int
    """

    experiment_name: str = MISSING
    trial_name: str = MISSING
    mode: str = "slurm"
    debug: bool = True
    partition: str = "dev"
    wandb_mode: str = "disabled"
    image_name: Optional[str] = None
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
        raise NotImplementedError(
            f"models is not implemented in {self.__class__}"
        )

    @property
    def rpcs(self) -> Dict[str, MFCDef]:
        raise NotImplementedError(
            f"rpcs is not implemented in {self.__class__}"
        )

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
        raise NotImplementedError(
            f"tokenizer_name_or_path is not implemented in {self.__class__}"
        )

    @property
    def exp_ctrl(self) -> ExperimentSaveEvalControl:
        raise NotImplementedError(
            f"expr_ctrl is not implemented in {self.__class__}"
        )

    @property
    def max_prompt_len(self) -> int:
        return None

    @property
    def search_kwargs(self) -> Dict[str, Any]:
        return {}

    @property
    def allocations(self) -> Dict[str, AllocationConfig]:
        return {}

    @property
    def global_device_mesh(self) -> DeviceMesh:
        return DeviceMesh(
            n_nodes=self.n_nodes,
            n_gpus_per_node=self.n_gpus_per_node,
            mapping=np.ones(
                (self.n_nodes, self.n_gpus_per_node), dtype=np.int32
            ),
            global_mesh_name=self.nodelist,
            name=self.nodelist,
        )

    def _heuristic_rpc_allocation(self) -> List[RPCAllocation]:
        raise NotImplementedError(
            f"_heuristic_rpc_allocation is not implemented in {self.__class__}"
        )

    def _search(self):
        # called in both api.main and controller
        gradient_checkpointing = any(
            model.gradient_checkpointing for model in self.models.values()
        )
        rpc_allocs: List[RPCAllocation] = search_rpc_allocations(
            device_mesh=self.global_device_mesh,
            rpcs=list(self.rpcs.values()),
            gradient_checkpointing=gradient_checkpointing,
            use_cache=self.allocation_use_cache,
            **self.search_kwargs,
        )
        return rpc_allocs

    def scheduling_setup(self) -> ExperimentScheduling:
        return ExperimentScheduling(
            master_worker=TasksGroup(
                count=1,
                scheduling=Scheduling.master_worker_default(
                    cpu=4, mem=20000, nodelist=self.nodelist
                ),
            ),
            model_worker=TasksGroup(
                count=self.n_nodes * self.n_gpus_per_node,
                scheduling=Scheduling.model_worker_default(
                    cpu=4,
                    gpu=1,
                    gpu_type="tesla",
                    mem=100000,
                    nodelist=self.nodelist,
                ),
            ),
        )

    def initial_setup(self) -> ExperimentConfig:
        if self.allocation_mode == "manual" and self.nodelist is None:
            logger.warning(
                "Warning: Nodelist is not set in manual allocation mode, "
                "in this case you cannot specify device mesh for each model RPC. "
                "All model RPC will be allocated on GPUs automatically "
                f"allocated according to n_nodes {self.n_nodes} "
                f"and n_gpus_per_node {self.n_gpus_per_node}."
            )
        self.__check_legal_experiment()

        rpcs = self.rpcs
        model_worker = []

        if self.allocation_mode == "search":
            # assert self.mode == "slurm"
            # assumes gradient checkpointing for all training RPCs if one is enabled
            # for the simplicity of search configurations
            rpc_allocs = self._search()
            for rpc_alloc in rpc_allocs:
                assert isinstance(rpc_alloc.rpc, str)
                for rpc in rpcs.values():
                    if rpc.name == rpc_alloc.rpc:
                        rpc_alloc.rpc = rpc
                        break
                else:
                    raise ValueError(f"RPC {rpc_alloc.rpc} not found in rpcs.")
        elif (
            self.allocation_mode == "pipe_data"
            or self.allocation_mode == "pipe_model"
        ):
            rpc_allocs: List[RPCAllocation] = [
                RPCAllocation(
                    rpc=rpc,
                    device_mesh=self.global_device_mesh,
                    parallel=ParallelismConfig(
                        data_parallel_size=(
                            self.n_gpus_per_node
                            if self.allocation_mode == "pipe_data"
                            else 1
                        ),
                        pipeline_parallel_size=self.n_nodes,
                        model_parallel_size=(
                            self.n_gpus_per_node
                            if self.allocation_mode == "pipe_model"
                            else 1
                        ),
                        use_sequence_parallel=(
                            rpc.interface_type == ModelInterfaceType.TRAIN_STEP
                            and self.allocation_mode == "pipe_model"
                        ),
                    ),
                )
                for rpc in self.rpcs.values()
            ]
        elif self.allocation_mode == "manual":
            rpc_allocs: List[RPCAllocation] = [
                RPCAllocation(
                    rpc=rpc,
                    device_mesh=(
                        make_device_mesh_from_name(
                            self.nodelist,
                            self.allocations[rpc_type].device_mesh,
                        )
                        if self.allocations[rpc_type].device_mesh is not None
                        else self.global_device_mesh
                    ),
                    parallel=self.allocations[rpc_type].parallel,
                )
                for rpc_type, rpc in self.rpcs.items()
            ]
        elif self.allocation_mode == "heuristic":
            rpc_allocs: List[RPCAllocation] = self._heuristic_rpc_allocation()
        else:
            raise NotImplementedError()

        shard_counter = defaultdict(lambda: 0)
        resolve_rpc_hooks(
            rpc_allocs
        )  # inplace modify MFCDefs in rpc allocations
        import pprint

        pprint.pprint(rpc_allocs)

        model_name_to_rpc_allocs: Dict[ModelName, List[RPCAllocation]] = (
            defaultdict(list)
        )
        for rpc_alloc in rpc_allocs:
            model_name_to_rpc_allocs[rpc_alloc.rpc.model_name].append(rpc_alloc)

        for i, j in itertools.product(
            range(self.n_nodes), range(self.n_gpus_per_node)
        ):
            mw = ModelWorker(
                seed=self.seed,
                shards=[],
                datasets=self.datasets,
                cuda_cache_cleanliness=False,
                cuda_cache_clear_freq=10,
                tokenizer_name_or_path=self.tokenizer_name_or_path,
            )
            # print(f"Setting up ModelWorker ({i},{j})")

            for (
                model_name,
                model_rpc_allocs,
            ) in model_name_to_rpc_allocs.items():
                rpcs = [rpc_alloc.rpc for rpc_alloc in model_rpc_allocs]
                rpc_alloc = model_rpc_allocs[0]
                model_cfg = self.models[model_name.role]
                model = make_model_config(model_cfg)
                mapping = rpc_alloc.device_mesh.mapping
                gradient_checkpointing = (
                    model_cfg.gradient_checkpointing
                    and any(
                        rpc.interface_type == ModelInterfaceType.TRAIN_STEP
                        for rpc in rpcs
                    )
                )

                topo = get_topo(
                    rpc_alloc.parallel,
                    gradient_checkpointing=gradient_checkpointing,
                    max_prompt_len=(
                        self.max_prompt_len
                        if any(
                            rpc.interface_type == ModelInterfaceType.GENERATE
                            for rpc in rpcs
                        )
                        else None
                    ),
                )

                if any(
                    rpc.interface_type == ModelInterfaceType.TRAIN_STEP
                    for rpc in rpcs
                ):
                    backend = make_train_backend_config(
                        model_cfg, rpc_alloc.parallel
                    )
                else:
                    backend = make_inf_backend_config(
                        model_cfg, rpc_alloc.parallel
                    )

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
                        )
                    )
                    shard_counter[model_name] += 1
            model_worker.append(mw)

        return ExperimentConfig(
            exp_ctrl=self.exp_ctrl,
            model_rpcs=[rpc_alloc.rpc for rpc_alloc in rpc_allocs],
            model_worker=model_worker,
        )

    def __check_legal_experiment(self):
        if self.n_nodes > 1 and self.mode == "local":
            raise ValueError(
                "Cannot run multi-node experiment in local mode, "
                "please setup slurm for distributed runs."
            )

        if self.n_gpus_per_node != 8 and self.allocation_mode in [
            "search",
            "heuristic",
        ]:
            raise ValueError(
                f"Cannot run search or heuristic allocation with "
                f"n_gpus_per_node {self.n_gpus_per_node}, "
                "please set n_gpus_per_node to 8."
            )

        for rpc_name, rpc in self.rpcs.items():
            if not check_is_realhf_native_model_interface(
                rpc.interface_impl.type_
            ) and self.allocation_mode in ["search", "heuristic"]:
                raise ValueError(
                    f"RPC {rpc.name} interface is not a realhf native implementation. "
                    f"Search and heuristic allocation mode are not available."
                )
            if (
                self.allocation_mode == "manual"
                and rpc_name not in self.allocations
            ):
                if rpc_name not in self.allocations:
                    raise ValueError(
                        f"RPC {rpc_name} is not in allocations, please implement "
                        f"`allocations()` method in your config class to enable "
                        f"manual allocation."
                    )

            if rpc.model_name.role not in self.models.keys():
                raise ValueError(
                    f"RPC {rpc.name} model name {rpc.model_name.role} is not in models."
                )
