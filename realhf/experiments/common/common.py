import contextlib
import dataclasses
import functools
import itertools
import os
import pprint
import re
from collections import defaultdict
from typing import *

import numpy as np
from omegaconf import MISSING

import realhf.base.logging as logging
from realhf.api.core.config import (
    DataLoaderAbstraction,
    DatasetAbstraction,
    ModelName,
    ModelShardID,
    StandaloneModelShardAbstraction,
)
from realhf.api.core.dfg import MFCDef, ModelInterfaceType, build_graph
from realhf.api.core.system_api import (
    Experiment,
    ExperimentConfig,
    ExperimentSaveEvalControl,
    ExperimentScheduling,
    ModelWorker,
    Scheduling,
    TasksGroup,
)
from realhf.api.quickstart.device_mesh import (
    DeviceMesh,
    MFCConfig,
    RPCAllocation,
    make_device_mesh_from_name,
)
from realhf.api.quickstart.model import (
    ModelTrainEvalConfig,
    ParallelismConfig,
    get_real_model_config,
)
from realhf.experiments.common.check import check_is_realhf_native_model_interface
from realhf.experiments.common.utils import (
    extract_symmetric_allocation,
    get_topo,
    make_inf_backend_config,
    make_train_backend_config,
    resolve_replica_ids,
    resolve_rpc_hooks,
)
from realhf.search_engine.search import search_rpc_allocations

logger = logging.getLogger("CommonExperimentConfig", "colored")


@dataclasses.dataclass
class CommonExperimentConfig(Experiment):
    """Configuration for quickstart experiments.

    All members can be modified via the command line. For example,

    .. code-block:: shell

        $ python3 -m realhf.apps.quickstart sft trial_name=my_trial seed=42 exp_ctrl.save_freq_steps=10 ...

    This command changes the ``trial_name``, ``seed``, and the ``save_freq_steps`` attribute
    of the ``exp_ctrl`` attribute in this class.

    ``recover_mode`` can be one of the following\:

    - ``auto``\: Automatically recover the last failed run.

    - ``save``\: Save recovery states if an error occurs.

    - ``resume``\: Resume from saved recovery states and save states if a failure occurs again.

    - ``disabled``\: Do nothing but raise an error if one occurs.

    If you are not familiar with ReaL's recovery mechanism, set this to ``disabled``.
    Normal checkpointing is usually sufficient in most cases.

    ``allocation_mode`` can be one of the following\:

    - ``manual``\: Manually allocate resources using the specified command-line options.

    - ``search``\: Allocate resources and configure parallel strategies using the search engine.

    - ``heuristic``\: Allocate resources and configure parallel strategies using heuristic strategies obtained from a search.

    - ``pipe_data``\: Identical parallelization (like DSChat) with pipe+data parallelism. For a world size under 8, only data parallelism will be used.

    - ``pipe_model``\: Identical parallelization (like DSChat) with pipe+model parallelism. For a world size under 8, only tensor-model parallelism will be used.

    - A regex pattern like ``d${DP}p${PP}m${TP}``\: Identical parallelization for all MFCs with ${DP}-way data parallelism, ${PP}-way pipeline parallelism, and ${TP}-way model parallelism.

    :param experiment_name: The name of the experiment.
        An arbitrary string without "_" and "/", e.g., ``ultra-chat-llama``.
        This parameter is required.
    :type experiment_name: str
    :param trial_name: The name of the trial.
        An arbitrary string without "-" and "/", e.g., ``lr1e-3wd0.05``.
        This parameter is required.
    :type trial_name: str
    :param mode: The experiment launching mode. Supported values are "local", "ray", or "slurm".
        "ray" mode requires launching the Ray cluster via CLI.
        "slurm" mode requires the Pyxis plugin with the Enroot container enabled.
        "local" mode implies ``n_nodes=1``.
    :type mode: str
    :param debug: Whether to run in debug mode.
        Setting this to `False` will disable all assertions, which will be faster but less safe.
    :type debug: bool
    :param partition: The SLURM partition for running the experiment.
    :type partition: str
    :param wandb_mode: The mode for WandB. Currently, WandB logging is not supported.
    :type wandb_mode: str
    :param image_name: The name of the Docker image used by the controller.
        This parameter is only used in SLURM mode.
    :type image_name: str or None
    :param recover_mode: The recovery mode. See above for details.
    :type recover_mode: str
    :param recover_retries: The number of retries for recovery.
        Effective only when ``recover_mode`` is set to "auto".
    :type recover_retries: int
    :param ignore_worker_error: Whether to ignore errors raised by
        workers during runtime. Only set this to `True` if you are certain that the error can be ignored.
        Effective only when ``recover_mode`` is set to "disabled".
    :type ignore_worker_error: bool
    :param allocation_mode: The mode for GPU parallel strategy allocation. See above for details.
    :type allocation_mode: str
    :param allocation_use_cache: Whether to use cache in allocation search.
        Effective only when ``allocation_mode`` is set to "search" and a cache is available in the log directory of the current experiment
        name and trial.
    :type allocation_use_cache: bool
    :param n_nodes: The number of nodes to run the experiment.
    :type n_nodes: int
    :param n_gpus_per_node: The number of GPUs per node.
        Thus, the total number of GPUs will be ``n_nodes * n_gpus_per_node``.
        ReaL supports a world size of 1, 2, 4, 8, ... within a single node,
        or multiple nodes with the same number of GPUs.
    :type n_gpus_per_node: int
    :param nodelist: Nodelist for the distributed setting in SLURM nodelist format.
        Required for the ``manual`` allocation mode.
        For multiple GPUs on a single node, it should be formatted as "NODE01:0,1,2,3",
        indicating the use of the first 4 GPUs on ``NODE01``.
        For multiple complete nodes, it should be formatted as "NODE[01-02,03,07],COM08",
        indicating the use of all GPUs on these nodes: [NODE01, NODE02, NODE03, NODE07, COM08].
    :type nodelist: str or None
    :param seed: The random seed.
    :type seed: int
    :param cache_clear_freq: The cache of data transfer will be cleared after each ``cache_clear_freq`` steps.
        If None, will not clear the cache. Set to a small number, e.g., 1, if OOM or CUDA OOM occurs.
    :type cache_clear_freq: int or None
    :param exp_ctrl: The control for saving and evaluating the experiment.
    :type exp_ctrl: ExperimentSaveEvalControl
    """

    experiment_name: str = MISSING
    trial_name: str = MISSING
    mode: str = dataclasses.field(
        metadata={"choices": ["slurm", "local", "ray"]}, default="slurm"
    )
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
    cache_clear_freq: Optional[int] = 10
    exp_ctrl: ExperimentSaveEvalControl = dataclasses.field(
        default_factory=ExperimentSaveEvalControl
    )

    def __post_init__(self):
        self.__run_model_sanity_check()

    @property
    def models(self) -> Dict[str, ModelTrainEvalConfig]:
        """A dict mapping from model roles to model configurations.

        Should be implemented in all subclasses.
        """
        raise NotImplementedError(f"models is not implemented in {self.__class__}")

    @property
    def rpcs(self) -> Dict[str, MFCDef]:
        """A dict mapping from model function call names to the MFCDef objects.

        Note that model function call names are different from model
        names. Should be implemented in all subclasses.

        NOTE: in implementation of ReaL, term RPC also refers to MFC.
        """
        raise NotImplementedError(f"rpcs is not implemented in {self.__class__}")

    @property
    def allocations(self) -> Dict[str, MFCConfig]:
        """The allocation configuration for each model function call.

        A dictionary mapping MFC names to its allocation configuration.
        Must be implemented in the subclass.
        """
        raise NotImplementedError(
            f"allocations is not implemented in {self.__class__}."
        )

    @property
    def datasets(self) -> List[DatasetAbstraction]:
        """A list of dataset configurations used for training.

        Should be implemented in all subclasses.
        """
        return NotImplementedError(f"datasets is not implemented in {self.__class__}")

    @property
    def eval_datasets(self) -> List[DatasetAbstraction]:
        """A list of dataset configurations used for evaluation.

        Can be None if runtime evaluation is not needed.
        """
        return None

    @property
    def eval_dataloader(self) -> DataLoaderAbstraction:
        """The dataloader configuration used for evaluation.

        Reserved to changed the evaluation batch size. Training does not
        require this property because the batch size is handled in MFC
        definitions.
        """
        return DataLoaderAbstraction("packed_eval", args=dict(batch_size=128))

    @property
    def tokenizer_name_or_path(self) -> str:
        """The tokenizer for tokenizing train/validation datasets.

        Required for all experiments.
        """
        raise NotImplementedError(
            f"tokenizer_name_or_path is not implemented in {self.__class__}."
        )

    @property
    def max_prompt_len(self) -> int:
        """The maximum prompt length for generation.

        Used by CUDAGraph-enabled generation. If no generation is used
        in the algorithm, this property can be None.
        """
        return None

    @property
    def search_kwargs(self) -> Dict[str, Any]:
        return {}

    @property
    def global_device_mesh(self) -> DeviceMesh:
        return DeviceMesh(
            n_nodes=self.n_nodes,
            n_gpus_per_node=self.n_gpus_per_node,
            mapping=np.ones((self.n_nodes, self.n_gpus_per_node), dtype=np.int32),
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
        """The resourced occupied by each worker.

        The resource requirements will be sent to SLURM or Ray, while
        being ignored in the local mode.
        """
        return ExperimentScheduling(
            master_worker=TasksGroup(
                count=1,
                scheduling=Scheduling.master_worker_default(
                    cpu=4,
                    mem=20000,
                    nodelist=self.nodelist,
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

    def _get_rpc_allocations(self) -> List[RPCAllocation]:
        if self.allocation_mode == "manual" and self.nodelist is None:
            logger.warning(
                "Warning: Nodelist is not set in manual allocation mode, "
                "in this case you cannot specify device mesh for each model RPC. "
                "All model RPC will be allocated on GPUs automatically "
                f"allocated according to n_nodes {self.n_nodes} "
                f"and n_gpus_per_node {self.n_gpus_per_node}."
            )

        self.__check_legal_allocation_options()

        rpcs = self.rpcs
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
            or extract_symmetric_allocation(self.allocation_mode)
        ):
            if self.allocation_mode == "pipe_data":
                dp, pp, mp = self.n_gpus_per_node, self.n_nodes, 1
            elif self.allocation_mode == "pipe_model":
                dp, pp, mp = 1, self.n_nodes, self.n_gpus_per_node
            else:
                para = extract_symmetric_allocation(self.allocation_mode)
                dp, pp, mp = para["d"], para["p"], para["m"]
                if dp * pp * mp != self.n_nodes * self.n_gpus_per_node:
                    raise ValueError(
                        "The multiplication of 3D parallel degrees "
                        "does not equal to the number of gpus. "
                        f"dp={dp}, pp={pp}, mp={mp}, "
                        f"n_nodes={self.n_nodes}, n_gpus_per_node={self.n_gpus_per_node}"
                    )
            rpc_allocs: List[RPCAllocation] = [
                RPCAllocation(
                    rpc=rpc,
                    device_mesh=self.global_device_mesh,
                    parallel=ParallelismConfig(
                        data_parallel_size=dp,
                        pipeline_parallel_size=pp,
                        model_parallel_size=mp,
                        use_sequence_parallel=(
                            rpc.interface_type == ModelInterfaceType.TRAIN_STEP
                            and mp > 1
                        ),
                    ),
                )
                for rpc in rpcs.values()
            ]
        elif self.allocation_mode == "manual":
            rpc_allocs: List[RPCAllocation] = [
                RPCAllocation(
                    rpc=rpc,
                    device_mesh=(
                        make_device_mesh_from_name(
                            self.global_device_mesh.name,
                            self.allocations[rpc_type].device_mesh,
                        )
                        if self.allocations[rpc_type].device_mesh is not None
                        else self.global_device_mesh
                    ),
                    parallel=self.allocations[rpc_type].parallel,
                )
                for rpc_type, rpc in rpcs.items()
            ]
        elif self.allocation_mode == "heuristic":
            rpc_allocs: List[RPCAllocation] = self._heuristic_rpc_allocation()
        else:
            raise NotImplementedError(
                f'Unknown allocation mode "{self.allocation_mode}".'
            )
        return rpc_allocs

    def _get_model_worker_configs(
        self, rpc_allocs: List[RPCAllocation]
    ) -> List[ModelWorker]:
        model_worker = []
        shard_counter = defaultdict(lambda: 0)

        model_name_to_rpc_allocs: Dict[ModelName, List[RPCAllocation]] = defaultdict(
            list
        )
        for rpc_alloc in rpc_allocs:
            model_name_to_rpc_allocs[rpc_alloc.rpc.model_name].append(rpc_alloc)

        for i, j in itertools.product(range(self.n_nodes), range(self.n_gpus_per_node)):
            mw = ModelWorker(
                seed=self.seed,
                shards=[],
                datasets=self.datasets,
                cuda_cache_cleanliness=self.cache_clear_freq is not None,
                cuda_cache_clear_freq=self.cache_clear_freq,
                tokenizer_name_or_path=self.tokenizer_name_or_path,
            )

            for (
                model_name,
                model_rpc_allocs,
            ) in model_name_to_rpc_allocs.items():
                rpcs = [rpc_alloc.rpc for rpc_alloc in model_rpc_allocs]
                rpc_alloc = model_rpc_allocs[0]
                model_cfg = self.models[model_name.role]
                model = get_real_model_config(
                    model_path=model_cfg.path,
                    hf_model_family=model_cfg.type._class,
                    is_critic=model_cfg.type.is_critic,
                    init_critic_from_actor=model_cfg.init_critic_from_actor,
                    dtype="bf16" if model_cfg.enable_bf16 else "fp16",
                    lora=model_cfg.lora,
                )
                mapping = rpc_alloc.device_mesh.mapping
                gradient_checkpointing = model_cfg.gradient_checkpointing and any(
                    rpc.interface_type == ModelInterfaceType.TRAIN_STEP for rpc in rpcs
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
                    gradient_accumulation_fusion=model_cfg.backend == "megatron",
                )

                if any(
                    rpc.interface_type == ModelInterfaceType.TRAIN_STEP for rpc in rpcs
                ):
                    backend = make_train_backend_config(model_cfg, rpc_alloc.parallel)
                else:
                    backend = make_inf_backend_config(model_cfg, rpc_alloc.parallel)

                if mapping[i, j]:
                    shard_idx = shard_counter[model_name]
                    mw.shards.append(
                        StandaloneModelShardAbstraction(
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
        return model_worker

    def initial_setup(self) -> ExperimentConfig:

        rpc_allocs = self._get_rpc_allocations()

        resolve_replica_ids(rpc_allocs)
        resolve_rpc_hooks(
            rpc_allocs, self.models
        )  # inplace modify MFCDefs in rpc allocations

        pprint.pprint(rpc_allocs)

        model_worker = self._get_model_worker_configs(rpc_allocs)

        return ExperimentConfig(
            exp_ctrl=self.exp_ctrl,
            model_rpcs=[rpc_alloc.rpc for rpc_alloc in rpc_allocs],
            model_worker=model_worker,
        )

    def __check_legal_allocation_options(self):
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
            if rpc_name != rpc.name:
                raise KeyError(
                    f"RPC name {rpc_name} does not match the name in the MFCDef object {rpc.name}."
                )
            if not check_is_realhf_native_model_interface(
                rpc.interface_impl.type_
            ) and self.allocation_mode in ["search"]:
                raise ValueError(
                    f"RPC {rpc.name} interface is not a realhf native implementation. "
                    f"The search allocation mode are not available."
                )
            if self.allocation_mode == "manual" and rpc_name not in self.allocations:
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

    def __run_model_sanity_check(self):
        for role, model in self.models.items():
            if model.enable_bf16 and model.enable_fp16:
                raise ValueError(
                    f"For model `{role}`, enable_bf16 and"
                    " enable_fp16 cannot be both True."
                )
            if (
                model.offload or model.optimizer.offload
            ) and model.backend != "deepspeed":
                raise ValueError(
                    f"For model `{role}`, offload is only"
                    " valid for the deepspeed backend."
                )
            if model.backend == "megatron" and model.zero_stage in [3]:
                raise ValueError(
                    f"For model `{role}`, the Megatron backend"
                    " only supports zero stage 0, 1 or 2."
                )
            if not os.path.exists(model.path):
                raise FileNotFoundError(
                    f"The model path `{model.path}` for `{role}` does not exist locally. "
                    "You must download the HuggingFace checkpoint before loading it."
                )
            if model.optimizer.min_lr_ratio < 0.0 or model.optimizer.min_lr_ratio > 1.0:
                raise ValueError(
                    f"Invalid min_lr_ratio: {model.optimizer.min_lr_ratio}"
                )
            if (
                model.optimizer.warmup_steps_proportion < 0.0
                or model.optimizer.warmup_steps_proportion > 1.0
            ):
                raise ValueError(
                    f"Invalid warmup_steps_proportion: {model.optimizer.warmup_steps_proportion}"
                )
