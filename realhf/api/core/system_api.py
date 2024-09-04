import collections
import copy
import dataclasses
import enum
import getpass
import itertools
import math
import os
import sys
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import realhf.api.core.dfg as dfg
import realhf.base.topology as topology
from realhf.api.core.config import (
    DataLoaderAbstraction,
    DatasetAbstraction,
    ModelAbstraction,
    ModelName,
    ModelShardID,
    StandaloneModelShardAbstraction,
)
from realhf.base.cluster import spec as cluster_spec
from realhf.base.constants import DATASET_CACHE_PATH, LOG_ROOT
from realhf.base.constants import experiment_name as get_experiment_name
from realhf.base.constants import trial_name as get_trial_name

_LLM_GPU_IMAGE = cluster_spec.gpu_image
_LLM_CPU_IMAGE = cluster_spec.cpu_image


@dataclasses.dataclass
class Scheduling:
    cpu: int
    gpu: int
    mem: int
    gpu_type: str = "tesla"
    node_type: str = None
    nodelist: str = None
    exclude: str = None
    container_image: str = _LLM_CPU_IMAGE
    env_vars: Dict[str, str] = dataclasses.field(default_factory=dict)
    # time utils from "https://slurm.schedmd.com/sbatch.html"
    time_limit: Optional[str] = None  # see  "--time" option for format
    begin: Optional[str] = None  # see "--begin" option for format
    deadline: Optional[str] = None  # see "--deadline" option for format

    @staticmethod
    def master_worker_default(**kwargs):
        return Scheduling(
            **{
                "cpu": 16,
                "mem": 20 * 1024,
                "gpu": 0,
                "container_image": _LLM_CPU_IMAGE,
                **kwargs,
            }
        )

    @staticmethod
    def model_worker_default(**kwargs):
        return Scheduling(
            **{
                "cpu": 2,
                "gpu": 1,
                "mem": 60 * 1024,
                "container_image": _LLM_GPU_IMAGE,
                **kwargs,
            }
        )


@dataclasses.dataclass
class WorkerInformation:
    """The basic information of an worker.

    To improve config readability, the experiment starter will fill the
    fields, instead of letting the users do so in experiment configs.
    """

    experiment_name: str = ""
    trial_name: str = ""  # Name of the trial of the experiment; e.g. "{USER}-0".
    worker_type: str = ""  # E.g. "policy", "actor", or "trainer".
    worker_index: int = (
        -1
    )  # The index of the worker of the specific type, starting from 0.
    worker_count: int = (
        0  # Total number of workers; hence, 0 <= worker_index < worker_count.
    )
    worker_tag: Optional[str] = (
        None  # For actor and policy worker, can be "training" or "evaluation".
    )
    host_key: Optional[str] = None  # Worker will update and keep this key alive.
    watch_keys: Union[str, List[str]] = (
        None  # Worker will exit if all of the watching keys are gone.
    )
    wandb_entity: Optional[str] = (
        None  # wandb_{config} are optional. They overwrite system wandb_configuration.
    )
    wandb_project: Optional[str] = None
    wandb_job_type: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_config: Optional[Dict] = None
    log_wandb: Optional[bool] = None

    def system_setup(
        self,
        experiment_name,
        trial_name,
        worker_type,
        worker_index,
        worker_count,
    ):
        """Setup system related worker information, while leaving the rest
        untouched."""
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.worker_type = worker_type
        self.worker_index = worker_index
        self.worker_count = worker_count


@dataclasses.dataclass
class ModelWorker:
    seed: int
    shards: List[StandaloneModelShardAbstraction]
    # dataset, for source model workers
    tokenizer_name_or_path: Optional[str] = None
    datasets: Optional[List[Union[str, DatasetAbstraction]]] = None
    dataloader: Union[str, DataLoaderAbstraction] = "packed"
    use_dataset_cache: bool = False
    dataset_cahce_root: str = DATASET_CACHE_PATH
    # cuda & cudnn config
    cudnn_benchmark: bool = False
    cudnn_deterministic: bool = False
    cuda_cache_cleanliness: bool = True
    cuda_cache_clear_freq: int = 10
    # model_topos and worker_info will be configured automatically
    model_rpcs: List[dfg.MFCDef] = None
    model_topos: Dict[ModelName, topology.PipeModelDataParallelTopology] = None
    msid2mwid: Dict[ModelShardID, int] = None
    data_transfer_pairs: List[Tuple[str, str]] = None
    sync_param_pairs: List[Tuple[str, str]] = None
    # profiling
    profile_mode: bool = False
    worker_info: Optional[WorkerInformation] = None

    def __post_init__(self):
        model_names = [s.id.model_name for s in self.shards]
        if len(set(model_names)) != len(model_names):
            raise ValueError(
                f"ModelWorker cannot have multiple shards of the same model name: {model_names}."
            )


@dataclasses.dataclass
class ExperimentSaveEvalControl:
    """Utility object for controlling the frequency of saving and evaluation
    during training.

    ``Epoch`` refers to the number of times the training loop iterates over the entire dataset.
    ``Step`` refers to the number of iterations running the algorithm dataflow.

    This object manages independent counters for epochs, steps, and seconds. The model will
    be saved or evaluated when any of the following conditions are met.

    :param total_train_epochs: The total number of epochs to train the model.
    :type total_train_epochs: int
    :param save_freq_epochs: Frequency in epochs at which to save the model. If None,
        the model will not be saved based on epoch changes during training.
    :type save_freq_epochs: Optional[int]
    :param save_freq_steps: Frequency in steps at which to save the model. If None,
        the model will not be saved based on step changes during training.
    :type save_freq_steps: Optional[int]
    :param save_freq_secs: Frequency in seconds at which to save the model. If None,
        the model will not be saved based on time changes during training.
    :type save_freq_secs: Optional[int]
    :param eval_freq_epochs: Frequency in epochs at which to evaluate the model. If None,
        the model will not be evaluated based on epoch changes during training.
    :type eval_freq_epochs: Optional[int]
    :param eval_freq_steps: Frequency in steps at which to evaluate the model. If None,
        the model will not be evaluated based on step changes during training.
    :type eval_freq_steps: Optional[int]
    :param eval_freq_secs: Frequency in seconds at which to evaluate the model. If None,
        the model will not be evaluated based on time changes during training.
    :type eval_freq_secs: Optional[int]
    :param benchmark_steps: Terminate training after this number of steps. Used for system
        benchmarking only. Set to None for normal training.
    :type benchmark_steps: Optional[int]
    """

    total_train_epochs: int = 1
    # save control
    save_freq_epochs: Optional[int] = None
    save_freq_steps: Optional[int] = None
    save_freq_secs: Optional[int] = None
    # eval control
    eval_freq_epochs: Optional[int] = None
    eval_freq_steps: Optional[int] = None
    eval_freq_secs: Optional[int] = None
    # benchmark
    benchmark_steps: Optional[int] = None


@dataclasses.dataclass
class MasterWorker:
    exp_ctrl: ExperimentSaveEvalControl
    # main components
    model_rpcs: List[dfg.MFCDef]
    n_model_workers: int
    model_topos: Dict[ModelName, topology.PipeModelDataParallelTopology]
    msid2mwid: Dict[ModelShardID, int] = None
    data_transfer_pairs: List[Tuple[str, str]] = None
    sync_param_pairs: List[Tuple[str, str]] = None
    worker_info: Optional[WorkerInformation] = None


@dataclasses.dataclass
class TasksGroup:
    count: int
    scheduling: Scheduling


@dataclasses.dataclass
class ExperimentScheduling:
    model_worker: Union[List[TasksGroup], TasksGroup] = dataclasses.field(
        default_factory=list
    )
    master_worker: Union[List[TasksGroup], TasksGroup] = dataclasses.field(
        default_factory=list
    )
    controller_image: str = _LLM_CPU_IMAGE


@dataclasses.dataclass
class ExperimentConfig:
    exp_ctrl: ExperimentSaveEvalControl
    # dataflow
    model_rpcs: List[dfg.MFCDef]
    model_worker: List[ModelWorker] = dataclasses.field(default_factory=list)
    # master_worker will be set automatically
    master_worker: Optional[List[MasterWorker]] = None

    def __post_init__(self):
        assert self.master_worker is None
        model_names = set()
        for w in self.model_worker:
            model_names = model_names.union([s.id.model_name for s in w.shards])
        model_names = sorted(list(model_names))

        assert get_trial_name() is not None
        assert get_experiment_name() is not None
        graph_path = os.path.join(
            LOG_ROOT, get_experiment_name(), get_trial_name(), "dataflow_graph.png"
        )
        G = dfg.build_graph(self.model_rpcs, verbose=True, graph_path=graph_path)
        for rpc in self.model_rpcs:
            rpc._G = G

        self._validate_model_names(model_names)

        model_topos = self._collect_topos(model_names)
        model_configs = self._collect_model_configs(model_names)

        data_transfer_pairs = self._resolve_data_transfer_pairs(model_names)

        sync_param_pairs = self._resolve_param_realloc_pairs(model_configs, model_topos)

        model_names_to_instantiate = self._resolve_model_names_to_instantiate(
            model_names
        )

        for mw in self.model_worker:
            for s in mw.shards:
                s.should_instantiate = s.id.model_name in model_names_to_instantiate

        msid2mwid = {}
        for i, mw in enumerate(self.model_worker):
            mw.model_topos = model_topos
            for m in mw.shards:
                msid2mwid[m.id] = i
        for m in self.model_worker:
            m.msid2mwid = msid2mwid
            m.data_transfer_pairs = data_transfer_pairs
            m.sync_param_pairs = sync_param_pairs

        for m in self.model_worker:
            m.model_rpcs = self.model_rpcs

        self.master_worker = [
            MasterWorker(
                exp_ctrl=self.exp_ctrl,
                model_topos=model_topos,
                model_rpcs=self.model_rpcs,
                n_model_workers=len(self.model_worker),
                msid2mwid=msid2mwid,
                sync_param_pairs=sync_param_pairs,
                data_transfer_pairs=data_transfer_pairs,
            )
        ]

    def set_worker_information(self, experiment_name, trial_name):
        if len(self.model_worker) > 0:
            assert len(self.master_worker) == 1

        for worker_type, workers in [
            ("model_worker", self.model_worker),
            ("master_worker", self.master_worker),
        ]:
            if len(workers) == 0:
                continue
            for i, worker in enumerate(workers):
                system_worker_info = dict(
                    experiment_name=experiment_name,
                    trial_name=trial_name,
                    worker_type=worker_type,
                    worker_index=i,
                    worker_count=len(workers),
                )
                if worker.worker_info is not None:
                    worker.worker_info.system_setup(**system_worker_info)
                else:
                    worker.worker_info = WorkerInformation(**system_worker_info)

    def _collect_topos(
        self, model_names: List[ModelName]
    ) -> Dict[ModelName, topology.PipeModelDataParallelTopology]:
        model_topos = {}
        model_allocations = {}
        for model_name in model_names:
            _this_mws_with_indicies = list(
                filter(
                    lambda i_mw: any(
                        x.id.model_name == model_name for x in i_mw[1].shards
                    ),
                    enumerate(self.model_worker),
                )
            )
            _this_mw_indices, _this_mws = zip(*_this_mws_with_indicies)
            _this_mw_indices = tuple(sorted(_this_mw_indices))
            all_shards: List[StandaloneModelShardAbstraction] = [
                next(filter(lambda x: x.id.model_name == model_name, mw.shards))
                for mw in _this_mws
            ]
            for k, v in model_topos.items():
                if (
                    k.role == model_name.role
                    and v == all_shards[0].id.topo
                    and _this_mw_indices == model_allocations[k]
                ):
                    raise ValueError(
                        f"If different RPCs have the same topology and allocation, "
                        f"they don't need to use multiple model names ({k}, {model_name})."
                    )
            model_topos[model_name] = all_shards[0].id.topo
            model_allocations[model_name] = tuple(sorted(_this_mw_indices))

            ##### Sanity check of parallelism ranks. #####
            ranks = [s.id.parallelism_rank for s in all_shards]
            _topos = [s.id.topo for s in all_shards]
            if set(ranks) != set(list(range(len(_this_mws)))) or any(
                _t.world_size() != _topos[0].world_size() for _t in _topos
            ):
                raise ValueError(
                    f"Parallelism rank check failed: model name {model_name}, "
                    f"model shard ids={[s.id for s in all_shards]}."
                )
            ##### Sanity check of parallelism ranks. #####
        return model_topos

    def _collect_model_configs(
        self, model_names: List[ModelName]
    ) -> Dict[ModelName, ModelAbstraction]:
        model_configs = {}
        for model_name in model_names:
            _this_mws = list(
                filter(
                    lambda mw: any(x.id.model_name == model_name for x in mw.shards),
                    self.model_worker,
                )
            )
            all_shards: List[StandaloneModelShardAbstraction] = [
                next(filter(lambda x: x.id.model_name == model_name, mw.shards))
                for mw in _this_mws
            ]
            model_configs[model_name] = all_shards[0].model
        return model_configs

    def _validate_model_names(self, model_names: List[ModelName]):
        model_names = sorted(model_names)
        _roles = set(mn.role for mn in model_names)
        _replica_ids = {
            _role: sorted([mn.replica_id for mn in model_names if mn.role == _role])
            for _role in _roles
        }
        for v in _replica_ids.values():
            if list(sorted(v)) != list(range(len(v))):
                raise ValueError(
                    f"Model replica ids should be 0, 1, 2, ... for each role: {_replica_ids}."
                )

    def _resolve_data_transfer_pairs(
        self, model_names: List[ModelName]
    ) -> List[Tuple[ModelName, ModelName]]:
        data_transfer_pairs: List[Tuple[ModelName, ModelName]] = []
        G = self.model_rpcs[0]._G
        for edge in G.edges():
            mn1 = G.nodes[edge[0]]["object"].model_name
            mn2 = G.nodes[edge[1]]["object"].model_name
            data_transfer_pairs.append((mn1, mn2))
        src_rpcs = [rpc for rpc in self.model_rpcs if rpc.is_src]
        data_src_rpc = src_rpcs[0]
        for r in src_rpcs[1:]:
            if (
                data_src_rpc.model_name,
                r.model_name,
            ) not in data_transfer_pairs:
                data_transfer_pairs.append((data_src_rpc.model_name, r.model_name))
        data_transfer_pairs += [(mn, mn) for mn in model_names]
        return data_transfer_pairs

    def _resolve_param_realloc_pairs(
        self, model_configs, model_topos
    ) -> List[Tuple[ModelName, ModelName]]:
        sync_param_pairs: List[Tuple[ModelName, ModelName]] = []
        for rpc in self.model_rpcs:
            for hook in rpc._pre_hooks + rpc._post_hooks:
                if not isinstance(hook, dfg.ParamReallocHook):
                    continue
                if (
                    hook.target is not None
                    and not (
                        model_configs[rpc.model_name].type_
                        == model_configs[hook.target].type_
                        == "real_model"
                    )
                ) or (
                    hook.source is not None
                    and not (
                        model_configs[rpc.model_name].type_
                        == model_configs[hook.source].type_
                        == "real_model"
                    )
                ):
                    raise ValueError(
                        "To synchronize parameters between two models, both models must be ReaLModel."
                    )
                other_model_name = (
                    hook.target if hook.target is not None else hook.source
                )
                other_topo = (
                    model_topos[hook.target]
                    if hook.target is not None
                    else model_topos[hook.source]
                )
                self_topo = model_topos[rpc.model_name]
                if (
                    self_topo.get_dim("model") % other_topo.get_dim("model") != 0
                    and other_topo.get_dim("model") % self_topo.get_dim("model") != 0
                ):
                    raise ValueError(
                        "To synchronize parameters between two models, "
                        "their model parallel size must be a multiple of each other."
                    )
                if rpc.model_name == other_model_name:
                    raise ValueError(
                        f"Cannot synchronize parameters within the same model "
                        f"(in {rpc}, {rpc.model_name} and {hook.target})."
                    )
                if hook.target is not None:
                    if not (rpc.model_name, hook.target) in sync_param_pairs:
                        sync_param_pairs.append((rpc.model_name, hook.target))
                else:
                    if not (hook.source, rpc.model_name) in sync_param_pairs:
                        sync_param_pairs.append((hook.source, rpc.model_name))
        return sync_param_pairs

    def _resolve_model_names_to_instantiate(
        self, model_names: List[ModelName]
    ) -> List[ModelName]:
        # Mark which shard of the same role should be instantiated.
        roles = set([model_name.role for model_name in model_names])
        role_is_trainable = {role: False for role in roles}
        role_trainable_idx = {}
        role_idx_collection = {role: set() for role in roles}
        for role in roles:
            for rpc in self.model_rpcs:
                if rpc.role != role:
                    continue
                if rpc.interface_type == dfg.ModelInterfaceType.TRAIN_STEP:
                    if role_is_trainable[role]:
                        raise ValueError(
                            f"Multiple train_step for the same role {role} is not allowed."
                        )
                    role_is_trainable[role] = True
                    role_trainable_idx[role] = rpc.model_name.replica_id
                role_idx_collection[role].add(rpc.model_name.replica_id)
        role_cnt = {role: len(v) for role, v in role_idx_collection.items()}

        model_names_to_instantiate = []
        for role in roles:
            if role_is_trainable[role]:
                model_names_to_instantiate.append(
                    ModelName(role, role_trainable_idx[role])
                )
            else:
                model_names_to_instantiate += [
                    ModelName(role, i) for i in range(role_cnt[role])
                ]

        return model_names_to_instantiate


class Experiment:
    """Base class for defining the procedure of an experiment."""

    def scheduling_setup(self) -> ExperimentScheduling:
        """Returns the Scheduling of all workers."""
        raise NotImplementedError()

    def initial_setup(self) -> ExperimentConfig | List[ExperimentConfig]:
        """Returns a list of workers to create when a trial of the experiment
        is initialized."""
        raise NotImplementedError()


ALL_EXPERIMENT_CLASSES = {}


def register_experiment(name, cls):
    assert name not in ALL_EXPERIMENT_CLASSES
    ALL_EXPERIMENT_CLASSES[name] = cls


def make_experiment(name) -> Experiment:
    cls = ALL_EXPERIMENT_CLASSES[name]
    return cls()
