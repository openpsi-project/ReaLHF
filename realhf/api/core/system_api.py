from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import collections
import copy
import dataclasses
import enum
import getpass
import itertools
import math
import os
import sys

from realhf.api.core.config import *
from realhf.base.cluster import spec as cluster_spec
from realhf.base.constants import (
    DATASET_CACHE_PATH,
    PYTORCH_KERNEL_CACHE_PATH,
    TORCH_EXTENSIONS_DIR,
    TRITON_CACHE_PATH,
)
import realhf.api.core.dfg as dfg
import realhf.base.topology as topology

_LLM_ENVVARS = {
    # "NCCL_P2P_DISABLE": "1",
    # "NCCL_IB_DISABLE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "PYTORCH_KERNEL_CACHE_PATH": PYTORCH_KERNEL_CACHE_PATH,
    "TRITON_CACHE_DIR": TRITON_CACHE_PATH,
    "TOKENIZERS_PARALLELISM": "true",
    "TORCH_EXTENSIONS_DIR": TORCH_EXTENSIONS_DIR,
    # "NCCL_DEBUG": "INFO",
    # "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
    # "NCCL_SOCKET_IFNAME": "ibp71s0",
    # "GLOO_SOCKET_IFNAME": "ibp71s0",
    # "TORCH_USE_CUDA_DSA": "1",
    # "NCCL_IGNORE_DISABLED_P2P": "1",
    # "CUDA_LAUNCH_BLOCKING": "1",  # NOTE: CUDAGraph Capturing will not work if CUDA_LAUNCH_BLOCKING is set to 1.
    # "NCCL_COMM_BLOCKING": "1",  # NOTE: CUDAGraph Capturing will not work if NCCL_COMM_BLOCKING is set to 1.
    # "NCCL_BLOCKING_WAIT": "1",  # NOTE: CUDAGraph Capturing will not work if NCCL_BLOCKING_WAIT is set to 1.
    # "TORCH_SHOW_CPP_STACKTRACES": "1",
    "RAY_DEDUP_LOGS": "0",  # disable ray log deduplication
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "PYTHONUSERBASE": "/nonsense",
    # torch.distributed.all_reduce does not free the input tensor until
    # the synchronization point. This causes the memory usage to grow
    # as the number of all_reduce calls increases. This env var disables
    # this behavior.
    # Related issue:
    # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
    # Whether to enable time mark to plot timelines.
    "REAL_CUDA_TMARK": "1",
}
for k, v in _LLM_ENVVARS.items():
    os.environ[k] = v

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
    env_vars: Dict[str, str] = dataclasses.field(
        default_factory=lambda: _LLM_ENVVARS
    )
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
    """The basic information of an worker. To improve config readability, the experiment starter will fill the
    fields, instead of letting the users do so in experiment configs.
    """

    experiment_name: str = ""
    trial_name: str = (
        ""  # Name of the trial of the experiment; e.g. "{USER}-0".
    )
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
    host_key: Optional[str] = (
        None  # Worker will update and keep this key alive.
    )
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
        """Setup system related worker information, while leaving the rest untouched."""
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.worker_type = worker_type
        self.worker_index = worker_index
        self.worker_count = worker_count


@dataclasses.dataclass
class ModelWorker:
    seed: int
    shards: List[StandaloneModelShard]
    # dataset, for source model workers
    tokenizer_name_or_path: Optional[str] = None
    datasets: Optional[List[Union[str, Dataset]]] = None
    dataloader: Union[str, DataLoader] = "packed"
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
    worker_info: Optional[WorkerInformation] = None

    def __post_init__(self):
        model_names = [s.id.model_name for s in self.shards]
        if len(set(model_names)) != len(model_names):
            raise ValueError(
                f"ModelWorker cannot have multiple shards of the same model name: {model_names}."
            )


@dataclasses.dataclass
class ExperimentSaveEvalControl:
    total_train_epochs: int = 1
    # save control
    save_frequency_epochs: Optional[int] = None
    save_frequency_steps: Optional[int] = None
    save_frequency_seconds: Optional[int] = None
    # eval control
    eval_frequency_epochs: Optional[int] = None
    eval_frequency_steps: Optional[int] = None
    eval_frequency_seconds: Optional[int] = None
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
    config: Optional[Any] = None

    def __post_init__(self):
        assert self.master_worker is None
        model_names = set()
        for w in self.model_worker:
            model_names = model_names.union([s.id.model_name for s in w.shards])
        model_names = sorted(list(model_names))

        ############### Sanity check of model names ###############
        _roles = set(mn.role for mn in model_names)
        _replica_ids = {
            _role: sorted(
                [mn.replica_id for mn in model_names if mn.role == _role]
            )
            for _role in _roles
        }
        for v in _replica_ids.values():
            if list(sorted(v)) != list(range(len(v))):
                raise ValueError(
                    f"Model replica ids should be 0, 1, 2, ... for each role: {_replica_ids}."
                )
        ############### Sanity check of model names ###############

        model_topos: Dict[ModelName, topology.PipeModelDataParallelTopology] = (
            {}
        )
        model_configs: Dict[ModelName, Model] = {}
        for model_name in model_names:
            _this_mws = list(
                filter(
                    lambda mw: any(
                        x.id.model_name == model_name for x in mw.shards
                    ),
                    self.model_worker,
                )
            )
            all_shards: List[StandaloneModelShard] = [
                next(filter(lambda x: x.id.model_name == model_name, mw.shards))
                for mw in _this_mws
            ]
            # TODO: same topo, diff nodelist?
            # for k, v in model_topos.items():
            # if k.role == model_name.role and v == all_shards[0].id.topo:
            #     raise ValueError(f"If different RPCs have the same topology, "
            #                      f"they don't need to use multiple model names ({k}, {model_name}).")
            model_topos[model_name] = all_shards[0].id.topo
            model_configs[model_name] = all_shards[0].model

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

        data_transfer_pairs: List[Tuple[ModelName, ModelName]] = []
        _rpc_nodes, edges = dfg.build_graph(self.model_rpcs, verbose=True)
        for i in range(len(self.model_rpcs)):
            for j in range(len(self.model_rpcs)):
                if len(edges[i][j]) > 0:
                    # NOTE: dependencies are reversed here
                    data_transfer_pairs.append(
                        (
                            self.model_rpcs[j].model_name,
                            self.model_rpcs[i].model_name,
                        )
                    )
                    # print(f"model_rpc j name {self.model_rpcs[j].name} i name {self.model_rpcs[i].name} "
                    #        "data transfer pair append {(self.model_rpcs[j].model_name, self.model_rpcs[i].model_name)} ")
        src_rpcs = [rpc for rpc in self.model_rpcs if rpc.is_src]
        data_src_rpc = src_rpcs[0]
        for r in src_rpcs[1:]:
            if (
                data_src_rpc.model_name,
                r.model_name,
            ) not in data_transfer_pairs:
                data_transfer_pairs.append(
                    (data_src_rpc.model_name, r.model_name)
                )
        data_transfer_pairs += [(mn, mn) for mn in model_names]

        sync_param_pairs: List[Tuple[ModelName, ModelName]] = []
        ######### sanity check of sync param hooks #########
        for rpc in self.model_rpcs:
            for hook in rpc.pre_hooks + rpc.post_hooks:
                if not isinstance(hook, dfg.SyncParamHook):
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
                    self_topo.get_dim("model") % other_topo.get_dim("model")
                    != 0
                    and other_topo.get_dim("model") % self_topo.get_dim("model")
                    != 0
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
        param_senders = set([x[0] for x in sync_param_pairs])
        assert len(set(param_senders)) == len(param_senders)
        param_receivers = set([x[1] for x in sync_param_pairs])
        assert len(set(param_receivers)) == len(param_receivers)
        for a, b in sync_param_pairs:
            if (b, a) not in sync_param_pairs:
                raise ValueError(
                    "The current implementation of parameter synchronization "
                    f"will throw local parameters away, "
                    f"so it is necceary to do bidirectional synchronization. "
                    f"{(a,b)} found in sync param pairs but not {(b,a)}"
                )
        ######### sanity check of sync param hooks #########

        # Mark which shard of the same role should be instantiated.
        _model_is_trainable = collections.defaultdict(list)
        for rpc in self.model_rpcs:
            _model_is_trainable[rpc.model_name].append(
                rpc.interface_type == dfg.ModelInterfaceType.TRAIN_STEP
            )

        _model_is_trainable = {
            model_name: any(values)
            for model_name, values in _model_is_trainable.items()
        }

        _roles = set([rpc.model_name.role for rpc in _rpc_nodes])
        _role_cnt = {
            role: len(
                set(
                    [
                        rpc.model_name
                        for rpc in _rpc_nodes
                        if rpc.model_name.role == role
                    ]
                )
            )
            for role in _roles
        }
        model_names_to_instantiate = []
        for role in _roles:
            _trainable_this_role = [
                _model_is_trainable[ModelName(role, i)]
                for i in range(_role_cnt[role])
            ]
            if _role_cnt[role] == 1 or not any(_trainable_this_role):
                model_names_to_instantiate.append(ModelName(role, 0))
                continue
            if any(_trainable_this_role):
                assert sum(_trainable_this_role) == 1
                _trainable_idx = _trainable_this_role.index(True)
                model_names_to_instantiate.append(
                    ModelName(role, _trainable_idx)
                )
        for mw in self.model_worker:
            for s in mw.shards:
                s.should_instantiate = (
                    s.id.model_name in model_names_to_instantiate
                )

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


class Experiment:
    """Base class for defining the procedure of an experiment."""

    def scheduling_setup(self, config) -> ExperimentScheduling:
        """Returns the Scheduling of all workers."""
        raise NotImplementedError()

    def initial_setup(self, config) -> ExperimentConfig:
        """Returns a list of workers to create when a trial of the experiment is initialized."""
        raise NotImplementedError()


def dump_config_to_yaml(config, file):
    import yaml

    with open(file, "w") as f:
        yaml.dump(dataclass_to_dict(config), f)


def load_config_from_yaml(file):
    import yaml

    with open(file, "r") as f:
        return config_to_dataclass(yaml.safe_load(f))


def dataclass_to_dict(dc):
    if isinstance(dc, (str, int, float)) or dc is None:
        pass
    elif isinstance(dc, enum.Enum):
        dc = dc.value
    elif isinstance(dc, (list, tuple)):
        dc = [dataclass_to_dict(d) for d in dc]
    elif isinstance(dc, dict):
        dc = {k: dataclass_to_dict(v) for k, v in dc.items()}
    elif dataclasses.is_dataclass(dc):
        root_name = dc.__class__.__name__
        dc = dict(
            config_class=root_name,
            config_value={
                k.name: dataclass_to_dict(getattr(dc, k.name))
                for k in dataclasses.fields(dc)
            },
        )
    else:
        raise f"{dc} of type {type(dc)} cannot be parse to dict."
    return dc


def config_to_dataclass(config: Union[List, Dict]):
    if isinstance(config, (list, tuple)):
        return [config_to_dataclass(c) for c in config]
    elif isinstance(config, dict):
        if "config_class" in config.keys():
            return getattr(sys.modules[__name__], config["config_class"])(
                **{
                    k: config_to_dataclass(v)
                    for k, v in config["config_value"].items()
                }
            )
        else:
            return config
    elif isinstance(config, (str, int, float)) or config is None:
        return config
    else:
        raise NotImplementedError(config)


ALL_EXPERIMENT_CLASSES = {}


def register_experiment(name, cls):
    assert name not in ALL_EXPERIMENT_CLASSES
    ALL_EXPERIMENT_CLASSES[name] = cls


def make_experiment(name) -> Experiment:
    cls = ALL_EXPERIMENT_CLASSES[name]
    return cls()
