from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import collections
import copy
import dataclasses
import enum
import getpass
import itertools
import math
import os
import sys

import yaml

from base.cluster import spec as cluster_spec
import api.dfg
import base.topology

PYTORCH_KERNEL_CACHE_PATH = f"{cluster_spec.fileroot}/.cache/{getpass.getuser()}/torch/kernels"
TRITON_CACHE_PATH = f"{cluster_spec.fileroot}/.cache/{getpass.getuser()}/triton"
DATASET_CACHE_PATH = f'{cluster_spec.fileroot}/.cache/{getpass.getuser()}/datasets'
TORCH_EXTENSIONS_DIR = f"{cluster_spec.fileroot}/.cache/{getpass.getuser()}/torch/extensions"

_LLM_ENVVARS = {
    # "NCCL_P2P_DISABLE": "1",
    # "NCCL_IB_DISABLE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "PYTORCH_KERNEL_CACHE_PATH": PYTORCH_KERNEL_CACHE_PATH,
    "TRITON_CACHE_DIR": TRITON_CACHE_PATH,
    "TOKENIZERS_PARALLELISM": "true",
    "TORCH_EXTENSIONS_DIR": TORCH_EXTENSIONS_DIR,
    # "CUDA_LAUNCH_BLOCKING": "1",
    # "TORCH_USE_CUDA_DSA": "1",
    "RAY_DEDUP_LOGS": "0",  # disable ray log deduplication
    "PYTHONUSERBASE": "/nonsense"
}
for k, v in _LLM_ENVVARS.items():
    os.environ[k] = v

_LLM_GPU_IMAGE = "llm/llm-gpu"
_LLM_CPU_IMAGE = "llm/llm-cpu:numba"


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
    env_vars: Dict[str, str] = dataclasses.field(default_factory=lambda: _LLM_ENVVARS)
    # time utils from "https://slurm.schedmd.com/sbatch.html"
    time_limit: Optional[str] = None  # see  "--time" option for format
    begin: Optional[str] = None  # see "--begin" option for format
    deadline: Optional[str] = None  # see "--deadline" option for format

    @staticmethod
    def master_worker_default(**kwargs):
        return Scheduling(**{
            "cpu": 1,
            "gpu": 0,
            "mem": 20 * 1024,
            "container_image": _LLM_CPU_IMAGE,
            **kwargs
        })

    @staticmethod
    def data_worker_default(**kwargs):
        return Scheduling(**{
            "cpu": 4,
            "gpu": 0,
            "mem": 20 * 1024,
            "container_image": _LLM_CPU_IMAGE,
            **kwargs
        })

    @staticmethod
    def model_worker_default(**kwargs):
        return Scheduling(**{
            "cpu": 2,
            "gpu": 1,
            "mem": 1024,
            "container_image": _LLM_GPU_IMAGE,
            **kwargs,
        })


@dataclasses.dataclass
class WorkerInformation:
    """The basic information of an worker. To improve config readability, the experiment starter will fill the
    fields, instead of letting the users do so in experiment configs.
    """
    experiment_name: str = ""
    trial_name: str = ""  # Name of the trial of the experiment; e.g. "{USER}-0".
    model_name: str = ""
    worker_type: str = ""  # E.g. "policy", "actor", or "trainer".
    worker_index: int = -1  # The index of the worker of the specific type, starting from 0.
    worker_count: int = 0  # Total number of workers; hence, 0 <= worker_index < worker_count.
    worker_tag: Optional[str] = None  # For actor and policy worker, can be "training" or "evaluation".
    host_key: Optional[str] = None  # Worker will update and keep this key alive.
    watch_keys: Union[str, List[str]] = None  # Worker will exit if all of the watching keys are gone.
    wandb_entity: Optional[
        str] = None  # wandb_{config} are optional. They overwrite system wandb_configuration.
    wandb_project: Optional[str] = None
    wandb_job_type: Optional[str] = None
    wandb_group: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_config: Optional[Dict] = None
    log_wandb: Optional[bool] = None

    def system_setup(self, experiment_name, trial_name, worker_type, worker_index, worker_count, model_name):
        """Setup system related worker information, while leaving the rest untouched.
        """
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.worker_type = worker_type
        self.worker_index = worker_index
        self.worker_count = worker_count
        self.model_name = model_name


@dataclasses.dataclass
class Dataset:
    type_: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class DataLoader:
    type_: str = "default"
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Model:
    type_: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ModelInterface:
    type_: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ModelBackend:
    type_: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class RequestReplyStream:
    stream_name: str
    serialization_method: str = 'raw_bytes'


@dataclasses.dataclass
class ModelWorker:
    seed: int
    model: Model
    interface: ModelInterface
    backend: ModelBackend
    model_name: str  # the name of this whole model, not this shard
    # parallelism ranks, used to reveal model shard's identity
    dp_rank: int = 0
    mp_rank: int = 0
    pp_rank: int = 0
    topo: Optional[base.topology.PipeModelDataParallelTopology] = dataclasses.field(
        default_factory=lambda x: base.topology.PipeModelDataParallelTopology(1, 1, 1))
    # evaluation
    eval_datasets: Optional[List[Dataset]] = None
    eval_dataloader: Optional[DataLoader] = None
    use_dataset_cache: bool = False
    dataset_cahce_root: str = DATASET_CACHE_PATH
    # cuda & cudnn config
    cudnn_benchmark: bool = False
    cudnn_deterministic: bool = False
    cuda_cache_cleanliness: bool = True
    cuda_cache_clear_freq: int = 10
    # stream and worker_info will be configured automatically
    stream: Optional[Union[str, RequestReplyStream]] = None
    worker_info: Optional[WorkerInformation] = None

    def __post_init__(self):
        assert '@' not in self.model_name


@dataclasses.dataclass
class DataWorker:
    tokenizer_name_or_path: str
    datasets: List[Union[str, Dataset]]
    dataloader: Union[str, DataLoader] = "default"
    seed: int = 1
    # dataset cache
    use_dataset_cache: bool = False
    dataset_cahce_root: str = DATASET_CACHE_PATH
    # stream and worker_info will be configured automatically
    stream: Optional[Union[str, RequestReplyStream]] = None
    worker_info: Optional[WorkerInformation] = None


@dataclasses.dataclass
class MasterWorker:
    total_train_epochs: int
    # save control
    save_frequency_epochs: int
    save_frequency_steps: int
    save_frequency_seconds: int
    # eval control
    eval_frequency_epochs: int
    eval_frequency_steps: int
    eval_frequency_seconds: int
    # main components
    model_rpcs: List[api.dfg.ModelRPC]
    data_streams: List[Union[str, RequestReplyStream]]
    model_streams: Dict[Tuple[str, int, int, int], Union[str, RequestReplyStream]]
    model_topos: Dict[str, base.topology.PipeModelDataParallelTopology]
    benchmark_steps: int
    worker_info: Optional[WorkerInformation] = None


@dataclasses.dataclass
class TasksGroup:
    count: int
    scheduling: Scheduling


@dataclasses.dataclass
class ExperimentScheduling:
    data_worker: Union[List[TasksGroup], TasksGroup] = dataclasses.field(default_factory=list)
    model_worker: Union[List[TasksGroup], TasksGroup] = dataclasses.field(default_factory=list)
    master_worker: Union[List[TasksGroup], TasksGroup] = dataclasses.field(default_factory=list)
    controller_image: str = _LLM_CPU_IMAGE


@dataclasses.dataclass
class ExperimentConfig:
    total_train_epochs: int
    # dataflow
    model_rpcs: List[api.dfg.ModelRPC]
    data_worker: List[DataWorker] = dataclasses.field(default_factory=list)
    model_worker: List[ModelWorker] = dataclasses.field(default_factory=list)
    # eval control
    save_frequency_epochs: Optional[int] = None
    save_frequency_steps: Optional[int] = None
    save_frequency_seconds: int = 3600
    # save control
    eval_frequency_epochs: Optional[int] = None
    eval_frequency_steps: Optional[int] = None
    eval_frequency_seconds: Optional[int] = None
    benchmark_steps: Optional[int] = None  # only used for benchmark
    # master_worker will be set automatically
    master_worker: Optional[List[MasterWorker]] = None
    config: Optional[Any] = None

    def __post_init__(self):
        assert self.master_worker is None

        model_streams = {}
        model_topos = {}
        model_names = list(set([w.model_name for w in self.model_worker]))
        for model_name in model_names:
            mws = [w for w in self.model_worker if w.model_name == model_name]
            ranks = [mw.topo.get_rank(pipe=mw.pp_rank, model=mw.mp_rank, data=mw.dp_rank) for mw in mws]
            # Sanity check of parallelism ranks.
            if set(ranks) != set(list(range(len(mws)))) or any(mw.topo.world_size() != len(mws)
                                                               for mw in mws):
                raise ValueError(f"Parallelism rank check failed: model name {model_name}, "
                                 f"parallelism ranks pipe={[mw.pp_rank for mw in mws]}, "
                                 f"model={[mw.mp_rank for mw in mws]}, "
                                 f"data={[mw.dp_rank for mw in mws]}, "
                                 f"flattened ranks {ranks}.")
            model_topos[model_name] = mws[0].topo

            # Set stream for model workers.
            for mw in mws:
                # Following the naming convention of deepspeed. Inner sep is '_' and outer sep is '-'.
                model_id = f"{mw.model_name}@pp_{mw.pp_rank:02d}-mp_{mw.mp_rank:02d}-dp_{mw.dp_rank:02d}"
                s = RequestReplyStream(model_id)
                mw.stream = s
                model_streams[model_id] = s

        for i, d in enumerate(self.data_worker):
            d.stream = RequestReplyStream(f"data_{i}")
        data_streams = [d.stream for d in self.data_worker]

        for w in itertools.chain(self.model_worker, self.data_worker):
            assert w.stream is not None

        self.master_worker = [
            MasterWorker(
                total_train_epochs=self.total_train_epochs,
                save_frequency_epochs=self.save_frequency_epochs,
                save_frequency_steps=self.save_frequency_steps,
                save_frequency_seconds=self.save_frequency_seconds,
                eval_frequency_epochs=self.eval_frequency_epochs,
                eval_frequency_steps=self.eval_frequency_steps,
                eval_frequency_seconds=self.eval_frequency_seconds,
                data_streams=data_streams,
                model_streams=model_streams,
                model_topos=model_topos,
                model_rpcs=self.model_rpcs,
                benchmark_steps=self.benchmark_steps,  # only used for benchmark
            )
        ]

    def set_worker_information(self, experiment_name, trial_name):
        assert len(self.master_worker) == 1
        for worker_type, workers in [
            ("model_worker", self.model_worker),
            ("master_worker", self.master_worker),
            ("data_worker", self.data_worker),
        ]:
            for i, worker in enumerate(workers):
                system_worker_info = dict(experiment_name=experiment_name,
                                          trial_name=trial_name,
                                          worker_type=worker_type,
                                          worker_index=i,
                                          worker_count=len(workers))
                if worker.worker_info is not None:
                    worker.worker_info.system_setup(**system_worker_info)
                else:
                    worker.worker_info = WorkerInformation(**system_worker_info)


class Experiment:
    """Base class for defining the procedure of an experiment.
    """

    def scheduling_setup(self) -> ExperimentScheduling:
        """Returns the Scheduling of all workers."""
        raise NotImplementedError()

    def initial_setup(self) -> ExperimentConfig:
        """Returns a list of workers to create when a trial of the experiment is initialized."""
        raise NotImplementedError()


def dump_config_to_yaml(config, file):
    with open(file, "w") as f:
        yaml.dump(dataclass_to_dict(config), f)


def load_config_from_yaml(file):
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
            config_value={k.name: dataclass_to_dict(getattr(dc, k.name))
                          for k in dataclasses.fields(dc)})
    else:
        raise f"{dc} of type {type(dc)} cannot be parse to dict."
    return dc


def config_to_dataclass(config: Union[List, Dict]):
    if isinstance(config, (list, tuple)):
        return [config_to_dataclass(c) for c in config]
    elif isinstance(config, dict):
        if "config_class" in config.keys():
            return getattr(sys.modules[__name__], config["config_class"])(
                **{k: config_to_dataclass(v)
                   for k, v in config["config_value"].items()})
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
