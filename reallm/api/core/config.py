from typing import *
import copy
import dataclasses

import reallm.base.topology


@dataclasses.dataclass
class Dataset:
    type_: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class DataLoader:
    type_: str = "default"
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ModelWrapper:
    type_: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Model:
    type_: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)
    wrappers: List[ModelWrapper] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ModelBackend:
    type_: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(unsafe_hash=True, order=True)
class ModelName:
    role: str
    replica_id: int


@dataclasses.dataclass(unsafe_hash=True)
class ModelType:
    _class: str
    size: int
    is_critic: bool

    def __repr__(self):
        return f"{self._class}-{self.size}"


@dataclasses.dataclass
class ModelShardID:
    model_name: ModelName
    dp_rank: int
    mp_rank: int
    pp_rank: int
    topo: reallm.base.topology.PipeModelDataParallelTopology = dataclasses.field(
        default_factory=lambda: reallm.base.topology.PipeModelDataParallelTopology(1, 1, 1))

    def __post_init__(self):
        assert self.dp_rank >= 0 and self.mp_rank >= 0 and self.pp_rank >= 0
        if "@" in self.model_name.role:
            raise ValueError("model_name cannot contain @")
        assert self.dp_rank < self.topo.get_dim("data")
        assert self.mp_rank < self.topo.get_dim("model")
        assert self.pp_rank < self.topo.get_dim("pipe")

    @property
    def parallelism_rank(self):
        return self.topo.get_rank(data=self.dp_rank, model=self.mp_rank, pipe=self.pp_rank)

    @classmethod
    def from_parallelism_rank(cls, model_name, topo, parallelism_rank):
        c = topo.get_coord(parallelism_rank)
        return cls(
            model_name=model_name,
            dp_rank=c.data,
            mp_rank=c.model,
            pp_rank=c.pipe,
            topo=topo,
        )

    def __repr__(self):
        return f"{self.model_name}@pp{self.pp_rank:02d}@mp{self.mp_rank:02d}@dp{self.dp_rank:02d}"

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        # Compare the key attribute for equality
        if isinstance(other, ModelShardID):
            return (self.model_name == other.model_name and self.dp_rank == other.dp_rank
                    and self.mp_rank == other.mp_rank and self.pp_rank == other.pp_rank)
        return False


@dataclasses.dataclass
class StandaloneModelShard:
    """A combination of model, model interface, and model backend. Representing
    a runnable model shard, which has topology `topo` and is indexed by `dp_rank`,
    `mp_rank`, and `pp_rank`. `StandaloneModelShard` resides in model workers,
    and each model worker can have multiple shards.
    """

    id: ModelShardID
    model: Model
    backend: ModelBackend
    # evaluation
    eval_datasets: Optional[List[Dataset]] = None
    eval_dataloader: Optional[DataLoader] = None


MODEL_TYPE_TO_PATH: Dict[ModelType, str] = {
    ModelType("llama", 0, True): "/lustre/public/pretrained_model_weights/testOnly/llama-2-16l/",
    ModelType("llama", 7, True): "/lustre/public/pretrained_model_weights/Llama-2-7b-hf/",
    ModelType("llama", 13, True): "/lustre/public/pretrained_model_weights/Llama-2-13b-hf/",
    ModelType("llama", 70, True): "/lustre/public/pretrained_model_weights/Llama-2-70b-hf/",
    ModelType("codellama", 34, True): "/lustre/public/pretrained_model_weights/CodeLlama-34b-hf",
}
_d = {}
for k, v in MODEL_TYPE_TO_PATH.items():
    k_ = copy.deepcopy(k)
    k_.is_critic = False
    _d[k_] = v
MODEL_TYPE_TO_PATH.update(_d)

SUPPORTED_MODELS = ["starcoder", "llama", "gpt2", "deepseek", "codellama"]
