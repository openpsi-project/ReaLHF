from typing import *
import dataclasses

import realhf.base.topology as topology


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
    """A unique identifier for a model.

    :param role: The role of the model, e.g. "actor", "critic".
    :type role: str
    :param replica_id: The replica id of the model.
    :type replica_id: int
    """

    role: str
    replica_id: int

    @property
    def name(self):
        return str(self)


@dataclasses.dataclass(unsafe_hash=True)
class ModelFamily:
    _class: str
    size: int
    is_critic: bool

    def __repr__(self):
        s = f"{self._class}-{self.size}"
        if self.is_critic:
            s += "-critic"
        return s


@dataclasses.dataclass
class ModelShardID:
    model_name: ModelName
    dp_rank: int
    mp_rank: int
    pp_rank: int
    topo: topology.PipeModelDataParallelTopology = dataclasses.field(
        default_factory=lambda: topology.PipeModelDataParallelTopology(1, 1, 1)
    )

    def __post_init__(self):
        assert self.dp_rank >= 0 and self.mp_rank >= 0 and self.pp_rank >= 0
        if "@" in self.model_name.role:
            raise ValueError("model_name cannot contain @")
        assert self.dp_rank < self.topo.get_dim("data")
        assert self.mp_rank < self.topo.get_dim("model")
        assert self.pp_rank < self.topo.get_dim("pipe")

    @property
    def parallelism_rank(self):
        return self.topo.get_rank(
            data=self.dp_rank, model=self.mp_rank, pipe=self.pp_rank
        )

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
            return (
                self.model_name == other.model_name
                and self.dp_rank == other.dp_rank
                and self.mp_rank == other.mp_rank
                and self.pp_rank == other.pp_rank
            )
        return False


@dataclasses.dataclass
class StandaloneModelShard:
    id: ModelShardID
    model: Model
    backend: ModelBackend
    # evaluation
    eval_datasets: Optional[List[Dataset]] = None
    eval_dataloader: Optional[DataLoader] = DataLoader(
        "packed_eval", args=dict(batch_size=128)
    )
    should_instantiate: bool = True
