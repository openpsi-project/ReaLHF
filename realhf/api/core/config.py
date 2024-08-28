import dataclasses
import enum
from typing import *

import realhf.base.topology as topology


@dataclasses.dataclass
class DatasetAbstraction:
    type_: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class DataLoaderAbstraction:
    type_: str = "default"
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ModelWrapperAbstraction:
    type_: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ModelAbstraction:
    type_: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)
    wrappers: List[ModelWrapperAbstraction] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class ModelBackendAbstraction:
    type_: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ModelInterfaceAbstraction:
    type_: str  # This type is the
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)


class ModelInterfaceType(enum.Enum):
    GENERATE = "generate"
    TRAIN_STEP = "train_step"
    EVALUATE = "evaluate"
    INFERENCE = "inference"


@dataclasses.dataclass(unsafe_hash=True, order=True, frozen=True)
class ModelName:
    """A unique identifier for a model.

    :param role: The role of the model, e.g., "actor" or "critic".
    :type role: str
    :param replica_id: The replica ID of the model. Different replicas
        of the same role have the same set of parameters but different
        memory locations. For example, if actor generation and training
        in PPO use different parallel strategies, they will have the
        same role but different replica IDs.
    :type replica_id: int
    """

    role: str
    replica_id: int

    @property
    def name(self):
        return str(self)


@dataclasses.dataclass(unsafe_hash=True)
class ModelFamily:
    """An identifier for the HF model type, such as llama, gpt2, etc.

    :param _class: The class of the model, e.g., "llama". This is the registered
        name in the ``register_hf_family`` function. Please refer to the files
        in ``realhf/api/from_hf`` for a list of all supported models.
    :type _class: str
    :param size: The size of the model. This parameter is only used by the ``search``
        allocation mode and will be ignored otherwise.
    :type size: int
    :param is_critic: Indicates whether the model is a critic or reward model,
        as opposed to a standard LLM.
    :type is_critic: bool
    """

    _class: str
    size: int = 0
    is_critic: bool = False

    def __repr__(self):
        s = f"{self._class}-{self.size}"
        if self.is_critic:
            s += "-critic"
        return s


@dataclasses.dataclass
class ModelShardID:
    """The ID of a model shard in a specific model worker.

    This ID is essentially a combination of the model name and the 3D
    parallelism rank, and can be used as a dictionary key. It represents
    the identity of a "model handler". The master worker maintains a
    lookup table mapping the ModelShardID to the model worker index,
    which can be a many-to-one mapping. Requests are created with the
    ModelShardID; for example, actors with ranks (dp=*, mp=0, pp=0)
    should transfer data to the critics. The ModelShardID is then mapped
    to the model worker index, and the requests are sent to the
    corresponding model workers.

    :param model_name: The name of the model.
    :type model_name: ModelName
    :param dp_rank: The data parallel rank.
    :type dp_rank: int
    :param mp_rank: The tensor-model parallel rank.
    :type mp_rank: int
    :param pp_rank: The pipeline-model parallel rank.
    :type pp_rank: int
    :param topo: The 3D parallelism topology of this model.
    :type topo: PipeModelDataParallelTopology
    """

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
class StandaloneModelShardAbstraction:
    id: ModelShardID
    model: ModelAbstraction
    backend: ModelBackendAbstraction
    # evaluation
    eval_datasets: Optional[List[DatasetAbstraction]] = None
    eval_dataloader: Optional[DataLoaderAbstraction] = DataLoaderAbstraction(
        "packed_eval", args=dict(batch_size=128)
    )
    should_instantiate: bool = True
