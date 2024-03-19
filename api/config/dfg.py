from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING, Union
import collections
import dataclasses
import enum
import itertools

import base.logging as logging
import base.namedarray as namedarray

logger = logging.getLogger("DataFlowGraph", "benchmark")


@dataclasses.dataclass
class OffloadHook:
    pass


@dataclasses.dataclass
class LoadToDeviceHook:
    pass


@dataclasses.dataclass
class SyncParamHook:
    target: str
    interval: int = 1


@dataclasses.dataclass
class ReparallelizeHook:
    dp_size: int
    mp_size: int
    pp_size: int


RPCHook = Union[OffloadHook, LoadToDeviceHook, SyncParamHook, ReparallelizeHook]


@dataclasses.dataclass
class ModelInterface:
    type_: str
    args: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __eq__(self, other: "ModelInterface"):
        return self.type_ == other.type_ and self.args == other.args


class ModelInterfaceType(enum.Enum):
    GENERATE = "generate"
    TRAIN_STEP = "train_step"
    EVALUATE = "evaluate"
    INFERENCE = "inference"


@dataclasses.dataclass
class ModelType:
    _class: str
    size: int
    is_critic: bool

    def __hash__(self):
        return (self._class, self.size, self.is_critic).__hash__()

    def __eq__(self, other):
        return self._class == other._class and self.size == other.size and self.is_critic == other.is_critic


@dataclasses.dataclass
class ModelRPC:
    model_name: str
    model_type: ModelType
    interface_type: ModelInterfaceType
    interface_impl: ModelInterface

    input_data: List[str] = dataclasses.field(default_factory=lambda: [])
    input_key_remap: Dict[str, str] = dataclasses.field(default_factory=lambda: {})
    output_data: List[str] = dataclasses.field(default_factory=lambda: [])
    output_key_remap: Dict[str, str] = dataclasses.field(default_factory=lambda: {})
    log_return_value: bool = False

    min_n_seqs_per_dp: int = 1
    balanced_dp: bool = False

    # batch sizes
    min_n_seqs: int = 1
    max_n_seqs: int = 1024
    min_n_tokens: int = 1
    max_n_tokens: int = 655360

    max_concurrent_calls: int = 1

    # hooks
    pre_hooks: List[RPCHook] = dataclasses.field(default_factory=lambda: [])
    post_hooks: List[RPCHook] = dataclasses.field(default_factory=lambda: [])

    # The followings will be automatically filled.
    max_min_flow_seqs: int = 1
    max_min_flow_tokens: int = 1

    parents: List[str] = dataclasses.field(default_factory=lambda: [])
    children: List[str] = dataclasses.field(default_factory=lambda: [])

    parent_rpcs: List["ModelRPC"] = dataclasses.field(default_factory=lambda: [])
    children_rpcs: List["ModelRPC"] = dataclasses.field(default_factory=lambda: [])

    # data key -> model name
    data_producers: Dict[str, str] = dataclasses.field(default_factory=lambda: {})
    # data key -> rpc names
    data2required_rpc_names: Dict[str, List[str]] = dataclasses.field(default_factory=lambda: {})

    def __post_init__(self):
        if self.min_n_seqs > self.max_n_seqs or self.min_n_tokens > self.max_n_tokens:
            raise RuntimeError("Invalid min/max n_seqs/n_tokens.")
        if self.is_src and self.max_n_seqs > 1e4 and self.max_n_tokens > 1e8:
            raise RuntimeError(
                "The maximum batch size of the source node in the dataflow graph is too large. "
                f"The maximum number of sequences is {self.max_n_seqs} > budget {int(1e4)} and "
                f"the maximum number of tokens is {self.max_n_tokens} > budget {int(1e8)}. "
                "Please set a smaller value.")
        if "@" in self.model_name or "@" in self.interface_type.value:
            raise ValueError(f"Invalid model name or interface type: {self.model_name}, {self.interface_type}.")

    def __repr__(self):
        return f"ModelRPC({self.model_name}, {self.interface_type})"

    @property
    def name(self):
        return f"{self.model_name}@{self.interface_type.value}"

    @property
    def is_src(self):
        return len(self.parents) == 0

    @property
    def is_dst(self):
        return len(self.children) == 0

    @property
    def is_dst_of_model(self):

        def _has_children_of_model_name(rpc: "ModelRPC", model_name: str):
            if rpc.is_dst:
                return False
            return any([
                r.model_name == model_name or _has_children_of_model_name(r, model_name)
                for r in rpc.children_rpcs
            ])

        return not _has_children_of_model_name(self, self.model_name)

    def remap_input_keys(self, input_batch: Dict) -> namedarray.NamedArray:
        data = {}
        for k in self.input_data:
            if k not in self.input_key_remap:
                data[k] = input_batch[k]
            else:
                data[self.input_key_remap[k]] = input_batch[k]
        return namedarray.from_dict(data)

    def remap_output_keys(self, output_batch: Dict) -> namedarray.NamedArray:
        res_data = {}
        for k, v in output_batch.items():
            if k not in self.output_data:
                continue
            if k in self.output_key_remap:
                res_data[self.output_key_remap[k]] = v
            else:
                res_data[k] = v
        return namedarray.from_dict(res_data)


def build_graph(rpcs: List[ModelRPC], verbose: bool = False) -> Tuple[List[ModelRPC], List[List[Tuple[str]]]]:
    # Resolve dependencies between model interfaces.
    children: List[List[str]] = [[] for _ in rpcs]
    parents: List[List[str]] = [[] for _ in rpcs]
    parent_rpcs: List[List[ModelRPC]] = [[] for _ in rpcs]
    children_rpcs: List[List[ModelRPC]] = [[] for _ in rpcs]
    edges: List[List[Tuple[str]]] = [[() for _ in rpcs] for _ in rpcs]

    required_data_entries: List[Tuple[str]] = [() for _ in rpcs]
    generated_data_entries: List[Tuple[str]] = [() for _ in rpcs]
    for i, rpc in enumerate(rpcs):
        required_data_entries[i] = (*required_data_entries[i], *rpc.input_data)
        generated_data_entries[i] = (
            *generated_data_entries[i],
            *[k if k not in rpc.output_key_remap else rpc.output_key_remap[k] for k in rpc.output_data],
        )
    data_producers = {}
    for rpc, gd in zip(rpcs, generated_data_entries):
        for k in gd:
            data_producers[k] = rpc.model_name
    data2required_rpc_names = collections.defaultdict(list)
    for rpc, data_keys in zip(rpcs, required_data_entries):
        for k in data_keys:
            data2required_rpc_names[k].append(rpc.name)

    for i, rpc in enumerate(rpcs):
        for k in required_data_entries[i]:
            for j, parent_rpc in enumerate(rpcs):
                if parent_rpc.name == rpc.name:
                    continue
                if k in generated_data_entries[j]:
                    if parent_rpc.name not in parents[i]:
                        parents[i].append(parent_rpc.name)
                        parent_rpcs[i].append(parent_rpc)
                    if rpc.name not in children[j]:
                        children[j].append(rpc.name)
                        children_rpcs[j].append(rpc)
                    edges[i][j] = (*edges[i][j], k)
                    if verbose:
                        logger.info(
                            f"Dependency added: {rpc.name} <- {parent_rpc.name} because of data entry `{k}`.")
    if verbose:
        for i, rpc in enumerate(rpcs):
            logger.info(
                f"Dependency: {rpc.name} <- { {x.name: deps for x, deps in zip(rpcs, edges[i]) if deps} }.")
    for rpc, p, c, pr, cr in zip(rpcs, parents, children, parent_rpcs, children_rpcs):
        rpc.parents = p
        rpc.children = c
        rpc.parent_rpcs = pr
        rpc.children_rpcs = cr

    for rpc in rpcs:
        rpc.max_min_flow_seqs = max([r.min_n_seqs for r in rpcs if r.model_name == rpc.model_name])
        rpc.max_min_flow_tokens = max([r.min_n_tokens for r in rpcs if r.model_name == rpc.model_name])
        rpc.data_producers = data_producers
        rpc.data2required_rpc_names = data2required_rpc_names

    # sanity check of hooks
    for rpc in rpcs:
        for h in itertools.chain(rpc.pre_hooks, rpc.post_hooks):
            assert isinstance(h, RPCHook), type(h)
            if isinstance(h, SyncParamHook):
                assert any(h.target == r.model_name for r in rpcs)
    return rpcs, edges
