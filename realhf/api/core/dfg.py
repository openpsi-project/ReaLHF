import collections
import dataclasses
from typing import *

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import realhf.base.logging as logging
from realhf.api.core.config import (
    ModelFamily,
    ModelInterfaceAbstraction,
    ModelInterfaceType,
    ModelName,
)

logger = logging.getLogger("DataFlowGraph", "benchmark")


@dataclasses.dataclass
class OffloadHook:
    pass


@dataclasses.dataclass
class SyncParamHook:
    source: Optional[ModelName] = None
    target: Optional[ModelName] = None
    interval: int = 1


RPCHook = Union[OffloadHook, SyncParamHook]


@dataclasses.dataclass
class MFCDef:
    """A model function call (MFC) object used by the workers.

    MFC is the abbreviation of Model Function Call.
    This object serves as the user interface for developing
    new algorithms.
    This object will be inserted in a nx.DiGraph as nodes.
    Edges will be automatically resolved by input/output keys.

    dataclasses.fields starting with an underscore will be filled automatically.

    :param name: The unique identifier of this model function call.
    :type name: str
    :param n_seqs: The number of sequences to be processed in a batch.
    :type n_seqs: int
    :param interface_type: The interface type to be used
        by the node (e.g., generate, train_step).
    :type interface_type: ModelInterfaceType
    :param model_name: The model identifier to be used by the node,
        corresponding to an unique LLM. The user-provided model name
        can just be a string. The replica ID will be resolved in ReaL.
    :type model_name: str or ModelName
    :param input_keys: Input data keys, used to resolve dependencies.
    :type input_keys: Tuple
    :param balanced_dp: Whether to balance the data parallelism such that
        each DP rank will get exactly n_seqs // dp_size sequences.
        If set to False, ReaL will do partition according to the number of tokens.
        If the lengths of sequences are not uniform, it may lead to unbalanced
        sequence numbers, but balanced memory usage.
    :type balanced_dp: bool
    :param log_return_value: Whether to log the return value of the interface implementation.
    :type log_return_value: bool
    :param model_type: The specification of the LLM, e.g., LLaMA-7B.
        Used by the profiler and search engine to produce an optimal execution plan.
        Can be omited if the search engine is not used.
    :type model_type: Optional[ModelFamily]
    :param model_path: The path to the model file. Used to get the config
        for the search engine.
        Can be omited if the search engine is not used.
    :type model_path: Optional[str]
    """

    # The unique identifier of this model function call.
    name: str

    # batch size
    n_seqs: int

    # The interface type to be used by the node (e.g., generate, train_step).
    interface_type: ModelInterfaceType
    interface_impl: ModelInterfaceAbstraction

    # The model identifier to be used by the node.
    model_name: str | ModelName

    # Input and output keys, used to resolve dependencies.
    input_keys: Tuple = dataclasses.field(default_factory=tuple)
    output_keys: Tuple = dataclasses.field(default_factory=tuple)

    balanced_dp: bool = False
    log_return_value: bool = False

    # Only used by search.
    model_type: Optional[Any | ModelFamily] = None
    model_path: Optional[str] = None

    # Reserved dataclasses.fields. Should not be set by the user.
    _G: nx.DiGraph = None
    _pre_hooks: List[RPCHook] = dataclasses.field(default_factory=lambda: [])
    _post_hooks: List[RPCHook] = dataclasses.field(default_factory=lambda: [])

    def __post_init__(self):
        if isinstance(self.model_name, str):
            self.model_name = ModelName(role=self.model_name, replica_id=0)

    def __repr__(self):
        return f"MFCDef[{self.name}]"

    def __hash__(self):
        return hash(self.name)

    @property
    def role(self):
        return self.model_name.role

    def add_pre_hook(self, h: RPCHook):
        assert isinstance(h, RPCHook), type(h)
        if isinstance(h, SyncParamHook):
            assert h.target is None or h.source is None
        if isinstance(h, OffloadHook):
            raise ValueError("Offload can only be post hooks!")
        self._pre_hooks.append(h)

    def add_post_hook(self, h: RPCHook):
        if isinstance(h, SyncParamHook):
            assert h.target is None or h.source is None
        self._post_hooks.append(h)

    @property
    def max_min_flow_seqs(self) -> int:
        return self._G.graph["max_min_flow_seqs"][self.name]

    @property
    def is_src(self):
        return len(list(self._G.predecessors(self.name))) == 0

    @property
    def is_dst(self):
        return len(list(self._G.successors(self.name))) == 0

    @property
    def data_producers(self) -> Dict[str, ModelName]:
        return self._G.graph["data_producers"]

    @property
    def data_consumers(self) -> Dict[str, List[str]]:
        return self._G.graph["data_consumers"]

    @property
    def parents(self) -> List["MFCDef"]:
        return [self._G.nodes[x]["object"] for x in self._G.predecessors(self.name)]

    @property
    def children(self) -> List["MFCDef"]:
        return [self._G.nodes[x]["object"] for x in self._G.successors(self.name)]

    @property
    def is_dst_of_model_role(self):

        def _has_children_of_model_name(rpc: "MFCDef", model_name: ModelName):
            if rpc.is_dst:
                return False
            return any(
                [
                    r.role == model_name.role
                    or _has_children_of_model_name(r, model_name)
                    for r in rpc.children
                ]
            )

        return not _has_children_of_model_name(self, self.model_name)


def _draw_topo_sorted_digraph(G: nx.DiGraph, graph_path: str):
    topological_order = list(nx.topological_sort(G))
    # Initialize a dictionary to store the depth of each node
    node_depth = {node: 0 for node in G.nodes()}

    # Calculate the depth of each node
    for node in topological_order:
        for neighbor in G.successors(node):
            node_depth[neighbor] = max(node_depth[neighbor], node_depth[node] + 1)

    layers = {
        i: [node for node, depth in node_depth.items() if depth == i]
        for i in range(max(node_depth.values()) + 1)
    }
    pos = nx.multipartite_layout(G, subset_key=layers)
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_size=4000,
        node_color="lightblue",
        arrows=True,
        arrowsize=20,
        width=1.5,
    )
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_color="black")
    plt.savefig(graph_path, dpi=300)


def build_graph(
    nodes: List[MFCDef],
    verbose: bool = False,
    graph_path: Optional[str] = None,
) -> nx.DiGraph:
    if len(set(node.name for node in nodes)) != len(nodes):
        raise ValueError(
            "Each model function call should have an unique name. "
            f"Got {[node.name for node in nodes]}."
        )

    _G = nx.DiGraph()
    _G.add_nodes_from([(node.name, dict(object=node)) for node in nodes])

    data_producers: Dict[str, MFCDef] = {}
    data_consumers: Dict[str, List[MFCDef]] = collections.defaultdict(list)
    for node in nodes:
        for k in node.output_keys:
            data_producers[k] = node
        for k in node.input_keys:
            data_consumers[k].append(node)

    for node in nodes:
        for k in node.input_keys:
            if k not in data_producers:
                # This is a key from the dataset.
                continue
            src, dst = data_producers[k].name, node.name
            if _G.has_edge(src, dst):
                _G[src][dst]["keys"].append(k)
            else:
                _G.add_edge(src, dst, keys=[k])
    if verbose:
        for u, v, data in _G.edges(data=True):
            logger.info(f"Edge: {u} -> {v} with keys {data['keys']}")
        if graph_path is not None:
            _draw_topo_sorted_digraph(_G, graph_path)
            logger.info(f"Graph illustration saved to: {graph_path}.")

    if len(nodes) != len(_G.nodes):
        raise ValueError("There are replicated nodes in the graph!")

    # Store useful metadata
    _G.graph["data_producers"] = {k: v.model_name for k, v in data_producers.items()}
    _G.graph["data_consumers"] = {
        k: [v.model_name for v in vs] for k, vs in data_consumers.items()
    }

    max_min_flow_seqs = {}
    for node in nodes:
        max_min_flow_seqs[node.name] = max(
            [r.n_seqs for r in nodes if r.role == node.role]
        )
    _G.graph["max_min_flow_seqs"] = max_min_flow_seqs

    return _G
