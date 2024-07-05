import collections
from typing import *

import networkx as nx
import numpy as np
from pydantic import Field
from pydantic import dataclasses as pdclasses

import realhf.base.logging as logging
import realhf.base.namedarray as namedarray
from realhf.api.core.config import ModelFamily, ModelInterfaceType, ModelName

logger = logging.getLogger("DataFlowGraph", "benchmark")


@pdclasses.dataclass
class OffloadHook:
    pass


@pdclasses.dataclass
class SyncParamHook:
    source: Optional[ModelName] = None
    target: Optional[ModelName] = None
    interval: int = 1


RPCHook = Union[OffloadHook, SyncParamHook]


@pdclasses.dataclass(unsafe_hash=True)
class MFCNode:
    """A node in the dataflow graph of the algorithm.

    MFC is the abbreviation of Model Function Call.
    This object serves as the user interface for developing
    new algorithms, so only configurable attributes are stored.
    This object will be inserted in a nx.DiGraph as nodes.
    Edges will be automatically resolved by input/output keys.

    This object does not provide enough information, e.g.,
    hooks and data transfer pairs. These additional information
    will be stored in the MFCDef object after building the graph.

    :param n_seqs: The number of sequences to be processed in a batch.
    :type n_seqs: int
    :param mfc_name: The unique identifier of this model function call.
    :type mfc_name: str
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

    n_seqs: int

    # The unique identifier of this model function call.
    mfc_name: str

    # The interface type to be used by the node (e.g., generate, train_step).
    interface_type: ModelInterfaceType

    # The model identifier to be used by the node.
    model_name: str | ModelName

    # Input and output keys, used to resolve dependencies.
    input_keys: Tuple = Field(default_factory=tuple)
    output_keys: Tuple = Field(default_factory=tuple)

    balanced_dp: bool = False
    log_return_value: bool = False

    # Only used by search.
    model_type: Optional[ModelFamily] = None
    model_path: Optional[str] = None

    @property
    def role(self):
        return (
            self.model_name.role
            if isinstance(self.model_name, ModelName)
            else self.model_name
        )


@pdclasses.dataclass(config=dict(arbitrary_types_allowed=True))
class MFCDef:
    """A model function call (MFC) object used by the workers.

    It holds to an MFCNode object and several helper attributes
    for the worker when running the graph.
    Attributes other than ``node`` are filled automatically
    after building the graph.
    """

    node: MFCNode

    G: nx.DiGraph = None
    _pre_hooks: List[RPCHook] = Field(default_factory=lambda: [])
    _post_hooks: List[RPCHook] = Field(default_factory=lambda: [])

    def __repr__(self):
        return f"MFCDef[{self.node.mfc_name}]"

    def add_pre_hook(self, h: RPCHook):
        assert isinstance(h, RPCHook), type(h)
        if isinstance(h, SyncParamHook):
            assert h.target == self.node.model_name or h.source == self.node.model_name
        if isinstance(h, OffloadHook):
            raise ValueError("Offload can only be post hooks!")
        self._pre_hooks.append(h)

    def add_post_hook(self, h: RPCHook):
        if isinstance(h, SyncParamHook):
            assert h.target == self.node.model_name or h.source == self.node.model_name
        self._post_hooks.append(h)

    @property
    def max_min_flow_seqs(self) -> int:
        return self.G.graph["max_min_flow_seqs"][self.node.role]

    @property
    def is_src(self):
        return len(self.G.predecessors(self.node)) == 0

    @property
    def is_dst(self):
        return len(self.G.successors(self.node)) == 0

    @property
    def data_producers(self) -> Dict[str, ModelName]:
        return self.G.graph["data_producers"]

    @property
    def data_consumers(self) -> Dict[str, List[str]]:
        return self.G.graph["data_consumers"]

    @property
    def parents(self) -> List["MFCDef"]:
        return list(self.G.predecessors(self.node))

    @property
    def children(self) -> List["MFCDef"]:
        return list(self.G.successors(self.node))

    @property
    def is_dst_of_model_role(self):

        def _has_children_of_model_name(rpc: "MFCDef", model_name: ModelName):
            if rpc.is_dst:
                return False
            return any(
                [
                    r.node.model_name.role == model_name.role
                    or _has_children_of_model_name(r, model_name)
                    for r in rpc.children
                ]
            )

        return not _has_children_of_model_name(self, self.node.model_name)


def build_graph(nodes: List[MFCNode], verbose: bool = False) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    data_producers: Dict[str, MFCNode] = {}
    data_consumers: Dict[str, List[MFCNode]] = collections.defaultdict(list)
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
            src, dst = data_producers[k], node
            if G.has_edge(src, dst):
                G[src][dst]["keys"].append(k)
            else:
                G.add_edge(data_producers[k], node, keys=[k])
    if verbose:
        for u, v, data in G.edges(data=True):
            logger.info(f"Edge: {u.mfc_name} -> {v.mfc_name} with keys {data['keys']}")

    # Store useful metadata
    G.graph["data_producers"] = data_producers
    G.graph["data_consumers"] = data_consumers

    max_min_flow_seqs = {}
    for node in nodes:
        max_min_flow_seqs[node] = max([r.n_seqs for r in nodes if r.role == node.role])
    G.graph["max_min_flow_seqs"] = max_min_flow_seqs

    return G
