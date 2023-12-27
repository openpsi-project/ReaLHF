from typing import Dict, List, Optional, Tuple, Callable
import dataclasses
import enum

import base.logging as logging
import base.namedarray as namedarray

logger = logging.getLogger("DataFlowGraph", "benchmark")


class ModelInterfaceType(enum.Enum):
    GENERATE = "generate"
    TRAIN_STEP = "train_step"
    EVALUATE = "evaluate"
    INFERENCE = "inference"


@dataclasses.dataclass
class ModelRPC:
    model_name: str
    interface_type: ModelInterfaceType
    input_data: List[str] = dataclasses.field(default_factory=lambda: [])
    input_key_remap: Dict[str, str] = dataclasses.field(default_factory=lambda: {})
    output_data: List[str] = dataclasses.field(default_factory=lambda: [])
    output_key_remap: Dict[str, str] = dataclasses.field(default_factory=lambda: {})
    dp_broker_type: str = "padded_batch"
    log_return_value: bool = False

    min_n_seqs: int = 1
    max_n_seqs: int = 1024
    min_n_tokens: int = 1
    max_n_tokens: int = 655360

    max_concurrent_calls: int = 2

    parents: List[str] = dataclasses.field(default_factory=lambda: [])
    children: List[str] = dataclasses.field(default_factory=lambda: [])

    def __post_init__(self):
        if self.min_n_seqs >= self.max_n_seqs or self.min_n_tokens >= self.max_n_tokens:
            raise RuntimeError("Invalid min/max n_seqs/n_tokens.")
        if self.is_src and self.max_n_seqs > 1e4 and self.max_n_tokens > 1e8:
            raise RuntimeError(
                "The maximum batch size of the source node in the dataflow graph is too large. "
                f"The maximum number of sequences is {self.max_n_seqs} > budget {int(1e4)} and "
                f"the maximum number of tokens is {self.max_n_tokens} > budget {int(1e8)}. "
                "Please set a smaller value."
            )

    @property
    def name(self):
        return f"{self.model_name}_{self.interface_type.value}"

    @property
    def is_src(self):
        return len(self.parents) == 0

    @property
    def is_dst(self):
        return len(self.children) == 0

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


def build_graph(rpcs: List[ModelRPC]) -> Tuple[List[ModelRPC], List[List[Tuple[str]]]]:
    # Resolve dependencies between model interfaces.
    children: List[List[str]] = [[] for _ in rpcs]
    parents: List[List[str]] = [[] for _ in rpcs]
    edges: List[List[Tuple[str]]] = [[() for _ in rpcs] for _ in rpcs]

    required_data_entries: List[Tuple[str]] = [() for _ in rpcs]
    generated_data_entries: List[Tuple[str]] = [() for _ in rpcs]
    for i, rpc in enumerate(rpcs):
        required_data_entries[i] = (*required_data_entries[i], *rpc.input_data)
        generated_data_entries[i] = (
            *generated_data_entries[i],
            *[k if k not in rpc.output_key_remap else rpc.output_key_remap[k] for k in rpc.output_data],
        )

    for i, rpc in enumerate(rpcs):
        for k in required_data_entries[i]:
            for j, parent_rpc in enumerate(rpcs):
                if parent_rpc.name == rpc.name:
                    continue
                if k in generated_data_entries[j]:
                    if parent_rpc.name not in parents[i]:
                        parents[i].append(parent_rpc.name)
                    if rpc.name not in children[j]:
                        children[j].append(rpc.name)
                    edges[i][j] = (*edges[i][j], k)
                    logger.info(
                        f"Dependency added: {rpc.name} <- {parent_rpc.name} because of data entry `{k}`."
                    )
    for i, rpc in enumerate(rpcs):
        logger.info(
            f"Dependency: {rpc.name} <- { {x.name: deps for x, deps in zip(rpcs, edges[i]) if deps} }."
        )
    for rpc, p, c in zip(rpcs, parents, children):
        rpc.parents = p
        rpc.children = c
    return rpcs, edges
