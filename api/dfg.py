from typing import Dict, List, Optional, Tuple
import dataclasses
import enum
import logging

logger = logging.getLogger("DataFlowGraph")


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

    @property
    def name(self):
        return f"{self.model_name}_{self.interface_type.value}"


def build_graph(rpcs: List[ModelRPC]) -> Tuple[List[List[str]], List[List[Tuple[str]]]]:
    # Resolve dependencies between model interfaces.
    parents: List[List[str]] = [[] for _ in rpcs]
    edges: List[List[Tuple[str]]] = [[() for _ in rpcs] for _ in rpcs]

    required_data_entries: List[Tuple[str]] = [() for _ in rpcs]
    generated_data_entries: List[Tuple[str]] = [() for _ in rpcs]
    for i, rpc in enumerate(rpcs):
        required_data_entries[i] = (*required_data_entries[i], *rpc.input_data)
        generated_data_entries[i] = (
            *generated_data_entries[i],
            *[k if k not in rpc.output_key_remap else rpc.output_key_remap[k] for k in rpc.output_data])

    for i, rpc in enumerate(rpcs):
        for k in required_data_entries[i]:
            for j, parent_rpc in enumerate(rpcs):
                if parent_rpc.name == rpc.name:
                    continue
                if k in generated_data_entries[j]:
                    if parent_rpc.name not in parents[i]:
                        parents[i].append(parent_rpc.name)
                    edges[i][j] = (*edges[i][j], k)
                    logger.info(
                        f"Dependency added: {rpc.name} <- {parent_rpc.name} because of data entry `{k}`.")
    for i, rpc in enumerate(rpcs):
        logger.info(
            f"Dependency: {rpc.name} <- { {x.name: deps for x, deps in zip(rpcs, edges[i]) if deps} }.")
    return parents, edges