from typing import List
import dataclasses

from profiler.device_mesh import DeviceMesh, ModelParallelStrategy

from api.config.dfg import ModelRPC


@dataclasses.dataclass
class RPC:
    """ simple RPC class for cpp search module input """
    rpc_name: str
    model_name: str
    interface_type: str

    @classmethod
    def from_config(cls, model_rpc: ModelRPC):
        return cls(rpc_name=model_rpc.name,
                   model_name=model_rpc.model_name.role,
                   interface_type=str(model_rpc.interface_type))


@dataclasses.dataclass
class RPCExecution:
    rpc: RPC
    device_mesh: DeviceMesh
    parallel_strategy: ModelParallelStrategy
    time_cost: int = None
    mem: int = None
    static_mem: int = None
    rpc_name: str = None

    def __post_init__(self):
        self.rpc_name = self.rpc.rpc_name

    def __hash__(self):
        return hash((self.rpc_name, self.device_mesh, self.parallel_strategy))


@dataclasses.dataclass
class RPCInstance:
    rpc: RPC
    epoch_id: int
    parents: List[RPC]
    children: List[RPC]
    name: str = None

    def __post_init__(self):
        self.name = f"{self.rpc.rpc_name}:{self.epoch_id}"

    def __repr__(self):
        if len(self.parents) == 0 and len(self.children) == 0:
            return f"RPCInstance({self.rpc.rpc_name}, {self.epoch_id})"
        else:
            return f"RPCInstance({self.rpc.rpc_name}, {self.epoch_id}, "\
                   f"{self.parents}, {self.children})"

    def __hash__(self):
        return hash((self.rpc.rpc_name, self.epoch_id))


@dataclasses.dataclass
class CommStats:
    local_send: int
    local_recv: int
    remote_send: int
    remote_recv: int
    offload_store: int
    offload_load: int
