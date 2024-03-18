import dataclasses

from profiler.device_mesh import DeviceMesh, ModelParallelStrategy

from api.dfg import ModelRPC


def model_rpc_name(model_rpc: ModelRPC):
    return f"{model_rpc.model_name}:{model_rpc.interface_type}"


class RPC:
    """ simple RPC class for cpp search module input """

    def __init__(self, model_rpc: ModelRPC):
        self.model_name = model_rpc.model_name
        self.interface_type = str(model_rpc.interface_type)
        self.rpc_name = model_rpc_name(model_rpc)


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
        self.rpc_name = model_rpc_name(self.rpc)

    def __hash__(self):
        return hash((self.rpc_name, self.device_mesh, self.parallel_strategy))
