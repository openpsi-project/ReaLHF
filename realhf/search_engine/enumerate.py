from typing import List

from realhf.api.core.dfg import MFCDef, ModelInterfaceType
from realhf.api.core.dfg import build_graph as build_dfg
from realhf.api.quickstart.device_mesh import DeviceMesh, find_parallel_strategies
from realhf.api.quickstart.search import MFCDef, RPCExecution, RPCInstance
from realhf.search_engine.estimate import (
    estimate_rpc_memory_cost,
    estimate_rpc_time_cost,
)

MEM_INDEX = 1.0  # heuristic value to scale estimated memory


def enumerate_rpc_executions(
    rpc: MFCDef,
    device_mesh: DeviceMesh,
    seq_len: int,
    num_gen_tokens: int,
    n_ppo_minibatches: int,
    gradient_checkpointing: bool,
) -> List[RPCExecution]:
    sub_device_meshes = device_mesh.sub_device_meshes()
    import pprint

    feasible = []
    for sub_device_mesh in sub_device_meshes:
        ps = find_parallel_strategies(sub_device_mesh)
        for parallel in ps:
            num_dp = parallel.data_parallel_size
            num_pp = parallel.pipeline_parallel_size
            num_mp = parallel.model_parallel_size
            bs = rpc.n_seqs
            # seq_len = seq_len
            min_bs = (
                2 * num_dp * num_pp * n_ppo_minibatches
                if rpc.interface_type == ModelInterfaceType.TRAIN_STEP
                else num_dp * num_pp
            )
            if min_bs > bs:
                # batch size too small
                continue
            # heuristic to filter out inherent slow configurations
            if (
                num_mp * num_dp > device_mesh.n_gpus_per_node
                and rpc.interface_type == ModelInterfaceType.TRAIN_STEP
            ):
                continue
            if num_mp > 8:
                continue
            if num_pp > max(device_mesh.n_nodes, 8):
                continue
            # memory and time estimation
            mem_cost, static_mem = estimate_rpc_memory_cost(
                rpc,
                parallel,
                bs,
                seq_len,
                gradient_checkpointing=gradient_checkpointing,
                n_ppo_minibatches=n_ppo_minibatches,
                num_gen_tokens=num_gen_tokens,
                offload=rpc.model_name.role in ["ref", "reward"],
            )
            mem_cost = int(mem_cost * MEM_INDEX)
            static_mem = int(static_mem * MEM_INDEX)
            time_cost = estimate_rpc_time_cost(
                rpc,
                parallel,
                bs=bs,
                seq_len=seq_len,
                num_gen_tokens=num_gen_tokens,
                gradient_checkpointing=gradient_checkpointing,
                n_ppo_minibatches=n_ppo_minibatches,
            )
            time_cost = int(time_cost)
            if mem_cost < device_mesh.gpu_memory_capacity:
                feasible.append(
                    RPCExecution(
                        rpc,
                        sub_device_mesh,
                        parallel,
                        time_cost,
                        mem_cost,
                        static_mem,
                    )
                )
    return feasible


def build_graph(
    rpcs: List[MFCDef],
    num_epoch: int = 5,
    epoch_dependency_interval: int = 1,
    if_print=False,
) -> List[RPCInstance]:
    """Build model function call graph of multiple training epochs,

    args:
        exp: ProfileExperiment, the experiment object
        num_epoch: int, number of training epochs
        epoch_dependency_interval: int, the interval of epoch dependency,
            e.g. if epoch_dependency_interval = 2, then the graph will have
            edges between epoch i and epoch i+2, i+4, ...

    returns:
        rpc_instances: List[RPCInstance], the list of RPCInstance objects
    """
    # one epoch dependency graph
    rpcs, edges = build_dfg(rpcs)
    rpc_names_mapping = {rpc.name: rpc for rpc in rpcs}
    rpc_instances = []

    # multi epoch graph
    for epoch_id in range(num_epoch):
        for rpc in rpcs:
            children = []
            parents = []
            if rpc.is_src and epoch_id >= epoch_dependency_interval:
                for other in rpcs:
                    if other.is_dst and other.model_name.role == rpc.model_name.role:
                        parents.append(
                            RPCInstance(
                                rpc,
                                epoch_id - epoch_dependency_interval,
                                [],
                                [],
                            )
                        )
            if rpc.is_dst and rpc.model_name.role == rpc.model_name.role:
                for other in rpcs:
                    if (
                        other.is_src
                        and epoch_id + epoch_dependency_interval < num_epoch
                    ):
                        children.append(
                            RPCInstance(
                                rpc,
                                epoch_id + epoch_dependency_interval,
                                [],
                                [],
                            )
                        )
            for parent in rpc.parents:
                p = rpc_names_mapping[parent]
                parents.append(RPCInstance(p, epoch_id, [], []))
            for child in rpc.children:
                c = rpc_names_mapping[child]
                children.append(RPCInstance(c, epoch_id, [], []))
            rpc_instance = RPCInstance(rpc, epoch_id, parents, children)
            rpc_instances.append(rpc_instance)
    if if_print:
        for ri in rpc_instances:
            print(ri)
    return rpc_instances
