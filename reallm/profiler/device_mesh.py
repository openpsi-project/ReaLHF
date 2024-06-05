from typing import Dict, List, Optional
import dataclasses

import numpy as np

from reallm.api.core.dfg import ModelRPC
from reallm.api.quickstart.device_mesh import ClusterDeviceMesh, RPCAllocation
from reallm.api.quickstart.model import ModelTrainEvalConfig, ParallelismConfig


@dataclasses.dataclass
class ModelParallelStrategy:
    num_pp: int
    num_mp: int
    num_dp: int

    def __repr__(self):
        return f"pp{self.num_pp}_mp{self.num_mp}_dp{self.num_dp}"

    @classmethod
    def from_config(cls, parallel: ParallelismConfig):
        return cls(
            num_pp=parallel.pipeline_parallel_size,
            num_mp=parallel.model_parallel_size,
            num_dp=parallel.data_parallel_size,
        )

    def to_config(self, use_sequence_parallel: bool):
        return ParallelismConfig(
            pipeline_parallel_size=self.num_pp,
            model_parallel_size=self.num_mp,
            data_parallel_size=self.num_dp,
            use_sequence_parallel=use_sequence_parallel,
        )


@dataclasses.dataclass
class DeviceMesh:
    """
    Args:
        device_mesh_name: str, naming rule: <slurm_nodelist>[:<gpu_ids>], slurm_nodelist
            is the name of slurm nodes the mesh is on, should follow slurm convention,
            for example "QH-com[40-43]" or "QH-com[01,11,13-14]", if n_nodes=1, gpu_ids are
            the gpu id list delimited by comma if n_gpus < 8, for example "0,1,2,3" or "0,1".
            An example of full device mesh name in this situation is "QH-com40:0,1,2,3"
        n_nodes: int
        n_gpus: int
        node_names: List[str]
        gpu_ids: Optional[List[int]] = None
        n_gpus_per_node: int = 8
    """

    device_mesh_name: str
    n_nodes: int
    n_gpus: int
    node_names: List[str]
    gpu_ids: Optional[List[int]] = None
    n_gpus_per_node: int = 8

    def __post_init__(self):
        assert len(self.node_names) == self.n_nodes
        if self.n_nodes == 1:
            assert self.n_gpus in [1, 2, 4, 8]
        else:
            assert self.n_gpus == self.n_nodes * self.n_gpus_per_node

    def __repr__(self):
        return self.device_mesh_name

    def to_config(
        self,
        device_mesh: "DeviceMesh",
        train_eval_config: ModelTrainEvalConfig,
        rpc: ModelRPC,
    ) -> RPCAllocation:
        """
        args:
            device_mesh: DeviceMesh, the entire device mesh of the experiment,
                         must contain this device mesh and contain whole nodes.
            parallel_strategy: model parallel strategy
            rpc: ModelRPC
            kwargs: additional arguments for ModelTrainEvalConfig
        """
        # ps = parallel_strategy.to_config()
        assert device_mesh.n_gpus_per_node == 8
        mapping = np.zeros((device_mesh.n_nodes, device_mesh.n_gpus_per_node), dtype=np.int32)

        # print(device_mesh.node_names, self.node_names)
        node_indices = [device_mesh.node_names.index(node_name) for node_name in self.node_names]
        gpu_indices = self.gpu_ids

        if self.n_gpus_per_node == 8:
            for node_index in node_indices:
                mapping[node_index] = 1
        else:
            assert len(gpu_indices) in [1, 2, 4]
            assert len(node_indices) == 1
            for node_index in node_indices:
                for gpu_index in gpu_indices:
                    mapping[node_index, gpu_index] = 1

        return RPCAllocation(rpc=rpc, mapping=mapping, train_eval_config=train_eval_config)


def is_overlap(device_mesh: DeviceMesh, other: DeviceMesh):
    if device_mesh.n_nodes > 1:
        return any([node_name in device_mesh.node_names for node_name in other.node_names])
    else:
        if device_mesh.node_names[0] not in other.device_mesh_name:
            return False
        if other.gpu_ids is None:
            other.gpu_ids = list(range(8))
        return any([gpu_id in device_mesh.gpu_ids for gpu_id in other.gpu_ids])


def is_contain(device_mesh: DeviceMesh, other: DeviceMesh):
    # other contain device mesh
    if device_mesh.n_nodes > 1:
        return all([node_name in device_mesh.node_names for node_name in other.node_names])
    else:
        if device_mesh.node_names[0] not in other.device_mesh_name:
            return False
        if other.gpu_ids is None:
            other.gpu_ids = list(range(8))
        return all([gpu_id in device_mesh.gpu_ids for gpu_id in other.gpu_ids])


def is_all_overlap(device_meshes: List[DeviceMesh], other: DeviceMesh):
    for i in range(len(device_meshes)):
        if not is_overlap(device_meshes[i], other):
            return False
    return True


def parse_node_id(node_name: str) -> int:
    return int(node_name.split("com")[-1])


def parse_slurm_nodelist(nodelist: str) -> List[str]:
    nodelist = nodelist.replace("QH-com", "")
    if "[" not in nodelist:
        return ["QH-com" + nodelist]
    else:
        nodelist = nodelist.strip("[]")
        node_ids = []
        nodelist = nodelist.split(",")
        for node_repr in nodelist:
            if "-" not in node_repr:
                node_ids.append(int(node_repr))
            else:
                start, end = map(int, node_repr.split("-"))
                node_ids += list(range(start, end + 1))
        return [f"QH-com{node_id:02d}" for node_id in node_ids]


def make_device_mesh_from_name(name: str):
    if ":" in name:
        node_name, gpu_ids = name.split(":")
        gpu_ids = list(map(int, gpu_ids.split(",")))
        return DeviceMesh(
            device_mesh_name=name,
            n_nodes=1,
            n_gpus=len(gpu_ids),
            node_names=[node_name],
            gpu_ids=gpu_ids,
            n_gpus_per_node=len(gpu_ids),
        )
    else:
        node_names = parse_slurm_nodelist(name)
        print(node_names)
        n_nodes = len(node_names)
        return DeviceMesh(
            device_mesh_name=name,
            n_nodes=n_nodes,
            n_gpus=n_nodes * 8,
            node_names=node_names,
            gpu_ids=list(range(8)),
            n_gpus_per_node=8,
        )


def slurm_nodelist_from_nodes(nodes: List[str]) -> str:
    node_ids = sorted([parse_node_id(node) for node in nodes])
    assert len(node_ids) > 0
    if len(node_ids) == 1:
        return f"QH-com{node_ids[0]:02d}"
    else:
        node_reprs = []
        start, end = node_ids[0], node_ids[0]
        for i in range(len(node_ids)):
            node_id = node_ids[i]
            next_node_id = node_ids[i + 1] if i + 1 < len(node_ids) else -1
            if node_id + 1 == next_node_id:
                end = next_node_id
            else:
                if start == end:
                    node_reprs.append(f"{start:02d}")
                else:
                    node_reprs.append(f"{start:02d}-{end:02d}")
                start = next_node_id
                end = next_node_id
        return f"QH-com[{','.join(node_reprs)}]"


def find_sub_device_meshes(device_mesh: DeviceMesh) -> List[DeviceMesh]:
    # In full device mesh NXM (M=8), sub-device mesh for models can only be 1x1, 1x2, 1x4, 1x8
    # or nx8 where n <= N, and in the same mesh, nodes should be consecutive
    # sub device meshes of single node device_mesh
    # this function only supports finding sub device meshes of integer number of nodes
    assert device_mesh.n_gpus_per_node == 8
    if device_mesh.n_nodes == 1:
        return [
            DeviceMesh(
                device_mesh_name=f"{device_mesh.node_names[0]}:"
                f"{gpu_id},{gpu_id+1},{gpu_id+2},{gpu_id+3}",
                n_nodes=1,
                n_gpus=4,
                node_names=device_mesh.node_names,
                gpu_ids=[gpu_id + i for i in range(4)],
            ) for gpu_id in [0, 4]
        ] + [device_mesh]
    # if device_mesh.n_gpus == 1:
    #     return [device_mesh]
    # elif device_mesh.n_gpus == 2:
    #     return [
    #         DeviceMesh(device_mesh_name=f"{device_mesh.node_names[0]}:{gpu_id}",
    #                    n_nodes=1,
    #                    n_gpus=1,
    #                    node_names=device_mesh.node_names,
    #                    gpu_ids=[gpu_id]) for gpu_id in device_mesh.gpu_ids
    #     ] + [device_mesh]
    # elif device_mesh.n_gpus == 4:
    #     return [
    #         DeviceMesh(device_mesh_name=f"{device_mesh.node_names[0]}:{gpu_id},{gpu_id+1}",
    #                    n_nodes=1,
    #                    n_gpus=2,
    #                    node_names=device_mesh.node_names,
    #                    gpu_ids=[gpu_id, gpu_id + 1]) for gpu_id in device_mesh.gpu_ids[:-1]
    #     ] + [
    #         DeviceMesh(device_mesh_name=f"{device_mesh.node_names[0]}:{gpu_id}",
    #                    n_nodes=1,
    #                    n_gpus=1,
    #                    node_names=device_mesh.node_names,
    #                    gpu_ids=[gpu_id]) for gpu_id in device_mesh.gpu_ids
    #     ] + [device_mesh]
    # elif device_mesh.n_gpus == 8:
    #     return [
    #         DeviceMesh(device_mesh_name=f"{device_mesh.node_names[0]}:"
    #                     f"{gpu_id},{gpu_id+1},{gpu_id+2},{gpu_id+3}",
    #                     n_nodes=1,
    #                     n_gpus=4,
    #                     node_names=device_mesh.node_names,
    #                     gpu_ids=[gpu_id + i for i in range(4)]) for gpu_id in [0, 4]
    #     ] + [
    #         DeviceMesh(device_mesh_name=f"{device_mesh.node_names[0]}:{gpu_id},{gpu_id+1}",
    #                     n_nodes=1,
    #                     n_gpus=2,
    #                     node_names=device_mesh.node_names,
    #                     gpu_ids=[gpu_id, gpu_id + 1]) for gpu_id in [0, 2, 4, 6]
    #     ] + [
    #         DeviceMesh(device_mesh_name=f"{device_mesh.node_names[0]}:{gpu_id}",
    #                     n_nodes=1,
    #                     n_gpus=1,
    #                     node_names=device_mesh.node_names,
    #                     gpu_ids=[gpu_id]) for gpu_id in device_mesh.gpu_ids
    #     ] + [device_mesh]

    # single node meshes
    res = []
    for node in device_mesh.node_names:
        res += find_sub_device_meshes(
            DeviceMesh(
                device_mesh_name=f"{node}",
                n_nodes=1,
                n_gpus=8,
                node_names=[node],
                gpu_ids=list(range(8)),
            ))

    # multi-node meshes
    node_ids = sorted([parse_node_id(node) for node in device_mesh.node_names])
    for i in range(2, device_mesh.n_nodes):
        for j in range(device_mesh.n_nodes - i + 1):
            sub_mesh_node_ids = node_ids[j:j + i]
            node_names = [f"QH-com{node_id:02d}" for node_id in sub_mesh_node_ids]
            res.append(
                DeviceMesh(
                    device_mesh_name=slurm_nodelist_from_nodes(node_names),
                    n_nodes=i,
                    n_gpus=8 * i,
                    node_names=node_names,
                    gpu_ids=list(range(8)),
                ))

    res += [device_mesh]
    return res


def find_parallel_strategies(device_mesh: DeviceMesh) -> List[ModelParallelStrategy]:
    n_gpus = device_mesh.n_gpus
    res = []
    for num_mp in [1, 2, 4, 8]:
        if n_gpus >= num_mp:
            assert n_gpus % num_mp == 0
            num_dp_pp = n_gpus // num_mp
            num_pp = 1
            while num_pp <= num_dp_pp:
                num_dp_mp = n_gpus // num_pp
                valid = ((num_dp_mp in [1, 2, 4, 8] or num_dp_mp % 8 == 0) and num_dp_pp % num_pp == 0
                         and num_pp <= 16)
                if valid:
                    res.append(ModelParallelStrategy(num_pp, num_mp, num_dp_pp // num_pp))
                num_pp += 1

    return res


if __name__ == "__main__":
    device_mesh_name = "QH-com[01,03,17-21,23]"
    device_mesh = make_device_mesh_from_name(device_mesh_name)
    sub_device_meshes = find_sub_device_meshes(device_mesh)
    print([m.device_mesh_name for m in sub_device_meshes], len(sub_device_meshes))
    num_model_device_mapping = 0
    parallel_strategies = find_parallel_strategies(device_mesh)
    print(parallel_strategies, len(parallel_strategies))
    for device_mesh in sub_device_meshes:
        num_model_device_mapping += len(find_parallel_strategies(device_mesh))
    print(num_model_device_mapping)
    print(num_model_device_mapping**6)
