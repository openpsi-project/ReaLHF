from typing import List, Optional, Union
import dataclasses
import json
import math

import numpy as np

from realhf.api.core.dfg import MFCDef
from realhf.api.quickstart.model import ParallelismConfig
from realhf.base.cluster import spec as cluster_spec
from realhf.base.slurm_utils import (
    are_ones_contiguous,
    nodelist_from_nodes,
    parse_nodelist,
)


@dataclasses.dataclass
class DeviceMesh:
    # number of total nodes, n_gpus_per_node=8
    n_nodes: int
    n_gpus_per_node: int
    # a 2D binary array of current device mesh name
    # shape: (n_nodes, n_gpus_per_node)
    mapping: np.ndarray
    # For slurm cluster: nodelist string of all
    # allocated nodes in the cluster
    global_mesh_name: str = None
    # For slurm cluster: nodelist string this device mesh
    name: str = None
    # cluster info, GPU memory cap in bytes
    gpu_memory_capacity: int = 80 * (1024**3)

    def to_dict(self):
        return dict(
            n_nodes=self.n_nodes,
            n_gpus_per_node=self.n_gpus_per_node,
            mapping=self.mapping.tolist(),
            global_mesh_name=self.global_mesh_name,
            name=self.name,
            gpu_memory_capacity=self.gpu_memory_capacity,
        )

    @staticmethod
    def from_dict(d):
        device_mesh = DeviceMesh(**d)
        device_mesh.mapping = np.array(d["mapping"])
        return device_mesh

    def __post_init__(self):
        if self.global_mesh_name is None:
            self.global_mesh_name = (
                f"{cluster_spec.node_name_prefix}[01-{self.n_nodes:02d}]"
                if self.n_nodes > 1
                else f"{cluster_spec.node_name_prefix}01"
            )

        if self.global_mesh_name is not None and self.name is None:
            self.name = device_mesh_name_from_mapping(
                self.global_mesh_name, self.mapping
            )

    def __eq__(self, other: "DeviceMesh"):
        assert (
            self.global_mesh_name is None
            or self.global_mesh_name == other.global_mesh_name
        ), "Only device meshes that on the same cluster mesh is comparable"
        return np.all(self.mapping == other.mapping)

    def __repr__(self):
        return f"DeviceMesh({self.name} in {self.global_mesh_name})"

    def __op_assertion(self, other: "DeviceMesh"):
        assert (
            self.global_mesh_name is None
            or self.global_mesh_name == other.global_mesh_name
        ), "operation only support device meshes on the same cluster nodes"
        assert self.n_nodes == other.n_nodes
        assert self.n_gpus_per_node == other.n_gpus_per_node

    def overlap(self, other: "DeviceMesh") -> bool:
        self.__op_assertion(other)
        return np.any(self.mapping & other.mapping)

    def contain(self, other: "DeviceMesh") -> bool:
        self.__op_assertion(other)
        return np.all(self.mapping & other.mapping == self.mapping)

    def contained_by(self, other: "DeviceMesh") -> bool:
        self.__op_assertion(other)
        return np.all(self.mapping & other.mapping == other.mapping)

    def sub_device_meshes(self, min_n_gpus: int = 4) -> List["DeviceMesh"]:
        """Find sub device meshes of this device mesh with at least min_n_gpus gpus.
        Sub device meshes have following constraints:
            1. Sub device meshes have the same cluster mesh.
            2. Sub device meshes of multiple nodes must contain consecutive nodes
               in the cluster mesh.
            3. Sub device meshes can only be of shape 1x1, 1x2, 1x4, 1x8 or Nx8
            4. If sub device meshes are of shape 1x2 or 1x4, the start GPU id
               must be 0, 2, 4, 6 for 1x2 and 0, 4 for 1x4.
        """
        sub_mappings = []
        rows, cols = np.where(self.mapping == 1)

        unique_rows = np.unique(rows)
        for row in unique_rows:
            this_cols = cols[rows == row]
            # print(row, this_cols)
            assert (
                self.n_gpus_per_node % min_n_gpus == 0
                or min_n_gpus % self.n_gpus_per_node == 0
            )
            n_gpus = min_n_gpus
            while n_gpus < min(self.n_gpus_per_node, np.sum(self.mapping)):
                for start in range(
                    np.min(this_cols), self.n_gpus_per_node, n_gpus
                ):
                    # print(row, start, start+n_gpus)
                    sub_mapping = np.zeros(
                        (self.n_nodes, self.n_gpus_per_node), dtype=np.int32
                    )
                    sub_mapping[row, start : start + n_gpus] = 1
                    sub_mappings.append(sub_mapping)
                n_gpus *= 2

        for n_rows in range(1, len(unique_rows) + 1):
            for start in range(0, len(unique_rows) - n_rows + 1):
                sub_mapping = np.zeros(
                    (self.n_nodes, self.n_gpus_per_node), dtype=np.int32
                )
                sub_mapping[start : start + n_rows, :] = 1
                sub_mappings.append(sub_mapping)

        return [
            DeviceMesh(
                n_nodes=self.n_nodes,
                n_gpus_per_node=self.n_gpus_per_node,
                mapping=sub_mapping,
                global_mesh_name=self.global_mesh_name,
                name=device_mesh_name_from_mapping(
                    self.global_mesh_name, sub_mapping
                ),
            )
            for sub_mapping in sub_mappings
        ]

    def _is_valid_mapping(self) -> bool:
        if self.mapping.shape != (self.n_nodes, self.n_gpus_per_node):
            raise RuntimeError(
                f"Invalid mapping shape {self.mapping.shape} " f"{self.name}"
            )
        if not np.all(np.logical_or(self.mapping == 0, self.mapping == 1)):
            raise RuntimeError(f"Invalid mapping value {self.mapping}")

        assert math.log(self.n_gpus_per_node, 2).is_integer()

        one_node_valid_gpus = [
            2**i for i in range(int(math.log(self.n_gpus_per_node, 2)))
        ]
        if self.mapping.sum() < self.n_gpus_per_node:
            if not any(self.mapping.sum() == g for g in one_node_valid_gpus):
                raise RuntimeError(f"Invalid mapping sum {self.mapping}")
        else:
            if not (
                self.mapping.sum() % self.n_gpus_per_node == 0
                and np.all(
                    np.logical_or(
                        self.mapping.sum(1) == self.n_gpus_per_node,
                        self.mapping.sum(1) == 0,
                    )
                )
            ):
                raise RuntimeError(f"Invalid mapping sum {self.mapping}")
        if not are_ones_contiguous(self.mapping.flatten()):
            raise RuntimeError(
                f"mapping devices are not contiguous {self.mapping}"
            )
        return True


def make_device_mesh_from_name(global_mesh_name: str, name: str):
    """
    DeviceMesh name format: <prefix><node_indices>[:<gpu_ids>]
        slurm_nodelist is the name of slurm nodes the mesh is on, should follow slurm convention,
        for example "NODE[40-43]" or "NODE[01,11,13-14]" with prefix NODE,
        if n_nodes=1, gpu_ids are the gpu id list delimited by comma if n_gpus < 8,
        for example "0,1,2,3" or "0,1". An example of full device mesh name
        in this situation is "NODE40:0,1,2,3"

    Note: cluster device mesh name must occupy entire nodes.
    """
    prefix = cluster_spec.node_name_prefix
    node_list = parse_nodelist(global_mesh_name, prefix)
    n_nodes = len(node_list)
    n_gpus_per_node = 8

    gpu_ids = None
    if ":" in name:
        node_names, gpu_ids = name.split(":")
        gpu_ids = list(map(int, gpu_ids.split(",")))
    else:
        node_names = name
    node_names = parse_nodelist(node_names, prefix)
    mapping = np.zeros((n_nodes, n_gpus_per_node), dtype=np.int32)
    if gpu_ids is None:
        node_indices = [node_list.index(node_name) for node_name in node_names]
        mapping[node_indices, :] = 1
    else:
        assert len(node_names) == 1
        node_index = node_list.index(node_names[0])
        mapping[node_index, gpu_ids] = 1

    return DeviceMesh(
        n_nodes=n_nodes,
        n_gpus_per_node=n_gpus_per_node,
        mapping=mapping,
        global_mesh_name=global_mesh_name,
        name=name,
    )


def device_mesh_name_from_mapping(global_mesh_name: str, mapping: np.ndarray):
    prefix = cluster_spec.node_name_prefix
    node_list = parse_nodelist(global_mesh_name, prefix)
    n_nodes = len(node_list)
    n_gpus_per_node = 8
    assert mapping.shape == (n_nodes, n_gpus_per_node)
    node_indices, gpu_ids = np.where(mapping == 1)

    if np.sum(mapping) < 8:
        node_name = node_list[node_indices[0]]
        gpu_ids = list(map(str, gpu_ids))
        return f"{node_name}:{','.join(gpu_ids)}"
    else:
        unique_node_indices = np.unique(node_indices)
        sub_node_list = [node_list[i] for i in unique_node_indices]
        node_name = nodelist_from_nodes(sub_node_list, prefix)
        return node_name


def find_parallel_strategies(
    device_mesh: DeviceMesh,
) -> List[ParallelismConfig]:
    n_gpus = np.sum(device_mesh.mapping)
    res = []
    for num_mp in [1, 2, 4, 8]:
        if n_gpus >= num_mp:
            assert n_gpus % num_mp == 0
            num_dp_pp = n_gpus // num_mp
            num_pp = 1
            while num_pp <= num_dp_pp:
                num_dp_mp = n_gpus // num_pp
                valid = (
                    num_dp_mp in [1, 2, 4, 8] or num_dp_mp % 8 == 0
                ) and num_dp_pp % num_pp == 0
                if valid:
                    res.append(
                        ParallelismConfig(num_pp, num_mp, num_dp_pp // num_pp)
                    )
                num_pp += 1
    return res


@dataclasses.dataclass
class RPCAllocation:
    rpc: Union[MFCDef, str]
    device_mesh: DeviceMesh
    parallel: ParallelismConfig

    def __post_init__(self):
        world_size = (
            self.parallel.model_parallel_size
            * self.parallel.pipeline_parallel_size
            * self.parallel.data_parallel_size
        )
        assert world_size == self.device_mesh.mapping.sum(), (
            "World size of ParallelismConfig does not match number of GPUs in device mesh"
            f"world_size {world_size} != n GPUs {self.device_mesh.mapping.sum()}"
        )

    def to_dict(self):
        return dict(
            rpc=self.rpc.name,
            device_mesh=self.device_mesh.to_dict(),
            parallel=dataclasses.asdict(self.parallel),
        )

    @staticmethod
    def from_dict(d):
        return RPCAllocation(
            rpc=d["rpc"],
            device_mesh=DeviceMesh.from_dict(d["device_mesh"]),
            parallel=ParallelismConfig(**d["parallel"]),
        )


@dataclasses.dataclass
class AllocationConfig:
    """Allocation configuration for one RPC, used for manual allocation.

    :param parallel: Parallelism strategy configuration.
    :type parallel: ParallelismConfig
    :param device_mesh: String representation for device mesh.
        If it is composed of multiple nodes, it should be in the form of
        slurm nodelist, e.g., node[01-02] or node01,node02.
        If it is a slice on a single node,
        we restrict it occupies 1, 2, 4, or 8 contiguous GPUs in the node.
        In this case, the string representation is similar to the MPI hostfile, e.g.,
        "node01:0-3" for the first 4 GPUs on node01.
    :type device_mesh: str
    """

    parallel: ParallelismConfig = dataclasses.field(
        default_factory=ParallelismConfig
    )
    device_mesh: Optional[str] = None
