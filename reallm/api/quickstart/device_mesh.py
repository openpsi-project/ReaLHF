from typing import List
import dataclasses
import math

import numpy as np

from reallm.api.core.config import ModelBackend
from reallm.api.core.dfg import ModelRPC
from reallm.api.quickstart.model import ModelTrainEvalConfig, ParallelismConfig
from reallm.base.slurm_utils import are_ones_contiguous, parse_slurm_nodelist, slurm_nodelist_from_nodes
from reallm.base.topology import PipeModelDataParallelTopology


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

    def __repr__(self):
        return self.name

    def __op_assertion(self, other: "DeviceMesh"):
        assert self.global_mesh_name == other.global_mesh_name,\
              "operation only support device meshes on the same cluster nodes"
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
        """ Find sub device meshes of this device mesh with at least min_n_gpus gpus.
        Sub device meshes have following constraits:
            1. Sub device meshes have the same cluster mesh.  
            2. Sub device meshes of multiple nodes must contain consecutive nodes
               in the cluster mesh.
            3. Sub device meshes of single node must start with GPU id:
                N * n_gpus_per_node / num_gpus where N is an integer. 
            4. Sub device meshes can only be of shape 1x1, 1x2, 1x4, 1x8 or Nx8
        """
        assert self.global_mesh_name is not None and self.name is not None,\
               "Only support device meshes in slurm cluster"
        sub_names = []
        # single node
        assert self.n_gpus_per_node % min_n_gpus == 0\
               or min_n_gpus % self.n_gpus_per_node == 0
        node_names = parse_slurm_nodelist(self.name)
        n_gpus = min_n_gpus
        while n_gpus < min(self.n_gpus_per_node, np.sum(self.mapping)):
            for node_name in node_names:
                if n_gpus == self.n_gpus_per_node:
                    sub_names.append(f"{node_name}")
                else:
                    for start in range(0, self.n_gpus_per_node, n_gpus):
                        sub_names.append(f"{node_name}:{','.join(map(str, range(start, start + n_gpus)))}")
            n_gpus *= 2

        if np.sum(self.mapping) >= self.n_gpus_per_node:
            # multiple nodes
            n_nodes = len(node_names)
            for start in range(0, n_nodes):
                for end in range(start, n_nodes):
                    if (end + 1 - start) * self.n_gpus_per_node < min_n_gpus:
                        continue
                    sub_node_names = node_names[start:end + 1]
                    sub_names.append(slurm_nodelist_from_nodes(sub_node_names))

        return [make_device_mesh_from_name(self.global_mesh_name, n) for n in sub_names]

    def _is_valid_mapping(self) -> bool:
        if self.mapping.shape != (self.n_nodes, self.n_gpus_per_node):
            raise RuntimeError(f"Invalid mapping shape {self.mapping.shape} "
                               f"{self.name}")
        if not np.all(np.logical_or(self.mapping == 0, self.mapping == 1)):
            raise RuntimeError(f"Invalid mapping value {self.mapping}")

        assert math.log(self.n_gpus_per_node, 2).is_integer()

        one_node_valid_gpus = [2**i for i in range(int(math.log(self.n_gpus_per_node, 2)))]
        if self.mapping.sum() < self.n_gpus_per_node:
            if not any(self.mapping.sum() == g for g in one_node_valid_gpus):
                raise RuntimeError(f"Invalid mapping sum {self.mapping}")
        else:
            if not (self.mapping.sum() % self.n_gpus_per_node == 0 and np.all(
                    np.logical_or(self.mapping.sum(1) == self.n_gpus_per_node,
                                  self.mapping.sum(1) == 0))):
                raise RuntimeError(f"Invalid mapping sum {self.mapping}")
        if not are_ones_contiguous(self.mapping.flatten()):
            raise RuntimeError(f"mapping devices are not contiguous {self.mapping}")
        return True


def make_device_mesh_from_name(global_mesh_name: str, name: str):
    """
    DeviceMesh name format for slurm: <slurm_nodelist>[:<gpu_ids>]
        slurm_nodelist is the name of slurm nodes the mesh is on, should follow slurm convention, 
        for example "QH-com[40-43]" or "QH-com[01,11,13-14]", if n_nodes=1, gpu_ids are
        the gpu id list delimited by comma if n_gpus < 8, for example "0,1,2,3" or "0,1".
        An example of full device mesh name in this situation is "QH-com40:0,1,2,3" 
    
    Note: cluster device mesh name must occupy entire nodes.
    """
    node_list = parse_slurm_nodelist(global_mesh_name)
    n_nodes = len(node_list)
    n_gpus_per_node = 8

    gpu_ids = None
    if ":" in name:
        name, gpu_ids = name.split(":")
        gpu_ids = list(map(int, gpu_ids.split(",")))
    node_names = parse_slurm_nodelist(name)
    mapping = np.zeros((n_nodes, n_gpus_per_node), dtype=np.int32)
    if gpu_ids is None:
        node_indices = [node_list.index(node_name) for node_name in node_names]
        mapping[node_indices, :] = 1
    else:
        assert len(node_names) == 1
        node_index = node_list.index(node_names[0])
        mapping[node_index, gpu_ids] = 1

    return DeviceMesh(n_nodes=n_nodes,
                      n_gpus_per_node=n_gpus_per_node,
                      mapping=mapping,
                      global_mesh_name=global_mesh_name,
                      name=name)


def find_parallel_strategies(device_mesh: DeviceMesh) -> List[ParallelismConfig]:
    n_gpus = np.sum(device_mesh.mapping)
    res = []
    for num_mp in [1, 2, 4, 8]:
        if n_gpus >= num_mp:
            assert n_gpus % num_mp == 0
            num_dp_pp = n_gpus // num_mp
            num_pp = 1
            while num_pp <= num_dp_pp:
                num_dp_mp = n_gpus // num_pp
                valid = (num_dp_mp in [1, 2, 4, 8] or num_dp_mp % 8 == 0)\
                        and num_dp_pp % num_pp == 0
                if valid:
                    res.append(ParallelismConfig(num_pp, num_mp, num_dp_pp // num_pp))
                num_pp += 1
    return res


@dataclasses.dataclass
class RPCAllocation:
    rpc: ModelRPC
    device_mesh: DeviceMesh
    parallel: ParallelismConfig
