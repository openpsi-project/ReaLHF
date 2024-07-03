import getpass
import json
import os
import re
from typing import Dict, List, Optional, Union

CLUSTER_SPEC_PATH = os.environ.get("CLUSTER_SPEC_PATH", "")


def get_user_tmp():
    user = getpass.getuser()
    user_tmp = os.path.join("/home", user, ".cache", "realhf")
    os.makedirs(user_tmp, exist_ok=True)
    return user_tmp


class ClusterSpec:
    def __init__(self):
        self.__loaded = False

    def load_spec_from_file(self, file_path: str):
        try:
            with open(file_path, "r") as f:
                spec: Dict = json.load(f)
        except FileNotFoundError:
            if file_path == "":
                spec = dict(
                    cluster_type="local",
                    cluster_name="local",
                    fileroot=get_user_tmp(),
                )
            else:
                raise FileNotFoundError(f"Cluster spec file not found: {file_path}")

        self.__cluster_type = spec["cluster_type"]
        self.__cluster_name = spec["cluster_name"]
        self.__fileroot = spec["fileroot"]
        self.__node_type_from_node_name_re = spec.get("node_type_from_node_name", None)
        self.__gpu_type_from_node_name_re = spec.get("gpu_type_from_node_name", None)
        self.__default_mount = spec.get("default_mount", None)
        self.__gpu_image = spec.get("gpu_image", None)
        self.__cpu_image = spec.get("cpu_image", None)
        self.__node_name_prefix = spec.get("node_name_prefix", "NODE")

        self.__loaded = True

    @property
    def name(self):
        assert self.__loaded
        return self.__cluster_name

    def node_type_from_node_name(self, node_name: str) -> str:
        """mapping nodename to slurm node type, including "g1", "g2", "g8", "a100" """
        if self.__cluster_type != "slurm":
            raise NotImplementedError(
                "Only slurm cluster uses node_type_from_node_name."
            )
        assert self.__loaded
        for regex, node_type in self.__node_type_from_node_name_re.items():
            if re.match(regex, node_name):
                return node_type
        raise NotImplementedError()

    def gpu_type_from_node_name(self, node_name: str) -> str:
        """mapping nodename to slurm GPU type, including "geforce" and "tesla" """
        if self.__cluster_type != "slurm":
            raise NotImplementedError(
                "Only slurm cluster uses gpu_type_from_node_name."
            )
        assert self.__loaded
        for regex, gpu_type in self.__gpu_type_from_node_name_re.items():
            if re.match(regex, node_name):
                return gpu_type
        raise NotImplementedError()

    @property
    def fileroot(self) -> str:
        """Return the root directory of the file system in the cluster.
        When running experiments, files such as logs, checkpoints, caches
        will be saved under this directory.
        """
        assert self.__loaded
        return self.__fileroot

    @property
    def default_mount(self) -> str:
        """Directories that should be mounted to container that runs workers."""
        assert self.__loaded
        return self.__default_mount

    @property
    def gpu_image(self) -> str:
        """Return the default image for containers of GPU workers."""
        assert self.__loaded
        return self.__gpu_image

    @property
    def cpu_image(self) -> str:
        """Return the default image for containers of CPU workers."""
        assert self.__loaded
        return self.__cpu_image

    @property
    def node_name_prefix(self) -> str:
        """Return the prefix of node names in slurm format"""
        assert self.__loaded
        return self.__node_name_prefix


def node_name_is_node_type(
    node_name: str, node_type: Optional[Union[List[str], str]] = None
) -> bool:
    assert spec is not None
    if node_type is None:
        return True
    if not isinstance(node_type, list):
        node_type = [node_type]
    nt_condition = []
    for nt in node_type:
        if nt not in ["g1", "g2", "g8", "a100"]:
            raise ValueError(f"Unknown node type {nt}.")
        else:
            cond = spec.node_type_from_node_name(node_name) == nt
        nt_condition.append(cond)
    return any(nt_condition)


spec = ClusterSpec()
spec.load_spec_from_file(CLUSTER_SPEC_PATH)
