from typing import List, Optional, Union
import abc
import getpass
import os
import re
import socket
import tempfile


def get_user_tmp():
    tmp = tempfile.gettempdir()
    user = getpass.getuser()
    user_tmp = os.path.join(tmp, user)
    os.makedirs(user_tmp, exist_ok=True)
    return user_tmp


def get_random_tmp():
    return tempfile.mkdtemp()


class ClusterSpec(abc.ABC):

    @abc.abstractproperty
    def name(self):
        ...

    def node_type_from_node_name(self, node_name: str) -> str:
        ...

    def gpu_type_from_node_name(self, node_name: str) -> str:
        ...

    @abc.abstractproperty
    def fileroot(self) -> str:
        ...

    @abc.abstractproperty
    def default_mount(self) -> str:
        ...


class QizhiClusterSpec(ClusterSpec):

    @property
    def name(self):
        return 'qizhi'

    def node_type_from_node_name(self, node_name: str) -> str:
        if 'frl1g' in node_name:
            return 'g1'
        if 'frl2g' in node_name:
            return 'g2'
        if 'frl8g' in node_name:
            return 'g8'
        if 'frl4a' in node_name or 'frl8a' in node_name:
            return 'a100'
        else:
            raise NotImplementedError()

    def gpu_type_from_node_name(self, node_name: str) -> str:
        if 'g' in self.node_type_from_node_name(node_name):
            return 'geforce'
        else:
            return 'tesla'

    @property
    def fileroot(self) -> str:
        return "/data/aigc/llm"

    @property
    def default_mount(self) -> str:
        return "/lustre:/lustre,/data:/data,/hddlustre:/hddlustre"


class QHClusterSpec(ClusterSpec):

    @property
    def name(self):
        return 'qh'

    def node_type_from_node_name(self, node_name: str) -> str:
        assert 'QH-com' in node_name
        return 'a800'

    def gpu_type_from_node_name(self, node_name: str) -> str:
        return 'tesla'

    @property
    def fileroot(self) -> str:
        return "/lustre/aigc/llm"

    @property
    def default_mount(self) -> str:
        return "/lustre:/lustre,/dev/infiniband:/dev/infiniband,/sys/class/infiniband_verbs:/sys/class/infiniband_verbs"


class YLClusterSpec(ClusterSpec):

    @property
    def name(self):
        return 'yl'

    def node_type_from_node_name(self, node_name: str) -> str:
        assert 'YL-com' in node_name
        return 'a800'

    def gpu_type_from_node_name(self, node_name: str) -> str:
        return 'tesla'

    @property
    def fileroot(self) -> str:
        return "/data/aigc/llm"

    @property
    def default_mount(self) -> str:
        return "/data:/data"


hostname = socket.gethostname()
if not (hostname.startswith("YL-ctrl0") or hostname.startswith("QH-ctrl0") or hostname.startswith("YL-com")
        or hostname.startswith("QH-com") or hostname.startswith("frl") or hostname.startswith("ctrl0")):
    raise RuntimeError(f"Unkown cluster with hostname {hostname}. "
                       "Please properly implement methods of `ClusterSpec` in base/cluster.py.")

spec = None
if hostname.startswith("YL-ctrl0") or hostname.startswith("YL-com"):
    spec = YLClusterSpec()
elif hostname.startswith("QH-ctrl0") or hostname.startswith("QH-com"):
    spec = QHClusterSpec()
else:
    spec = QizhiClusterSpec()


def node_name_is_node_type(node_name: str, node_type: Optional[Union[List[str], str]] = None) -> bool:
    if node_type is None:
        return True
    if not isinstance(node_type, list):
        node_type = [node_type]
    nt_condition = []
    for nt in node_type:
        if nt not in ['g1', 'g2', 'g8', 'a100', 'a800']:
            raise ValueError(f"Unknown node type {nt}.")
        else:
            cond = spec.node_type_from_node_name(node_name) == nt
        nt_condition.append(cond)
    return any(nt_condition)
