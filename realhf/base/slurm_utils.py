import os
import re
import subprocess
from typing import List

import numpy as np


def parse_node_id(node_name: str, prefix: str) -> int:
    return int(node_name.split(prefix)[-1])


def parse_nodelist(nodelist: str, prefix: str) -> List[str]:
    if not nodelist.startswith(prefix):
        raise ValueError(
            f"Node list `{nodelist}` does not start with hostname prefix `{prefix}`."
        )
    nodelist = nodelist.replace(prefix, "")
    if "[" not in nodelist:
        return [prefix + nodelist]
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
        return [f"{prefix}{node_id:02d}" for node_id in node_ids]


def nodelist_from_nodes(nodes: List[str], prefix: str) -> str:
    node_ids = sorted([parse_node_id(node, prefix) for node in nodes])
    assert len(node_ids) > 0
    if len(node_ids) == 1:
        return f"{prefix}{node_ids[0]:02d}"
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
        return f"{prefix}[{','.join(node_reprs)}]"


def are_ones_contiguous(binary_array: np.ndarray):
    one_indices = np.where(binary_array == 1)[0]
    if len(one_indices) == 0:
        return False
    return np.all(np.diff(one_indices) == 1)


def slurm_hostname_key(hostname):
    """Custom sorting key function to sort Slurm hostnames."""
    # Extract node number from hostname
    match = re.match(r"(\D+)(\d+)", hostname)
    if match:
        prefix, number = match.groups()
        return (prefix, int(number))
    else:
        return (hostname,)


def check_slurm_availability():

    slurm_available = (
        int(
            subprocess.run(
                "squeue",
                shell=True,
                stdout=open(os.devnull, "wb"),
                stderr=open(os.devnull, "wb"),
            ).returncode
        )
        == 0
    )
    return slurm_available
