import dataclasses
import enum
import os
import subprocess

import base.names


def log_path(job_name, task_name):
    root_dir = "/home"
    if task_name is None:
        return f"{root_dir}/aigc/llm/{base.names.USER_NAMESPACE}/logs/{job_name}"
    else:
        return f"{root_dir}/aigc/llm/{base.names.USER_NAMESPACE}/logs/{job_name}/{task_name}"


@dataclasses.dataclass
class SlurmResource:
    # a data class that represents a slurm resource quota
    mem: int = 0
    cpu: int = 0
    gpu_type: str = "geforce"
    gpu: int = 0

    def __mul__(self, other):
        if isinstance(other, int):
            return SlurmResource(mem=self.mem * other,
                                 cpu=self.cpu * other,
                                 gpu=self.gpu * other,
                                 gpu_type=self.gpu_type)
        else:
            raise TypeError("ResourceRequirement can only be multiplied by int.")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if isinstance(other, SlurmResource):
            # adding two resources with different gpu types
            if self.gpu_type != other.gpu_type:
                # when two gpu types are different, use the gpu type whose corresponding gpu count is non-zero
                # if both non-zero, raise error
                new_gpu_type = self.gpu_type if other.gpu_type == 0 else other.gpu_type
                if self.gpu and other.gpu:
                    raise ValueError("Cannot add two different gpu types.")
            return SlurmResource(mem=self.mem + other.mem,
                                 cpu=self.cpu + other.cpu,
                                 gpu=self.gpu + other.gpu,
                                 gpu_type=new_gpu_type)
        else:
            raise TypeError("ResourceRequirement can only add another ResourceRequirement instance.")

    def __sub__(self, other):
        if isinstance(other, SlurmResource):
            if self.gpu_type != other.gpu_type:
                new_gpu_type = self.gpu_type if other.gpu == 0 else other.gpu_type
                if self.gpu > 0 and other.gpu > 0:
                    raise ValueError("Cannot subtract two different gpu types.")
            else:
                new_gpu_type = self.gpu_type

            return SlurmResource(mem=self.mem - other.mem,
                                 cpu=self.cpu - other.cpu,
                                 gpu=self.gpu - other.gpu,
                                 gpu_type=new_gpu_type)
        else:
            raise TypeError("ResourceRequirement can only subtract another ResourceRequirement instance.")

    def __neg__(self):
        return SlurmResource(mem=-self.mem, cpu=-self.cpu, gpu=-self.gpu, gpu_type=self.gpu_type)

    def valid(self):
        # check if it is a valid resource requirement
        if self.gpu_type not in ["geforce", "tesla"]:
            return False
        if self.mem < 0 or self.cpu < 0 or self.gpu < 0:
            return False
        return True


@dataclasses.dataclass
class SlurmTaskSpecification:
    # contain all informations required for a slurm task
    job_name: str
    task_name: str
    ntasks: int
    resource_requirement: SlurmResource
    cmd: str
    container_image: str
    container_mounts: str
    env_vars: dict
    nodelist: str
    exclude: str
    hostfile: bool


def parse_output_status_line(status):
    assert status.startswith("State=")
    status = status.split(" ")[0]
    status = status.split("=")[1]
    return status.split("+")


def parse_output_tres_line(tres):
    tres = tres.split("=", maxsplit=1)[1]
    tres = tres.split(",")
    res = SlurmResource()
    if len(tres) == 0:
        return SlurmResource()
    for t in tres:
        if t.startswith("mem"):
            if t.endswith("M"):
                res.mem = int(t.split("=")[1].strip("M"))
            elif t.endswith("G"):
                res.mem = int(t.split("=")[1].strip("G")) * 1024
            else:
                raise ValueError("Unknown memory unit.")
        elif t.startswith("cpu"):
            res.cpu = int(t.split("=")[1])
        elif t.startswith("gres/gpu"):
            prefix, sgpu = t.split("=")
            if ":" in prefix:
                res.gpu_type = prefix.split(":")[1]
            else:
                res.gpu = int(sgpu)
    return res


def get_slurm_node_resources():
    # execute `scontrol show node` to get node resources
    # return a list of SlurmResource
    # o = subprocess.check_output(["scontrol", "show", "node"]).decode("utf-8")
    # nodes = o.split("\n\n")
    # all_rres = {}
    # for node in nodes:
    #     if len(node) <= 1:
    #         continue
    #     ls = node.split("\n")
    #     node_name = ls[0].split(" ")[0].split("=")[1]
    #     ctres = SlurmResource()
    #     atres = SlurmResource()
    #     for l in ls:
    #         l = l.strip("\n").strip()
    #         if l.startswith("State"):
    #             status = parse_output_status_line(l)
    #             if "DOWN" in status or "DRAIN" in status or "NOT_RESPONDING" in status:
    #                 break
    #         if l.startswith("CfgTRES"):
    #             ctres = parse_output_tres_line(l)
    #         if l.startswith("AllocTRES"):
    #             atres = parse_output_tres_line(l)
    #     if "8a" in node_name or "4a" in node_name or "slurm" in node_name:
    #         ctres.gpu_type = atres.gpu_type = "tesla"
    #     else:
    #         ctres.gpu_type = atres.gpu_type = "geforce"
    #     rres = ctres - atres
    #     if rres.valid():
    #         all_rres[node_name] = rres
    #     else:
    #         all_rres[node_name] = SlurmResource()

    # return all_rres
    return dict(slurm01=SlurmResource(mem=1000000, cpu=112, gpu_type='tesla', gpu=8),
                slurm02=SlurmResource(mem=1000000, cpu=112, gpu_type='tesla', gpu=8))


def write_hostfile(res, num_tasks, hostfile_dir):
    # only support homogeneous tasks now
    # may cause conflict when other are allocating jobs concurrently
    # greedy allocation, search for the node that has the most gpu
    n = num_tasks
    arres = get_slurm_node_resources()
    arres = sorted([(k, v) for k, v in arres.items()],
                   key=lambda x: (x[1].gpu, x[1].cpu, x[1].mem),
                   reverse=True)
    # print(arres)
    allocated = {}
    for k, v in arres:
        task_count = 0
        while n > 0:
            try:
                v = v - res
            except ValueError as e:
                # print(e)
                break
            if not v.valid():
                # print(v)
                break
            task_count += 1
            n -= 1
        allocated[k] = task_count
        # print(k, v, task_count)
    allocated = {k: v for k, v in allocated.items() if v > 0}
    if n > 0:
        raise ValueError("Not enough resources.")

    with open(hostfile_dir, "w") as f:
        for k, v in allocated.items():
            for _ in range(v):
                f.write(f"{k}\n")
    return allocated


def show_tesla():
    all_rres = get_slurm_node_resources()
    for k, v in all_rres.items():
        if v.gpu_type == "tesla":
            print(k, v)


if __name__ == "__main__":
    show_tesla()
