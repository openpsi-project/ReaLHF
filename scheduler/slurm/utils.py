from __future__ import annotations  # python3.7+ feature to allow self-referencing type hints

from typing import Callable, Dict, List, Literal, Optional, Union
import dataclasses
import datetime
import logging
import math
import os
import shutil
import subprocess

from scheduler.client import TaskException, TaskInfo, TaskState
import base.names

logger = logging.getLogger("scheduler.slurm.utils")

SQUEUE_FIELDS = [
    "JobID",
    "State",
    "SubmitTime",
    "StartTime",
    "Name",
    "NodeList",
    "UserName",
    "MaxCPUs",
    "cpus-per-task",
    "NumTasks",
    "tres-alloc",
]
STATUS_MAPPING = {
    "RUNNING": TaskState.RUNNING,
    "COMPLETING": TaskState.RUNNING,
    "PENDING": TaskState.PENDING,
    "CANCELLED": TaskState.CANCELLED,
    "FAILED": TaskState.FAILED,
    "COMPLETED": TaskState.COMPLETED,
    "OUT_OF_MEMORY": TaskState.FAILED,
    "DEADLINE": TaskState.COMPLETED,
    "TIMEOUT": TaskState.COMPLETED
}
LOG_BASE_PATH = "/data/aigc/llm/logs/"


class SlurmResourceNotEnoughException(Exception):
    pass


# dataclasses


@dataclasses.dataclass
class SlurmResource:
    # a data class that represents a slurm resource quota
    mem: int = 0
    cpu: int = 0
    gpu_type: Optional[Literal["tesla", "geforce"]] = None
    gpu: Union[float, int] = 0

    def __check_gpu_type(self, other: SlurmResource) -> str:
        self_gpu_type = None if self.gpu == 0 else self.gpu_type
        other_gpu_type = None if other.gpu == 0 else other.gpu_type
        valid_gpu_type = self_gpu_type == other_gpu_type or (self_gpu_type or other_gpu_type)
        assert valid_gpu_type, f"Cannot add two different gpu types {self_gpu_type}, {other_gpu_type}."
        return self_gpu_type if self_gpu_type else other_gpu_type

    def __str__(self):
        return "SlurmResource: \n" + \
               "mem: " + str(self.mem) + " MB \n" + \
               "cpu: " + str(self.cpu) + " \n" + \
               "gpu: " + str(self.gpu) + " \n" + \
               "gpu_type: " + str(self.gpu_type)

    def __mul__(self, other: int) -> SlurmResource:
        assert isinstance(other, int), "ResourceRequirement can only be multiplied by int."
        return SlurmResource(mem=self.mem * other,
                             cpu=self.cpu * other,
                             gpu=self.gpu * other,
                             gpu_type=self.gpu_type)

    def __rmul__(self, other: int) -> SlurmResource:
        return self.__mul__(other)

    def __add__(self, other: SlurmResource) -> SlurmResource:
        assert isinstance(other, SlurmResource), "SlurmResource can only add another SlurmResource instance."
        return SlurmResource(mem=self.mem + other.mem,
                             cpu=self.cpu + other.cpu,
                             gpu=self.gpu + other.gpu,
                             gpu_type=self.__check_gpu_type(other))

    def __sub__(self, other: SlurmResource) -> SlurmResource:
        assert isinstance(other,
                          SlurmResource), "SlurmResource can only subtract another SlurmResource instance."
        return SlurmResource(mem=self.mem - other.mem,
                             cpu=self.cpu - other.cpu,
                             gpu=self.gpu - other.gpu,
                             gpu_type=self.__check_gpu_type(other))

    def __neg__(self) -> SlurmResource:
        return SlurmResource(mem=-self.mem, cpu=-self.cpu, gpu=-self.gpu, gpu_type=self.gpu_type)

    def __eq__(self, other: SlurmResource) -> bool:
        return self.mem == other.mem and \
               self.cpu == other.cpu and \
               self.gpu == other.gpu and \
               self.gpu_type == other.gpu_type

    def __lt__(self, other: SlurmResource) -> bool:
        if self.gpu_type != other.gpu_type:
            if self.gpu_type is None:
                return True
            if self.gpu_type == "geforce":
                return other.gpu_type == "tesla"
            if self.gpu_type == "tesla":
                return False
        if self.gpu != other.gpu:
            return self.gpu < other.gpu
        if self.cpu != other.cpu:
            return self.cpu < other.cpu
        if self.mem != other.mem:
            return self.mem < other.mem

    def valid(self) -> bool:
        # check if it is a valid resource requirement
        if self.gpu_type not in ["geforce", "tesla"]:
            return False
        if self.mem < 0 or self.cpu < 0 or self.gpu < 0:
            return False
        return True


@dataclasses.dataclass
class SlurmTaskInfo:
    """ Contain all informantion required to **launch** a slurm task
    Matching one TasksGroup in expr config
    """
    # common
    job_name: str
    task_name: str  # hierarchy: job_name:task_name:task_id,
    # specifically, GPU in the same communication group should share a task_name
    task_id: int  # for heterogeneous task scheduling
    ntasks: int
    resource_requirement: SlurmResource
    cmd: str
    container_image: str
    container_mounts: str
    env_vars: dict
    node_type: str
    nodelist: str
    exclude: str
    #
    partition: Optional[str] = None
    workers_per_task: int = 1
    nworkers: int = 0
    # time configs
    time_limit: Optional[str] = None
    begin: Optional[str] = None  # scheduled worker start time
    deadline: Optional[str] = None  # scheduled worker end time
    # hostfile
    hostfile: bool = True
    hostfile_content: Optional[str] = None
    # multiprog options, override cmd
    multiprog: bool = True
    multiprog_content: Optional[str] = None
    # addtional info for worker scheduling
    # submitted task infos
    task_info: Optional[TaskInfo] = None
    # state: TaskState = TaskState.NOT_FOUND
    # host: str = None  # The host on which the task is/was running. None if the task had not run.
    # real_start_time: str = None # The real start time of the task.
    # slurm_id: str = None  # The Slurm id of the task.

    @property
    def slurm_name(self) -> str:
        # unique slurm name for a task
        return f"{self.job_name}:{self.task_name}:{self.task_id}"

    @property
    def slurm_id(self) -> Optional[str]:
        if self.task_info:
            return self.task_info.slurm_id
        else:
            return None

    @property
    def log_path(self) -> str:
        return os.path.join(LOG_BASE_PATH, self.job_name, f"{self.task_name}-{self.task_id}")

    @property
    def multiprog_path(self) -> str:
        return os.path.join(LOG_BASE_PATH, self.job_name, f"{self.task_name}-{self.task_id}.multiprog")

    @property
    def hostfile_path(self) -> str:
        return os.path.join(LOG_BASE_PATH, self.job_name, f"{self.task_name}-{self.task_id}.hostfile")

    def show_log(self):
        try:
            terminal_columns = os.get_terminal_size().columns
        except OSError:
            terminal_columns = shutil.get_terminal_size().columns
        logger.info(f"Showing log of task: {self.task_name}-{self.task_id}\n\n{'-'*terminal_columns}")
        subprocess.Popen(["tail", "-n50", self.log_path]).wait(timeout=3)
        logger.info(f"End of log: {self.task_name}-{self.task_id}\n\n{'-'*terminal_columns}")

    def update(self):
        task_infos = query_tasks(slurm_names=[self.slurm_name])
        # assert len(task_infos) <= 1, "Multiple tasks with the same name, have all previous tasks been canceled?"
        # if returned multiple task_infos, pick the latest one
        task_infos = sorted(task_infos, key=lambda x: parse_formatted_time(x.submit_time), reverse=True)
        self.task_info = task_infos[0] if len(task_infos) > 0 else None
        # logger.info(f"All task infos: {task_infos}")
        # logger.info(f"Updated task info for {self.slurm_name}: {self.task_info}")

    def cancel(self):
        cancel_tasks(slurm_names=[self.slurm_name])
        self.task_info = TaskInfo(name=self.slurm_name, state=TaskState.CANCELLED)

    def resolve_gpu_requirement(self):
        """ resolve fractional GPU resource requirement
        """
        gpu_per_worker = self.resource_requirement.gpu
        assert gpu_per_worker <= 1 and gpu_per_worker >= 0
        self.nworkers = self.ntasks
        if gpu_per_worker < 1 and gpu_per_worker > 0:
            self.resource_requirement.gpu = 1
            self.workers_per_task = math.ceil(1 / gpu_per_worker)
            self.ntasks = math.ceil(self.ntasks / self.workers_per_task)
            logger.info(f"Resolved fractional GPU requirement for {self.slurm_name}")
            logger.info(f"GPU per worker {gpu_per_worker}, workers per task {self.workers_per_task}, "
                        f"ntasks {self.ntasks}")

    def __str__(self):
        s = f"SlurmTaskInfo [{self.slurm_name}] \n"
        s += f"Resources: [\n{self.resource_requirement}\n]\n"
        s += f"Multiprog Filepath: [{self.multiprog_path}]\n"
        s += f"Multiprog Content: [\n{self.multiprog_content}\n]\n"
        s += f"Hostfile Filepath: [{self.hostfile_path}]\n"
        s += f"Hostfile Content: [\n{self.hostfile_content}\n]\n"
        if self.task_info is None:
            task_info_str = "None"
        else:
            task_info_str = "\n".join([f"{k}: {v}" for k, v in self.task_info.__dict__.items()])
        s += f"Runtime TaskInfo: [\n{task_info_str}\n]\n"
        env_var_str = "\n".join([f"{k}: {v}" for k, v in self.env_vars.items()])
        s += f"Env vars: [\n{env_var_str}\n]\n"
        return s

    def commit(self):
        """ Commit a task to slurm scheduler
        """
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True, mode=0o775)

        ntasks = self.ntasks
        mem = self.resource_requirement.mem
        cpu = self.resource_requirement.cpu
        gpu = self.resource_requirement.gpu

        cmd = self.cmd

        assert gpu == 1 or gpu == 0, "Slurm task GPU requirement should be resolved to a integer."
        gpu_type = self.resource_requirement.gpu_type

        with open(self.multiprog_path, "w") as f:
            f.write(self.multiprog_content)
        with open(self.hostfile_path, "w") as f:
            f.write(self.hostfile_content)

        # reconstruct worker index in remote.py by adding env var
        # start : start_index + args.group_offset * args.task_size
        # end : start_index + min((args.group_offset + 1) * args.task_size, spec.ntasks))

        logger.info(
            f"Allocating {ntasks} task(s) \"{self.task_name}\" task id {self.task_id} with {cpu} cpu, {gpu} gpu and {mem} MB memory."
        )
        logger.info(f"To check the output, run \n\t\t\t\t\t\t`tail -f {self.log_path}`.")

        # Setup sbatch
        # head
        lines = [
            '#!/bin/bash',
            f'#SBATCH --job-name={self.slurm_name}',
            f'#SBATCH --output={self.log_path}',
            f'#SBATCH --ntasks={ntasks}',
            f'#SBATCH --gpus-per-task={gpu_type}:1' if gpu == 1 else "",
            f'#SBATCH --cpus-per-task={cpu}',
            f'#SBATCH --mem-per-cpu={mem // max(1, cpu)}M',
            f'#SBATCH --partition={self.partition}' if self.partition else "",
            "#SBATCH --distribution=arbitrary" if self.hostfile else "",
            # f'#SBATCH --nodelist={spec.nodelist}' if spec.nodelist is not None else "",
            # f'#SBATCH --exclude={spec.exclude}' if spec.exclude is not None else "",
            f"#SBATCH --time={self.time_limit}" if self.time_limit else "",
            f"#SBATCH --begin={self.begin}" if self.begin else "",
            f"#SBATCH --deadline={self.deadline}" if self.deadline else "",
        ]

        srun_env = os.environ.copy()
        if self.hostfile:
            srun_env["SLURM_HOSTFILE"] = self.hostfile_path
        # Setup step command.
        srun_flags = [
            f"--ntasks={ntasks}",
            f"--cpus-per-task={cpu}",
            f"--gpus-per-task={gpu_type}:1" if gpu == 1 else "",
            f"--mem-per-cpu={mem // max(1, cpu)}",
            f"--export={','.join(str(k)+'='+str(v) for k, v in self.env_vars.items())}"
            if self.env_vars else "",
            f"--multi-prog" if self.multiprog else "",
            f"--container-image={self.container_image}",
            f"--container-mounts={self.container_mounts}",
            f"--container-mount-home",
        ]

        if self.multiprog:
            srun_cmd = f'srun -l {" ".join(srun_flags)} {self.multiprog_path}'
        else:
            srun_cmd = f'srun -l {" ".join(srun_flags)} {cmd}'

        lines += [
            'echo "*************************************"',
            'echo "[Runner] StartTime: $(date -u)"',
            'echo "[Runner] Host: $(hostname)"',
            "echo '[Runner] Command: {}'".format(srun_cmd),
            "echo '[Runner] Log: {}'".format(self.log_path),
            'echo "[Runner] CudaVisible: $CUDA_VISIBLE_DEVICES"',
            'echo "[Runner] CudaMpsPerc: $CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"',
            srun_cmd,
            'RETCODE=$?',
            'echo "[Runner] FinishTime: $(date -u)"',
            'echo "[Runner] RetCode: $RETCODE"',
            'echo "[Runner] ------------"',
            'exit $RETCODE',
        ]

        script_strs = '\n'.join(lines)
        script = script_strs.encode('ascii')

        with open(self.log_path, "w") as f:
            f.write("=" * 20 + " SBATCH SCRIPT " + "=" * 20 + "\n")
            f.write(script_strs)
            f.write("=" * 20 + " SLURM TASK INFO " + "=" * 20 + "\n")
            f.write(str(self))
            f.write("=" * 20 + " OUTPUT " + "=" * 20 + "\n")
        r = subprocess.check_output(['sbatch', '--parsable'], input=script,
                                    env=srun_env).decode('ascii').strip()
        self.task_info = TaskInfo(name=self.slurm_name, state=TaskState.PENDING)


def parse_formatted_time(time_string: str) -> int:
    if time_string == "N/A":
        return -1
    d = datetime.datetime.strptime(time_string, "%Y-%m-%dT%H:%M:%S")
    return int(datetime.datetime.timestamp(d))


def unparse_formatted_time(timestamp: int) -> str:
    if timestamp == -1:
        return "N/A"
    d = datetime.datetime.fromtimestamp(timestamp)
    return d.strftime("%Y-%m-%dT%H:%M:%S")


# slurm command execute and output parsing
def query_tasks(slurm_names: Optional[List[str]] = None,
                slurm_ids: Optional[List[str]] = None,
                status: str = "all",
                delimiter: str = "__PSI__") -> List[TaskInfo]:
    squeue_format = f":.{delimiter},".join(SQUEUE_FIELDS)
    cmd = ["squeue", "-O", squeue_format, f"-t{status}"]
    if slurm_names is not None:
        cmd += ["-n", ",".join(slurm_names)]
    if slurm_ids is not None:
        cmd += ["-j", ",".join([str(s) for s in slurm_ids])]
    output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("ascii").strip()
    rs = []
    for line in output.split("\n")[1:]:
        job_id, state, submit_time, start_time, slurm_name, nodelist, *_ = line.split(delimiter)
        rs.append(
            TaskInfo(name=slurm_name,
                     state=STATUS_MAPPING[state],
                     host=nodelist,
                     submit_time=submit_time,
                     start_time=start_time,
                     slurm_id=job_id.strip()))
    return rs


def cancel_tasks(slurm_names: Optional[List[str]] = None, slurm_ids: Optional[List[str]] = None):
    assert slurm_names is not None or slurm_ids is not None, "Must specify slurm_names or slurm_ids."
    assert not (slurm_names and slurm_ids), "Cannot specify both slurm_names and slurm_ids."
    cmd = ["scancel"]
    if slurm_names is not None:
        cmd += ["-n", ",".join(slurm_names)]
    elif slurm_ids is not None:
        cmd += ["-j", ",".join([str(s) for s in slurm_ids])]
    subprocess.check_call(cmd)
    logger.info(f"Cancelled Slurm task: slurm identifiers {slurm_names if slurm_ids is None else slurm_ids}")


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
                res.mem = float(t.split("=")[1].strip("M"))
            elif t.endswith("G"):
                res.mem = float(t.split("=")[1].strip("G")) * 1024
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


def _parse_output_status_line(status):
    assert status.startswith("State=")
    status = status.split(" ")[0]
    status = status.split("=")[1]
    return status.split("+")


def _parse_output_tres_line(tres):
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
                res.mem = int(float(t.split("=")[1].strip("G")) * 1024)
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


def available_hostnames(
    node_type: Optional[List[str]] = None,
    nodelist: Optional[str] = None,
    exclude: Optional[str] = None,
) -> List[str]:
    all_nodelist: str = subprocess.check_output("sinfo -o \"%N\" --noheader",
                                                shell=True).decode("utf-8").strip()
    all_hostnames: List[str] = subprocess.check_output([
        'scontrol',
        'show',
        'hostnames',
        all_nodelist,
    ]).decode('utf-8').strip().split('\n')

    if nodelist is not None:
        valid_hostnames: List[str] = subprocess.check_output([
            'scontrol',
            'show',
            'hostnames',
            nodelist,
        ]).decode('utf-8').strip().split('\n')
    else:
        valid_hostnames = all_hostnames

    if exclude is not None:
        excluded_hostnames: List[str] = subprocess.check_output([
            'scontrol',
            'show',
            'hostnames',
            exclude,
        ]).decode('utf-8').strip().split('\n')
        for hn in excluded_hostnames:
            if hn in valid_hostnames:
                valid_hostnames.remove(hn)

    for hn in valid_hostnames:
        if hn not in all_hostnames:
            raise ValueError(f"Invalid host name: {hn}. Available host names: {all_hostnames}.")

    def _filter_node_type(node_type, node_name):
        if node_type is not None:
            if not isinstance(node_type, list):
                node_type = [node_type]
            nt_condition = []
            for nt in node_type:
                if nt == 'g1' and 'frl1g' not in node_name:
                    cond = False
                elif nt == 'g2' and 'frl2g' not in node_name:
                    cond = False
                elif nt == 'g8' and 'frl8g' not in node_name:
                    cond = False
                elif nt == 'a100' and 'frl8a' not in node_name and 'frl4a' not in node_name:
                    cond = False
                elif nt == 'a800' and "YL-com" not in node_name:
                    cond = False
                elif nt not in ['g1', 'g2', 'g8', 'a100', 'a800']:
                    raise ValueError("Unknown node type.")
                else:
                    cond = True
                nt_condition.append(cond)
            return any(nt_condition)
        else:
            return True

    return list(filter(lambda x: _filter_node_type(node_type, x), valid_hostnames))


def get_all_node_resources() -> Dict[str, SlurmResource]:
    """ Execute `scontrol show node` to get all node resources
    available in the slurm cluster. 
    Return a list of SlurmResource
    """
    # TODO: refactor this with "scontrol show -o node"
    o = subprocess.check_output(["scontrol", "show", "node"]).decode("utf-8")
    nodes = o.split("\n\n")
    all_rres = {}
    for node in nodes:
        if len(node) <= 1:
            continue
        ls = node.split("\n")
        node_name = ls[0].split(" ")[0].split("=")[1]
        ctres = SlurmResource()
        atres = SlurmResource()
        for l in ls:
            l = l.strip("\n").strip()
            if l.startswith("State"):
                status = _parse_output_status_line(l)
                if "DOWN" in status or "DRAIN" in status or "NOT_RESPONDING" in status:
                    break
            if l.startswith("CfgTRES"):
                ctres = _parse_output_tres_line(l)
            if l.startswith("AllocTRES"):
                atres = _parse_output_tres_line(l)
        if "8a" in node_name or "4a" in node_name or "YL-com" in node_name:
            ctres.gpu_type = atres.gpu_type = "tesla"
        else:
            ctres.gpu_type = atres.gpu_type = "geforce"
        rres = ctres - atres
        if rres.valid():
            all_rres[node_name] = rres
        else:
            all_rres[node_name] = SlurmResource(gpu_type=ctres.gpu_type)

    return all_rres


def allocate_resources(infos: List[SlurmTaskInfo],
                       # strategy: Literal["pack", "plane"] = "pack",
                       ) -> List[SlurmTaskInfo]:
    """ Allocate all slurm task specs, fill in the hostfile field of the specs
    Only support simple greedy strategy now, which allocates tasks to node with the most 
    available resources without considering other tasks.
    """
    all_resources = get_all_node_resources()
    # all_resources = sorted([(k, v) for k, v in all_resources.items()],
    #                 key=lambda x: x[1],
    #                 reverse=True)
    infos = sorted(infos, key=lambda x: x.resource_requirement * x.ntasks, reverse=True)
    for info in infos:
        valid_hostnames = available_hostnames(
            node_type=info.node_type,
            nodelist=info.nodelist,
            exclude=info.exclude,
        )
        valid_hostnames = list(filter(lambda x: x in all_resources, valid_hostnames))
        valid_resources = {hn: all_resources[hn] for hn in valid_hostnames}
        valid_resources = sorted(valid_resources.items(), key=lambda x: x[1], reverse=True)
        task_left = info.ntasks
        allocated = dict()
        for hostname, resource in valid_resources:
            tmp = task_left
            while task_left > 0:
                try:
                    resource = resource - info.resource_requirement
                except ValueError:
                    break
                if not resource.valid():
                    break
                task_left -= 1
            if tmp - task_left > 0:
                allocated[hostname] = tmp - task_left
            all_resources[hostname] = resource
        if task_left > 0:
            raise SlurmResourceNotEnoughException()
        hostlist = []
        for hostname, task_num in allocated.items():
            hostlist += [hostname] * task_num
        info.hostfile_content = "\n".join(hostlist)
    return infos


def show_tesla():
    all_rres = get_all_node_resources()
    for k in available_hostnames(node_type=["a100"]):
        print(k, all_rres[k])


def show_all():
    all_rres = get_all_node_resources()
    for k, v in all_rres.items():
        print(k, v)


if __name__ == "__main__":
    # show_all()
    # print(available_hostnames(node_type=["a100"]))
    show_tesla()
