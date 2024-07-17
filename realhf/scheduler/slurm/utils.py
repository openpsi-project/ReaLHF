from __future__ import (
    annotations,  # python3.7+ feature to allow self-referencing type hints
)

import collections
import dataclasses
import datetime
import getpass
import math
import os
import shutil
import socket
import subprocess
from typing import Callable, Dict, List, Literal, Optional, Union

import pandas as pd

import realhf.base.cluster
import realhf.base.logging as logging
from realhf.base.constants import LOG_ROOT
from realhf.scheduler.client import JobException, JobInfo, JobState

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
    "RUNNING": JobState.RUNNING,
    "COMPLETING": JobState.RUNNING,
    "PENDING": JobState.PENDING,
    "CANCELLED": JobState.CANCELLED,
    "FAILED": JobState.FAILED,
    "COMPLETED": JobState.COMPLETED,
    "OUT_OF_MEMORY": JobState.FAILED,
    "DEADLINE": JobState.COMPLETED,
    "TIMEOUT": JobState.COMPLETED,
}


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
        valid_gpu_type = self_gpu_type == other_gpu_type or (
            self_gpu_type or other_gpu_type
        )
        assert (
            valid_gpu_type
        ), f"Cannot add two different gpu types {self_gpu_type}, {other_gpu_type}."
        return self_gpu_type if self_gpu_type else other_gpu_type

    def __str__(self):
        return (
            "SlurmResource: \n"
            + "mem: "
            + str(self.mem)
            + " MB \n"
            + "cpu: "
            + str(self.cpu)
            + " \n"
            + "gpu: "
            + str(self.gpu)
            + " \n"
            + "gpu_type: "
            + str(self.gpu_type)
        )

    def __mul__(self, other: int) -> SlurmResource:
        assert isinstance(
            other, int
        ), "ResourceRequirement can only be multiplied by int."
        return SlurmResource(
            mem=self.mem * other,
            cpu=self.cpu * other,
            gpu=self.gpu * other,
            gpu_type=self.gpu_type,
        )

    def __rmul__(self, other: int) -> SlurmResource:
        return self.__mul__(other)

    def __add__(self, other: SlurmResource) -> SlurmResource:
        assert isinstance(
            other, SlurmResource
        ), "SlurmResource can only add another SlurmResource instance."
        return SlurmResource(
            mem=self.mem + other.mem,
            cpu=self.cpu + other.cpu,
            gpu=self.gpu + other.gpu,
            gpu_type=self.__check_gpu_type(other),
        )

    def __sub__(self, other: SlurmResource) -> SlurmResource:
        assert isinstance(
            other, SlurmResource
        ), "SlurmResource can only subtract another SlurmResource instance."
        return SlurmResource(
            mem=self.mem - other.mem,
            cpu=self.cpu - other.cpu,
            gpu=self.gpu - other.gpu,
            gpu_type=self.__check_gpu_type(other),
        )

    def __neg__(self) -> SlurmResource:
        return SlurmResource(
            mem=-self.mem, cpu=-self.cpu, gpu=-self.gpu, gpu_type=self.gpu_type
        )

    def __eq__(self, other: SlurmResource) -> bool:
        return (
            self.mem == other.mem
            and self.cpu == other.cpu
            and self.gpu == other.gpu
            and self.gpu_type == other.gpu_type
        )

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
        if self.gpu_type not in ["geforce", "tesla", None]:
            return False
        if self.mem < 0 or self.cpu < 0 or self.gpu < 0:
            return False
        return True


@dataclasses.dataclass
class SlurmLaunchInfo:
    """A SlurmLaunchInfo contains all informantion required to **launch** a
    slurm job.

    Matching one `TasksGroup` in `SchedulingConfig` and one slurm job.

    The naming conventions:
        - `job`: Literally a slurm job with a (maybe non-unique) job name and an unique job ID,
            which may contain multiple job steps and processes. It corresponds an `sbatch` or `srun` call.
            Job names are guaranteed to be unique using the scheduler within this repo.
        - `jobstep`: Literally a slurm job step with a unique job step ID, i.e., ${jobID}.${stepID},
            which corresponds to a running instance `apps.remote` script, but may still contain multiple processes.
            A job step occupies at most one GPU. Processes in the same job step must share the same GPU.
        - `wproc`: A single worker process launched by `apps.remote` script, which may occupy less than 1 GPU.
            A worker just corresponds to a process.
        - `task`: The alias of `jobstep`. It is easier to under stand this concept in the context of `srun` command.
            `--ntasks' is just the number of jobsteps. We use the alternative term `jobstep` to avoid confusion.

    Attributes:
        run_name (str): Identifier of this run, typically ${exp_name}_${trial_name}.
        worker_type (str): Type of workers to be launched, e.g. model_worker, data_worker, etc.
        worker_submission_idx (int): For heterogeneous scheduling, we submit jobs of the same worker_type to slurm
            for multiple times. `worker_submission_idx` is used to distinguish them, so the (global) slurm job name will
            be ${run_name}:${worker_type}:${worker_submission_idx}.
        wprocs_in_job: The number of worker processes in this slurm job (of all job steps).
        n_jobsteps (int): The number of job steps of this slurm job. This is also the group size of the multiprog file.
            Will be resolved automatically according to GPU requirement.
        wprocs_per_jobstep: The number of worker processes in each job step, as well as the number of sub-processes
            spawned by `apps.remote`. Will be resolved automatically according to GPU requirement.

        resource_requirement (SlurmResource): The resource requirement of this job, including all job steps.
        cmd (str): The command to be executed.
        container_image (str): .
        container_mounts (str): .
        env_vars (dict): .
        node_type (str): .
        nodelist (str): .
        exclude (str): .
        partition (str, optional): .
        time_limit (str, optional): Slurm job time limit.
        begin (str, optional): Scheduled worker start time.
        deadline (str, optional): Scheduled worker end time.
        hostfile (bool): Whether to use hostfile for `--distribution=arbitrary` scheduling.
        hostfile_content (str, optional): The content of the hostfile.
        multiprog (bool): Whether to use multiprog file for `--multi-prog` job submission.
        multiprog_content (str, optional): The content of the multiprog file.
    """

    run_name: str
    exper_name: str
    trial_name: str
    worker_type: str
    worker_submission_idx: int
    wprocs_in_job: int

    resource_requirement: SlurmResource
    cmd: str
    container_image: str
    container_mounts: str
    env_vars: dict
    node_type: str
    nodelist: str
    exclude: str
    partition: Optional[str] = None
    time_limit: Optional[str] = None
    begin: Optional[str] = None
    deadline: Optional[str] = None
    # hostfile
    hostfile: bool = True
    hostfile_content: Optional[str] = None
    # multiprog options, override cmd
    multiprog: bool = True
    multiprog_content: Optional[str] = None

    n_jobsteps: int = None
    wprocs_per_jobstep: int = None

    job_info: Optional[JobInfo] = None

    def __post_init__(self):
        """Resolve fractional GPU resource requirement."""
        gpu_per_worker = self.resource_requirement.gpu
        # assert gpu_per_worker <= 1 and gpu_per_worker >= 0
        if gpu_per_worker < 1 and gpu_per_worker > 0:
            self.resource_requirement.gpu = 1
            self.wprocs_per_jobstep = math.floor(1 / gpu_per_worker)
            self.resource_requirement.cpu *= self.wprocs_per_jobstep
            self.resource_requirement.mem *= self.wprocs_per_jobstep
            self.n_jobsteps = math.ceil(self.wprocs_in_job / self.wprocs_per_jobstep)
            logger.info(f"Resolved fractional GPU requirement for {self.slurm_name}")
            logger.info(
                f"GPU per worker {gpu_per_worker}, workers per jobstep (process size in `apps.remote`) {self.wprocs_per_jobstep}, "
                f"number of jobsteps (instance of running `apps.remote`) {self.n_jobsteps}"
            )
        elif gpu_per_worker == 0:
            self.wprocs_per_jobstep = self.wprocs_in_job
            self.n_jobsteps = 1
        elif gpu_per_worker == 1:
            self.n_jobsteps = self.wprocs_in_job
            self.wprocs_per_jobstep = 1
        else:
            self.n_jobsteps = 1
            self.wprocs_per_jobstep = 1

    @property
    def slurm_name(self) -> str:
        return f"{self.run_name}:{self.worker_type}:{self.worker_submission_idx}"

    @property
    def slurm_id(self) -> Optional[str]:
        if self.job_info:
            return self.job_info.slurm_id
        else:
            return None

    @property
    def log_path(self) -> str:
        return os.path.join(
            LOG_ROOT,
            self.exper_name,
            self.trial_name,
            f"{self.worker_type}-{self.worker_submission_idx}",
        )

    @property
    def multiprog_path(self) -> str:
        return os.path.join(
            LOG_ROOT,
            self.exper_name,
            self.trial_name,
            f"{self.worker_type}-{self.worker_submission_idx}.multiprog",
        )

    @property
    def hostfile_path(self) -> str:
        return os.path.join(
            LOG_ROOT,
            self.exper_name,
            self.trial_name,
            f"{self.worker_type}-{self.worker_submission_idx}.hostfile",
        )

    def show_log(self):
        try:
            terminal_columns = os.get_terminal_size().columns
        except OSError:
            terminal_columns = shutil.get_terminal_size().columns
        logger.info(
            f"Showing log of slurm job: {self.worker_type}-{self.worker_submission_idx}\n\n{'-'*terminal_columns}"
        )
        subprocess.Popen(["tail", "-n50", self.log_path]).wait(timeout=3)
        logger.info(
            f"End of log: {self.worker_type}-{self.worker_submission_idx}\n\n{'-'*terminal_columns}"
        )

    def update(self):
        job_infos = query_jobs(slurm_names=[self.slurm_name])
        job_infos = sorted(
            job_infos,
            key=lambda x: parse_formatted_time(x.submit_time),
            reverse=True,
        )
        self.job_info = job_infos[0] if len(job_infos) > 0 else None
        if self.job_info:
            return self.job_info.state
        else:
            return None

    def cancel(self, signal: Literal["SIGINT", "SIGKILL"] = "SIGKILL"):
        cancel_jobs(slurm_names=[self.slurm_name], signal=signal)
        self.job_info = JobInfo(name=self.slurm_name, state=JobState.CANCELLED)

    def __str__(self):
        s = f"SlurmLaunchInfo [{self.slurm_name}] \n"
        s += f"Resources: [\n{self.resource_requirement}\n]\n"
        s += f"Multiprog Filepath: [{self.multiprog_path}]\n"
        s += f"Multiprog Content: [\n{self.multiprog_content}\n]\n"
        s += f"Hostfile Filepath: [{self.hostfile_path}]\n"
        s += f"Hostfile Content: [\n{self.hostfile_content}\n]\n"
        if self.job_info is None:
            job_info_str = "None"
        else:
            job_info_str = "\n".join(
                [f"{k}: {v}" for k, v in self.job_info.__dict__.items()]
            )
        s += f"Runtime JobInfo: [\n{job_info_str}\n]\n"
        env_var_str = "\n".join([f"{k}: {v}" for k, v in self.env_vars.items()])
        s += f"Env vars: [\n{env_var_str}\n]\n"
        return s

    def commit(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True, mode=0o775)

        ntasks = self.n_jobsteps
        mem = self.resource_requirement.mem
        cpu = self.resource_requirement.cpu
        gpu = self.resource_requirement.gpu

        cmd = self.cmd

        # assert gpu == 1 or gpu == 0, "Slurm job GPU requirement should be resolved to a integer."
        gpu_type = self.resource_requirement.gpu_type

        if self.multiprog:
            with open(self.multiprog_path, "w") as f:
                f.write(self.multiprog_content)
        if self.hostfile:
            with open(self.hostfile_path, "w") as f:
                f.write(self.hostfile_content)

        logger.info(
            f'Allocating {ntasks} jobstep(s) "{self.worker_type}" submission index {self.worker_submission_idx}'
            f" with {cpu} cpu, {gpu} gpu and {mem} MB memory."
        )
        logger.info(f"To check the output, run \n\t`tail -f {self.log_path}`.")

        # Setup sbatch
        # head
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={self.slurm_name}",
            f"#SBATCH --output={self.log_path}",
            f"#SBATCH --ntasks={ntasks}",
            f"#SBATCH --gpus-per-task={gpu_type}:{gpu}" if gpu >= 1 else "",
            f"#SBATCH --cpus-per-task={cpu}",
            f"#SBATCH --mem-per-cpu={mem // max(1, cpu)}M",
            f"#SBATCH --partition={self.partition}" if self.partition else "",
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
        # add current directory into container mounts to ensure editable mode for realhf package
        container_mounts = (
            f"{os.environ.get('REAL_PACKAGE_PATH', '$PWD')}:/realhf,"
            + self.container_mounts
        )
        srun_flags = [
            f"--ntasks={ntasks}",
            f"--cpus-per-task={cpu}",
            f"--gpus-per-task={gpu_type}:{gpu}" if gpu >= 1 else "",
            f"--mem-per-cpu={mem // max(1, cpu)}",
            (
                f"--export={','.join(str(k)+'='+str(v) for k, v in self.env_vars.items())}"
                if self.env_vars
                else ""
            ),
            f"--multi-prog" if self.multiprog else "",
            f"--container-image={self.container_image}",
            f"--container-mounts={container_mounts}",
            f"--container-mount-home",
        ]

        if self.multiprog:
            srun_cmd = f'srun -l {" ".join(srun_flags)} {self.multiprog_path}'
        else:
            srun_cmd = f'srun -l {" ".join(srun_flags)} {cmd}'

        lines += [
            'echo "[Runner] StartTime: $(date -u)"',
            'echo "[Runner] Host: $(hostname)"',
            "echo '[Runner] Command: {}'".format(srun_cmd),
            "echo '[Runner] Log: {}'".format(self.log_path),
            'echo "[Runner] CudaVisible: $CUDA_VISIBLE_DEVICES"',
            'echo "[Runner] CudaMpsPerc: $CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"',
            srun_cmd,
            "RETCODE=$?",
            'echo "[Runner] FinishTime: $(date -u)"',
            'echo "[Runner] RetCode: $RETCODE"',
            'echo "[Runner] ------------"',
            "exit $RETCODE",
        ]

        script_strs = "\n".join(list(filter(lambda x: x, lines))) + "\n"
        script = script_strs.encode("ascii")

        def pad_output_str_to_length(s: str, pad_s: str, length: int):
            assert len(pad_s) == 1
            assert len(s) + 2 <= length
            n_pads = (length - len(s) - 2) // 2
            return pad_s * n_pads + " " + s + " " + pad_s * n_pads

        with open(self.log_path, "a") as f:
            f.write(pad_output_str_to_length("SBATCH SCRIPT BEGIN", "=", 80) + "\n")
            f.write(script_strs)
            f.write(pad_output_str_to_length("SBATCH SCRIPT END", "=", 80) + "\n")
            f.write(pad_output_str_to_length("SBATCH JOB INFO BEGIN", "=", 80) + "\n")
            f.write(str(self))
            f.write(pad_output_str_to_length("SBATCH JOB INFO END", "=", 80) + "\n")
            f.write(pad_output_str_to_length("JOB OUTPUT BEGIN", "=", 80) + "\n")
        r = (
            subprocess.check_output(
                ["sbatch", "--parsable"], input=script, env=srun_env
            )
            .decode("ascii")
            .strip()
        )
        self.job_info = JobInfo(name=self.slurm_name, state=JobState.PENDING)


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
def query_jobs(
    slurm_names: Optional[List[str]] = None,
    slurm_ids: Optional[List[str]] = None,
    status: str = "all",
    delimiter: str = "__PSI__",
) -> List[JobInfo]:
    squeue_format = f":.{delimiter},".join(SQUEUE_FIELDS)
    cmd = ["squeue", "-O", squeue_format, f"-t{status}"]
    if slurm_names is not None:
        cmd += ["-n", ",".join(slurm_names)]
    if slurm_ids is not None:
        cmd += ["-j", ",".join([str(s) for s in slurm_ids])]
    output = (
        subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("ascii").strip()
    )
    rs = []
    for line in output.split("\n")[1:]:
        job_id, state, submit_time, start_time, slurm_name, nodelist, *_ = line.split(
            delimiter
        )
        rs.append(
            JobInfo(
                name=slurm_name,
                state=STATUS_MAPPING[state],
                host=nodelist,
                submit_time=submit_time,
                start_time=start_time,
                slurm_id=job_id.strip(),
            )
        )
    return rs


def cancel_jobs(
    slurm_names: Optional[List[str]] = None,
    slurm_ids: Optional[List[str]] = None,
    signal: Literal["SIGINT", "SIGKILL"] = "SIGKILL",
):
    assert (
        slurm_names is not None or slurm_ids is not None
    ), "Must specify slurm_names or slurm_ids."
    assert not (
        slurm_names and slurm_ids
    ), "Cannot specify both slurm_names and slurm_ids."
    cmd = ["scancel", "-s", signal]
    if slurm_names is not None:
        cmd += ["-n", ",".join(slurm_names)]
    elif slurm_ids is not None:
        cmd += ["-j", ",".join([str(s) for s in slurm_ids])]
    subprocess.check_call(cmd)
    logger.info(
        f"Cancelled Slurm job with signal {signal}: "
        f"slurm identifiers {slurm_names if slurm_ids is None else slurm_ids}"
    )


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
    all_nodelist: str = (
        subprocess.check_output('sinfo -o "%N" --noheader', shell=True)
        .decode("utf-8")
        .strip()
    )
    all_hostnames: List[str] = (
        subprocess.check_output(
            [
                "scontrol",
                "show",
                "hostnames",
                all_nodelist,
            ]
        )
        .decode("utf-8")
        .strip()
        .split("\n")
    )

    if nodelist is not None:
        valid_hostnames: List[str] = (
            subprocess.check_output(
                [
                    "scontrol",
                    "show",
                    "hostnames",
                    nodelist,
                ]
            )
            .decode("utf-8")
            .strip()
            .split("\n")
        )
    else:
        valid_hostnames = all_hostnames

    if exclude is not None:
        excluded_hostnames: List[str] = (
            subprocess.check_output(
                [
                    "scontrol",
                    "show",
                    "hostnames",
                    exclude,
                ]
            )
            .decode("utf-8")
            .strip()
            .split("\n")
        )
        for hn in excluded_hostnames:
            if hn in valid_hostnames:
                valid_hostnames.remove(hn)

    for hn in valid_hostnames:
        if hn not in all_hostnames:
            raise ValueError(
                f"Invalid host name: {hn}. Available host names: {all_hostnames}."
            )

    return list(
        filter(
            lambda x: realhf.base.cluster.node_name_is_node_type(x, node_type),
            valid_hostnames,
        )
    )


def get_all_node_resources() -> Dict[str, SlurmResource]:
    """Execute `scontrol show node` to get all node resources available in the
    slurm cluster.

    Return a list of SlurmResource
    """
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
        ctres.gpu_type = atres.gpu_type = (
            realhf.base.cluster.spec.gpu_type_from_node_name(node_name)
        )
        rres = ctres - atres
        if rres.valid():
            all_rres[node_name] = rres
        else:
            all_rres[node_name] = SlurmResource(gpu_type=ctres.gpu_type)

    return all_rres


def resource_to_string(resources: Dict[str, SlurmResource]) -> str:
    resource_list = [
        {
            **{"NodeName": k},
            **{
                field.name: getattr(r, field.name)
                for field in r.__dataclass_fields__.values()
            },
        }
        for k, r in resources.items()
    ]
    return pd.DataFrame(resource_list).to_string(index=False)


def allocate_resources(
    infos: List[SlurmLaunchInfo],
    # strategy: Literal["pack", "plane"] = "pack",
) -> List[SlurmLaunchInfo]:
    """Allocate all slurm task specs, fill in the hostfile field of the specs
    Only support simple greedy strategy now, which allocates tasks to node with
    the most available resources without considering other tasks."""
    all_resources = get_all_node_resources()
    # all_resources = sorted([(k, v) for k, v in all_resources.items()],
    #                 key=lambda x: x[1],
    #                 reverse=True)
    infos = sorted(infos, key=lambda x: x.resource_requirement, reverse=True)
    for info_idx, info in enumerate(infos):
        valid_hostnames = available_hostnames(
            node_type=info.node_type,
            nodelist=info.nodelist,
            exclude=info.exclude,
        )
        valid_hostnames = list(filter(lambda x: x in all_resources, valid_hostnames))
        valid_resources = {hn: all_resources[hn] for hn in valid_hostnames}
        valid_resources = sorted(
            valid_resources.items(), key=lambda x: x[1], reverse=True
        )
        task_left = info.n_jobsteps
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
            # logger.warning(f"Current resources in the cluster:\n {resource_to_string(get_all_node_resources())}")
            logger.warning(
                f'Unable to allocate {info.n_jobsteps} Jobs with name "{info.slurm_name}". '
                f"Resource Requirement of this job is: {dataclasses.asdict(info.resource_requirement)}. "
                f"Valid resources for this job is "
                f"(according to NodeType={info.node_type}, NodeList={info.nodelist}, "
                f"and Exclude={info.exclude}):\n {resource_to_string({k: v for k, v in get_all_node_resources().items() if k in valid_hostnames})}"
            )
            for pinfo in infos[:info_idx]:
                if (
                    len(
                        set(pinfo.hostfile_content.split("\n")).intersection(
                            set(valid_hostnames)
                        )
                    )
                    == 0
                ):
                    continue
                palloc = collections.defaultdict(lambda: 0)
                for _n in pinfo.hostfile_content.split("\n"):
                    palloc[_n] += 1
                logger.warning(
                    f'Found previous job "{pinfo.slurm_name}" (ntasks={pinfo.n_jobsteps}) '
                    f"has been allocated to the same set of nodes. "
                    f"Resource requirement of this job is: {dataclasses.asdict(pinfo.resource_requirement)}, "
                    f"allocation of this job is {dict(palloc)}."
                )
            raise SlurmResourceNotEnoughException()
        hostlist = []
        for hostname, task_num in allocated.items():
            hostlist += [hostname] * task_num
        info.hostfile_content = "\n".join(hostlist)
    return infos


def show_tesla():
    all_rres = get_all_node_resources()
    hostname = socket.gethostname()
    for k in available_hostnames(node_type=["a100"]):
        print(k, all_rres[k])


def show_all():
    all_rres = get_all_node_resources()
    for k, v in all_rres.items():
        print(k, v)


if __name__ == "__main__":
    show_tesla()
