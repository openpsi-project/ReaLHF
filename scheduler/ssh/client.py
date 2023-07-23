from typing import List, Literal, Optional, Tuple
import collections
import dataclasses
import getpass
import logging
import os
import re
import shutil
import subprocess
import time

from scheduler.client import SchedulerClient, SchedulerError, TaskException, TaskInfo, TaskState
import base.names

logger = logging.getLogger("SSH scheduler")


def log_path(job_name, task_name):
    if task_name is None:
        return f"/home/{base.names.USER_NAMESPACE}/llm/logs/{job_name}"
    else:
        return f"/home/{base.names.USER_NAMESPACE}/llm/logs/{job_name}/{task_name}"


@dataclasses.dataclass
class SSHTaskSpecification:
    cpu: int
    mem: int  # in MB
    gpu: Literal[0, 1]
    task_name: str  # aka model, master, etc.
    ntasks: int
    cmd: str
    job_name: str
    container_image: str
    container_mounts: str
    env_vars: dict


@dataclasses.dataclass
class SSHResource:
    mem: int = 0
    cpu: int = 0
    gpu: int = 0

    def __mul__(self, other):
        assert isinstance(other, int)
        return SSHResource(
            mem=self.mem * other,
            cpu=self.cpu * other,
            gpu=self.gpu * other,
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        assert isinstance(other, SSHResource)
        return SSHResource(
            mem=self.mem + other.mem,
            cpu=self.cpu + other.cpu,
            gpu=self.gpu + other.gpu,
        )

    def __sub__(self, other):
        assert isinstance(other, SSHResource)
        return SSHResource(
            mem=self.mem - other.mem,
            cpu=self.cpu - other.cpu,
            gpu=self.gpu - other.gpu,
        )

    def __neg__(self):
        return SSHResource(mem=-self.mem, cpu=-self.cpu, gpu=-self.gpu)

    def valid(self):
        # check if it is a valid resource requirement
        if self.mem < 0 or self.cpu < 0 or self.gpu < 0:
            return False
        return True


A800_RESOUCE = collections.OrderedDict({
    "10.122.2.11": SSHResource(cpu=112, mem=1008e3, gpu=8),
    "10.122.2.12": SSHResource(cpu=112, mem=1008e3, gpu=8),
})


class SSHSchedulerClient(SchedulerClient):
    """Uses SSH and docker on two A800 machines.
    """

    STATUS_MAPPING = {
        "RUNNING": TaskState.RUNNING,
        "COMPLETING": TaskState.RUNNING,
        "CANCELLED": TaskState.CANCELLED,
        "FAILED": TaskState.FAILED,
        "COMPLETED": TaskState.COMPLETED,
        "OUT_OF_MEMORY": TaskState.FAILED,
    }

    def __init__(self, job_name):
        super().__init__(job_name)
        self._tasks = {}
        self.__pending_task_specs: List[SSHTaskSpecification] = []

        self.remaining_resources = A800_RESOUCE.copy()

    def submit(self, task_name, cmd, **kwargs):
        self.submit_array(task_name, cmd, count=1, **kwargs)

    def submit_array(self,
                     task_name,
                     cmd,
                     count,
                     cpu=1,
                     gpu=0,
                     mem=1024,
                     env_vars=None,
                     container_image="llm-gpu",
                     container_mounts=None,
                     nodelist=None,
                     exclude=None,
                     hostfile=False):
        assert gpu == 1 or gpu == 0, "GPU count must be 0 or 1."
        task_spec = SSHTaskSpecification(
            cpu=cpu,
            mem=mem,
            gpu=gpu,
            task_name=task_name,
            ntasks=count,
            cmd=cmd,
            job_name=self.job_name,
            container_image=container_image,
            container_mounts=container_mounts,
            env_vars=env_vars,
        )
        self.__pending_task_specs.append(task_spec)
        logger.info("Registered SSH task: %s (count=%s)", task_name, count)

    def __commit_one(self, spec: SSHTaskSpecification, ip_address: str, task_idx: int):
        task_name = f"{spec.task_name}_{task_idx}"
        container_name = self.__container_name_from_task_and_idx(spec.task_name, task_idx)
        output = log_path(self.job_name, task_name)
        os.makedirs(os.path.dirname(output), exist_ok=True, mode=0o775)
        cmd = spec.cmd

        if task_idx == 0:
            logger.info(
                f"Allocating {spec.ntasks} task(s) \"{task_name}\" with {spec.cpu} cpu, {spec.gpu} gpu and {spec.mem} MB memory."
            )
            logger.info(
                f"To check the output, run \n\t\t\t\t\t\t`tail -f {output}` (change to your interested index)."
            )

        env_vars = os.environ.copy()
        for k, v in spec.env_vars.items():
            env_vars[k] = v
        docker_flags = [f"--env={k}={v}" for k, v in env_vars.items()]
        docker_flags += [
            "--rm",
            f"--name={container_name}",
            f"--cpus={spec.cpu}",
            f"--memory={int(1e6 * spec.mem)}",  # MBytes to Bytes
            f"--gpus=all" if spec.gpu > 0 else "",
            f"-v={spec.container_mounts}",
            "--ipc=host",
            "--ulimit=memlock=-1",
            "--ulimit=stack=67108864",
            "--net=host",
            f"--shm-size={int(500e6)}",  # 500MB
        ]
        docker_cmd = f"docker run {' '.join(docker_flags)} {spec.container_image} {cmd.format(index=task_idx)} >> {output} 2>&1 &"

        lines = [
            'echo "[Runner] StartTime: $(date -u)"',
            'echo "[Runner] Host: $(hostname)"',
            "echo '[Runner] Docker command: {}'".format(docker_cmd),
            "echo '[Runner] Log: {}'".format(output),
            'echo "[Runner] CudaVisible: $CUDA_VISIBLE_DEVICES"',
            'echo "[Runner] CudaMpsPerc: $CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"',
            docker_cmd,
            'RETCODE=$?',
            'echo "[Runner] FinishTime: $(date -u)"',
            'echo "[Runner] RetCode: $RETCODE"',
            'echo "[Runner] ------------"',
            'exit $RETCODE',
        ]

        script = '\n'.join(lines).encode('ascii')
        r = subprocess.check_output(['ssh', f'{getpass.getuser()}@{ip_address}'],
                                    input=script).decode('ascii').strip()
        self._tasks.update({r: TaskInfo(name=task_name, state=TaskState.PENDING)})

    def __commit_all(self):
        to_commit = []
        for task_spec in self.__pending_task_specs:
            for i in range(task_spec.ntasks):
                nodes = list(self.remaining_resources.keys())
                ip_address = None
                for node in nodes:
                    if (self.remaining_resources[node] - task_spec.resource_requirement).valid():
                        ip_address = node
                        self.remaining_resources[node] -= task_spec.resource_requirement
                        break
                if ip_address is None:
                    raise RuntimeError(f"Not enough resources node to run task: {task_spec}.")
                to_commit.append((task_spec, ip_address, i))
        for args in to_commit:
            self.__commit_one(*args)
        self.__pending_task_specs = []

    def stop(self, task_name):
        r = self.find(task_name)
        if r is not None and r.state in {TaskState.RUNNING, TaskState.PENDING}:
            subprocess.check_call(["scancel", str(r.slurm_id)])
            logger.info("Cancelled Slurm task %d: %s", r.slurm_id, self.__slurm_name(task_name))
            time.sleep(0.2)
            self.__update_subset([r.slurm_id])

    def stop_all(self):
        rs = self.__query_tasks(list(self._tasks.keys()))
        ids = [r.slurm_id for r in rs if r.state in {TaskState.RUNNING, TaskState.PENDING}]
        group_ids = set([i.split("_")[0] for i in ids])
        logger.info(f"STOPPING SLURM IDS: {group_ids}")
        if len(ids) == 0:
            logger.info("No task to stop, skipping")
        else:
            subprocess.check_call(["scancel", ",".join(group_ids)])
            logger.info("Cancelled %d Slurm tasks: %s", len(group_ids), ",".join(group_ids))
        time.sleep(0.2)
        self.wait(check_status=(),
                  remove_status=(TaskState.CANCELLED, TaskState.NOT_FOUND, TaskState.FAILED,
                                 TaskState.COMPLETED))

    def find(self, task_name):
        for r in self._tasks.values():
            if r.task_name == task_name:
                self.__update_subset(r.slurm_id)
                return self._tasks[r.slurm_id]
        return TaskInfo(name=task_name, state=TaskState.NOT_FOUND)

    def find_all(self, task_name_regex=".*"):
        self.__update_all()
        rs = []
        for r in self._tasks.values():
            if re.fullmatch(task_name_regex, r.name):
                rs.append(r)
        return rs

    def __show_log(self, task_name):
        try:
            terminal_columns = os.get_terminal_size().columns
        except OSError:
            terminal_columns = shutil.get_terminal_size().columns
        logger.info(f"Showing log of task: {task_name}\n\n{'-'*terminal_columns}")
        subprocess.Popen(["tail", "-n50", log_path(self.job_name, task_name)]).wait(timeout=3)
        logger.info(f"End of log: {task_name}\n\n{'-'*terminal_columns}")

    def wait(
            self,
            timeout=None,
            check_status: Tuple[TaskState,
                                ...] = (TaskState.CANCELLED, TaskState.FAILED, TaskState.NOT_FOUND),
            remove_status: Tuple[TaskState, ...] = (TaskState.COMPLETED,),
            update=False,
    ):
        # before wait, commit all remaining pending task specs
        self.__commit_all()
        # begin wait
        deadline = None if timeout is None else time.time() + timeout
        left = set(self._tasks)
        logger.info(str(self._tasks))
        num_jobs_left = len(left)
        logger.info(f"Waiting for {num_jobs_left} jobs.")
        while len(left) > 0:
            if len(left) < num_jobs_left:
                num_jobs_left = len(left)
                logger.info(f"Waiting for {num_jobs_left} jobs.")
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(f"Timeout waiting for {self.job_name}: {', '.join(sorted(left))}")
            try:
                self.__update_all()
            except subprocess.CalledProcessError:
                logger.warning(
                    "Calling squeue failed. Check slurm manually if you continue to see this warning.")
                time.sleep(30)
                continue
            for i in list(left):
                r = self._tasks[i]
                if r.slurm_id is None:
                    continue
                if r.state in check_status:
                    self.__show_log(r.name)
                    raise TaskException(job_name=self.job_name,
                                        task_name=r.name + "_" + i.split("_")[-1],
                                        host=r.host,
                                        reason=r.state)
                if r.state in remove_status:
                    logger.info(f"Task {r.name + '_' + i.split('_')[-1]} is {r.state}.(Removed)")
                    left.remove(r.slurm_id)
                    if update:
                        self._tasks.pop(r.slurm_id)
            time.sleep(2)

    def __container_name_from_task_and_idx(self, task_name, task_idx):
        return f"{self.job_name}##{task_name}__{task_idx}"

    def __task_and_idx_from_container_name(self, container_name):
        task_idx = container_name.split("__")[-1]
        job_name, task_name = container_name.split("__")[0].split("##")
        return task_name, task_idx

    # def __query_tasks(self, slurm_ids, status="all", delimiter="__PSI__"):
    #     docker_query_cmd = ["docker", "ps", "-a"]
    #     squeue_format = f":.{delimiter},".join(SlurmSchedulerClient.SQUEUE_FIELDS)
    #     cmd = ["squeue", "-O", squeue_format, f"-t{status}"]
    #     if slurm_ids is not None:
    #         cmd += ["-j", ",".join([str(s) for s in slurm_ids])]
    #     output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("ascii").strip()
    #     rs = []
    #     for line in output.split("\n")[1:]:
    #         job_id, state, start_time, slurm_name, node_list, *_ = line.split(delimiter)
    #         if slurm_ids is not None:
    #             assert slurm_name.startswith(f"{self.job_name}:")
    #         elif not slurm_name.startswith(f"{self.job_name}:"):
    #             continue
    #         task_name = self.__task_name(slurm_name)
    #         job_ids = self.__parse_job_ids(job_id)
    #         for ji in job_ids:
    #             rs.append(
    #                 TaskInfo(name=task_name,
    #                          state=SlurmSchedulerClient.STATUS_MAPPING[state],
    #                          host=node_list,
    #                          start_time=start_time,
    #                          slurm_id=ji.strip()))
    #     return rs

    def __parse_job_ids(self, job_id):
        """This method may be optimized as we no longer user array jobs.
        """
        if "[" in job_id and "]" in job_id and "-" in job_id:
            batch_id, idx_start, idx_end, _ = re.split("\[|]|-", job_id)
            job_ids = [batch_id + str(idx) for idx in range(int(idx_start), int(idx_end) + 1)]
        elif "[" in job_id and "]" in job_id:
            job_ids = [job_id.replace("[", "").replace("]", "")]
        else:
            job_ids = [job_id]
        return job_ids

    def __update_all(self):
        if not self._tasks:
            tasks = self.__query_tasks(None)
            self._tasks = {r.slurm_id: r for r in tasks}
        else:
            tasks = self.__query_tasks(list(self._tasks.keys()))
        for r in tasks:
            self._tasks[r.slurm_id] = r

    def __update_subset(self, slurm_ids):
        tasks = self.__query_tasks(slurm_ids=slurm_ids)
        for r in tasks:
            self._tasks[r.slurm_id] = r