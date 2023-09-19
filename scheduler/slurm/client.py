from collections import defaultdict
from typing import List, Optional, Tuple
import logging
import math
import os
import re
import shutil
import subprocess
import time

from scheduler.client import SchedulerClient, TaskException, TaskInfo, TaskState
from scheduler.slurm.utils import *

logger = logging.getLogger("Slurm scheduler")


class SlurmSchedulerClient(SchedulerClient):
    """Uses Slurm (https://slurm.schedmd.com/overview.html).
    """
    SQUEUE_FIELDS = [
        "JobID",
        "State",
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
    }

    def __init__(self, expr_name, trial_name):
        super().__init__(expr_name, trial_name)
        self._tasks = {}
        self.__pending_task_specs = []
        self.__taskname_to_ntasks = defaultdict(int)  # record ntasks for each type of task
        self.__allocated_ntasks = defaultdict(int)
        self.__taskname_to_nspecs = defaultdict(int)  # record commited #specs for each type of task
        self.__allocated_workers = defaultdict(int)  # dict for worker indices

    def submit(self, task_name, cmd, **kwargs):
        self.submit_array(task_name, cmd, count=1, **kwargs)

    def submit_array(
        self,
        task_name,
        cmd,  # XXX: should be None for workers
        count,
        cpu=1,
        gpu_type: str = "geforce",
        gpu=0,
        mem=1024,
        env_vars=None,
        container_image="llm/llm-cpu",
        container_mounts="/data:/data",
        nodelist=None,
        exclude=None,
        hostfile=False,
    ):
        # record information of the task, do not submit to slurm until `wait()` is called
        resource_requirement = SlurmResource(mem=mem, cpu=cpu, gpu=gpu, gpu_type=gpu_type)
        # TODO: temporary fix
        if gpu == 0:
            hostfile = False
        task_spec = SlurmTaskSpecification(task_name=task_name,
                                           ntasks=count,
                                           resource_requirement=resource_requirement,
                                           cmd=cmd,
                                           job_name=self.job_name,
                                           container_image=container_image,
                                           container_mounts=container_mounts,
                                           env_vars=env_vars,
                                           nodelist=nodelist,
                                           exclude=exclude,
                                           hostfile=hostfile)
        self.__pending_task_specs.append(task_spec)
        if gpu > 0:
            task_size = math.ceil(1 / gpu)
            real_count = math.ceil(count / task_size)
        else:
            real_count = count
        self.__taskname_to_ntasks[task_name] += real_count
        logger.info("Registered Slurm task: %s (count=%s, real_count=%s)", task_name, count, real_count)

    def __commit_one(self, spec: SlurmTaskSpecification):
        """Commit one spec to slurm."""
        # logging, all tasks with the same name shares the same log file
        # spec will not be changed throughout the function
        # sepearte task name with index
        task_name = f"{spec.task_name}-{self.__taskname_to_nspecs[spec.task_name]}"
        group_index = self.__taskname_to_nspecs[spec.task_name]
        self.__taskname_to_nspecs[spec.task_name] += 1

        output = log_path(self.job_name, task_name)
        slurm_name = f"{self.job_name}:{task_name}"
        os.makedirs(os.path.dirname(output), exist_ok=True, mode=0o775)

        ntasks = spec.ntasks
        mem = spec.resource_requirement.mem
        cpu = spec.resource_requirement.cpu
        gpu = spec.resource_requirement.gpu
        cmd = spec.cmd

        # assert gpu == 1 or gpu == 0, "GPU count must be 0 or 1 for one slurm task."
        # support fractional GPU count
        assert gpu <= 1 and gpu >= 0, "GPU count must be between 0 and 1."
        gpu_type = spec.resource_requirement.gpu_type

        multi_prog_file = log_path(self.job_name, task_name) + ".multiprog"
        hostfile = log_path(self.job_name, task_name) + ".hostfile"

        task_size = 1
        # extra_task_size = 0
        if gpu < 1 and gpu > 0:
            # merge multiple tasks that has fraction GPU count into
            # one task s.t. GPU count = 1
            task_size = math.ceil(1 / gpu)
            gpu = 1
            cpu = cpu * task_size
            mem = mem * task_size
            # XXX: resource in task size is not scaled properly because i dont want to introduce
            #      different resource requirements in one submit

            ntasks = math.ceil(ntasks / task_size)
            # extra_task_size = ntasks % task_size
            # if extra_task_size > 0:
            #     ntasks += 1

        new_rr = SlurmResource(mem=mem, cpu=cpu, gpu=gpu, gpu_type=gpu_type)
        start_group_id = self.__allocated_ntasks[spec.task_name]
        total_ntasks = self.__taskname_to_ntasks[spec.task_name]
        commit_index_start = self.__allocated_workers[spec.task_name]
        self.__allocated_ntasks[spec.task_name] += ntasks
        self.__allocated_workers[spec.task_name] += spec.ntasks

        with open(multi_prog_file, "w") as f:
            if "index" in cmd:
                cmd = cmd.format(index=str(start_group_id),
                                 offset='%t',
                                 count=ntasks,
                                 group_index=str(group_index))
            f.write(f"0-{ntasks-1} {cmd}\n")

        logger.info(f"{task_name} cpu: {cpu}, gpu: {gpu}, mem: {mem}, ntasks: {ntasks},"
                    f"task_size: {task_size}, start_group_id: {start_group_id},"
                    f"total_ntasks: {total_ntasks}, commit_index_start: {commit_index_start}")

        if spec.hostfile:
            write_hostfile(new_rr,
                           ntasks,
                           hostfile,
                           node_type=None,
                           nodelist=spec.nodelist,
                           exclude=spec.exclude)

        # reconstruct worker index in remote.py by adding env var
        # start : start_index + args.group_offset * args.task_size
        # end : start_index + min((args.group_offset + 1) * args.task_size, spec.ntasks))
        env_vars = spec.env_vars.copy() if spec.env_vars is not None else {}
        env_vars.update({
            "COMMIT_INDEX_START": str(commit_index_start),
            "TASK_SIZE": str(task_size),
            "COMMIT_N_WORKERS": str(spec.ntasks)
        })
        logger.info("Setting env var")

        logger.info(
            f"Allocating {ntasks} task(s) \"{task_name}\" with {cpu} cpu, {gpu} gpu and {mem} MB memory.")
        logger.info(
            f"To check the output, run \n\t\t\t\t\t\t`tail -f {log_path(spec.job_name, spec.task_name)}`.")

        # Setup sbatch
        # head
        lines = [
            '#!/bin/bash',
            f'#SBATCH --job-name={slurm_name}',
            f'#SBATCH --output={output}',
            f'#SBATCH --ntasks={ntasks}',
            f'#SBATCH --gpus-per-task={gpu_type}:1' if gpu == 1 else "",
            f'#SBATCH --cpus-per-task={cpu}',
            f'#SBATCH --mem-per-cpu={mem // max(1, cpu)}M',
            # '#SBATCH --partition=cpu' if gpu == 0 else "",
            "#SBATCH --distribution=arbitrary" if spec.hostfile else "",
            # f'#SBATCH --nodelist={spec.nodelist}' if spec.nodelist is not None else "",
            # f'#SBATCH --exclude={spec.exclude}' if spec.exclude is not None else "",
        ]

        srun_env = os.environ.copy()
        if spec.hostfile:
            srun_env["SLURM_HOSTFILE"] = hostfile
        # Setup step command.
        srun_flags = [
            f"--ntasks={ntasks}",
            f"--cpus-per-task={cpu}",
            f"--gpus-per-task={gpu_type}:1" if gpu == 1 else "",
            f"--mem-per-cpu={mem // max(1, cpu)}",
            f"--export={','.join(str(k)+'='+str(v) for k, v in env_vars.items())}"
            if env_vars is not None else "",
            f"--multi-prog",
            f"--container-image={spec.container_image}",
            f"--container-mounts={spec.container_mounts}",
            f"--container-mount-home",
        ]

        srun_cmd = f'srun -l {" ".join(srun_flags)} {multi_prog_file}'

        lines += [
            'echo "*************************************"',
            'echo "[Runner] StartTime: $(date -u)"',
            'echo "[Runner] Host: $(hostname)"',
            "echo '[Runner] Command: {}'".format(srun_cmd),
            "echo '[Runner] Log: {}'".format(output),
            'echo "[Runner] CudaVisible: $CUDA_VISIBLE_DEVICES"',
            'echo "[Runner] CudaMpsPerc: $CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"',
            srun_cmd,
            'RETCODE=$?',
            'echo "[Runner] FinishTime: $(date -u)"',
            'echo "[Runner] RetCode: $RETCODE"',
            'echo "[Runner] ------------"',
            'exit $RETCODE',
        ]

        script = '\n'.join(lines).encode('ascii')
        r = subprocess.check_output(['sbatch', '--parsable'], input=script,
                                    env=srun_env).decode('ascii').strip()
        self._tasks.update({r: TaskInfo(name=task_name, state=TaskState.PENDING)})

    def __commit_all(self):
        for task_spec in self.__pending_task_specs:
            time.sleep(2)  # TODO: temo fix for multiple commits
            self.__commit_one(task_spec)
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
        # task_name = task_name.split("-")[0]
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

    def __slurm_name(self, task_name):
        return f"{self.job_name}:{task_name}"

    def __task_name(self, slurm_name):
        prefix = f"{self.job_name}:"
        if not slurm_name.startswith(prefix):
            raise ValueError(f"Slurm name '{slurm_name}' does not start with '{prefix}'")
        return slurm_name[len(prefix):]

    def __query_tasks(self, slurm_ids, status="all", delimiter="__PSI__"):
        squeue_format = f":.{delimiter},".join(SlurmSchedulerClient.SQUEUE_FIELDS)
        cmd = ["squeue", "-O", squeue_format, f"-t{status}"]
        if slurm_ids is not None:
            cmd += ["-j", ",".join([str(s) for s in slurm_ids])]
        output = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("ascii").strip()
        rs = []
        for line in output.split("\n")[1:]:
            job_id, state, start_time, slurm_name, nodelist, *_ = line.split(delimiter)
            if slurm_ids is not None:
                assert slurm_name.startswith(f"{self.job_name}:")
            elif not slurm_name.startswith(f"{self.job_name}:"):
                continue
            task_name = self.__task_name(slurm_name)
            job_ids = self.__parse_job_ids(job_id)
            for ji in job_ids:
                rs.append(
                    TaskInfo(name=task_name,
                             state=SlurmSchedulerClient.STATUS_MAPPING[state],
                             host=nodelist,
                             start_time=start_time,
                             slurm_id=ji.strip()))
        return rs

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
