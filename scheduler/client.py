from typing import List, Optional, Tuple
import dataclasses
import enum
import logging
import os
import re
import shutil
import subprocess
import time

logger = logging.getLogger("scheduler")


class TaskState(enum.Enum):
    NOT_FOUND = 0
    PENDING = 1
    RUNNING = 2
    COMPLETED = 3
    FAILED = 4
    CANCELLED = 5

    def active(self):
        return self == self.PENDING or self == self.RUNNING


class SchedulerError(Exception):
    pass


class TaskException(Exception):

    def __init__(self, job_name, task_name, host, reason: TaskState):
        super().__init__(f"Task {job_name}:{task_name} {reason} at node {host}")
        self.job_name = job_name
        self.task_name = task_name
        self.host = host
        self.reason = reason


@dataclasses.dataclass
class TaskInfo:
    name: str
    state: TaskState
    host: str = None  # The host on which the task is/was running. None if the task had not run.
    start_time: str = None
    slurm_id: str = None  # Slurm only. The Slurm id of the task.


class SchedulerClient:

    def __init__(self, job_name):
        self.job_name = job_name

    def submit(self, task_name, cmd, **kwargs):
        """Submits a task to the scheduler. Raises exception if the task is already running.

        Args:
            task_name: Name of the task. The job name is specified when initializing the client.
            cmd (str or List[str]): The command of the task process. If this is str, the command is parsed by
                shell; otherwise it is executed directly.
        """
        raise NotImplementedError()

    def submit_array(self, task_name, cmd, count, **kwargs):
        """Submits an array of tasks to the scheduler.

        Args:
            task_name: The tasks share the same name.
            cmd: Command template of the tasks that may contain an "{index}" format placeholder.
            count: Number of tasks. The indices of the tasks shall be 0..count-1.
        """
        for index in range(count):
            self.submit(task_name + "_" + str(index), cmd.format(index=index), **kwargs)

    def stop(self, task_name):
        """Stops a running task. Raises exception if there is no such task, but passes if the task has stopped
        either successfully or not.
        """
        raise NotImplementedError()

    def stop_all(self):
        """Stops the whole job.
        """
        raise NotImplementedError()

    def find(self, task_name) -> Optional[TaskInfo]:
        """Gets the status of a task of this job.

        Args:
            task_name: Name of the task.

        Returns:
            A TaskInfo if the task is found, or None otherwise.
        """
        raise NotImplementedError()

    def find_all(self, task_name_regex=".*") -> List[TaskInfo]:
        """Finds tasks.

        Args:
            task_name_regex: Task name regex.

        Returns:
            A list of found TaskInfo.
        """
        raise NotImplementedError()

    def wait(self, timeout=None, **kwargs):
        """Waits until all tasks submitted via this client instance finish.
        """
        raise NotImplementedError()


def remote_worker_cmd(expr_name, trial_name, debug, worker_type):
    return f"python3 {'' if debug else '-O'} -m apps.remote worker -w {worker_type} " \
           f"-e {expr_name} -f {trial_name} -i {{index}}"


def setup_cmd(expr_name, trial_name, debug):
    return f"python3 {'' if debug else '-O'} -m apps.remote reset_name_resolve -e {expr_name} -f {trial_name}"


def control_cmd(expr_name, trial_name, debug, ignore_worker_error):
    return (f"python3 {'' if debug else '-O'} -m apps.remote controller -e {expr_name} -f {trial_name} "
            f"--{'ignore_worker_error' if ignore_worker_error else 'raise_worker_error'}")


def make(mode, job_name, **kwargs) -> SchedulerClient:
    if mode == "slurm":
        from scheduler.slurm import SlurmSchedulerClient
        return SlurmSchedulerClient(job_name)
    else:
        raise NotImplementedError(f"Scheduler {mode} not found")
