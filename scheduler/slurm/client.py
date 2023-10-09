from collections import defaultdict
from typing import Dict, List, Literal, Optional, Tuple
import fcntl
import logging
import math
import os
import re
import shutil
import subprocess
import time

from scheduler.client import SchedulerClient, TaskException, TaskInfo, TaskState
from scheduler.slurm.utils import (allocate_resources, SlurmResource, SlurmResourceNotEnoughException,
                                   SlurmTaskInfo)

logger = logging.getLogger("Slurm scheduler")

SCHEDULING_RETRY_INTERVAL_SECONDS = 30
SCHEDULING_TIMEOUT_MAX_SECONDS = 3600 * 24
LOCK_FILE_NAME = "/data/aigc/llm/logs/slurm_scheduler.lock"


class SlurmSchedulerClient(SchedulerClient):
    """Uses Slurm (https://slurm.schedmd.com/overview.html).
    """

    def __init__(self, expr_name, trial_name):
        super().__init__(expr_name, trial_name)
        self.__pending_tasks: Dict[str, SlurmTaskInfo] = dict()
        self.__committed_tasks: Dict[str, SlurmTaskInfo] = dict()
        self.__pending_task_array_counter = defaultdict(int)
        self.__pending_task_counter = defaultdict(int)
        self.__pending_worker_counter = defaultdict(int)

    def submit(self, task_name, cmd, **kwargs):
        self.submit_array(task_name, cmd, count=1, **kwargs)

    def submit_array(
            self,
            task_name: str,
            cmd: str,  # XXX: should be None for workers
            count: int,
            cpu: int = 1,
            gpu_type: str = "geforce",
            gpu: int = 0,
            mem: int = 1024,  # MB
            env_vars: Optional[Dict] = None,
            container_image: str = "llm/llm-gpu",
            container_mounts: str = "/data:/data,/lustre:/lustre",
            node_type: Optional[str] = None,
            nodelist: Optional[str] = None,
            exclude: Optional[str] = None,
            hostfile: bool = True,
            multiprog: bool = True,
            begin: str = None,
            deadline: str = None,
            time_limit: str = None):
        # record information of the task, do not submit to slurm until `wait()` is called
        resource_requirement = SlurmResource(mem=mem, cpu=cpu, gpu=gpu, gpu_type=gpu_type)
        task_info = SlurmTaskInfo(task_name=task_name,
                                  ntasks=count,
                                  resource_requirement=resource_requirement,
                                  cmd=cmd,
                                  job_name=self.job_name,
                                  container_image=container_image,
                                  container_mounts=container_mounts,
                                  env_vars=env_vars,
                                  node_type=node_type,
                                  nodelist=nodelist,
                                  exclude=exclude,
                                  hostfile=hostfile,
                                  multiprog=multiprog,
                                  task_id=self.__pending_task_array_counter[task_name],
                                  begin=begin,
                                  deadline=deadline,
                                  time_limit=time_limit)
        if task_info.slurm_name in self.__pending_tasks \
            or task_info.slurm_name in self.__committed_tasks:
            raise ValueError(f"Task name {task_info.slurm_name} already existed.")
        self.__pending_task_array_counter[task_name] += 1
        # fractional GPU count to integer
        task_info.resolve_gpu_requirement()
        if task_info.multiprog:
            task_info = self.__resolve_multiprog_file(task_info)
            task_info = self.__resolve_envvar(task_info)
        self.__pending_task_counter[task_name] += task_info.ntasks
        self.__pending_worker_counter[task_name] += task_info.nworkers
        self.__pending_tasks[task_info.slurm_name] = task_info
        logger.info(f"Registered Slurm task {task_info.slurm_name} to scheduler.")

    def __resolve_multiprog_file(self, task_info: SlurmTaskInfo):
        cmd = task_info.cmd.format(group_id=str(self.__pending_task_counter[task_info.task_name]),
                                   group_offset='%t',
                                   group_size=str(task_info.ntasks),
                                   group_index=str(self.__pending_task_array_counter[task_info.task_name]))
        task_info.multiprog_content = f"0-{task_info.ntasks-1} {cmd}\n"
        return task_info

    def __resolve_envvar(self, task_info: SlurmTaskInfo):
        env_vars = task_info.env_vars.copy() if task_info.env_vars is not None else {}
        env_vars.update({
            "COMMIT_INDEX_START": str(self.__pending_worker_counter[task_info.task_name]),
            "TASK_SIZE": str(task_info.workers_per_task),
            "COMMIT_N_WORKERS": str(task_info.nworkers)
        })
        task_info.env_vars = env_vars
        return task_info

    def __allocate_and_commit_pending_tasks(self):
        """Allocate resources to all pending task specs.
        Generate hostfiles for each task info
        """
        start_time = time.monotonic()
        while True:
            try:
                fp = open(LOCK_FILE_NAME, "w")
                fcntl.flock(fp, fcntl.LOCK_EX)
                infos = list(self.__pending_tasks.values())
                infos = allocate_resources(infos)
                self.__pending_tasks = {info.slurm_name: info for info in infos}
                # logger.info("Allocated tasks: ")
                # for info in infos:
                #     logger.info(info)
                break
            except SlurmResourceNotEnoughException:
                logger.info("Not enough resources to allocate all pending tasks. Retrying ...")
                logger.info("Time since start: %d seconds", time.monotonic() - start_time)
                fcntl.flock(fp, fcntl.LOCK_UN)
                time.sleep(SCHEDULING_RETRY_INTERVAL_SECONDS)
                if time.monotonic() - start_time > SCHEDULING_TIMEOUT_MAX_SECONDS:
                    raise TimeoutError(f"Timeout waiting for {self.job_name} to schedule.")

        try:
            for slurm_name, task_info in self.__pending_tasks.items():
                task_info.commit()
                self.__committed_tasks[slurm_name] = task_info
            self.__pending_tasks = dict()
            states = [None for _ in self.__committed_tasks]
            while TaskState.PENDING in states or None in states:
                time.sleep(0.1)
                states = self.__update_all()
            # time.sleep(2)
            fcntl.flock(fp, fcntl.LOCK_UN)
            # self.__pending_task_array_counter = defaultdict(int)
            # self.__pending_task_counter = defaultdict(int)
            # self.__pending_worker_counter = defaultdict(int)
        except Exception as e:
            for task_info in self.__committed_tasks.values():
                task_info.cancel()
            fcntl.flock(fp, fcntl.LOCK_UN)
            raise e

    def stop(self, slurm_name: str):
        task_info = self.__committed_tasks.get(slurm_name, None)
        if task_info:
            task_info.cancel()

    def stop_all(self):
        for task_info in self.__committed_tasks.values():
            logger.info(f"Canceling task {task_info.slurm_name}")
            task_info.cancel()
        time.sleep(0.2)
        # print("before stop wait", self.__pending_tasks)
        self.wait(check_status=(),
                  remove_status=(TaskState.CANCELLED, TaskState.NOT_FOUND, TaskState.FAILED,
                                 TaskState.COMPLETED))

    def find(self, slurm_name: str) -> TaskInfo:
        task_info = self.__committed_tasks.get(slurm_name, None)
        if task_info is None or task_info.task_info is None:
            return TaskInfo(name=slurm_name, state=TaskState.NOT_FOUND)
        else:
            return task_info.task_info

    def find_all(self, task_name_regex: str = ".*") -> List[TaskInfo]:
        self.__update_all()
        infos = []
        for r in self.__committed_tasks.values():
            if r.task_info is None:
                continue
            if re.fullmatch(task_name_regex, r.slurm_name):
                infos.append(r.task_info)
        return infos

    def wait(
            self,
            timeout=None,
            check_status: Tuple[TaskState,
                                ...] = (TaskState.CANCELLED, TaskState.FAILED, TaskState.NOT_FOUND),
            remove_status: Tuple[TaskState, ...] = (TaskState.COMPLETED,),
            update=False,
    ):
        # before wait, commit all remaining pending task specs
        # TODO: grab global file lock to avoid multi-experiment deadlocks
        self.__allocate_and_commit_pending_tasks()
        # begin wait
        deadline = None if timeout is None else time.time() + timeout
        left = set(self.__committed_tasks.keys())
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
            for task_slurm_name in list(left):
                task_info = self.__committed_tasks[task_slurm_name]
                if task_info.slurm_id is None:
                    continue
                if task_info.task_info.state in check_status:
                    task_info.show_log()
                    raise TaskException(job_name=self.job_name,
                                        task_name=task_info.slurm_name,
                                        host=task_info.task_info.host,
                                        reason=task_info.task_info.state)
                if task_info.task_info.state in remove_status:
                    logger.info(f"Task {task_info.slurm_name} is {task_info.task_info.state}.(Removed)")
                    left.remove(task_slurm_name)
                    if update:
                        self.__committed_tasks.pop(task_slurm_name)
            time.sleep(2)

    def __update_all(self):
        states = []
        for task_info in self.__committed_tasks.values():
            state = task_info.update()
            states.append(state)
        return states

    # def __update_subset(self, slurm_names):
