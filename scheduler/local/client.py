from typing import Dict, List, Optional
import logging
import os
import re
import subprocess

from scheduler.client import SchedulerClient, SchedulerError, TaskInfo, TaskState

logger = logging.getLogger("Local Scheduler")


class LocalSchedulerClient(SchedulerClient):
    """Instead of talking to the scheduler server (the typical behaviour), this client starts tasks directly
    on the local host and keeps a collection of task processes.
    """

    def __init__(self, job_name):
        super().__init__(job_name)
        self._tasks: Dict[str, subprocess.Popen] = {}
        self._gpu_counter = 0
        self._cuda_devices: List[str] = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
        if len(self._cuda_devices) < 1:
            raise RuntimeError(
                f"Local mode can only run when there is at least one GPU. "
                f"CUDA_VISIBLE_DEVICES is currently set to {os.environ['CUDA_VISIBLE_DEVICES']}.")

    def __del__(self):
        self.wait()

    def submit(
        self,
        task_name,
        cmd,
        gpu: int = 0,
        env_vars: Optional[Dict] = None,
        **kwargs,
    ):
        assert task_name not in self._tasks
        if env_vars is None:
            env_vars = {}
        if gpu > 0:
            available_device_id = self._gpu_counter % len(self._cuda_devices)
            env_vars['CUDA_VISIBLE_DEVICES'] = str(self._cuda_devices[available_device_id])
            self._gpu_counter += 1
        cmd = ' '.join(str(k) + '=' + str(v) for k, v in env_vars.items()) + ' ' + cmd
        logger.info("Starting local process with command: %s", cmd)
        process = subprocess.Popen(cmd, shell=isinstance(cmd, str))
        self._tasks[task_name] = process

    def submit_array(self, task_name, cmd, count, gpu: int = 0, **kwargs):
        if gpu > 0 and count > len(self._cuda_devices):
            raise RuntimeError(f"Number of \"{task_name}\" exceeds the number of GPUs.")
        return super().submit_array(task_name, cmd, count, gpu=gpu, **kwargs)

    def stop(self, task_name):
        assert task_name in self._tasks
        logger.info("Stopping local process, pid: %d", self._tasks[task_name].pid)
        self._tasks[task_name].kill()
        self._tasks[task_name].wait()
        del self._tasks[task_name]

    def stop_all(self):
        for name in list(self._tasks):
            self.stop(name)

    def find(self, task_name):
        if task_name in self._tasks:
            return TaskInfo(name=task_name, state=TaskState.RUNNING, host="localhost")
        else:
            return TaskInfo(name=task_name, state=TaskState.NOT_FOUND)

    def find_all(self, task_name_regex=".*"):
        rs = []
        for name in self._tasks:
            if re.fullmatch(task_name_regex, name):
                rs.append(self.find(name))
        return rs

    def wait(self, timeout=None, update=False, **kwargs):
        logger.info("Waiting %d local running processes, pids: %s", len(self._tasks),
                    " ".join(str(task.pid) for task in self._tasks.values()))
        to_remove = []
        try:
            for key, task in self._tasks.items():
                task.wait(timeout)
                if update:
                    to_remove.append(key)
        except subprocess.TimeoutExpired:
            raise TimeoutError()
        finally:
            for k in to_remove:
                self._tasks.pop(k)
