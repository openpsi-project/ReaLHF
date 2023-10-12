from typing import Dict, List, Optional
import logging
import os
import re
import subprocess

from scheduler.client import SchedulerClient, SchedulerError, JobInfo, JobState

logger = logging.getLogger("Local Scheduler")


class LocalSchedulerClient(SchedulerClient):
    """Instead of talking to the scheduler server (the typical behaviour), this client starts jobs directly
    on the local host and keeps a collection of job processes.
    """

    def __init__(self, expr_name, trial_name):
        super().__init__(expr_name, trial_name)
        self._jobs: Dict[str, subprocess.Popen] = {}
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
        worker_type: str,
        cmd: str,
        gpu: int = 0,
        env_vars: Optional[Dict] = None,
        multiprog: bool = False,
        **kwargs,
    ):
        assert worker_type not in self._jobs
        if env_vars is None:
            env_vars = {}
        if gpu > 0:
            available_device_id = self._gpu_counter % len(self._cuda_devices)
            env_vars['CUDA_VISIBLE_DEVICES'] = str(self._cuda_devices[available_device_id])
            self._gpu_counter += 1
        cmd_envvar = ' '.join(str(k) + '=' + str(v) for k, v in env_vars.items()) + ' '
        if not multiprog:
            cmd = cmd_envvar + cmd
        else:
            cmd = cmd_envvar + cmd.format()
        logger.info("Starting local process with command: %s", cmd)
        process = subprocess.Popen(cmd, shell=isinstance(cmd, str))
        self._jobs[worker_type] = process

    def submit_array(self, worker_type, cmd, count, gpu: int = 0, **kwargs):
        if gpu > 0 and count > len(self._cuda_devices):
            raise RuntimeError(f"Number of \"{worker_type}\" exceeds the number of GPUs.")
        return super().submit_array(worker_type, cmd, count, gpu=gpu, **kwargs)

    def stop(self, worker_type):
        assert worker_type in self._jobs
        logger.info("Stopping local process, pid: %d", self._jobs[worker_type].pid)
        self._jobs[worker_type].kill()
        self._jobs[worker_type].wait()
        del self._jobs[worker_type]

    def stop_all(self):
        for name in list(self._jobs):
            self.stop(name)

    def find(self, job_name):
        if job_name in self._jobs:
            return JobInfo(name=job_name, state=JobState.RUNNING, host="localhost")
        else:
            return JobInfo(name=job_name, state=JobState.NOT_FOUND)

    def find_all(self, job_name_regex=".*"):
        rs = []
        for name in self._jobs:
            if re.fullmatch(job_name_regex, name):
                rs.append(self.find(name))
        return rs

    def wait(self, timeout=None, update=False, **kwargs):
        logger.info("Waiting %d local running processes, pids: %s", len(self._jobs),
                    " ".join(str(job.pid) for job in self._jobs.values()))
        to_remove = []
        try:
            for key, job in self._jobs.items():
                job.wait(timeout)
                if update:
                    to_remove.append(key)
        except subprocess.TimeoutExpired:
            raise TimeoutError()
        finally:
            for k in to_remove:
                self._jobs.pop(k)
