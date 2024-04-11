from collections import defaultdict
from typing import Dict, List, Optional
import os
import re
import subprocess

import psutil

from scheduler.client import JobInfo, JobState, SchedulerClient, SchedulerError
import base.logging as logging

logger = logging.getLogger("Local Scheduler")


def terminate_process_and_children(pid: int):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            terminate_process_and_children(child.pid)
        parent.terminate()
    except psutil.NoSuchProcess:
        pass


class LocalSchedulerClient(SchedulerClient):
    # TODO: log to file
    """Instead of talking to the scheduler server (the typical behaviour), this client starts jobs directly
    on the local host and keeps a collection of job processes.
    """

    def __init__(self, expr_name, trial_name):
        super().__init__(expr_name, trial_name)
        self._jobs: Dict[str, subprocess.Popen] = {}
        self._running_worker_types = []

        self._gpu_counter = 0
        self._cuda_devices: List[str] = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")

        self._job_counter: Dict[str, int] = defaultdict(int)
        self._job_with_gpu: Dict[str, bool] = defaultdict(int)
        self._job_env_vars: Dict[str, Dict] = defaultdict(int)
        self._job_cmd = {}

        if len(self._cuda_devices) < 1:
            raise RuntimeError(
                f"Local mode can only run when there is at least one GPU. "
                f"CUDA_VISIBLE_DEVICES is currently set to {os.environ['CUDA_VISIBLE_DEVICES']}.")

    def __del__(self):
        self.wait(commit=False)

    def submit_array(
        self,
        worker_type: str,
        cmd: str,
        count: int = 1,
        gpu: int = 0,
        env_vars: Optional[Dict] = None,
        **kwargs,
    ):
        if env_vars is None:
            env_vars = {}

        self._job_counter[worker_type] += count
        if worker_type in self._job_with_gpu:
            assert self._job_with_gpu[worker_type] == (
                gpu > 0), "All workers of the same type must either use GPU or not use GPU."
        else:
            self._job_with_gpu[worker_type] = gpu > 0

        if worker_type in self._job_env_vars:
            assert (self._job_env_vars[worker_type] == env_vars
                    ), "All workers of the same type must have the same env vars."
        else:
            self._job_env_vars[worker_type] = env_vars

        if worker_type in self._job_cmd:
            assert self._job_cmd[worker_type] == cmd, "All workers of the same type must have the same cmd."
        else:
            self._job_cmd[worker_type] = cmd

    def submit(self, worker_type, cmd, **kwargs):
        self.submit_array(worker_type, cmd, count=1, **kwargs)

    def __commit_all(self):
        for worker_type, count, use_gpu, env_vars in zip(
                self._job_counter.keys(),
                self._job_counter.values(),
                self._job_with_gpu.values(),
                self._job_env_vars.values(),
        ):
            for i in range(count):
                if use_gpu:
                    available_device_id = self._gpu_counter % len(self._cuda_devices)
                    env_vars["CUDA_VISIBLE_DEVICES"] = str(self._cuda_devices[available_device_id])
                    self._gpu_counter += 1
                cmd = (" ".join(str(k) + "=" + str(v)
                                for k, v in env_vars.items()) + " " + self._job_cmd[worker_type])
                # Run `apps.remote` with a single process.
                # This simulates a multi-prog slurm job with `count` jobsteps, with each jobstep having a single process.
                cmd = cmd.format(
                    jobstep_id=i,
                    n_jobsteps=count,
                    worker_submission_index=0,
                    wprocs_per_jobstep=1,
                    wprocs_in_job=count,
                    wproc_offset=0,
                )
                logger.info("Starting local process with command: %s", cmd)
                process = subprocess.Popen(cmd, shell=isinstance(cmd, str))
                self._jobs[f"{worker_type}/{i}"] = process
            self._running_worker_types.append(worker_type)

    def stop(self, worker_type):
        assert any(k.startswith(worker_type) for k in self._jobs)
        keys = [k for k, p in self._jobs.items() if k.startswith(worker_type)]
        procs = [p for k, p in self._jobs.items() if k.startswith(worker_type)]
        logger.info("Stopping local process, pid: %s", [p.pid for p in procs])
        for p in procs:
            terminate_process_and_children(p.pid)
        for p in procs:
            p.wait()
        for k, p in zip(keys, procs):
            self._jobs.pop(k)
            del p
        self._running_worker_types.remove(worker_type)

    def stop_all(self):
        for name in self._job_counter:
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

    def wait(self, timeout=None, update=False, commit=True, **kwargs):
        if commit:
            self.__commit_all()
        logger.info(
            "Waiting %d local running processes, pids: %s",
            len(self._jobs),
            " ".join(str(job.pid) for job in self._jobs.values()),
        )
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
                worker_type = k.split("/")[0]
                assert worker_type in self._job_counter
                assert worker_type in self._job_with_gpu
                assert worker_type in self._job_env_vars
                assert worker_type in self._job_cmd
                self._job_counter.pop(worker_type)
                self._job_with_gpu.pop(worker_type)
                self._job_env_vars.pop(worker_type)
                self._job_cmd.pop(worker_type)
