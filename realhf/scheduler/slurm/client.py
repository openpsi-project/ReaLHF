import fcntl
import re
import subprocess
import time
from collections import defaultdict
from typing import Dict, List, Literal, Optional, Tuple

import realhf.base.logging as logging
from realhf.base.cluster import spec as cluster_spec
from realhf.base.constants import SLURM_LOCK_FILE_NAME as LOCK_FILE_NAME
from realhf.scheduler.client import JobException, JobInfo, JobState, SchedulerClient
from realhf.scheduler.slurm.utils import (
    SlurmLaunchInfo,
    SlurmResource,
    SlurmResourceNotEnoughException,
    allocate_resources,
)

logger = logging.getLogger("Slurm-scheduler")

SCHEDULING_RETRY_INTERVAL_SECONDS = 30
SCHEDULING_TIMEOUT_MAX_SECONDS = 3600 * 24


class SlurmSchedulerClient(SchedulerClient):
    """Uses Slurm (https://slurm.schedmd.com/overview.html)."""

    def __init__(self, expr_name, trial_name):
        super().__init__(expr_name, trial_name)

        self.__pending_jobs: Dict[str, SlurmLaunchInfo] = dict()
        self.__committed_jobs: Dict[str, SlurmLaunchInfo] = dict()

        self.__submission_counter = defaultdict(int)
        self.__wprocs_counter = defaultdict(int)

    def submit(self, worker_type, cmd, **kwargs):
        self.submit_array(worker_type, cmd, count=1, **kwargs)

    def submit_array(
        self,
        worker_type: str,
        cmd: str,  # XXX: should be None for workers
        count: int,
        cpu: int = 1,
        gpu_type: str = "geforce",
        gpu: int = 0,
        mem: int = 1024,  # MB
        env_vars: Optional[Dict] = None,
        container_image: str = cluster_spec.gpu_image,
        container_mounts: str = cluster_spec.default_mount,
        node_type: Optional[str] = None,
        nodelist: Optional[str] = None,
        exclude: Optional[str] = None,
        hostfile: bool = True,
        multiprog: bool = True,
        begin: str = None,
        deadline: str = None,
        time_limit: str = None,
    ):
        # record launch information, do not submit to slurm until `wait()` is called
        # NOTE: fractional GPU requirement will be resolved automatically in `__post_init__` of SlurnLaunchInfo
        launch_info = SlurmLaunchInfo(
            worker_type=worker_type,
            wprocs_in_job=count,
            resource_requirement=SlurmResource(
                mem=mem, cpu=cpu, gpu=gpu, gpu_type=gpu_type
            ),
            cmd=cmd,
            run_name=self.run_name,
            exper_name=self.expr_name,
            trial_name=self.trial_name,
            container_image=container_image,
            container_mounts=container_mounts,
            env_vars=env_vars,
            node_type=node_type,
            nodelist=nodelist,
            exclude=exclude,
            hostfile=hostfile,
            multiprog=multiprog,
            worker_submission_idx=self.__submission_counter[worker_type],
            begin=begin,
            deadline=deadline,
            time_limit=time_limit,
        )

        if (
            launch_info.slurm_name in self.__pending_jobs
            or launch_info.slurm_name in self.__committed_jobs
        ):
            raise ValueError(f"job name {launch_info.slurm_name} already existed.")

        if launch_info.multiprog:
            launch_info = self.__resolve_multiprog_file(launch_info)

        self.__submission_counter[worker_type] += 1
        self.__wprocs_counter[worker_type] += count

        self.__pending_jobs[launch_info.slurm_name] = launch_info
        logger.info(f"Registered Slurm job {launch_info.slurm_name} to scheduler.")

    def __resolve_multiprog_file(self, launch_info: SlurmLaunchInfo):
        worker_type = launch_info.worker_type
        cmd = launch_info.cmd.format(
            jobstep_id="%t",
            n_jobsteps=launch_info.n_jobsteps,
            worker_submission_index=self.__submission_counter[worker_type],
            wprocs_per_jobstep=launch_info.wprocs_per_jobstep,
            wprocs_in_job=launch_info.wprocs_in_job,
            wproc_offset=self.__wprocs_counter[worker_type],
        )
        launch_info.multiprog_content = f"0-{launch_info.n_jobsteps - 1} {cmd}\n"
        return launch_info

    def __allocate_and_commit_pending_jobs(self):
        """Allocate resources to all pending job specs.

        Generate hostfiles for each job info
        """
        start_time = time.monotonic()
        while True:
            try:
                fp = open(LOCK_FILE_NAME, "w")
                fcntl.flock(fp, fcntl.LOCK_EX)
                infos = list(self.__pending_jobs.values())
                infos = allocate_resources(infos)
                self.__pending_jobs = {info.slurm_name: info for info in infos}
                # logger.info("Allocated jobs: ")
                # for info in infos:
                #     logger.info(info)
                break
            except SlurmResourceNotEnoughException:
                logger.critical(
                    "Not enough resources to allocate all pending jobs. Retrying ..."
                )
                logger.warning(
                    "Time since start: %d seconds",
                    time.monotonic() - start_time,
                )
                fcntl.flock(fp, fcntl.LOCK_UN)
                time.sleep(SCHEDULING_RETRY_INTERVAL_SECONDS)
                if time.monotonic() - start_time > SCHEDULING_TIMEOUT_MAX_SECONDS:
                    raise TimeoutError(
                        f"Timeout waiting for {self.run_name} to schedule."
                    )

        try:
            for slurm_name, launch_info in self.__pending_jobs.items():
                launch_info.commit()
                self.__committed_jobs[slurm_name] = launch_info
            self.__pending_jobs = dict()
            states = [None for _ in self.__committed_jobs]
            while JobState.PENDING in states or None in states:
                time.sleep(0.1)
                states = self.__update_all()
            # time.sleep(2)
            fcntl.flock(fp, fcntl.LOCK_UN)
        except Exception as e:
            for launch_info in self.__committed_jobs.values():
                launch_info.cancel()
            fcntl.flock(fp, fcntl.LOCK_UN)
            raise e

    def stop(self, slurm_name: str):
        launch_info = self.__committed_jobs.get(slurm_name, None)
        if launch_info:
            launch_info.cancel()

    def stop_all(self, signal: Literal["SIGINT", "SIGKILL"] = "SIGKILL"):
        for launch_info in self.__committed_jobs.values():
            logger.info(f"Canceling job {launch_info.slurm_name}")
            launch_info.cancel(signal)
        time.sleep(0.2)
        self.wait(
            check_status=(),
            remove_status=(
                JobState.CANCELLED,
                JobState.NOT_FOUND,
                JobState.FAILED,
                JobState.COMPLETED,
            ),
        )

    def find(self, slurm_name: str) -> JobInfo:
        launch_info = self.__committed_jobs.get(slurm_name, None)
        if launch_info is None or launch_info.job_info is None:
            return JobInfo(name=slurm_name, state=JobState.NOT_FOUND)
        else:
            return launch_info.job_info

    def find_all(self, job_name_regex: str = ".*") -> List[JobInfo]:
        self.__update_all()
        infos = []
        for r in self.__committed_jobs.values():
            if r.job_info is None:
                continue
            if re.fullmatch(job_name_regex, r.slurm_name):
                infos.append(r.job_info)
        return infos

    def wait(
        self,
        timeout=None,
        check_status: Tuple[JobState, ...] = (
            JobState.CANCELLED,
            JobState.FAILED,
            JobState.NOT_FOUND,
        ),
        remove_status: Tuple[JobState, ...] = (JobState.COMPLETED,),
        update=False,
    ):
        # before wait, commit all remaining pending jobs
        # TODO: grab global file lock to avoid multi-experiment deadlocks
        self.__allocate_and_commit_pending_jobs()
        # begin wait
        deadline = None if timeout is None else time.time() + timeout
        left = set(self.__committed_jobs.keys())
        num_jobs_left = len(left)
        logger.info(
            f"Waiting for {num_jobs_left} jobs. Jobs IDs: "
            f"{','.join(sorted([x.job_info.slurm_id for x in self.__committed_jobs.values()]))}."
        )
        while len(left) > 0:
            if len(left) < num_jobs_left:
                num_jobs_left = len(left)
                logger.info(f"Waiting for {num_jobs_left} jobs.")
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(
                    f"Timeout waiting for {self.run_name}: {', '.join(sorted(left))}"
                )
            try:
                self.__update_all()
            except subprocess.CalledProcessError:
                logger.warning(
                    "Calling squeue failed. Check slurm manually if you continue to see this warning."
                )
                time.sleep(30)
                continue
            for job_slurm_name in list(left):
                launch_info = self.__committed_jobs[job_slurm_name]
                if launch_info.slurm_id is None:
                    continue
                if launch_info.job_info.state in check_status:
                    launch_info.show_log()
                    raise JobException(
                        run_name=self.run_name,
                        worker_type=launch_info.worker_type,
                        host=launch_info.job_info.host,
                        reason=launch_info.job_info.state,
                    )
                if launch_info.job_info.state in remove_status:
                    logger.info(
                        f"Job {launch_info.slurm_name} is {launch_info.job_info.state}.(Removed)"
                    )
                    left.remove(job_slurm_name)
                    if update:
                        self.__committed_jobs.pop(job_slurm_name)
            time.sleep(2)

    def __update_all(self):
        states = []
        for launch_info in self.__committed_jobs.values():
            state = launch_info.update()
            states.append(state)
        return states
