from typing import List, Optional
import dataclasses
import enum


class JobState(enum.Enum):
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


class JobException(Exception):

    def __init__(self, run_name, worker_type, host, reason: JobState):
        super().__init__(f"Job {run_name}:{worker_type} {reason} at node {host}")
        self.run_name = run_name
        self.worker_type = worker_type
        self.host = host
        self.reason = reason


@dataclasses.dataclass
class JobInfo:
    name: str
    state: JobState
    host: str = None  # The host on which the job is/was running. None if the job had not run.
    submit_time: str = None
    start_time: str = None
    slurm_id: str = None  # Slurm only. The Slurm id of the job.


class SchedulerClient:

    def __init__(self, expr_name, trial_name):
        self.expr_name = expr_name
        self.trial_name = trial_name
        self.run_name = f"{expr_name}_{trial_name}"

    def submit(self, worker_type, cmd, **kwargs):
        """Submits a job to the scheduler. Raises exception if the job is already running.

        Args:
            worker_type: The worker type to be submitted. The job name is specified when initializing the client.
            cmd (str or List[str]): The command of this job. If this is str, the command is parsed by
                shell; otherwise it is executed directly.
        """
        raise NotImplementedError()

    def submit_array(self, worker_type, cmd, count, **kwargs):
        """Submits an array of jobs to the scheduler.

        Args:
            worker_type: The worker type to be submitted, shared by all jobs.
            cmd: Command template of the jobs that may contain an "{index}" format placeholder.
            count: Number of jobs. The indices of the jobs shall be 0..count-1.
        """
        for index in range(count):
            self.submit(worker_type + "_" + str(index), cmd.format(index=index, count=count), **kwargs)

    def stop(self, job_name):
        """Stops a running job. Raises exception if there is no such job, but passes if the job has stopped
        either successfully or not.
        """
        raise NotImplementedError()

    def stop_all(self):
        """Stops the whole job.
        """
        raise NotImplementedError()

    def find(self, job_name) -> Optional[JobInfo]:
        """Gets the status of a job of this job.

        Args:
            job_name: Name of the job.

        Returns:
            A JobInfo if the job is found, or None otherwise.
        """
        raise NotImplementedError()

    def find_all(self, job_name_regex=".*") -> List[JobInfo]:
        """Finds jobs.

        Args:
            job_name_regex: job name regex.

        Returns:
            A list of found JobInfo.
        """
        raise NotImplementedError()

    def wait(self, timeout=None, **kwargs):
        """Waits until all jobs submitted via this client instance finish.
        """
        raise NotImplementedError()


def remote_worker_cmd(expr_name, trial_name, debug, worker_type):
    # requires information in scheduler package
    return f"python3 {'' if debug else '-O'} -m apps.remote worker -w {worker_type} " \
           f"-e {expr_name} -f {trial_name} -i {{jobstep_id}} -g {{n_jobsteps}} -r {{worker_submission_index}} " \
           f"-p {{wprocs_per_jobstep}} -j {{wprocs_in_job}} -o {{wproc_offset}}"


def setup_cmd(expr_name, trial_name, debug):
    return f"python3 {'' if debug else '-O'} -m apps.remote reset_name_resolve -e {expr_name} -f {trial_name}"


def control_cmd(expr_name, trial_name, debug, ignore_worker_error, controller_type):
    return (f"python3 {'' if debug else '-O'} -m apps.remote controller -e {expr_name} -f {trial_name} "
            f"--{'ignore_worker_error' if ignore_worker_error else 'raise_worker_error'} "
            f"--type {controller_type}")


def ray_cluster_cmd(expr_name, trial_name, worker_type):
    flags = [f"-e {expr_name}", f"-f {trial_name}", f"-w {worker_type}"]
    return (f"python3 -m apps.remote ray -i {{index}} -g {{count}} {' '.join(flags)}")


def make(mode, expr_name, trial_name, **kwargs) -> SchedulerClient:
    if mode == "slurm":
        from scheduler.slurm.client import SlurmSchedulerClient
        return SlurmSchedulerClient(expr_name, trial_name)
    elif mode == 'local':
        from scheduler.local.client import LocalSchedulerClient
        return LocalSchedulerClient(expr_name, trial_name)
    else:
        raise NotImplementedError(f"Scheduler {mode} not found")
