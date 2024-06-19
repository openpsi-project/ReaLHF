from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import copy
import dataclasses
import enum
import getpass
import json
import os
import time
import traceback

import ray
import ray.util.queue as rq

from realhf.base import logging, name_resolve, names
from realhf.base.cluster import spec as cluster_spec
from realhf.system import load_worker, worker_base, worker_control, WORKER_TYPES
from realhf.system.worker_base import WorkerServerStatus as Wss
import realhf.api.core.system_api as system_api

CONNECTION_RETRY_AFTER_SECONDS = 360

logger = logging.getLogger("controller")


@dataclasses.dataclass
class TrialStatus:
    experiment_name: str
    trial_name: str
    running_workers: Dict[str, List[str]] = dataclasses.field(
        default_factory=dict
    )


@dataclasses.dataclass
class TrialHistory:
    experiment_name: str
    trial_name: str
    age_days: int


class ControllerExitStatus(enum.Enum):
    SUCCESS = 0
    TIMEOUT = 1
    INTERRUPTED = 9
    FAIL = 101
    LOST = 102
    UNKNOWN = 404


class Controller:

    def __init__(
        self, experiment_name, trial_name, panel: worker_base.WorkerControlPanel
    ):
        assert "_" not in experiment_name, (
            f"_ not allowed in experiment_name (args: -e) "
            f"{experiment_name}, use '-' instead."
        )
        assert (
            "_" not in trial_name
        ), f"_ not allowed in trial_name (args: -f) {trial_name}, use '-' instead."
        self.experiment_name = experiment_name
        self.trial_name = trial_name

        logger.info("Experiment: %s %s", self.experiment_name, self.trial_name)

        self.__control = panel
        self.json_config_file_path = f"{cluster_spec.fileroot}/logs/{getpass.getuser()}/{self.experiment_name}_{self.trial_name}"

    def reconnect(self):
        """Automatically reconnect to workers. And list all jobs to scheduler."""
        self.__control.auto_connect()

    def start(
        self, experiment: system_api.Experiment, ignore_worker_error=False
    ):
        if ignore_worker_error:
            check_worker_status = ()
            remove_worker_status = (
                Wss.COMPLETED,
                Wss.ERROR,
                Wss.LOST,
                Wss.UNKNOWN,
            )
        else:
            check_worker_status = (Wss.ERROR, Wss.LOST, Wss.UNKNOWN)
            remove_worker_status = (Wss.COMPLETED,)

        scheduling: system_api.ExperimentScheduling = (
            experiment.scheduling_setup()
        )
        setup = experiment.initial_setup()
        setup.set_worker_information(
            experiment_name=self.experiment_name, trial_name=self.trial_name
        )

        if setup.config is not None:
            os.makedirs(self.json_config_file_path, exist_ok=True)
            with open(
                os.path.join(self.json_config_file_path, "config.json"), "w"
            ) as f:
                json.dump(asdict(setup.config), f, indent=4)

        # Scheduling and connecting to workers.
        # TODO for heterogeneous workers of the same type k, list scheduling[k] and setup[k] should match.
        workers_configs = [
            (k, getattr(setup, k), getattr(scheduling, k)) for k in WORKER_TYPES
        ]

        for name, config, schedule in workers_configs:
            count = (
                sum([s.count for s in schedule])
                if isinstance(schedule, list)
                else schedule.count
            )
            if len(config) != count:
                logger.error(
                    "Scheduling and config mismatch, interrupting all workers."
                )
                self.interrupt()
                raise IndexError(
                    f"Configuration has {len(config)} {name}, {count} scheduled."
                )
            logger.info(f"Configuration has {len(config)} {name}.")

        name_resolve.add(
            names.trial_registry(self.experiment_name, self.trial_name),
            value=datetime.now().strftime("%Y%m%d"),
            delete_on_exit=False,
            replace=True,
        )
        name_resolve.add(
            names.worker_status(
                experiment_name=self.experiment_name,
                trial_name=self.trial_name,
                worker_name="ctl",
            ),
            value="READY",
            delete_on_exit=True,
        )

        while True:
            try:
                logger.info("Connecting to workers...")
                self.__control.connect(
                    [
                        self.__control.name(name, i)
                        for name, cfgs, _ in workers_configs
                        for i in range(len(cfgs))
                    ],
                    progress=True,
                    timeout=CONNECTION_RETRY_AFTER_SECONDS,
                    raises_timeout_error=True,
                )
                break

            except TimeoutError:
                logger.info("Connecting to workers timeout. Retrying...")
            except KeyboardInterrupt as e:
                logger.info("Interrupted by user. Stopping all and exiting...")
                raise e

        name_resolve.delete(
            names.worker_status(
                experiment_name=self.experiment_name,
                trial_name=self.trial_name,
                worker_name="ctl",
            )
        )

        # Configure workers.
        try:
            for name, cfgs, _ in workers_configs:
                logger.info(f"Configuring Workers: {name}...")
                self.__control.group_request(
                    "configure",
                    worker_names=[
                        self.__control.name(name, i) for i in range(len(cfgs))
                    ],
                    worker_kwargs=[dict(config=cfg) for cfg in cfgs],
                    progress=True,
                )
        except Exception as e:
            logger.error(f"Configuring Failed: {e}. Exiting Workers.")
            logger.error(traceback.format_exc())
            self.interrupt(wait_timeout=120)
            raise e

        logger.info("Start workers...")
        self.__control.group_request("start")
        logger.info("Started.")
        try:
            self.wait(
                timeout=None,
                check_status=check_worker_status,
                remove_status=remove_worker_status,
            )
        except worker_base.WorkerException as e:
            logger.error(e)
            self.interrupt(wait_timeout=30)
        except KeyboardInterrupt:
            logger.info("Interrupted.")
            self.interrupt(wait_timeout=30)

    def wait(
        self,
        timeout: Optional[int],
        check_status: Tuple[Wss, ...],
        remove_status: Tuple[Wss, ...],
    ):
        deadline = None if timeout is None else time.time() + timeout
        left = set(self.__control.worker_names)
        num_jobs_left = len(left)
        logger.info(f"Waiting for {num_jobs_left} jobs.")
        current_status = {
            name: Wss.UNKNOWN for name in self.__control.worker_names
        }
        while len(left) > 0:
            logger.debug(
                f"JOBS LEFT: {[str(len([l for l in left if job_type in l])) + ' ' + job_type for job_type in set([job_id.split('/')[0] for job_id in left])]}"
            )
            if len(left) < num_jobs_left:
                num_jobs_left = len(left)
                logger.info(f"Waiting for {num_jobs_left} jobs.")
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(
                    f"Timeout waiting for {self.experiment_name, self.trial_name}: {', '.join(sorted(left))}"
                )
            for worker_name, worker_status in self.__control.pulse().items():
                if worker_status in check_status:
                    raise worker_base.WorkerException(
                        worker_name, worker_status, "experiment is running."
                    )
                if worker_status in remove_status:
                    if worker_name in current_status:
                        logger.debug(
                            f"Worker {worker_name} is {worker_status}. Removed from waiting list."
                        )
                        current_status.pop(worker_name)
                    else:
                        pass
                else:
                    if current_status.get(worker_name, None) != worker_status:
                        current_status.update({worker_name: worker_status})
                        logger.debug(
                            f"Update worker status: {worker_name} -> {worker_status}"
                        )

            left = set(current_status.keys())
            time.sleep(10)

    def stop(self):
        """Stop the experiment.
        Note:
            This method assumes that the controller and scheduler is connected to the correct workers. To ensure this,
            call controller.reconnect before your call controller.stop.
        """
        raise NotImplementedError()

    def interrupt(self, wait_timeout=120):
        """Interrupt the experiment."""
        logger.info("Interrupting experiment")
        self.__control.group_request("interrupt", wait_response=False)
        try:
            self.wait(
                timeout=wait_timeout,
                check_status=(),
                remove_status=(
                    Wss.ERROR,
                    Wss.LOST,
                    Wss.COMPLETED,
                    Wss.INTERRUPTED,
                ),
            )
        except TimeoutError:
            raise RuntimeError(
                f"Fail to interrupt workers, timeout={wait_timeout}."
            )


def run_ray_worker(
    worker_type,
    idx,
    experiment_name,
    trial_name,
    comm: Tuple[rq.Queue, rq.Queue],
):
    worker_name = f"{worker_type}/{idx}"
    server = worker_control.make_server(
        "ray",
        worker_name=worker_name,
        experiment_name=experiment_name,
        trial_name=trial_name,
        comm=comm,
    )
    worker = load_worker(worker_type)(server=server)
    try:
        worker.run()
    except Exception as e:
        logging.error("Worker %s failed with exception: %s", worker_name, e)
        logging.error(traceback.format_exc())
        raise e


class RayController:
    """A controller that uses Ray to manage workers.

    It uses the basic Controller to configure workers.
    Besides, it launchs all remote workers using Ray,
    instead of submitting them to the scheduelr.
    """

    def __init__(self, experiment_name, trial_name, local_mode: bool):
        # base controller will be lazier initialized when launching workers.
        self.__experiment_name = experiment_name
        self.__trial_name = trial_name
        self.__base_controller = None

        self.__workers_reply_comm = None
        self.__workers_request_comm = None
        self.__workers_ref = None

        self.__local_mode = local_mode

    def _launch_workers(
        self, workers_configs: List[Tuple[str, List, system_api.TasksGroup]]
    ):
        # Launch remote workers.
        logger.info("Launching remote workers using Ray...")
        self.__workers_ref: Dict[str, ray.ObjectRef] = {}
        self.__workers_request_comm: Dict[str, rq.Queue] = dict()
        self.__workers_reply_comm: Dict[str, rq.Queue] = dict()
        for worker_type, config, schedule in workers_configs:
            count = len(config)
            all_schedules: List[system_api.TasksGroup] = []
            if isinstance(schedule, List):
                for s in schedule:
                    for _ in range(s.count):
                        s_ = copy.deepcopy(s)
                        s_.count = 1
                        all_schedules.append(s_)
            else:
                for _ in range(schedule.count):
                    s_ = copy.deepcopy(schedule)
                    s_.count = 1
                    all_schedules.append(s_)
            assert len(all_schedules) == len(config)
            comms = [
                (rq.Queue(maxsize=8), rq.Queue(maxsize=8))
                for _ in all_schedules
            ]
            jobs = [
                ray.remote(
                    num_cpus=sch.scheduling.cpu,
                    num_gpus=sch.scheduling.gpu,
                    memory=sch.scheduling.mem,
                    name=f"{worker_type}/{idx}",
                )(run_ray_worker).remote(
                    worker_type,
                    idx,
                    self.__experiment_name,
                    self.__trial_name,
                    comm,
                )
                for idx, (comm, sch) in enumerate(zip(comms, all_schedules))
            ]
            for idx, (job, c) in enumerate(zip(jobs, comms)):
                name = f"{worker_type}/{idx}"
                self.__workers_ref[name] = job
                self.__workers_request_comm[name] = c[0]
                self.__workers_reply_comm[name] = c[1]
            logger.info(f"Launched {count} {worker_type}.")

        panel = worker_control.make_control(
            "ray",
            self.__experiment_name,
            self.__trial_name,
            request_comms=self.__workers_request_comm,
            reply_comms=self.__workers_reply_comm,
        )
        self.__base_controller = Controller(
            self.__experiment_name, self.__trial_name, panel
        )
        logger.info("All Ray workers are lauched.")

    def start(
        self, experiment: system_api.Experiment, ignore_worker_error=False
    ):
        scheduling: system_api.ExperimentScheduling = (
            experiment.scheduling_setup()
        )
        setup = experiment.initial_setup()
        setup.set_worker_information(
            experiment_name=self.__experiment_name, trial_name=self.__trial_name
        )
        workers_configs = [
            (k, getattr(setup, k), getattr(scheduling, k)) for k in WORKER_TYPES
        ]
        workers_configs: List[Tuple[str, List, system_api.TasksGroup]]

        if self.__local_mode:
            ray.init()
        else:
            for name, config, schedule in workers_configs:
                count = (
                    sum([s.count for s in schedule])
                    if isinstance(schedule, list)
                    else schedule.count
                )
                if len(config) != count:
                    logger.error(
                        "Scheduling and config mismatch, interrupting all workers."
                    )
                    raise IndexError(
                        f"Configuration has {len(config)} {name}, {count} scheduled."
                    )
                for idx in range(count):
                    try:
                        name_resolve.wait(
                            names.ray_cluster(
                                self.__experiment_name,
                                self.__trial_name,
                                f"{name}/{idx}",
                            ),
                            timeout=300,
                        )
                    except TimeoutError:
                        raise RuntimeError(
                            f"Timeout waiting for Ray cluster node {name}/{idx} to start."
                        )
            logger.info("Ray cluster started.")

            try:
                ray_head_addr = name_resolve.wait(
                    names.ray_cluster(
                        self.__experiment_name, self.__trial_name, "address"
                    ),
                    timeout=300,
                )
            except TimeoutError:
                raise RuntimeError(
                    "Timeout waiting for ray cluster head address."
                )
            ray.init(address=ray_head_addr)

        logger.info("Ray initialized! Ready to run workers.")

        try:
            self._launch_workers(workers_configs)
            self.__base_controller.start(experiment, ignore_worker_error)
        except Exception as e:
            self.shutdown()

    def shutdown(self):
        ray_exiting_name = names.ray_cluster(
            self.__experiment_name, self.__trial_name, "exiting"
        )
        name_resolve.add(ray_exiting_name, value="1", delete_on_exit=True)
        del self.__workers_reply_comm
        del self.__workers_request_comm
        del self.__workers_ref
        ray.shutdown()
