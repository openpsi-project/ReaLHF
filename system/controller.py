from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import dataclasses
import enum
import logging
import time
import copy

import ray
import ray.util.queue as rq

from system import WORKER_TYPES, Controller, run_worker, load_worker
from system.worker_base import WorkerServerStatus as Wss
from system.worker_control import RayServer
import api.config
import base.monitoring
import base.name_resolve
import base.names as names
import system.worker_base
import system.worker_control

CONNECTION_RETRY_AFTER_SECONDS = 360

logger = logging.getLogger("controller")


@dataclasses.dataclass
class TrialStatus:
    experiment_name: str
    trial_name: str
    running_workers: Dict[str, List[str]] = dataclasses.field(default_factory=dict)


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


class ZmqController(Controller):

    def __init__(self, experiment_name, trial_name):
        super().__init__(experiment_name, trial_name)

        self.__control = system.worker_control.make_control(
            type_='zmq',
            experiment_name=self.experiment_name,
            trial_name=self.trial_name,
        )

    def reconnect(self):
        """Automatically reconnect to workers. And list all jobs to scheduler.
        """
        self.__control.auto_connect()

    def start(self, experiment: api.config.Experiment, ignore_worker_error=False):
        if ignore_worker_error:
            check_worker_status = ()
            remove_worker_status = (Wss.COMPLETED, Wss.ERROR, Wss.LOST, Wss.UNKNOWN)
        else:
            check_worker_status = (Wss.ERROR, Wss.LOST, Wss.UNKNOWN)
            remove_worker_status = (Wss.COMPLETED,)

        scheduling: api.config.ExperimentScheduling = experiment.scheduling_setup()
        setup = experiment.initial_setup()
        setup.set_worker_information(experiment_name=self.experiment_name, trial_name=self.trial_name)

        # Scheduling and connecting to workers.
        workers_configs = []
        workers_configs = [(k, getattr(setup, k), getattr(scheduling, k)) for k in WORKER_TYPES]

        for name, config, schedule in workers_configs:
            count = sum([s.count for s in schedule]) if isinstance(schedule, list) else schedule.count
            if len(config) != count:
                logger.error("Scheduling and config mismatch, interrupting all workers.")
                self.interrupt()
                raise IndexError(f"Configuration has {len(config)} {name}, {count} scheduled.")
            logger.info(f"Configuration has {len(config)} {name}.")

        # State clean-up.
        logger.info("Cleaning up previous states")
        base.name_resolve.clear_subtree(names.trial_root(self.experiment_name, self.trial_name))
        base.name_resolve.add(names.trial_registry(self.experiment_name, self.trial_name),
                              value=datetime.now().strftime("%Y%m%d"),
                              delete_on_exit=False,
                              replace=True)
        base.name_resolve.add(names.worker_status(experiment_name=self.experiment_name,
                                                  trial_name=self.trial_name,
                                                  worker_name="ctl"),
                              value="READY",
                              delete_on_exit=True)

        while True:
            try:
                logger.info("Connecting to workers...")
                self.__control.connect([
                    self.__control.name(name, i) for name, cfgs, _ in workers_configs
                    for i in range(len(cfgs))
                ],
                                       progress=True,
                                       timeout=CONNECTION_RETRY_AFTER_SECONDS,
                                       raises_timeout_error=True)
                break

            except TimeoutError:
                logger.info("Connecting to workers timeout. Retrying...")
            except KeyboardInterrupt as e:
                logger.info("Interrupted by user. Stopping all and exiting...")
                raise e

        base.name_resolve.delete(
            names.worker_status(experiment_name=self.experiment_name,
                                trial_name=self.trial_name,
                                worker_name="ctl"))

        # Configure workers.
        try:
            for name, cfgs, _ in workers_configs:
                logger.info(f"Configuring Workers: {name}...")
                self.__control.group_request(
                    "configure",
                    worker_names=[self.__control.name(name, i) for i in range(len(cfgs))],
                    worker_kwargs=[dict(config=cfg) for cfg in cfgs],
                    progress=True)
        except Exception as e:
            logger.error("Configuring Failed. Exiting Workers.")
            self.interrupt(wait_timeout=120)
            raise e

        # # Configure monitoring.
        logger.info("Configuring monitoring")
        mon_addresses = []
        mon_repo = base.monitoring.TargetRepository()
        workers = None
        for _ in range(10):
            rs = self.__control.group_request("start_monitoring", worker_names=workers, timeout=3)
            workers = []
            for r in rs:
                if r.timed_out:
                    workers.append(r.worker_name)
                else:
                    mon_addresses.append(f"{r.result.host}:{r.result.prometheus_port}")
            if len(workers) == 0:
                break
            logger.warning("Failed start monitoring for %d workers, reconnecting and trying again",
                           len(workers))
            self.__control.connect(workers, reconnect=True)
        else:
            raise RuntimeError("Failed to start monitoring.")

        with mon_repo.add_target_group(f"{self.experiment_name}.{self.trial_name}",
                                       mon_addresses,
                                       delete_on_exit=True):
            logger.info("Start workers...")
            self.__control.group_request("start")
            logger.info("Started.")
            try:
                self.wait(timeout=None, check_status=check_worker_status, remove_status=remove_worker_status)
            except system.worker_base.WorkerException as e:
                logger.error(e)
                self.interrupt(wait_timeout=30)
            except KeyboardInterrupt:
                logger.info("Interrupted.")
                self.interrupt(wait_timeout=30)

    def wait(self, timeout: Optional[int], check_status: Tuple[Wss, ...], remove_status: Tuple[Wss, ...]):
        deadline = None if timeout is None else time.time() + timeout
        left = set(self.__control.worker_names)
        num_jobs_left = len(left)
        logger.info(f"Waiting for {num_jobs_left} jobs.")
        current_status = {name: Wss.UNKNOWN for name in self.__control.worker_names}
        while len(left) > 0:
            logger.debug(
                f"JOBS LEFT: {[str(len([l for l in left if job_type in l])) + ' ' + job_type for job_type in set([job_id.split('/')[0] for job_id in left])]}"
            )
            if len(left) < num_jobs_left:
                num_jobs_left = len(left)
                logger.info(f"Waiting for {num_jobs_left} jobs.")
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(
                    f"Timeout waiting for {self.experiment_name, self.trial_name}: {', '.join(sorted(left))}")
            for worker_name, worker_status in self.__control.pulse().items():
                if worker_status in check_status:
                    raise system.worker_base.WorkerException(worker_name, worker_status,
                                                             "experiment is running.")
                if worker_status in remove_status:
                    if worker_name in current_status:
                        logger.debug(f"Worker {worker_name} is {worker_status}. Removed from waiting list.")
                        current_status.pop(worker_name)
                    else:
                        pass
                else:
                    if current_status.get(worker_name, None) != worker_status:
                        current_status.update({worker_name: worker_status})
                        logger.debug(f"Update worker status: {worker_name} -> {worker_status}")

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
        """Interrupt the experiment.
        """
        logger.info("Interrupting experiment")
        self.__control.group_request("interrupt", wait_response=False)
        try:
            self.wait(timeout=wait_timeout,
                      check_status=(),
                      remove_status=(Wss.ERROR, Wss.LOST, Wss.COMPLETED, Wss.INTERRUPTED))
        except TimeoutError:
            raise RuntimeError(f"Fail to interrupt workers, timeout={wait_timeout}.")


class RayController(Controller):

    def __init__(self, experiment_name, trial_name):
        super().__init__(experiment_name, trial_name)

        self.__control = system.worker_control.make_control(
            type_='ray',
            experiment_name=self.experiment_name,
            trial_name=self.trial_name,
        )

        self.__workers_reply_comm = None
        self.__workers_request_comm = None
        self.__workers_run_refs = None
        self.__workers_ref = None

    def start(self, *args, **kwargs):
        # FIXME: for debug only
        time.sleep(10000000)
        try:
            self._start(*args, **kwargs)
        except Exception as e:
            self.shutdown()

    def _start(self, experiment: api.config.Experiment, ignore_worker_error=False):
        # TODO: wait for ray cluster worker nodes ready
        if ignore_worker_error:
            check_worker_status = ()
            remove_worker_status = (Wss.COMPLETED, Wss.ERROR, Wss.LOST, Wss.UNKNOWN)
        else:
            check_worker_status = (Wss.ERROR, Wss.LOST, Wss.UNKNOWN)
            remove_worker_status = (Wss.COMPLETED,)

        ray.init('auto')

        scheduling: api.config.ExperimentScheduling = experiment.scheduling_setup()
        setup = experiment.initial_setup()
        setup.set_worker_information(experiment_name=self.experiment_name, trial_name=self.trial_name)

        # Scheduling and connecting to workers.
        workers_configs = []
        workers_configs: List[str, Any, api.config.TasksGroup] = [(k, getattr(setup,
                                                                              k), getattr(scheduling, k))
                                                                  for k in WORKER_TYPES]

        for name, config, schedule in workers_configs:
            count = sum([s.count for s in schedule]) if isinstance(schedule, list) else schedule.count
            if len(config) != count:
                logger.error("Scheduling and config mismatch, exit.")
                raise IndexError(f"Configuration has {len(config)} {name}, {count} scheduled.")
            logger.info(f"Configuration has {len(config)} {name}.")

        # State clean-up.
        logger.info("Cleaning up previous states")
        base.name_resolve.clear_subtree(names.trial_root(self.experiment_name, self.trial_name))
        base.name_resolve.add(names.trial_registry(self.experiment_name, self.trial_name),
                              value=datetime.now().strftime("%Y%m%d"),
                              delete_on_exit=False,
                              replace=True)

        # Launch remote workers.
        logger.info("Launching remote workers using Ray...")
        self.__workers_ref: Dict[str, ray.ObjectRef] = {}
        self.__workers_request_comm: Dict[str, rq.Queue] = dict()
        self.__workers_reply_comm: Dict[str, rq.Queue] = dict()
        self.__workers_run_refs: Dict[str, ray.ObjectRef] = {}
        for worker_type, config, schedule in workers_configs:
            count = len(config)
            all_schedules: List[api.config.TasksGroup] = []
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
            comms = [(rq.Queue(maxsize=8), rq.Queue(maxsize=8)) for _ in all_schedules]
            worker_servers = [RayServer(comm) for comm in comms]
            worker_cls = load_worker(worker_type)
            refs = [
                ray.remote(
                    num_cpus=sch.scheduling.cpu,
                    num_gpus=sch.scheduling.gpu,
                    memory=sch.scheduling.mem,
                    name=f"{worker_type}/{idx}",
                )(worker_cls).remote(server=server)
                for idx, (server, sch) in enumerate(zip(worker_servers, all_schedules))
            ]
            _run_refs = [ref.run.remote() for ref in refs]

            for idx, (ref, c, run_ref) in enumerate(zip(refs, comms, _run_refs)):
                name = f"{worker_type}/{idx}"
                self.__workers_ref[name] = ref
                self.__workers_request_comm[name] = c[0]
                self.__workers_reply_comm[name] = c[1]
                self.__workers_run_refs[name] = run_ref
            logger.info(f"Launched {count} {worker_type}.")

        # Setup control panel.
        self.__control.connect(list(self.__workers_ref.keys()))

        # Configure workers.
        try:
            for name, cfgs, _ in workers_configs:
                logger.info(f"Configuring Workers: {name}...")
                self.__control.group_request(
                    "configure",
                    worker_names=[self.__control.name(name, i) for i in range(len(cfgs))],
                    worker_kwargs=[dict(config=cfg) for cfg in cfgs],
                    progress=True)
        except Exception as e:
            logger.error("Configuring Failed. Exiting Workers.")
            self.interrupt(wait_timeout=120)
            raise e

        # Configure monitoring.
        logger.info("Configuring monitoring")
        mon_addresses = []
        mon_repo = base.monitoring.TargetRepository()
        workers = None
        for _ in range(10):
            rs = self.__control.group_request("start_monitoring", worker_names=workers, timeout=3)
            workers = []
            for r in rs:
                if r.timed_out:
                    workers.append(r.worker_name)
                else:
                    mon_addresses.append(f"{r.result.host}:{r.result.prometheus_port}")
            if len(workers) == 0:
                break
            logger.warning("Failed start monitoring for %d workers, trying again", len(workers))
        else:
            raise RuntimeError("Failed to start monitoring.")

        with mon_repo.add_target_group(f"{self.experiment_name}.{self.trial_name}",
                                       mon_addresses,
                                       delete_on_exit=True):
            logger.info("Start workers...")
            self.__control.group_request("start")
            logger.info("Started.")
            try:
                self.wait(timeout=None, check_status=check_worker_status, remove_status=remove_worker_status)
            except system.worker_base.WorkerException as e:
                logger.error(e)
                self.interrupt(wait_timeout=30)
            except KeyboardInterrupt:
                logger.info("Interrupted.")
                self.interrupt(wait_timeout=30)

    def wait(self, timeout: Optional[int], check_status: Tuple[Wss, ...], remove_status: Tuple[Wss, ...]):
        deadline = None if timeout is None else time.time() + timeout
        left = set(self.__control.worker_names)
        num_jobs_left = len(left)
        logger.info(f"Waiting for {num_jobs_left} jobs.")
        current_status = {name: Wss.UNKNOWN for name in self.__control.worker_names}
        while len(left) > 0:
            logger.debug(
                f"JOBS LEFT: {[str(len([l for l in left if job_type in l])) + ' ' + job_type for job_type in set([job_id.split('/')[0] for job_id in left])]}"
            )
            if len(left) < num_jobs_left:
                num_jobs_left = len(left)
                logger.info(f"Waiting for {num_jobs_left} jobs.")
            if deadline is not None and time.time() > deadline:
                raise TimeoutError(
                    f"Timeout waiting for {self.experiment_name, self.trial_name}: {', '.join(sorted(left))}")
            for worker_name, worker_status in self.__control.pulse().items():
                if worker_status in check_status:
                    raise system.worker_base.WorkerException(worker_name, worker_status,
                                                             "experiment is running.")
                if worker_status in remove_status:
                    if worker_name in current_status:
                        logger.debug(f"Worker {worker_name} is {worker_status}. Removed from waiting list.")
                        current_status.pop(worker_name)
                    else:
                        pass
                else:
                    if current_status.get(worker_name, None) != worker_status:
                        current_status.update({worker_name: worker_status})
                        logger.debug(f"Update worker status: {worker_name} -> {worker_status}")

            left = set(current_status.keys())
            time.sleep(10)

    def interrupt(self, wait_timeout=120):
        """Interrupt the experiment.
        """
        logger.info("Interrupting experiment")
        self.__control.group_request("interrupt", wait_response=False)
        try:
            self.wait(timeout=wait_timeout,
                      check_status=(),
                      remove_status=(Wss.ERROR, Wss.LOST, Wss.COMPLETED, Wss.INTERRUPTED))
        except TimeoutError:
            raise RuntimeError(f"Fail to interrupt workers, timeout={wait_timeout}.")

    def shutdown(self):
        ray_exiting_name = names.ray_cluster(self.experiment_name, self.trial_name, "exiting")
        base.name_resolve.add(ray_exiting_name, value="1", delete_on_exit=True)
        del self.__workers_reply_comm
        del self.__workers_request_comm
        del self.__workers_run_refs
        del self.__workers_ref
        ray.shutdown()
