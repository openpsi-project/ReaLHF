from hashlib import new
from typing import Any, Dict, List, Optional
import dataclasses
import enum
import logging
import queue
import re
import socket
import threading
import time

import prometheus_client
import wandb

from api import config as config_pkg
from base.gpu_utils import set_cuda_device
import base.cluster
import base.monitoring
import base.name_resolve
import base.names
import base.network

logger = logging.getLogger("worker")

_WANDB_LOG_FREQUENCY_SECONDS = 10
_MAX_SOCKET_CONCURRENCY = 1000


class WorkerException(Exception):

    def __init__(self, worker_name, worker_status, scenario):
        super(WorkerException, self).__init__(f"Worker {worker_name} is {worker_status} while {scenario}")
        self.worker_name = worker_name
        self.worker_status = worker_status
        self.scenario = scenario


class WorkerServerStatus(str, enum.Enum):
    """List of all possible Server status. This is typically set by workers hosting the server, and
    read by the controller.
    """
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"

    UNKNOWN = "UNKNOWN"  # CANNOT be set.
    INTERRUPTED = "INTERRUPTED"
    ERROR = "ERROR"
    LOST = "LOST"  # CANNOT be set


class WorkerServer:
    """The abstract class that defines how a worker exposes RPC stubs to the controller.
    """

    def __init__(self, worker_name):
        """Specifies the name of the worker that WorkerControlPanel can used to find and manage.
        Args:
            worker_name: Typically "<worker_type>/<worker_index>".
        """
        self.worker_name = worker_name

    def register_handler(self, command, fn):
        """Registers an RPC command. The handler `fn` shall be called when `self.handle_requests()` sees an
        incoming command of the registered type.
        """
        raise NotImplementedError()

    def handle_requests(self, max_count=None):
        raise NotImplementedError()

    def set_status(self, status: WorkerServerStatus):
        raise NotImplementedError()


class WorkerControlPanel:
    """The abstract class that defines the management utilities to all the workers of an experiment trial.
    """

    class Future:

        def result(self, timeout=None):
            raise NotImplementedError()

    @dataclasses.dataclass
    class Response:
        worker_name: str
        result: Any
        timed_out: bool = False

    def __init__(self):
        self.__closed = False

    def __del__(self):
        if not self.__closed:
            self.close()
            self.__closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.__closed:
            self.close()
            self.__closed = True

    def close(self):
        raise NotImplementedError()

    @staticmethod
    def name(worker_type, worker_index):
        return f"{worker_type}/{worker_index}"

    @staticmethod
    def parse_name(worker_name):
        type_, index = worker_name.split("/")
        return type_, index

    @property
    def worker_names(self) -> List[str]:
        """Returns current connected workers. A WorkerControlPanel initializes with no connected workers.
        Workers are connected via either self.connect(names), or self.auto_connect().
        """
        raise NotImplementedError()

    def connect(self,
                worker_names: List[str],
                timeout=None,
                raises_timeout_error=False,
                reconnect=False,
                progress=False) -> List[str]:
        """Waits until all the workers specified by the given names are ready for receiving commands.

        Args:
            worker_names: A list of workers to connect.
            timeout: The maximum waiting time in seconds, or None if infinite.
            raises_timeout_error: If True, any connection failure will result in the raise of an TimeoutError.
                If False, such exception can be detected via the returned succeeded list.
            reconnect: If True, this will reconnect to the workers that has already connected. If False, any
                worker in `worker_names` that has already connected will be ignored.
            progress: Whether to show a progress bar.

        Returns:
            A list of successfully connected or reconnected workers. If a specified worker is missing, it can
            either be that it is already connected and `reconnect` is False, or that the connection timed out.
        """
        raise NotImplementedError()

    def auto_connect(self) -> List[str]:
        """Auto-detects available workers belonging to the experiment trial, and connects to them.

        Returns:
            Names of successfully connected workers.
        """
        raise NotImplementedError()

    def request(self, worker_name: str, command, **kwargs) -> Any:
        """Sends an request to the specified worker.
        """
        return self.async_request(worker_name, command, **kwargs).result()

    def async_request(self, worker_name: str, command, **kwargs) -> Future:
        """Posts an request to the specified worker.
        """
        raise NotImplementedError()

    def group_request(self,
                      command,
                      worker_names: Optional[List[str]] = None,
                      worker_regex: Optional[str] = None,
                      timeout=None,
                      progress=False,
                      worker_kwargs: Optional[List[Dict[str, Any]]] = None,
                      wait_response=True,
                      **kwargs) -> List[Response]:
        """Requests selected workers, or all connected workers if not specified.

        Args:
            command: RPC command.
            worker_names: Optional selection of workers.
            worker_regex: Optional regex selector of workers.
            timeout: Optional timeout.
            progress: Whether to show a progress bar.
            worker_kwargs: RPC arguments, but one for each worker instead of `kwargs` where every worker share
                the arguments. If this is specified, worker_names must be specified, and worker_regex, kwargs
                must be None.
            wait_response: Whether to wait for server response.
            kwargs: RPC arguments.
        """
        selected = self.worker_names
        if worker_names is not None:
            assert len(set(worker_names).difference(selected)) == 0
            selected = worker_names
        if worker_regex is not None:
            selected = [x for x in selected if re.fullmatch(worker_regex, x)]
        if worker_kwargs is not None:
            assert worker_names is not None
            assert worker_regex is None
            assert len(kwargs) == 0
            assert len(worker_names) == len(worker_kwargs), f"{len(worker_names)} != {len(worker_kwargs)}"
        else:
            worker_kwargs = [kwargs for _ in selected]

        # connect _MAX_SOCKET_CONCURRENCY sockets at most
        rs = []
        deadline = time.monotonic() + (timeout or 0)
        for j in range(0, len(selected), _MAX_SOCKET_CONCURRENCY):
            sub_rs = []
            sub_selected = selected[j:j + _MAX_SOCKET_CONCURRENCY]
            sub_worker_kwargs = worker_kwargs[j:j + _MAX_SOCKET_CONCURRENCY]
            for name, kwargs in zip(sub_selected, sub_worker_kwargs):
                sub_rs.append(
                    self.Response(worker_name=name,
                                  result=self.async_request(name, command, wait_response, **kwargs)))

            if not wait_response:
                continue

            bar = range(len(sub_rs))
            if progress:
                try:
                    import tqdm
                    bar = tqdm.tqdm(bar, leave=False)
                except ModuleNotFoundError:
                    pass
            for r, _ in zip(sub_rs, bar):
                if timeout is not None:
                    timeout = max(0, deadline - time.monotonic())
                try:
                    r.result = r.result.result(timeout=timeout)
                except TimeoutError:
                    r.timed_out = True
            rs.extend(sub_rs)
        return rs

    def get_worker_status(self, worker_name) -> WorkerServerStatus:
        """Get status of a connected worker.
        Raises:
            ValueError if worker is not connected.
        """
        raise NotImplementedError()

    def pulse(self):
        return {name: self.get_worker_status(name) for name in self.worker_names}


@dataclasses.dataclass
class MonitoringInformation:
    host: str
    prometheus_port: int


@dataclasses.dataclass
class PollResult:
    # Number of total samples and batches processed by the worker. Specifically:
    # - For an actor worker, sample_count = batch_count = number of env.step()-s being executed.
    # - For a policy worker, number of inference requests being handled, versus how many batches were made.
    # - For a trainer worker, number of samples & batches fed into the trainer (typically GPU).
    sample_count: int
    batch_count: int


class Worker:
    """The worker base class that provides general methods and entry point.

    For simplicity, we use a single-threaded pattern in implementing the worker RPC server. Logic
    of every worker are executed via periodical calls to the poll() method, instead of inside
    another thread or process (e.g. the gRPC implementation). A subclass only needs to implement
    poll() without duplicating the main loop.

    The typical code on the worker side is:
        worker = make_worker()  # Returns instance of Worker.
        worker.run()
    and the later is standardized here as:
        while exit command is not received:
            if worker is started:
                worker.poll()
    """

    def __init__(self, server: Optional[WorkerServer] = None):
        """Initializes a worker server.

        Args:
            server: The RPC server API for the worker to register handlers and poll requests.
        """
        self.__running = False
        self.__exiting = False
        self.config = None
        self.__is_configured = False
        self._monitoring_info = None

        self._server = server
        if server is not None:
            server.register_handler('configure', self.configure)
            server.register_handler('reconfigure', self.reconfigure)
            server.register_handler('start_monitoring', self.start_monitoring)
            server.register_handler('start', self.start)
            server.register_handler('pause', self.pause)
            server.register_handler('exit', self.exit)
            server.register_handler('interrupt', self.interrupt)
            server.register_handler('ping', lambda: "pong")

        self.logger = logging.getLogger("worker")
        self.__worker_type = None
        self.__worker_index = None
        self.__last_successful_poll_time = None
        self.__worker_info = None

        # Monitoring related.
        self._start_time_ns = None
        self.__wandb_run = None

        self.__set_status(WorkerServerStatus.READY)

    def __set_status(self, status: WorkerServerStatus):
        if self._server is not None:
            self.logger.debug(f"Setting worker server status to {status}")
            self._server.set_status(status)

    def __del__(self):
        if self.__wandb_run is not None:
            self.__wandb_run.finish()

    @property
    def is_configured(self):
        return self.__is_configured

    def _reconfigure(self, **kwargs) -> config_pkg.WorkerInformation:
        """Implemented by sub-classes."""
        raise NotImplementedError()

    def _configure(self, config) -> config_pkg.WorkerInformation:
        """Implemented by sub-classes."""
        raise NotImplementedError()

    def _poll(self) -> PollResult:
        """Implemented by sub-classes."""
        raise NotImplementedError()

    def _stats(self) -> Dict[str, Any]:
        """Implemented by sub-classes. For wandb logging only"""
        return {}

    def configure(self, config):
        assert not self.__running
        self.logger.info("Configuring with: %s", config)

        # For passing tests
        self.monitor = base.monitoring.DummyMonitor(None)
        self.monitor_thread = base.monitoring.DummyMonitorThread()

        r = self._configure(config)
        self.__worker_info = r
        self.__worker_type = r.worker_type
        self.__worker_index = r.worker_index
        self.logger = logging.getLogger(r.worker_type + "-worker")
        if r.host_key is not None:
            self.__host_key(
                base.names.worker_key(experiment_name=r.experiment_name,
                                      trial_name=r.trial_name,
                                      key=r.host_key))
        if r.watch_keys is not None:
            keys = [r.watch_keys] if isinstance(r.watch_keys, str) else r.watch_keys
            self.__watch_keys([
                base.names.worker_key(experiment_name=r.experiment_name, trial_name=r.trial_name, key=k)
                for k in keys
            ])

        self.__is_configured = True
        self.logger.info("Configured successfully")

    def reconfigure(self, **kwargs):
        assert not self.__running
        self.__is_configured = False
        self.logger.info(f"Reconfiguring with: {kwargs}")
        self._reconfigure(**kwargs)
        self.__is_configured = True
        self.logger.info("Reconfigured successfully")

    def print_monitor_info(self):
        # for debugging in workers
        self.monitor_thread.print()

    def start_monitoring(self):
        """ Start monitoring and define monitoring metrics for prometheus.
        Subclasses should add metrics in this method if different monitoring metrics is needed.
        """
        prometheus_port = base.monitoring.start_prometheus_server()
        r = self._monitoring_info = MonitoringInformation(host=socket.gethostname(),
                                                          prometheus_port=prometheus_port)
        self.logger.info("Started prometheus server on %s:%s", r.host, r.prometheus_port)

        if not self.__worker_info:
            raise Exception("Initializing monitoring before configuration.")

        model_name = "" if not self.__worker_info.model_name else self.__worker_info.model_name
        prometheus_labels = dict(host=base.network.gethostname(),
                                 experiment=self.__worker_info.experiment_name,
                                 trial=self.__worker_info.trial_name,
                                 worker=self.__worker_info.worker_type,
                                 worker_id=self.__worker_info.worker_index,
                                 policy=model_name)
        wandb_args = dict(
            entity=self.__worker_info.wandb_entity,
            project=self.__worker_info.wandb_project or f"{self.__worker_info.experiment_name}",
            group=self.__worker_info.wandb_group or self.__worker_info.trial_name,
            job_type=self.__worker_info.wandb_job_type or f"{self.__worker_info.worker_type}",
            name=self.__worker_info.wandb_name
            or f"{self.__worker_info.model_name or self.__worker_info.worker_index}",
            id=
            f"{self.__worker_info.experiment_name}_{self.__worker_info.trial_name}_{self.__worker_info.model_name or 'unnamed'}"
            f"_{self.__worker_info.worker_type}_{self.__worker_info.worker_index}",
            settings=wandb.Settings(start_method="fork"),
        )

        if_log_wandb = (self.__worker_index == 0 and self.__worker_type == 'trainer') \
                       if self.__worker_info.log_wandb is None else self.__worker_info.log_wandb

        prometheus_metrics = dict(marl_worker_sample_count="Counter",
                                  marl_worker_batch_count="Counter",
                                  marl_worker_wait_seconds="Histogram",
                                  marl_worker_cpu_percent="Summary",
                                  marl_worker_memory_rss_mb="Summary",
                                  marl_worker_memory_vms_mb="Summary",
                                  marl_worker_memory_shared_mb="Summary",
                                  marl_worker_gpu_percent="Summary",
                                  marl_worker_gpu_mem_util_percent="Summary",
                                  marl_worker_gpu_memory_mb="Summary")
        monitor_info = base.monitoring.MonitorInfo(
            prometheus_labels=prometheus_labels,
            prometheus_metrics=prometheus_metrics,
            if_log_wandb=if_log_wandb,
            wandb_args=wandb_args,
        )
        self.monitor = base.monitoring.Monitor(monitor_info)
        self.monitor_thread = base.monitoring.MonitorThread(self.monitor)
        self.monitor_thread.start()
        return r

    def start(self):
        self.logger.info("Starting worker")
        self.__running = True
        self.__set_status(WorkerServerStatus.RUNNING)

    def pause(self):
        self.logger.info("Pausing worker")
        self.__running = False
        self.__set_status(WorkerServerStatus.PAUSED)

    def exit(self):
        self.logger.info("Exiting worker")
        self.__set_status(WorkerServerStatus.COMPLETED)
        self.__exiting = True

    def interrupt(self):
        self.logger.info("Worker interrupted by remote control.")
        self.__set_status(WorkerServerStatus.INTERRUPTED)
        raise WorkerException(worker_name="worker",
                              worker_status=WorkerServerStatus.INTERRUPTED,
                              scenario="running")

    def run(self):
        self._start_time_ns = time.monotonic_ns()
        self.__last_update_ns = None
        self.logger.info("Running worker now")
        try:
            while not self.__exiting:
                if self.__running:
                    if not self.__is_configured:
                        raise RuntimeError("Worker is not configured")
                    start_time = time.monotonic_ns()
                    r = self._poll()
                    poll_time = (time.monotonic_ns() - start_time) / 1e9
                    wait_seconds = 0.0
                    if self.__last_successful_poll_time is not None:
                        # Account the waiting time since the last successful step.
                        wait_seconds = (start_time - self.__last_successful_poll_time) / 1e9
                    self.__last_successful_poll_time = time.monotonic_ns()

                    if r.sample_count == r.batch_count == 0:
                        if self.__worker_type != "actor":
                            time.sleep(0.002)
                    else:
                        self.monitor.metric("marl_worker_sample_count").inc(r.sample_count)
                        self.monitor.metric("marl_worker_batch_count").inc(r.batch_count)
                        self.monitor.metric("marl_worker_wait_seconds").observe(wait_seconds)

                        now = time.monotonic_ns()
                        if self.__last_update_ns is not None:  # Update new stats with 10 seconds frequency.
                            if (now - self.__last_update_ns) / 1e9 >= 10:
                                duration = (time.monotonic_ns() - self._start_time_ns) / 1e9
                                new_stats = dict(
                                    samples=self.monitor.metric("marl_worker_sample_count")._value.get() /
                                    duration,
                                    batches=self.monitor.metric("marl_worker_batch_count")._value.get() /
                                    duration,
                                    idleTime=self.monitor.metric("marl_worker_wait_seconds")._sum.get() /
                                    duration,
                                    **self._stats())
                                self.monitor_thread.update_stats(new_stats)
                                t1, t2, perc = self.monitor_thread.thread_profiles()
                                self.logger.debug(
                                    f"Monitoring thread time: {t1}, total CPU time: {t2}, monitor thread time percentage: {perc}"
                                )
                                self.__last_update_ns = now
                        else:
                            self.__last_update_ns = now
                else:
                    time.sleep(0.05)
                self._server.handle_requests()
        except KeyboardInterrupt:
            self.exit()
        except Exception as e:
            if isinstance(e, WorkerException):
                raise e
            self.__set_status(WorkerServerStatus.ERROR)
            raise e

    def __host_key(self, key: str):
        self.logger.info(f"Hosting key: {key}")
        base.name_resolve.add(key, "up", keepalive_ttl=15, replace=True, delete_on_exit=True)

    def __watch_keys(self, keys: List[str]):
        self.logger.info(f"Watching keys: {keys}")
        base.name_resolve.watch_names(keys, call_back=self.exit)


class MappingThread:
    """Wrapped of a mapping thread.
    A mapping thread gets from up_stream_queue, process data, and puts to down_stream_queue.
    """

    def __init__(self,
                 map_fn,
                 interrupt_flag,
                 upstream_queue,
                 downstream_queue: queue.Queue = None,
                 cuda_device=None):
        """Init method of MappingThread for Policy Workers.

        Args:
            map_fn: mapping function.
            interrupt_flag: main thread sets this value to True to interrupt the thread.
            upstream_queue: the queue to get data from.
            downstream_queue: the queue to put data after processing. If None, data will be discarded after processing.
        """
        self.__map_fn = map_fn
        self.__interrupt = interrupt_flag
        self.__upstream_queue = upstream_queue
        self.__downstream_queue = downstream_queue
        self.__thread = threading.Thread(target=self._run, daemon=True)
        self.__cuda_device = cuda_device

    def is_alive(self) -> bool:
        """Check whether the thread is alive.

        Returns:
            alive: True if the wrapped thread is alive, False otherwise.
        """
        return self.__interrupt or self.__thread.is_alive()

    def start(self):
        """Start the wrapped thread.
        """
        self.__thread.start()

    def join(self):
        """Join the wrapped thread.
        """
        self.__thread.join()

    def _run(self):
        if self.__cuda_device is not None:
            set_cuda_device(self.__cuda_device)
        while not self.__interrupt:
            self._run_step()

    def _run_step(self):
        try:
            data = self.__upstream_queue.get(timeout=1)
            data = self.__map_fn(data)
            if self.__downstream_queue is not None:
                self.__downstream_queue.put(data)
        except queue.Empty:
            pass

    def stop(self):
        """Stop the wrapped thread.
        """
        self.__interrupt = True
        if self.__thread.is_alive():
            self.__thread.join()
