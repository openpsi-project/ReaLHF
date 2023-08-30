from typing import Tuple, List, Dict, Any, Optional, Union
import re
import logging
import pickle
import socket
import time

import zmq
import ray.util.queue as rq

import base.name_resolve
import base.names as names
import system.worker_base as worker_base
from system.worker_base import WorkerServerStatus

logger = logging.getLogger("worker-control")
WORKER_WAIT_FOR_CONTROLLER_SECONDS = 3600
WORKER_JOB_STATUS_LINGER_SECONDS = 60


class ZmqServer(worker_base.WorkerServer):
    """A light-weight implementation of an RPC server.

    Note that this server only allows a single client connection for now, as that is sufficient for
    workers which respond to the controller only.

    Example:
        # Server side.
        server = RpcServer(port)
        server.register_handler('foo', foo)
        while True:
            server.handle_requests()

        # Client side.
        client = RpcClient(host, port)
        client.request('foo', x=42, y='str') # foo(x=42, y='str') will be called on the server side.
    """

    def set_status(self, status: worker_base.WorkerServerStatus):
        """On graceful exit, worker status is cleared.
        """
        base.name_resolve.add(
            names.worker_status(experiment_name=self.__experiment_name,
                                trial_name=self.__trial_name,
                                worker_name=self.worker_name),
            value=status.value,
            keepalive_ttl=WORKER_JOB_STATUS_LINGER_SECONDS,  # Job Status lives one minutes after worker exit.
            replace=True,
            delete_on_exit=False)

    def __init__(self, port=0, experiment_name=None, trial_name=None, worker_name=None):
        super().__init__(worker_name)
        self.__experiment_name = experiment_name
        self.__trial_name = trial_name
        self.__handlers = {}
        self.__context = zmq.Context()
        self.__socket = self.__context.socket(zmq.REP)
        host_ip = socket.gethostbyname(socket.gethostname())
        if port == 0:
            self.__port = self.__socket.bind_to_random_port(f"tcp://{host_ip}")
        else:
            self.__socket.bind(f"tcp://{host_ip}:{port}")
            self.__port = port

        try:
            controller_status = base.name_resolve.wait(names.worker_status(experiment_name, trial_name,
                                                                           "ctl"),
                                                       timeout=WORKER_WAIT_FOR_CONTROLLER_SECONDS)
        except TimeoutError:
            raise TimeoutError(
                f"Worker ({experiment_name, trial_name, worker_name}) connect to controller timeout from host {socket.gethostname()}."
            )

        if controller_status != "READY":
            raise RuntimeError(f"Abnormal controller state on experiment launch {controller_status}.")

        if experiment_name is not None and trial_name is not None:
            key = names.worker(experiment_name, trial_name, worker_name)
            address = f"{host_ip}:{self.__port}"
            base.name_resolve.add(key, address, keepalive_ttl=10, delete_on_exit=True)
            logger.info("Added name_resolve entry %s for worker server at %s", key, address)

    def __del__(self):
        self.__socket.close()

    @property
    def port(self):
        return self.__port

    def register_handler(self, command, fn):
        if command in self.__handlers:
            raise KeyError(f"Command '{command}' exists")
        self.__handlers[command] = fn

    def handle_requests(self, max_count=None):
        """Handles queued requests in order, optionally limited by `max_count`.

        Returns:
            The count of requests handled.
        """
        count = 0
        while max_count is None or count < max_count:
            try:
                data = self.__socket.recv(zmq.NOBLOCK)
            except zmq.ZMQError:
                # Currently no request in the queue.
                break
            command, kwargs = pickle.loads(data)
            logger.debug("Handle request: %s, len(data)=%d", command, len(data))
            if command in self.__handlers:
                try:
                    response = self.__handlers[command](**kwargs)
                    logger.debug("Handle request: %s, ok", command)
                except worker_base.WorkerException:
                    raise
                except Exception as e:
                    logger.error("Handle request: %s, error", command)
                    logger.error(e, exc_info=True)
                    response = e
            else:
                logger.error("Handle request: %s, no such command", command)
                response = KeyError(f'No such command: {command}')
            self.__socket.send(pickle.dumps(response))
            logger.debug("Handle request: %s, sent reply", command)
            count += 1
        return count


class RayServer(worker_base.WorkerServer):

    def set_status(self, status: worker_base.WorkerServerStatus):
        """On graceful exit, worker status is cleared.
        """
        base.name_resolve.add(
            names.worker_status(experiment_name=self.__experiment_name,
                                trial_name=self.__trial_name,
                                worker_name=self.worker_name),
            value=status.value,
            keepalive_ttl=WORKER_JOB_STATUS_LINGER_SECONDS,  # Job Status lives one minutes after worker exit.
            replace=True,
            delete_on_exit=False)

    def __init__(self, comm: Tuple[rq.Queue, rq.Queue], worker_name: str, experiment_name: str,
                 trial_name: str):
        super().__init__(worker_name)
        self.__handlers = {}

        self.__experiment_name = experiment_name
        self.__trial_name = trial_name

        recv_queue, send_queue = comm
        self.__recv_queue = recv_queue
        self.__send_queue = send_queue

    def register_handler(self, command, fn):
        if command in self.__handlers:
            raise KeyError(f"Command '{command}' exists")
        self.__handlers[command] = fn

    def handle_requests(self, max_count=None):
        count = 0
        while max_count is None or count < max_count:
            try:
                command, kwargs = self.__recv_queue.get_nowait()
            except rq.Empty:
                break
            logger.debug("Handle request: %s with kwargs %s", command, kwargs)
            if command in self.__handlers:
                try:
                    response = self.__handlers[command](**kwargs)
                    logger.debug("Handle request: %s, ok", command)
                except worker_base.WorkerException:
                    raise
                except Exception as e:
                    logger.error("Handle request: %s, error", command)
                    logger.error(e, exc_info=True)
            else:
                logger.error("Handle request: %s, no such command", command)
                response = KeyError(f'No such command: {command}')
            self.__send_queue.put(response)
            logger.debug("Handle request: %s, sent reply", command)
            count += 1
        return count


class ZmqWorkerControl(worker_base.WorkerControlPanel):

    class ZmqFuture(worker_base.WorkerControlPanel.Future):
        # Every ZmqFuture connect one socket, close after returning results.
        def __init__(self, payload, context: zmq.Context, address, worker_name, wait_response=True):
            self.__worker_name = worker_name
            self.__socket = context.socket(zmq.REQ)
            self.__socket.setsockopt(zmq.LINGER, 0)
            self.__socket.connect(f"tcp://{address}")
            self.__socket.send(payload, flags=zmq.NOBLOCK)
            if not wait_response:
                self.__socket.close()

        def result(self, timeout=None):
            if timeout is not None:
                self.__socket.RCVTIMEO = int(timeout * 1000)
            else:
                self.__socket.RCVTIMEO = int(1e9)
            try:
                r = pickle.loads(self.__socket.recv())
            except zmq.error.Again as e:
                raise TimeoutError(f"Waiting for RPC server response timeout: {e}")
            if isinstance(r, Exception):
                logger.error(f"Error configuring worker {self.__worker_name}")
                raise r
            self.__socket.close()
            return r

    def __init__(self, experiment_name, trial_name):
        super().__init__()
        self.__experiment_name = experiment_name
        self.__trial_name = trial_name
        self.__context = zmq.Context()
        self.__context.set(zmq.MAX_SOCKETS, 20480)
        self.__worker_addresses = {}

    def close(self):
        logger.info("Closing ZMQ worker control panel.")

    @property
    def worker_names(self):
        return list(self.__worker_addresses.keys())

    def connect(self,
                worker_names,
                timeout=None,
                raises_timeout_error=False,
                reconnect=False,
                progress=False):
        rs = []
        deadline = time.monotonic() + (timeout or 0)
        if progress:
            try:
                import tqdm
                worker_names = tqdm.tqdm(worker_names, leave=False)
            except ModuleNotFoundError:
                pass
        for name in worker_names:
            if name in self.__worker_addresses:
                if reconnect:
                    del self.__worker_addresses[name]
                else:
                    continue
            try:
                if timeout is not None:
                    timeout = max(0, deadline - time.monotonic())
                server_address = base.name_resolve.wait(names.worker(self.__experiment_name,
                                                                     self.__trial_name, name),
                                                        timeout=timeout)
            except TimeoutError as e:
                if raises_timeout_error:
                    raise e
                continue
            # self.__worker_addresses[name] stores address
            self.__worker_addresses[name] = server_address
            rs.append(name)
        return rs

    def auto_connect(self):
        name_root = names.worker_root(self.__experiment_name, self.__trial_name)
        worker_names = [r[len(name_root):] for r in base.name_resolve.find_subtree(name_root)]
        return self.connect(worker_names, timeout=0, raises_timeout_error=True)

    def async_request(self, worker_name, command, wait_response=True, **kwargs):
        if worker_name not in self.__worker_addresses:
            raise KeyError(f"No such connected worker: {worker_name}")
        return self.ZmqFuture(pickle.dumps((command, kwargs)),
                              self.__context,
                              self.__worker_addresses[worker_name],
                              worker_name,
                              wait_response=wait_response)

    def get_worker_status(self, worker_name) -> worker_base.WorkerServerStatus:
        try:
            status_str = base.name_resolve.wait(
                names.worker_status(experiment_name=self.__experiment_name,
                                    trial_name=self.__trial_name,
                                    worker_name=worker_name),
                timeout=1,
            )
            status = worker_base.WorkerServerStatus(status_str)
        except base.name_resolve.NameEntryNotFoundError:
            status = worker_base.WorkerServerStatus.LOST
        return status


class RayControlPanel(worker_base.WorkerControlPanel):

    class RayQueueFuture(worker_base.WorkerControlPanel.Future):

        def __init__(self, worker_name: str, queue: rq.Queue):
            self.__queue = queue
            self.__worker_name = worker_name

        def result(self, timeout=None):
            try:
                return self.__queue.get(timeout=timeout)
            except rq.Empty:
                raise TimeoutError(f"Waiting for Ray worker {self.__worker_name} response timeout.")
            except Exception as e:
                raise RuntimeError(f"Error waiting for Ray queue future {self.__worker_name}.") from e

    def __init__(self, experiment_name, trial_name):
        super().__init__()
        self.__experiment_name = experiment_name
        self.__trial_name = trial_name

        self.__worker_names = set()
        self.__request_comms: Dict[str, rq.Queue] = dict()
        self.__reply_comms: Dict[str, rq.Queue] = dict()

    def close(self):
        logger.info("Closing Ray worker control panel.")

    @property
    def worker_names(self):
        return self.__worker_names

    def connect(self, worker_names, *args, **kwargs):
        self.__worker_names = self.__worker_names.union(set(worker_names))

    def auto_connect(self) -> List[str]:
        raise NotImplementedError("auto_connect not supported for RayControlPanel")

    def async_request(self, worker_name, command, **kwargs):
        if worker_name not in self.__worker_names:
            raise KeyError(f"No such connected worker: {worker_name}")
        request_queue = self.__request_comms[worker_name]
        request_queue.put((command, kwargs))
        reply_queue = self.__reply_comms[worker_name]
        return self.RayQueueFuture(reply_queue)

    def get_worker_status(self, worker_name) -> WorkerServerStatus:
        try:
            status_str = base.name_resolve.wait(
                names.worker_status(experiment_name=self.__experiment_name,
                                    trial_name=self.__trial_name,
                                    worker_name=worker_name),
                timeout=1,
            )
            status = worker_base.WorkerServerStatus(status_str)
        except base.name_resolve.NameEntryNotFoundError:
            status = worker_base.WorkerServerStatus.LOST
        return status


def make_server(type_, **kwargs):
    if type_ == "zmq":
        return ZmqServer(**kwargs)
    if type_ == 'ray':
        return RayServer(**kwargs)
    raise NotImplementedError(type_)


def make_control(type_, **kwargs):
    if type_ == "zmq":
        return ZmqWorkerControl(**kwargs)
    if type_ == 'ray':
        return RayControlPanel(**kwargs)
    raise NotImplementedError(type_)
