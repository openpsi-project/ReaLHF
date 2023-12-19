# Request-reply stream between model workers and the master worker.
# The stream is composed of a pair of ZMQ sockets, one PUSH and one PULL, for asynchronous communication,
# i.e., the model worker can buffer requests from the master and execute them in any order under the hood.
from typing import Dict, List, Optional, Union
import asyncio
import dataclasses
import pickle
import socket
import time
import uuid

import torch
import zmq

import api.config
import api.dfg
import base.logging as logging
import base.name_resolve as name_resolve
import base.namedarray as namedarray
import base.names as names

logger = logging.getLogger("Request-Replay Stream")
ZMQ_IO_THREADS = 8

PUBSUB_BARRIER_NAME = "__{name}_pubsub_barrier__"


class NoMessage(Exception):
    pass


@dataclasses.dataclass
class Payload:
    request_id: Optional[str] = None
    handle_name: Optional[str] = None
    data: Optional[Union[Dict, namedarray.NamedArray]] = None

    def __post_init__(self):
        if self.request_id is None:
            self.request_id = str(uuid.uuid4())


def post(socket: zmq.Socket, payload: Payload, serialization_method: str):
    tik = time.monotonic()
    if isinstance(payload.data, namedarray.NamedArray):
        assert isinstance(payload.data, namedarray.NamedArray), type(payload.data)
        payload.data = namedarray.recursive_apply(payload.data, lambda x: x.cpu().numpy())
        encoded_data = namedarray.dumps(payload.data, method=serialization_method)
        encoding = b"01"
    else:
        encoded_data = [pickle.dumps(payload.data)]
        encoding = b"00"
    socket.send_multipart([
        pickle.dumps(tik),
        payload.handle_name.encode("ascii"),
        payload.request_id.encode("ascii"),
        encoding,
    ] + encoded_data)
    return payload.request_id


def poll(socket: zmq.Socket, block: bool = False) -> Payload:
    try:
        time_bytes, handle_name, request_id, encoding, *data = socket.recv_multipart(
            flags=0 if block else zmq.NOBLOCK)
    except zmq.ZMQError:
        raise NoMessage()

    send_time = pickle.loads(time_bytes)
    handle_name = handle_name.decode("ascii")
    request_id = request_id.decode("ascii")
    if encoding == b"01":
        data = namedarray.loads(data)
        data = namedarray.recursive_apply(data, lambda x: torch.from_numpy(x))
    elif encoding == b"00":
        assert len(data) == 1
        data = pickle.loads(data[0])
    else:
        raise NotImplementedError()
    logger.debug(f"Payload transfer time: {time.monotonic() - send_time:.4f}s")
    return Payload(handle_name=handle_name, request_id=request_id, data=data)


class RequestReplyStream:

    def post(self, payload: Payload):
        return post(self.send_socket, payload, self.serialization_method)

    def poll(self, socket_name: Optional[str] = None, block: bool = False) -> Payload:
        if socket_name is None:
            if len(self.recv_sockets) == 1:
                socket_name = next(iter(self.recv_sockets))
            else:
                raise RuntimeError(
                    f"Must specify socket name if there are multiple recv sockets. Recv socket names: {list(self.recv_sockets.keys())}"
                )
        return poll(self.recv_sockets[socket_name], block=block)

    def poll_all_blocked(self) -> Dict[str, Payload]:
        return {k: self.poll(k, block=True) for k in self.recv_sockets}

    async def async_poll_all(self) -> Dict[str, Payload]:
        all_res: Dict[str, Payload] = {}
        for recv_socket_name in self.recv_sockets:
            while True:
                try:
                    res = self.poll(recv_socket_name, block=False)
                    break
                except NoMessage:
                    await asyncio.sleep(0.01)
            all_res[recv_socket_name] = res
        return all_res

    @property
    def context(self) -> zmq.Context:
        raise NotImplementedError()

    @property
    def send_socket(self) -> zmq.Socket:
        raise NotImplementedError()

    @property
    def recv_sockets(self) -> Dict[str, zmq.Socket]:
        raise NotImplementedError()

    @property
    def serialization_method(self) -> str:
        raise NotImplementedError()

    def close(self):
        for socket in self.recv_sockets.values():
            socket.close()
        self.send_socket.close()
        self.context.destroy()

    def __del__(self):
        self.close()


class IpRequestReplyMasterStream(RequestReplyStream):

    def __init__(self, serialization_method: str):
        self._context = zmq.Context.instance(io_threads=ZMQ_IO_THREADS)

        self._send_socket = self._context.socket(zmq.PUB)
        host_ip = socket.gethostbyname(socket.gethostname())
        send_port = self._send_socket.bind_to_random_port(f"tcp://{host_ip}")
        # self._send_socket.setsockopt(zmq.LINGER, 0)
        self._send_address = f"{host_ip}:{send_port}"

        # NOTE: We cannot use a single end-point PULL socket for receiving
        # because we care about the order of data returned by different data parallel ranks.
        # Use multiple sockets will resolve this issue.
        self._recv_sockets = dict()

        self._serialization_method = serialization_method

    def accept(self, socket_name: str, server_addr: str):
        recv_socket = self.context.socket(zmq.PULL)
        recv_socket.connect(f"tcp://{server_addr}")
        # recv_socket.setsockopt(zmq.LINGER, 0)
        self.recv_sockets[socket_name] = recv_socket

    @property
    def context(self) -> zmq.Context:
        return self._context

    @property
    def address(self) -> str:
        return self._send_address

    @property
    def send_socket(self) -> zmq.Socket:
        return self._send_socket

    @property
    def recv_sockets(self) -> Dict[str, zmq.Socket]:
        return self._recv_sockets

    @property
    def serialization_method(self) -> str:
        return self._serialization_method


class IpRequestReplyWorkerStream(RequestReplyStream):

    def __init__(self, serialization_method: str):
        self._context = zmq.Context.instance(io_threads=ZMQ_IO_THREADS)

        self._recv_sockets = dict()

        self._send_socket = self._context.socket(zmq.PUSH)
        host_ip = socket.gethostbyname(socket.gethostname())
        send_port = self._send_socket.bind_to_random_port(f"tcp://{host_ip}")
        # self._send_socket.setsockopt(zmq.LINGER, 0)
        self._send_address = f"{host_ip}:{send_port}"

        self._serialization_method = serialization_method

    def accept(self, socket_name: str, server_addr: str):
        recv_socket: zmq.Socket = self.context.socket(zmq.SUB)
        recv_socket.connect(f"tcp://{server_addr}")
        recv_socket.setsockopt(zmq.SUBSCRIBE, b"")
        # recv_socket.setsockopt(zmq.LINGER, 0)
        self.recv_sockets[socket_name] = recv_socket

    @property
    def context(self) -> zmq.Context:
        return self._context

    @property
    def address(self) -> str:
        return self._send_address

    @property
    def send_socket(self) -> zmq.Socket:
        return self._send_socket

    @property
    def recv_sockets(self) -> Dict[str, zmq.Socket]:
        return self._recv_sockets

    @property
    def serialization_method(self) -> str:
        return self._serialization_method


class NameResolvingRequstReplyMasterStream(IpRequestReplyMasterStream):

    def __init__(
        self,
        experiment_name: str,
        trial_name: str,
        push_stream_name: str,
        pull_stream_names: List[str],
        serialization_method: str,
    ):
        super().__init__(serialization_method)
        assert isinstance(push_stream_name, str), push_stream_name
        assert isinstance(pull_stream_names, list) and isinstance(pull_stream_names[0],
                                                                  str), pull_stream_names

        master_send_name = names.request_reply_stream(experiment_name, trial_name, push_stream_name)
        name_resolve.add(name=master_send_name, value=self.address)

        logger.info(f"Add master send address {self.address} as {master_send_name}")

        for pull_stream_name in pull_stream_names:
            master_recv_name = names.request_reply_stream(experiment_name, trial_name, pull_stream_name)
            try:
                master_recv_address = name_resolve.wait(master_recv_name, timeout=60)
            except TimeoutError as e:
                logger.error(f"Master timeout waiting for worker send stream {pull_stream_name}")
                raise e
            self.accept(master_recv_name, master_recv_address)
            logger.info(f"Get master receive address: {master_recv_address} from {master_recv_name}")

        # master needs to wait all peers (subscribers) to connect
        while len(
                name_resolve.get_subtree(
                    names.request_reply_stream(
                        experiment_name, trial_name,
                        PUBSUB_BARRIER_NAME.format(name=push_stream_name)))) < len(pull_stream_names):
            time.sleep(0.1)
        logger.info(
            f"Master discovered all {len(pull_stream_names)} "
            f"subscribers: {name_resolve.get_subtree(names.request_reply_stream(experiment_name, trial_name, PUBSUB_BARRIER_NAME.format(name=push_stream_name)))}."
        )


class NameResolvingRequstReplyWorkerStream(IpRequestReplyWorkerStream):

    def __init__(
        self,
        experiment_name: str,
        trial_name: str,
        push_stream_name: str,
        pull_stream_names: List[str],
        serialization_method: str,
    ):
        super().__init__(serialization_method)

        send_name = names.request_reply_stream(experiment_name, trial_name, push_stream_name)
        name_resolve.add(name=send_name, value=self.address)

        logger.info(f"Add worker send address {self.address} as {send_name}")

        assert len(pull_stream_names) == 1
        recv_name = names.request_reply_stream(experiment_name, trial_name, pull_stream_names[0])
        try:
            master_send_addr = name_resolve.wait(recv_name, timeout=60)
        except TimeoutError as e:
            logger.error(f"Worker timeout waiting for master send stream {pull_stream_names[0]}")
            raise e
        self.accept(recv_name, master_send_addr)
        logger.info(f"Get worker receive address: {master_send_addr} from {recv_name}")

        name_resolve.add_subentry(
            name=names.request_reply_stream(experiment_name, trial_name,
                                            PUBSUB_BARRIER_NAME.format(name=pull_stream_names[0])),
            value=self.address,
            keepalive_ttl=60,
        )


def make_master_stream(
    worker_info: api.config.WorkerInformation,
    config: api.config.RequestReplyStream,
) -> NameResolvingRequstReplyMasterStream:
    return NameResolvingRequstReplyMasterStream(
        experiment_name=worker_info.experiment_name,
        trial_name=worker_info.trial_name,
        push_stream_name=config.push_stream_name,
        pull_stream_names=config.pull_stream_names,
        serialization_method=config.serialization_method,
    )


def make_worker_stream(
    worker_info: api.config.WorkerInformation,
    config: api.config.RequestReplyStream,
) -> NameResolvingRequstReplyWorkerStream:
    return NameResolvingRequstReplyWorkerStream(
        experiment_name=worker_info.experiment_name,
        trial_name=worker_info.trial_name,
        push_stream_name=config.push_stream_name,
        pull_stream_names=config.pull_stream_names,
        serialization_method=config.serialization_method,
    )
