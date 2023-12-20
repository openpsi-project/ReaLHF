# Request-reply stream between model workers and the master worker.
# The stream is composed of a pair of ZMQ sockets, one PUSH and one PULL, for asynchronous communication,
# i.e., the model worker can buffer requests from the master and execute them in any order under the hood.
from typing import Dict, Optional, Union
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


class RequestReplyStream:
    # in master server
    def post(self, payload: Payload):
        raise NotImplementedError()

    def poll(self, block: bool = False) -> Payload:
        raise NotImplementedError()


class IpRequestReplyStream(RequestReplyStream):

    def __init__(self, server_address: str, serialization_method: str):
        self._context = zmq.Context(io_threads=ZMQ_IO_THREADS)
        self._send_socket = self._context.socket(zmq.PUSH)
        host_ip = socket.gethostbyname(socket.gethostname())
        port = self._send_socket.bind_to_random_port(f"tcp://{host_ip}")
        self._send_socket.setsockopt(zmq.LINGER, 0)

        self.address = f"{host_ip}:{port}"

        self._recv_socket = self._context.socket(zmq.PULL)
        self._recv_socket.connect(f"tcp://{server_address}")
        self._recv_socket.setsockopt(zmq.LINGER, 0)

        self._serialization_method = serialization_method

    def post(self, payload: Payload):
        tik = time.monotonic()
        if isinstance(payload.data, namedarray.NamedArray):
            assert isinstance(payload.data, namedarray.NamedArray), type(payload.data)
            # payload.data = namedarray.recursive_apply(payload.data,
            #                                           lambda x: x.cpu().numpy() if not x.is_sparse else pickle.dumps(x))
            payload.data = namedarray.dumps(payload.data, method=self._serialization_method)
            encoding = b"01"
        else:
            payload.data = [pickle.dumps(payload.data)]
            encoding = b"00"
        self._send_socket.send_multipart([
            pickle.dumps(tik),
            payload.handle_name.encode("ascii"),
            payload.request_id.encode("ascii"),
            encoding,
        ] + payload.data)

    def poll(self, block: bool = False) -> Payload:
        try:
            time_bytes, handle_name, request_id, encoding, *data = self._recv_socket.recv_multipart(
                flags=0 if block else zmq.NOBLOCK)
        except zmq.ZMQError:
            raise NoMessage()

        send_time = pickle.loads(time_bytes)
        handle_name = handle_name.decode("ascii")
        request_id = request_id.decode("ascii")
        if encoding == b"01":
            data = namedarray.loads(data)
            # data = namedarray.recursive_apply(data,
            #                                   lambda x: torch.from_numpy(x) if not isinstance(x, bytes) else pickle.loads(x))
        elif encoding == b"00":
            assert len(data) == 1
            data = pickle.loads(data[0])
        else:
            raise NotImplementedError()
        logger.debug(f"Payload transfer time: {time.monotonic() - send_time:.4f}s")
        return Payload(handle_name=handle_name, request_id=request_id, data=data)


class NameResolvingRequstReplyStream(IpRequestReplyStream):

    def __init__(
        self,
        experiment_name: str,
        trial_name: str,
        push_stream_name: str,
        pull_stream_name: str,
        serialization_method: str,
    ):
        self._context = zmq.Context(io_threads=ZMQ_IO_THREADS)
        self._send_socket = self._context.socket(zmq.PUSH)
        host_ip = socket.gethostbyname(socket.gethostname())
        port = self._send_socket.bind_to_random_port(f"tcp://{host_ip}")
        self._send_socket.setsockopt(zmq.LINGER, 0)

        address = f"{host_ip}:{port}"
        name = names.request_reply_stream(experiment_name, trial_name, push_stream_name)
        name_resolve.add(name=name, value=address)

        logger.info(f"Added push stream {push_stream_name}.")

        name = names.request_reply_stream(experiment_name, trial_name, pull_stream_name)

        try:
            server_address = name_resolve.wait(name, timeout=60)
        except TimeoutError as e:
            logger.error(f"Timeout waiting for {pull_stream_name}")
            raise e

        self._recv_socket = self._context.socket(zmq.PULL)
        self._recv_socket.connect(f"tcp://{server_address}")
        self._recv_socket.setsockopt(zmq.LINGER, 0)

        self._serialization_method = serialization_method


def make_stream(
    worker_info: api.config.WorkerInformation,
    config: api.config.RequestReplyStream,
) -> RequestReplyStream:
    return NameResolvingRequstReplyStream(
        experiment_name=worker_info.experiment_name,
        trial_name=worker_info.trial_name,
        push_stream_name=config.push_stream_name,
        pull_stream_name=config.pull_stream_name,
        serialization_method=config.serialization_method,
    )
