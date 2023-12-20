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

    send_time: float = None
    is_tensor: bool = True
    dtypes: Optional[Dict[str, str]] = None
    shapes: Optional[Dict[str, List[int]]] = None

    data: Optional[Union[Dict, namedarray.NamedArray]] = None

    def __post_init__(self):
        if self.request_id is None:
            self.request_id = str(uuid.uuid4())


class RequestReplyStream:

    def post(self, payload: Payload):
        raise NotImplementedError()

    def poll(self, block: bool = False) -> Payload:
        raise NotImplementedError()

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
    def recv_socket(self) -> zmq.Socket:
        return self._recv_socket

    @property
    def serialization_method(self) -> str:
        return self._serialization_method

    def close(self):
        if hasattr(self, "_recv_socket"):
            self.recv_socket.close()
        if hasattr(self, "_send_socket"):
            self.send_socket.close()
        if hasattr(self, "_context"):
            self.context.destroy()

    def __del__(self):
        self.close()


class IpRequestClient(RequestReplyStream):

    def __init__(self, serialization_method: str):
        self._context = zmq.Context.instance(io_threads=ZMQ_IO_THREADS)

        self._send_socket = self._context.socket(zmq.PUB)
        host_ip = socket.gethostbyname(socket.gethostname())
        send_port = self._send_socket.bind_to_random_port(f"tcp://{host_ip}")
        # self._send_socket.setsockopt(zmq.LINGER, 0)
        self._send_address = f"{host_ip}:{send_port}"

        self._serialization_method = serialization_method

    def accept(self, server_addr: str):
        recv_socket = self.context.socket(zmq.PULL)
        recv_socket.connect(f"tcp://{server_addr}")
        # recv_socket.setsockopt(zmq.LINGER, 0)
        self._recv_socket = recv_socket

    def post(self, payload: Payload):
        assert payload.request_id is not None and payload.handle_name is not None
        if isinstance(payload.data, namedarray.NamedArray):
            assert isinstance(payload.data, namedarray.NamedArray), type(payload.data)
            encoded_data = namedarray.recursive_apply(payload.data, lambda x: x.cpu().numpy())
            payload.is_tensor = True
            payload.dtypes = {k: namedarray._numpy_dtype_to_str(v.dtype) for k, v in encoded_data.items()}
            payload.shapes = {k: v.shape for k, v in encoded_data.items()}
            encoded_data = namedarray.dumps(encoded_data, method=self.serialization_method)
            encoding = b"01"
        else:
            encoded_data = [pickle.dumps(payload.data)]
            payload.is_tensor = False
            encoding = b"00"

        # For all workers.
        self.send_socket.send_multipart([
            b"metadata",
            pickle.dumps(
                Payload(
                    request_id=payload.request_id,
                    handle_name=payload.handle_name,
                    is_tensor=payload.is_tensor,
                    dtypes=payload.dtypes,
                    shapes=payload.shapes,
                    send_time=time.monotonic(),
                )),
        ])
        # For DP heads.
        self.send_socket.send_multipart([b"data", encoding] + encoded_data)
        return payload.request_id

    def poll(self, block: bool = False) -> Payload:
        try:
            p_bytes, encoding, *data = self.recv_socket.recv_multipart(flags=0 if block else zmq.NOBLOCK)
        except zmq.ZMQError:
            raise NoMessage()

        payload: Payload = pickle.loads(p_bytes)
        if encoding == b"01":
            data = namedarray.loads(data)
            data = namedarray.recursive_apply(data, lambda x: torch.from_numpy(x))
        elif encoding == b"00":
            assert len(data) == 1
            data = pickle.loads(data[0])
        else:
            raise NotImplementedError()
        logger.info(f"Payload transfer time: {time.monotonic() - payload.send_time:.4f}s")
        payload.data = data
        return payload


class IpReplyServer(RequestReplyStream):

    def __init__(self, serialization_method: str):
        self._context = zmq.Context.instance(io_threads=ZMQ_IO_THREADS)

        self._send_socket = self._context.socket(zmq.PUSH)
        host_ip = socket.gethostbyname(socket.gethostname())
        send_port = self._send_socket.bind_to_random_port(f"tcp://{host_ip}")
        # self._send_socket.setsockopt(zmq.LINGER, 0)
        self._send_address = f"{host_ip}:{send_port}"

        self._serialization_method = serialization_method

    def accept(self, server_addr: str, is_dp_head: bool):
        recv_socket: zmq.Socket = self.context.socket(zmq.SUB)
        recv_socket.connect(f"tcp://{server_addr}")
        if is_dp_head:
            recv_socket.setsockopt(zmq.SUBSCRIBE, b"")
        else:
            recv_socket.setsockopt(zmq.SUBSCRIBE, b"metadata")
        # recv_socket.setsockopt(zmq.LINGER, 0)
        self._recv_socket = recv_socket
        self._is_dp_head = is_dp_head

    def post(self, payload: Payload):
        assert payload.request_id is not None and payload.handle_name is not None
        if isinstance(payload.data, namedarray.NamedArray):
            assert isinstance(payload.data, namedarray.NamedArray), type(payload.data)
            encoded_data = namedarray.recursive_apply(payload.data, lambda x: x.cpu().numpy())
            encoded_data = namedarray.dumps(encoded_data, method=self.serialization_method)
            encoding = b"01"
        else:
            encoded_data = [pickle.dumps(payload.data)]
            encoding = b"00"

        msg = [
            pickle.dumps(
                Payload(
                    request_id=payload.request_id,
                    handle_name=payload.handle_name,
                    send_time=time.monotonic(),
                )),
            encoding,
        ] + encoded_data
        self.send_socket.send_multipart(msg)
        return payload.request_id

    def poll(self, block: bool = False) -> Payload:
        try:
            topic, *msg = self.recv_socket.recv_multipart(flags=0 if block else zmq.NOBLOCK)
        except zmq.ZMQError:
            raise NoMessage()

        assert len(msg) == 1, len(msg)
        assert topic == b"metadata"
        payload: Payload = pickle.loads(msg[0])
        if self._is_dp_head:
            _, encoding, *data = self.recv_socket.recv_multipart()
            if encoding == b"01":
                data = namedarray.loads(data)
                data = namedarray.recursive_apply(data, lambda x: torch.from_numpy(x))
            elif encoding == b"00":
                assert len(data) == 1
                data = pickle.loads(data[0])
            else:
                raise NotImplementedError()
            payload.data = data
        logger.debug(f"Payload transfer time: {time.monotonic() - payload.send_time:.4f}s")
        return payload


class NameResolvingRequstClient(IpRequestClient):

    def __init__(
        self,
        experiment_name: str,
        trial_name: str,
        push_stream_name: str,
        pull_stream_name: str,
        n_subscribers: int,
        serialization_method: str,
    ):
        super().__init__(serialization_method)
        assert isinstance(push_stream_name, str), push_stream_name
        assert isinstance(pull_stream_name, str), pull_stream_name

        master_send_name = names.request_reply_stream(experiment_name, trial_name, push_stream_name)
        name_resolve.add(name=master_send_name, value=self.address)

        logger.info(f"Add master send address {self.address} as {master_send_name}")

        master_recv_name = names.request_reply_stream(experiment_name, trial_name, pull_stream_name)
        try:
            master_recv_address = name_resolve.wait(master_recv_name, timeout=60)
        except TimeoutError as e:
            logger.error(f"Master timeout waiting for worker send stream {pull_stream_name}")
            raise e
        self.accept(master_recv_address)
        logger.info(f"Get master receive address: {master_recv_address} from {master_recv_name}")

        # master needs to wait all peers (subscribers) to connect
        while (len(
                name_resolve.get_subtree(
                    names.request_reply_stream(experiment_name, trial_name,
                                               PUBSUB_BARRIER_NAME.format(name=push_stream_name))))
               < n_subscribers):
            time.sleep(0.1)
        logger.info(
            f"Master discovered all {n_subscribers} "
            f"subscribers: {name_resolve.get_subtree(names.request_reply_stream(experiment_name, trial_name, PUBSUB_BARRIER_NAME.format(name=push_stream_name)))}."
        )


class NameResolvingReplyServer(IpReplyServer):

    def __init__(
        self,
        experiment_name: str,
        trial_name: str,
        push_stream_name: str,
        pull_stream_name: str,
        is_dp_head: bool,
        serialization_method: str,
    ):
        super().__init__(serialization_method)

        send_name = names.request_reply_stream(experiment_name, trial_name, push_stream_name)
        name_resolve.add(name=send_name, value=self.address)

        logger.info(f"Add worker send address {self.address} as {send_name}")

        recv_name = names.request_reply_stream(experiment_name, trial_name, pull_stream_name)
        try:
            master_send_addr = name_resolve.wait(recv_name, timeout=60)
        except TimeoutError as e:
            logger.error(f"Worker timeout waiting for master send stream {pull_stream_name}")
            raise e
        self.accept(master_send_addr, is_dp_head)
        logger.info(f"Get worker receive address: {master_send_addr} from {recv_name}")

        name_resolve.add_subentry(
            name=names.request_reply_stream(experiment_name, trial_name,
                                            PUBSUB_BARRIER_NAME.format(name=pull_stream_name)),
            value=self.address,
            keepalive_ttl=60,
        )


def make_master_stream(
    worker_info: api.config.WorkerInformation,
    config: api.config.RequestReplyStream,
    n_subscribers: int,
) -> NameResolvingRequstClient:
    return NameResolvingRequstClient(
        experiment_name=worker_info.experiment_name,
        trial_name=worker_info.trial_name,
        push_stream_name=config.push_stream_name,
        pull_stream_name=config.pull_stream_name,
        n_subscribers=n_subscribers,
        serialization_method=config.serialization_method,
    )


def make_worker_stream(
    worker_info: api.config.WorkerInformation,
    config: api.config.RequestReplyStream,
    is_dp_head: bool,
) -> NameResolvingReplyServer:
    return NameResolvingReplyServer(
        experiment_name=worker_info.experiment_name,
        trial_name=worker_info.trial_name,
        push_stream_name=config.push_stream_name,
        pull_stream_name=config.pull_stream_name,
        serialization_method=config.serialization_method,
        is_dp_head=is_dp_head,
    )
