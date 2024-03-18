# Request-reply stream between model workers and the master worker.
# The stream is composed of a pair of ZMQ sockets, one PUSH and one PULL, for asynchronous communication,
# i.e., the model worker can buffer requests from the master and execute them in any order under the hood.
from typing import Any, Dict, List, Optional, Tuple, Union
import asyncio
import dataclasses
import pickle
import re
import socket
import time
import uuid

import torch
import zmq

import api.config.config_system
import api.config.dfg
import base.logging as logging
import base.name_resolve as name_resolve
import base.namedarray as namedarray
import base.names as names

logger = logging.getLogger("Request-Replay Stream")
ZMQ_IO_THREADS = 8

PUBSUB_BARRIER_NAME = "__pubsub_barrier__"


class NoMessage(Exception):
    pass


@dataclasses.dataclass
class Payload:
    handler: Union[api.config.config_system.ModelShardID, str]
    handle_name: str

    request_id: uuid.UUID = None
    ack_reply_id: uuid.UUID = None

    send_time: float = None

    # Non-tensor data
    data: Any = None

    # Specs of tensor data. Tensors will be trasnferred with NCCL.
    buffer_indices: Optional[List[int]] = None
    seqlens: Optional[List[int]] = None

    def __post_init__(self):
        if self.request_id is None:
            self.request_id = uuid.uuid4()
        if self.ack_reply_id is None:
            self.ack_reply_id = uuid.uuid4()


class RequestReplyStream:

    def post(self, payload: Payload) -> uuid.UUID:
        raise NotImplementedError()

    def post_batch(self, payloads: List[Payload]) -> List[uuid.UUID]:
        raise NotImplementedError()

    def poll(self, pattern: re.Pattern | None = None, block: bool = False) -> Payload:
        raise NotImplementedError()

    def poll_batch(self, pattern: re.Pattern | None = None, block: bool = False) -> List[Payload]:
        raise NotImplementedError()

    @property
    def context(self) -> zmq.Context:
        return self._context

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

    def __init__(self):
        self._context = zmq.Context.instance(io_threads=ZMQ_IO_THREADS)
        host_ip = socket.gethostbyname(socket.gethostname())

        self._send_socket = self._context.socket(zmq.PUB)
        send_port = self._send_socket.bind_to_random_port(f"tcp://{host_ip}")
        self._send_socket.setsockopt(zmq.LINGER, 0)
        self.send_address = f"{host_ip}:{send_port}"

        self._recv_socket = self.context.socket(zmq.PULL)
        recv_port = self._recv_socket.bind_to_random_port(f"tcp://{host_ip}")
        self._recv_socket.setsockopt(zmq.LINGER, 0)
        self.recv_address = f"{host_ip}:{recv_port}"

        self._response_buffer: Dict[uuid.UUID, Payload] = {}

    def post(self, payload: Payload) -> uuid.UUID:
        assert payload.request_id is not None and payload.handle_name is not None
        payload.send_time = time.monotonic()
        self.send_socket.send_multipart([str(payload.handler).encode("utf-8"), pickle.dumps(payload)])
        return payload.request_id

    def poll(self, pattern: re.Pattern | None = None, block: bool = False) -> Payload:
        payloads = self.poll_batch(pattern=pattern, block=block)
        for p in payloads[1:]:
            self._response_buffer[p.request_id] = p
        return payloads[0]

    def poll_batch(self, pattern: re.Pattern | None = None, block: bool = False) -> List[Payload]:
        """Collect responses that match some pattern from the stream.

        This function may NOT actually pull from the stream. It may fetch something
        from the buffer, which records mismatched responses.

        Args:
            pattern (Optional[re.Pattern], optional): Only responses with this
                specific regex pattern will be returned.
                None means no pattern specified. Defaults to None.
            block (bool, optional): Whether to block to receive a
                response (with the given pattern). Defaults to False.
        """
        if not block:
            return self._poll_batch_nonblock(pattern)
        else:
            while True:
                try:
                    return self._poll_batch_nonblock(pattern)
                except NoMessage:
                    time.sleep(0.05)

    def _poll_batch_nonblock(self, pattern: Optional[re.Pattern] = None) -> List[Payload]:
        # Check whether there's response in the buffer.
        # If so, return immediately.
        if pattern is None:
            pattern = re.compile(".*")

        payloads = []
        for req_id, p in self._response_buffer.items():
            if pattern.match(str(req_id)):
                payloads.append(p)
        for p in payloads:
            self._response_buffer.pop(p.request_id)
        if len(payloads) > 0:
            return payloads

        # Otherwise, pull from the socket.
        try:
            p_bytes = self.recv_socket.recv(flags=zmq.NOBLOCK)
        except zmq.ZMQError:
            raise NoMessage()
        payload: Payload = pickle.loads(p_bytes)
        # logger.info(f"Payload transfer time: {time.monotonic() - payload.send_time:.4f}s")
        self._response_buffer[payload.request_id] = payload

        payloads = []
        for req_id, p in self._response_buffer.items():
            if pattern.match(str(req_id)):
                payloads.append(p)
        for p in payloads:
            self._response_buffer.pop(p.request_id)
        if len(payloads) > 0:
            return payloads
        raise NoMessage()


class IpReplyServer(RequestReplyStream):

    def __init__(self):
        self._context = zmq.Context.instance(io_threads=ZMQ_IO_THREADS)

    def accept(self, server_send_addr: str, server_recv_addr: str, sub_partterns: List[str] = None):
        recv_socket: zmq.Socket = self.context.socket(zmq.SUB)
        recv_socket.connect(f"tcp://{server_send_addr}")
        recv_socket.setsockopt(zmq.LINGER, 0)
        if sub_partterns is not None:
            for p in sub_partterns:
                recv_socket.setsockopt(zmq.SUBSCRIBE, str(p).encode("utf-8"))
        self._recv_socket = recv_socket

        send_socket: zmq.Socket = self._context.socket(zmq.PUSH)
        send_socket.connect(f"tcp://{server_recv_addr}")
        send_socket.setsockopt(zmq.LINGER, 0)
        self._send_socket = send_socket

    def post(self, payload: Payload) -> uuid.UUID:
        assert payload.request_id is not None and payload.handle_name is not None
        payload.send_time = time.monotonic()
        self.send_socket.send(pickle.dumps(payload))
        return payload.request_id

    def poll(self, block: bool = False) -> Payload:
        try:
            _, payload_bytes = self.recv_socket.recv_multipart(flags=0 if block else zmq.NOBLOCK)
        except zmq.ZMQError:
            raise NoMessage()

        payload: Payload = pickle.loads(payload_bytes)
        # logger.debug(f"Payload transfer time: {time.monotonic() - payload.send_time:.4f}s")
        return payload


class NameResolvingRequstClient(IpRequestClient):

    def __init__(
        self,
        experiment_name: str,
        trial_name: str,
        n_subscribers: int,
    ):
        super().__init__()

        master_send_name = names.request_reply_stream(experiment_name, trial_name, "master_send")
        name_resolve.add(name=master_send_name, value=self.send_address)
        logger.info(f"Add master send address {self.send_address} as {master_send_name}")

        master_recv_name = names.request_reply_stream(experiment_name, trial_name, "master_recv")
        name_resolve.add(name=master_recv_name, value=self.recv_address)
        logger.info(f"Add master send address {self.recv_address} as {master_recv_name}")

        # master needs to wait all peers (subscribers) to connect
        while (len(
                name_resolve.get_subtree(
                    names.request_reply_stream(experiment_name, trial_name, PUBSUB_BARRIER_NAME)))
               < n_subscribers):
            time.sleep(0.1)
        logger.info(
            f"Master discovered all {n_subscribers} "
            f"subscribers: {name_resolve.get_subtree(names.request_reply_stream(experiment_name, trial_name, PUBSUB_BARRIER_NAME))}."
        )


class NameResolvingReplyServer(IpReplyServer):

    def __init__(
        self,
        experiment_name: str,
        trial_name: str,
        sub_patterns: List[str] = None,
    ):
        super().__init__()

        send_name = names.request_reply_stream(experiment_name, trial_name, "master_recv")
        try:
            master_recv_addr = name_resolve.wait(send_name, timeout=300)
        except TimeoutError as e:
            logger.error(f"Worker timeout waiting for master receive stream.")
            raise e

        recv_name = names.request_reply_stream(experiment_name, trial_name, "master_send")
        try:
            master_send_addr = name_resolve.wait(recv_name, timeout=300)
        except TimeoutError as e:
            logger.error(f"Worker timeout waiting for master send stream")
            raise e

        self.accept(master_send_addr, master_recv_addr, sub_patterns)

        name_resolve.add_subentry(
            name=names.request_reply_stream(experiment_name, trial_name, PUBSUB_BARRIER_NAME),
            value=socket.gethostbyname(socket.gethostname()),
            keepalive_ttl=60,
        )


def make_master_stream(
    worker_info: api.config.config_system.WorkerInformation,
    n_subscribers: int,
) -> NameResolvingRequstClient:
    return NameResolvingRequstClient(
        experiment_name=worker_info.experiment_name,
        trial_name=worker_info.trial_name,
        n_subscribers=n_subscribers,
    )


def make_worker_stream(
    worker_info: api.config.config_system.WorkerInformation,
    sub_patterns: List[Union[str, api.config.config_system.ModelShardID]],
) -> NameResolvingReplyServer:
    return NameResolvingReplyServer(
        experiment_name=worker_info.experiment_name,
        trial_name=worker_info.trial_name,
        sub_patterns=sub_patterns,
    )
