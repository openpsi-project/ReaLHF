# Request-reply stream between model workers and the master worker.
# The stream is composed of a pair of ZMQ sockets, one PUSH and one PULL, for asynchronous communication,
# i.e., the model worker can buffer requests from the master and execute them in any order under the hood.
import asyncio
import dataclasses
import pickle
import re
import socket
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import zmq

import realhf.api.core.system_api as system_api
from realhf.base import logging, name_resolve, names

logger = logging.getLogger("Request-Replay Stream")
ZMQ_IO_THREADS = 8

PUBSUB_BARRIER_NAME = "__pubsub_barrier__"


class NoMessage(Exception):
    pass


class NoResponse:
    pass


@dataclasses.dataclass
class Payload:
    handler: Union[system_api.ModelShardID, str]
    handle_name: str

    request_id: uuid.UUID = None
    syn_reply_id: uuid.UUID = None
    ack_reply_id: uuid.UUID = None

    send_time: float = None

    # Non-tensor data
    data: Any = None

    # RPC hooks
    pre_hooks: List[str] = dataclasses.field(default_factory=list)
    pre_hook_data: List[Any] = dataclasses.field(default_factory=list)

    post_hooks: List[str] = dataclasses.field(default_factory=list)
    post_hook_data: List[Any] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if self.request_id is None:
            self.request_id = uuid.uuid4()
        if self.syn_reply_id is None:
            self.syn_reply_id = uuid.uuid4()
        if self.ack_reply_id is None:
            self.ack_reply_id = uuid.uuid4()


class NameResolvingRequestClient:

    def __init__(
        self,
        experiment_name: str,
        trial_name: str,
        n_subscribers: int,
        handler_routing: Dict[str | system_api.ModelShardID, int],
    ):

        self.context = zmq.Context.instance(io_threads=ZMQ_IO_THREADS)
        host_ip = socket.gethostbyname(socket.gethostname())

        self.send_sockets: List[zmq.Socket] = []
        for i in range(n_subscribers):
            s: zmq.Socket = self.context.socket(zmq.PUSH)
            send_port = s.bind_to_random_port(f"tcp://{host_ip}")
            s.setsockopt(zmq.LINGER, 0)

            master_send_name = names.request_reply_stream(
                experiment_name, trial_name, f"master_send_{i}"
            )
            name_resolve.add(name=master_send_name, value=f"{host_ip}:{send_port}")
            logger.debug(
                f"Add master send address {host_ip}:{send_port} as {master_send_name}"
            )
            self.send_sockets.append(s)

        self.recv_socket: zmq.Socket = self.context.socket(zmq.PULL)
        recv_port = self.recv_socket.bind_to_random_port(f"tcp://{host_ip}")
        self.recv_socket.setsockopt(zmq.LINGER, 0)
        self.recv_address = f"{host_ip}:{recv_port}"

        master_recv_name = names.request_reply_stream(
            experiment_name, trial_name, "master_recv"
        )
        name_resolve.add(name=master_recv_name, value=self.recv_address)
        logger.debug(
            f"Add master send address {self.recv_address} as {master_recv_name}"
        )

        self._response_buffer: Dict[uuid.UUID, Payload] = {}
        self._handler_routing = handler_routing

        # master needs to wait all peers (subscribers) to connect
        while (
            len(
                name_resolve.get_subtree(
                    names.request_reply_stream(
                        experiment_name, trial_name, PUBSUB_BARRIER_NAME
                    )
                )
            )
            < n_subscribers
        ):
            time.sleep(0.1)
        logger.debug(
            f"Master discovered all {n_subscribers} "
            f"subscribers: {name_resolve.get_subtree(names.request_reply_stream(experiment_name, trial_name, PUBSUB_BARRIER_NAME))}."
        )

    def close(self):
        self.recv_socket.close()
        for send_socket in self.send_sockets:
            send_socket.close()
        self.context.destroy()

    def __del__(self):
        self.close()

    def post(self, payload: Payload) -> uuid.UUID:
        assert payload.request_id is not None and payload.handle_name is not None
        payload.send_time = time.monotonic()
        idx = self._handler_routing[payload.handler]
        self.send_sockets[idx].send(pickle.dumps(payload))
        return payload.request_id

    def poll(self, pattern: re.Pattern | None = None, block: bool = False) -> Payload:
        payloads = self.poll_batch(pattern=pattern, block=block)
        for p in payloads[1:]:
            self._response_buffer[p.request_id] = p
        return payloads[0]

    def poll_batch(
        self, pattern: re.Pattern | None = None, block: bool = False
    ) -> List[Payload]:
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

    def _poll_batch_nonblock(
        self, pattern: Optional[re.Pattern] = None
    ) -> List[Payload]:
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


class NameResolvingReplyServer:

    def __init__(
        self,
        experiment_name: str,
        trial_name: str,
        idx: int,
    ):
        self.context = zmq.Context.instance(io_threads=ZMQ_IO_THREADS)

        send_name = names.request_reply_stream(
            experiment_name, trial_name, "master_recv"
        )
        try:
            master_recv_addr = name_resolve.wait(send_name, timeout=300)
        except TimeoutError as e:
            logger.error(f"Worker timeout waiting for master receive stream.")
            raise e

        recv_name = names.request_reply_stream(
            experiment_name, trial_name, f"master_send_{idx}"
        )
        try:
            master_send_addr = name_resolve.wait(recv_name, timeout=300)
        except TimeoutError as e:
            logger.error(f"Worker timeout waiting for master send stream")
            raise e

        self.accept(master_send_addr, master_recv_addr)

        name_resolve.add_subentry(
            name=names.request_reply_stream(
                experiment_name, trial_name, PUBSUB_BARRIER_NAME
            ),
            value=socket.gethostbyname(socket.gethostname()),
            keepalive_ttl=60,
        )

    def accept(self, server_send_addr: str, server_recv_addr: str):
        recv_socket: zmq.Socket = self.context.socket(zmq.PULL)
        recv_socket.connect(f"tcp://{server_send_addr}")
        recv_socket.setsockopt(zmq.LINGER, 0)
        self.recv_socket = recv_socket

        send_socket: zmq.Socket = self.context.socket(zmq.PUSH)
        send_socket.connect(f"tcp://{server_recv_addr}")
        send_socket.setsockopt(zmq.LINGER, 0)
        self.send_socket = send_socket

    def post(self, payload: Payload) -> uuid.UUID:
        assert payload.request_id is not None and payload.handle_name is not None
        payload.send_time = time.monotonic()
        self.send_socket.send(pickle.dumps(payload))
        return payload.request_id

    def poll(self, block: bool = False) -> Payload:
        try:
            payload_bytes = self.recv_socket.recv(flags=0 if block else zmq.NOBLOCK)
        except zmq.ZMQError:
            raise NoMessage()

        payload: Payload = pickle.loads(payload_bytes)
        # logger.debug(f"Payload transfer time: {time.monotonic() - payload.send_time:.4f}s")
        return payload

    def close(self):
        self.recv_socket.close()
        self.send_socket.close()
        self.context.destroy()

    def __del__(self):
        self.close()


def make_master_stream(
    worker_info: system_api.WorkerInformation,
    n_subscribers: int,
    handler_routing: Dict[str | system_api.ModelShardID, int],
) -> NameResolvingRequestClient:
    return NameResolvingRequestClient(
        experiment_name=worker_info.experiment_name,
        trial_name=worker_info.trial_name,
        n_subscribers=n_subscribers,
        handler_routing=handler_routing,
    )


def make_worker_stream(
    worker_info: system_api.WorkerInformation,
    idx: int,
) -> NameResolvingReplyServer:
    return NameResolvingReplyServer(
        experiment_name=worker_info.experiment_name,
        trial_name=worker_info.trial_name,
        idx=idx,
    )
