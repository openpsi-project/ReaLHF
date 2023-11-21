import pickle
import queue
import threading
import time

import zmq

from .schedule import type_to_schedule
import base.name_resolve as name_resolve
import base.names as names
import base.network as network


# TODO: try zmq.asyncio
class EngineScheduleController:
    """ Centralized scheduler that schedules pipeline instruction for each engine.
    only exists on every engine that has pp_rank=0
    """

    def __init__(self, expr_name: str, trial_name: str, model_name: str, dp_rank: int, mp_rank: int,
                 schedule_queue: queue.Queue):
        address = network.gethostip()
        self.context = zmq.Context()

        self.instruction_socket = self.context.socket(zmq.PUB)
        instruction_port = network.find_free_port()
        self.instruction_socket.bind(f"tcp://*:{instruction_port}")

        self.signal_socket = self.context.socket(zmq.PULL)
        signal_port = network.find_free_port()
        self.signal_socket.bind(f"tcp://*:{signal_port}")

        name = names.stream_pipe_engine_schedule(expr_name, trial_name, model_name, dp_rank, mp_rank)
        name_resolve.add(name, f"{address};{instruction_port};{signal_port}", keepalive_ttl=30)

        # receive schedule command from interface
        self.schedule_queue = schedule_queue
        self.__terminated = False

        self.thread = threading.Thread(target=self.run)
        self.thread.start()

        self.schedules = []
        self.pending_instructions = []

    def issue_schedule(self, schedule_type: str, **schedule_args):
        """ issue a schedule command to all engines
        """
        self.schedule_queue.put((schedule_type, schedule_args))

    def check_and_start_schedule(self):
        """ check if there is any schedule command 
        """
        try:
            schedule_type, schedule_args = self.schedule_queue.get()
            new_sched = type_to_schedule(schedule_type)(**schedule_args)
            self.schedules.append(new_sched)
        except queue.Empty:
            pass

    def advance(self, pp_rank, signal):
        """ advance into next schedule steps according to received signal from client
        """
        pass

    def ready_steps(self):
        """ return current ready steps
        """
        pass

    def check_signal(self):
        try:
            pp_rank, signal = self.signal_socket.recv_multipart(flags=zmq.NOBLOCK)
            signal = signal.decode(encoding="utf-8")
            self.advance(pp_rank, signal)
        except:
            pass

    def check_and_send_instruction(self):
        ready_steps = self.ready_steps()
        for pp_rank, instruction in ready_steps:
            msg = [str(pp_rank).encode(encoding="utf-8"), instruction.encode(encoding="utf-8")]
            self.instruction_socket.send_multipart(msg)

    def run(self):
        while not self.__terminated:
            schedule_args = self.schedule_queue.get()

    def terminate(self):
        self.__terminated = True


class EngineScheduleClient:
    """ Client on every enigne instance to receive instruction from EngineScheduleController
    """

    def __init__(self, expr_name: str, trial_name: str, model_name: str, pp_rank: int, dp_rank: int,
                 mp_rank: int):
        self.pp_rank = pp_rank
        name = names.stream_pipe_engine_schedule(expr_name, trial_name, model_name, dp_rank, mp_rank)
        address, instruction_port, signal_port = name_resolve.get(name).split(";")

        self.context = zmq.Context()
        self.instruction_socket = self.context.socket(zmq.SUB)
        self.instruction_socket.setsockopt_string(zmq.SUBSCRIBE, str(pp_rank))
        self.instruction_socket.connect(f"tcp://{address}:{instruction_port}")

        self.signal_socket = self.context.socket(zmq.PUSH)
        self.signal_socket.connect(f"tcp://{address}:{signal_port}")

        self.signal_queue = queue.Queue(maxsize=1024)
        self.instruction_queue = queue.Queue(maxsize=1024)
        self.__terminated = False

        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def signal(self, signal: str):
        """ Send a signal to EngineScheduleController to notify that is has completed 
        some vital instruction
        """
        self.signal_queue.put(signal)

    def check_instruction(self):
        try:
            return self.instruction_queue.get_nowait()
        except queue.Empty:
            return None

    def check_signal_and_send(self):
        try:
            signal: str = self.signal_queue.get_nowait()
            msg = [str(self.pp_rank).encode(encoding="utf-8"), signal.encode(encoding="utf-8")]
            self.signal_socket.send(msg)
        except queue.Empty:
            pass

    def poll(self):
        """ try receive instruction
        """
        try:
            pp_rank, msg = self.instruction_socket.recv_multipart(flags=zmq.NOBLOCK)
            msg = msg.decode(encoding="utf-8")
            self.instruction_queue.put(msg)
        except zmq.ZMQError:
            pass

    def run(self):
        while not self.__terminated:
            self.check_signal_and_send()
            self.poll()
            time.sleep(1)

    def terminate(self):
        self.__terminated = True
