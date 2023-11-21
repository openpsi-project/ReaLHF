# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List
import copy
import queue
import threading
import time

from deepspeed.runtime.utils import call_to_str
import torch
import zmq

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

        self.schedule_cls = {
            "inference": InferenceSchedule,
            "train1f1b": Train1F1BSchedule,
        }

    def issue_schedule(self, schedule_type: str, **schedule_args):
        """ issue a schedule command to all engines
        """
        self.schedule_queue.put((schedule_type, schedule_args))

    def check_and_start_schedule(self, schedule_type: str, **schedule_args):
        """ check if there is any schedule command 
        """
        try:
            schedule, schedule_args = self.schedule_queue.get()
            new_sched = self.schedule_cls[schedule](**schedule_args)
            self.schedules.append(new_sched)
        except queue.Empty:
            pass

    def advance(self, pp_rank, signal):
        """ advance into next schedule steps according to received signal from client
        """
        pass

    def ready_steps(self):
        """ return current ready steps are not sent
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


class PipeSchedule(ABC):
    """ Define an execution sequence of instructions of a stream pipeline engine 
    in the centralized EngineScheduleController.

    Args:
        num_micro_batches (int): Number of micro-batches to split the batch into
        num_stages (int): Number of pipeline stages
        priority (int): Priority of the schedule, higher priority schedules are
                        scheduled first in the centralized controller.
    """

    def __init__(self, num_micro_batches: int, num_stages: int, priority: int):
        super().__init__()
        self.num_micro_batches = num_micro_batches
        self.num_stages = num_stages
        self.priority = priority
        self.__executed = {}
        self.__invar_instructions = {stage_id: self.steps(stage_id) for stage_id in range(self.num_stages)}
        self.__instructions = copy.deepcopy(self.__invar_instructions)
        self.__current_ready = defaultdict(list)
        self.update_ready()

    @abstractmethod
    def steps(self, stage_id):
        """Return a list of :class:`PipeInstruction` for each step in the schedule
        of stage `stage_id`.

        .. note::
            Schedules must implement ``steps()`` to define the schedule.

        Returns:
            Instructions to be executed as one step of the pipeline
        """
        raise NotImplementedError()

    @property
    def all_instructions(self):
        """ Return all instructions in the schedule
        """
        return self.__invar_instructions

    @property
    def current_instructions(self):
        """ Generate and return not executed instructions for all pipeline stages
        """
        return self.__instructions

    @property
    def current_ready(self):
        """ Return ready instructions for all pipeline stages
        """
        self.update_ready()
        return self.__current_ready

    def update_ready(self):
        self.__current_ready = defaultdict(list)
        for i in range(self.num_stages):
            insts = self.__instructions[i]
            for inst in insts:
                if self.is_ready(inst):
                    self.__current_ready[i].append(inst)

    def record_execution(self, inst: 'PipeInstruction'):
        if inst.stage_id not in self.__executed:
            self.__executed[inst.stage_id] = dict()
        if inst.micro_batch_id not in self.__executed[inst.stage_id]:
            self.__executed[inst.stage_id][inst.micro_batch_id] = defaultdict(int)

        to_remove = []
        # cannot use list.remove() directly because of input instruction is not the same object
        # in the instruction storage
        for cur_inst in self.__instructions[inst.stage_id]:
            if cur_inst == inst:
                to_remove.append(cur_inst)

        for cur_inst in to_remove:
            self.__instructions[inst.stage_id].remove(cur_inst)

        self.__executed[inst.stage_id][inst.micro_batch_id][inst.name] += 1
        self.update_ready()

    def is_executed(self, inst: 'PipeInstruction'):
        if inst.stage_id not in self.__executed:
            return False
        if inst.micro_batch_id not in self.__executed[inst.stage_id]:
            return False
        if inst.name not in self.__executed[inst.stage_id][inst.micro_batch_id]:
            return False
        return self.__executed[inst.stage_id][inst.micro_batch_id][inst.name] > inst.step_id

    def is_ready(self, inst: 'PipeInstruction'):
        if inst.deps is None:
            return True
        for dep in inst.deps:
            if not self.is_executed(dep):
                return False
        return True


class InferenceSchedule(PipeSchedule):
    """A schedule for inferencing batches using pipeline parallelism.
    """

    def steps(self):
        """"""
        # TODO: add store activation option
        pass


class Train1F1BSchedule(PipeSchedule):
    """A schedule for training a batch using hybrid parallelism.

    Pipeline parallelism is extracted through gradient accumulation and thus
    convergence follows that of a data parallel approach with the same batch
    size.
    """

    def __post_init__(self):
        assert self.num_micro_batches >= self.num_stages, \
               "In Train1F1BSchedule, num_micro_batches must be >= num_stages"

    def _is_first_half(self, micro_batch_id):
        """ In 1F1B, the first half of micro-batches in the first stage 
        has no dependencies other than last micro batch
        """
        return 0 <= micro_batch_id < self.num_stages

    def steps(self, stage_id):
        cmds = []
        # forward passes
        for micro_batch_id in range(self.num_micro_batches):
            forward_deps = []
            send_deps = []
            if stage_id == 0:
                load_deps = [LoadMicroBatch(stage_id=0, micro_batch_id=micro_batch_id - 1)]
                if not self._is_first_half(micro_batch_id):
                    # in second half of the micro batches, forward microbatch `m`` only if
                    # microbatch `m-self.num_stages` backward is finished
                    load_deps.append(BackwardPass(stage_id=0,
                                                  micro_batch_id=micro_batch_id - self.num_stages))
                cmds.append(LoadMicroBatch(stage_id=stage_id, micro_batch_id=micro_batch_id, deps=load_deps))
            else:
                recv_deps = [ForwardPass(stage_id=stage_id - 1, micro_batch_id=micro_batch_id)]
                cmds.append(RecvActivation(stage_id=stage_id, micro_batch_id=micro_batch_id, deps=recv_deps))
                forward_deps(RecvActivation(stage_id=stage_id, micro_batch_id=micro_batch_id))

            if stage_id < self.num_stages - 1:
                # when not last stage, send activation to next stage
                send_deps.append(ForwardPass(stage_id=stage_id, micro_batch_id=micro_batch_id))
                # depend on last stage forward pass to ensure send and recv activation appear in pair.
                cmds.append(SendActivation(stage_id=stage_id, micro_batch_id=micro_batch_id, deps=send_deps))
            cmds.append(ForwardPass(stage_id=stage_id, micro_batch_id=micro_batch_id, deps=forward_deps))

            backward_deps = [ForwardPass(stage_id=stage_id, micro_batch_id=micro_batch_id)]
            if stage_id > 0:
                # send grad to last stage
                send_grad_deps = [BackwardPass(stage_id=stage_id, micro_batch_id=micro_batch_id)]
                cmds.append(SendGrad(stage_id=stage_id, micro_batch_id=micro_batch_id, deps=send_grad_deps))
            if stage_id < self.num_stages - 1:
                recv_grad_deps = [BackwardPass(stage_id=stage_id + 1, micro_batch_id=micro_batch_id)]
                # depend on next stage backward pass to ensure send and recv activation appear in pair.
                cmds.append(RecvGrad(stage_id=stage_id, micro_batch_id=micro_batch_id, deps=recv_grad_deps))
                backward_deps.append(RecvGrad(stage_id=stage_id, micro_batch_id=micro_batch_id))
            cmds.append(BackwardPass(stage_id=stage_id, micro_batch_id=micro_batch_id, deps=backward_deps))
        return cmds


class DataParallelSchedule(PipeSchedule):
    """An example schedule that trains using traditional data parallelism with gradient
    accumulation.
    """

    def steps(self):
        pass


class PipeInstruction:
    """Base class for all instructions to be executed by the pipeline engine.

    All keyword arguments are stored as members similar to a ``namedtuple``. These are
    then accessible to the :class:`PipeEngine` during execution.

    Args:
        kwargs (optional): keyword arguments to store as members
    """

    def __init__(self,
                 stage_id: int,
                 micro_batch_id: int,
                 deps: List['PipeInstruction'] = None,
                 step_id: int = 0,
                 **kwargs):
        self.stage_id = stage_id
        self.micro_batch_id = micro_batch_id
        self.deps = deps
        self.name = self.__class__.__name__
        self.step_id = step_id

        self.args = (stage_id, micro_batch_id, step_id)
        self.kwargs = kwargs
        for key, val in self.kwargs.items():
            setattr(self, key, val)

    def __repr__(self):
        return call_to_str(self.name, self.args, self.kwargs)

    def __eq__(self, other: 'PipeInstruction'):
        return self.stage_id == other.stage_id and \
               self.micro_batch_id == other.micro_batch_id and \
               self.step_id == other.step_id and \
               self.name == other.name


class OptimizerStep(PipeInstruction):
    """Performs one step with the optimizer and zeros gradients.

    .. note:: Should be issued after :class:`ReduceGrads` and :class:`ReduceTiedGrads`.

    .. note:: Can be a synchronization point among data-parallel ranks.
    """
    pass


class ReduceGrads(PipeInstruction):
    """Reduce the computed gradients among data-parallel processes within the stage.
    """
    pass


class BufferOpInstruction(PipeInstruction):
    """A pipeline instruction that operates on pipeline buffer(s).
    """
    pass


# IO
class LoadMicroBatch(BufferOpInstruction):
    """Load a micro-batch into a buffer.
    """
    pass


# Compute
class ForwardPass(BufferOpInstruction):
    """Compute a forward pass.

    Roughly:

    .. code-block:: python

        buffers['outputs'][buffer_id] = forward(buffers['inputs'][buffer_id])
    """
    pass


class BackwardPass(BufferOpInstruction):
    """Compute a backward pass and accumulate gradients.
    """
    pass


# Communication
class SendActivation(BufferOpInstruction):
    """Send activations to the next stage in the pipeline.
    """
    pass


class RecvActivation(BufferOpInstruction):
    """Receive activations from the previous stage in the pipeline.
    """
    pass


class SendGrad(BufferOpInstruction):
    """Send computed gradients to the previous pipeline stage.
    with respect to the received activations
    """
    pass


class RecvGrad(BufferOpInstruction):
    """Receive computed gradients the next pipeline stage.
    """
    pass


def _is_even(x):
    return x % 2 == 0


def _is_odd(x):
    return x % 2 != 0
