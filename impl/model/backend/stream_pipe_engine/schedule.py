from abc import ABC, abstractmethod
from collections import defaultdict
from enum import IntEnum
import copy

from impl.model.backend.stream_pipe_engine.instruction import *


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


class InferenceSchedule(PipeSchedule):
    """A schedule for inferencing batches using pipeline parallelism.
    """

    def steps(self):
        """"""
        # TODO: add store activation option
        pass


class ScheduleType(IntEnum):
    TRAIN_1F1B = 0
    INFERENCE = 2


SCHEDULE_TYPE_TO_CLS = {
    ScheduleType.TRAIN_1F1B: Train1F1BSchedule,
    ScheduleType.INFERENCE: InferenceSchedule,
}

SCHEDULE_CLS_TO_TYPE = {v: k for k, v in SCHEDULE_TYPE_TO_CLS.items()}


def type_to_schedule(schedule_type: ScheduleType) -> PipeSchedule:
    return SCHEDULE_TYPE_TO_CLS[schedule_type]


def schedule_to_type(schedule_cls: PipeSchedule) -> ScheduleType:
    return SCHEDULE_CLS_TO_TYPE[schedule_cls]
