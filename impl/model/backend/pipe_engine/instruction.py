from enum import IntEnum
from typing import List

from deepspeed.runtime.utils import call_to_str


# schedule executor
class PipeInstruction:
    """Base class for all instructions to be executed by the pipeline engine.

    All keyword arguments are stored as members similar to a ``namedtuple``. These are
    then accessible to the :class:`PipeEngine` during execution.

    Each instruction has following attributes: 
    1. stage_id: pipeline stage that the instruction should be executed by.
    2. micro_batch_id: the ID of data micro batch that this instruction should be executed on.
    3. step_id: usually used by generation schedules, identical to generation token id.
    4. priority: priority of the instruction, higher priority instructions should be scheduled first in 
                 the dynamic schedule.
    5. deps: list of instructions that this instruction depends on.

    Note:: The dependency between instructions should follows rules for simplicity:
    Instructions are classified into two types: 1. compute instructions, 2. communication instructions.
    For compute instructions, their dependency should only be other instructions that has the same 
    stage_id. For communication instructions, they should appear in pairs with their communication target stage.
    For example, SendActivation(stage_id=0, micro_batch_id=M) should appear in pairs with 
    RecvActivation(stage_id=1, micro_batch_id=M). They should have the same dependency. 
    
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


# Compute
class ForwardPass(PipeInstruction):
    """Compute a forward pass.
    """
    pass


class BackwardPass(PipeInstruction):
    """Compute a backward pass and accumulate gradients.
    """
    pass


# Communication
class SendActivation(PipeInstruction):
    """Send activations to the next stage in the pipeline.
    """
    pass


class RecvActivation(PipeInstruction):
    """Receive activations from the previous stage in the pipeline.
    """
    pass


class SendGrad(PipeInstruction):
    """Send computed gradients to the previous pipeline stage.
    with respect to the received activations
    """
    pass


class RecvGrad(PipeInstruction):
    """Receive computed gradients the next pipeline stage.
    """
    pass


# generation
class SendNextTokens(PipeInstruction):
    """ In GenerateSchedule, send next tokens to the first stage. Only available in the last stage.
    """
    pass


class RecvNextTokens(PipeInstruction):
    """ In GenerateSchedule, recv next tokens from the last stage. Only available in the first stage.
    """
    pass


class InstructionType(IntEnum):
    FORWARD_PASS = 1
    BACKWARD_PASS = 2
    SEND_ACTIVATION = 3
    RECV_ACTIVATION = 4
    SEND_GRAD = 5
    RECV_GRAD = 6
    REDUCE_GRADS = 7
    OPTIMIZER_STEP = 8
    SEND_NEXT_TOKENS = 9
    RECV_NEXT_TOKENS = 10


INSTRUCTION_TYPE_TO_CLASS = {
    InstructionType.FORWARD_PASS: ForwardPass,
    InstructionType.BACKWARD_PASS: BackwardPass,
    InstructionType.SEND_ACTIVATION: SendActivation,
    InstructionType.RECV_ACTIVATION: RecvActivation,
    InstructionType.SEND_GRAD: SendGrad,
    InstructionType.RECV_GRAD: RecvGrad,
    InstructionType.REDUCE_GRADS: ReduceGrads,
    InstructionType.OPTIMIZER_STEP: OptimizerStep,
    InstructionType.SEND_NEXT_TOKENS: SendNextTokens,
    InstructionType.RECV_NEXT_TOKENS: RecvNextTokens,
}

INSTRUCTION_CLASS_TO_TYPE = {v: k for k, v in INSTRUCTION_TYPE_TO_CLASS.items()}


def instruction_to_type(instruction: PipeInstruction) -> InstructionType:
    return INSTRUCTION_CLASS_TO_TYPE[type(instruction)]


def type_to_instruction(instruction_type: InstructionType) -> PipeInstruction:
    return INSTRUCTION_TYPE_TO_CLASS[instruction_type]
