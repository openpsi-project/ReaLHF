from enum import IntEnum
from typing import List

from deepspeed.runtime.utils import call_to_str


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


# IO
class LoadMicroBatch(PipeInstruction):
    """Load a micro-batch into a buffer.
    """
    pass


# Compute
class ForwardPass(PipeInstruction):
    """Compute a forward pass.

    Roughly:

    .. code-block:: python

        buffers['outputs'][buffer_id] = forward(buffers['inputs'][buffer_id])
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


class InstructionType(IntEnum):
    LOAD_MICRO_BATCH = 0
    FORWARD_PASS = 1
    BACKWARD_PASS = 2
    SEND_ACTIVATION = 3
    RECV_ACTIVATION = 4
    SEND_GRAD = 5
    RECV_GRAD = 6
    REDUCE_GRADS = 7
    OPTIMIZER_STEP = 8


INSTRUCTION_TYPE_TO_CLASS = {
    InstructionType.LOAD_MICRO_BATCH: LoadMicroBatch,
    InstructionType.FORWARD_PASS: ForwardPass,
    InstructionType.BACKWARD_PASS: BackwardPass,
    InstructionType.SEND_ACTIVATION: SendActivation,
    InstructionType.RECV_ACTIVATION: RecvActivation,
    InstructionType.SEND_GRAD: SendGrad,
    InstructionType.RECV_GRAD: RecvGrad,
    InstructionType.REDUCE_GRADS: ReduceGrads,
    InstructionType.OPTIMIZER_STEP: OptimizerStep,
}

INSTRUCTION_CLASS_TO_TYPE = {v: k for k, v in INSTRUCTION_TYPE_TO_CLASS.items()}


def instruction_to_type(instruction: PipeInstruction) -> InstructionType:
    return INSTRUCTION_CLASS_TO_TYPE[type(instruction)]


def type_to_instruction(instruction_type: InstructionType) -> PipeInstruction:
    return INSTRUCTION_TYPE_TO_CLASS[instruction_type]
