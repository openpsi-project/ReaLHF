from collections import defaultdict
from enum import IntEnum
from typing import List, Optional, Set, Union
import sys

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
    6. bind: Instruction that this instruction is binded with. Binded instructions are send/recv instructions 
             that should appear in pair, belong to different (adjancent) stages and have the same micro_batch_id.
             Two binded instructions should have the same dependency list. This feature is used to avoid NCCL
             communication deadlock.
    
    """

    def __init__(self,
                 stage_id: int,
                 micro_batch_id: int,
                 deps: List['PipeInstruction'] = [],
                 bind: List['PipeInstruction'] = [],
                 step_id: int = 0):
        self.stage_id = stage_id
        self.micro_batch_id = micro_batch_id
        self.deps = deps
        self.bind = bind
        self.name = self.__class__.__name__
        self.step_id = step_id
        self.args = (stage_id, micro_batch_id, step_id)

    def __repr__(self):
        return f"{self.name}{self.args}"
        # return call_to_str(self.name, self.args, self.kwargs)

    def __eq__(self, other: 'PipeInstruction'):
        return self.stage_id == other.stage_id and \
               self.micro_batch_id == other.micro_batch_id and \
               self.step_id == other.step_id and \
               self.name == other.name

    def __lt__(self, other: 'PipeInstruction'):
        # order by stage_id, micro_batch_id, step_id
        # used to sort finded results in InstructionSet
        return self.stage_id < other.stage_id or \
               (self.stage_id == other.stage_id and self.micro_batch_id < other.micro_batch_id) or \
               (self.stage_id == other.stage_id and self.micro_batch_id == other.micro_batch_id and self.step_id < other.step_id)

    def encode_str(self):
        return f"{self.name};{self.stage_id};{self.micro_batch_id};{self.step_id}"

    def encode(self):
        return self.encode_str().encode(encoding="utf-8")

    @classmethod
    def decode(cls, encoded: Union[bytes, str]) -> 'PipeInstruction':
        if isinstance(encoded, bytes):
            s = encoded.decode(encoding="utf-8")
        else:
            s = encoded
        cls_name, stage_id, micro_batch_id, step_id = s.split(";")
        cls_ = getattr(sys.modules[__name__], cls_name)
        return cls_(stage_id=int(stage_id), micro_batch_id=int(micro_batch_id), step_id=int(step_id))


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


class EndSchedule(PipeInstruction):
    """ Force terminate schedule.
    """


class InstructionSet:
    """ A set of instructions that can be indexed by stage_id, micro_batch_id and step_id.
    Instructions are stored in their string representation, dependency of instructions are stored 
    separately, update only when new instruction has a different dependency with length > 0.
    """

    def __init__(self, max_size=None):
        # TODO: check performance, if slow or memory consuming
        # use a large dict that record number of steps instead.
        self.__storage = dict()
        self.__stage_sets = defaultdict(set)
        self.__mbid_sets = defaultdict(set)
        self.__name_sets = defaultdict(set)
        self.__step_sets = defaultdict(set)
        # self.__deps_storage = dict()
        # self.__bind_storage = dict()
        self.__size = 0
        self.__stage_sizes = defaultdict(int)
        self.__mbid_sizes = defaultdict(int)
        self.__name_sizes = defaultdict(int)
        self.__step_sizes = defaultdict(int)
        self.max_size = max_size  # maximum number of instructions in the set

    def add(self, inst: Union[List[PipeInstruction], PipeInstruction]):
        """ Add one or multiple instructions to the set.
        """
        if isinstance(inst, list):
            [self.add(x) for x in inst]
        else:
            if self.max_size is not None:
                if self.size >= self.max_size:
                    raise RuntimeError(
                        f"InstructionSet size {self.size} exceeds maximum size {self.max_size}.")
            s = inst.encode_str()
            # if len(inst.deps) > 0:
            #     self.__deps_storage[s] = inst.deps
            # if inst.bind is not None:
            #     self.__bind_storage[s] = inst.bind
            self.__storage[s] = inst

            self.__stage_sets[inst.stage_id].add(s)
            self.__mbid_sets[inst.micro_batch_id].add(s)
            self.__name_sets[inst.name].add(s)
            self.__step_sets[inst.step_id].add(s)
            self.__stage_sizes[inst.stage_id] += 1
            self.__mbid_sizes[inst.micro_batch_id] += 1
            self.__name_sizes[inst.name] += 1
            self.__step_sizes[inst.step_id] += 1

            self.__size += 1

    def contain(self, inst: PipeInstruction):
        """ Check if the set contains the instruction.
        """
        s = inst.encode_str()
        return s in self.__stage_sets[inst.stage_id]

    def remove(self, inst: Union[List[PipeInstruction], PipeInstruction]):
        """ Remove one or multiple instructions from the set.
        """
        if isinstance(inst, list):
            [self.remove(x) for x in inst]
        else:
            s = inst.encode_str()
            if self.contain(inst):
                self.__stage_sets[inst.stage_id].remove(s)
                self.__mbid_sets[inst.micro_batch_id].remove(s)
                self.__name_sets[inst.name].remove(s)
                self.__step_sets[inst.step_id].remove(s)
                # if s in self.__deps_storage:
                #     self.__deps_storage.pop(s)
                # if s in self.__bind_storage:
                #     self.__bind_storage.pop(s)
                self.__storage.pop(s)
                self.__size -= 1
                self.__stage_sizes[inst.stage_id] -= 1
                self.__mbid_sizes[inst.micro_batch_id] -= 1
                self.__name_sizes[inst.name] -= 1
                self.__step_sizes[inst.step_id] -= 1
            else:
                raise KeyError(f"Instruction {inst} not in set.")

    def find(self,
             stage_id: Optional[int] = None,
             micro_batch_id: Optional[int] = None,
             name: Optional[str] = None,
             step_id: Optional[int] = None,
             unmutable_result: bool = False) -> List[PipeInstruction]:
        """ Find all instructions in set that satisfies arguments as a list.
        """
        related_sets: List[Set] = []
        if stage_id is not None:
            related_sets.append(self.__stage_sets[stage_id])
        if micro_batch_id is not None:
            related_sets.append(self.__mbid_sets[micro_batch_id])
        if name is not None:
            related_sets.append(self.__name_sets[name])
        if step_id is not None:
            related_sets.append(self.__step_sets[step_id])

        if len(related_sets) > 0:
            res_set = related_sets[0].intersection(*related_sets)
            return [self.__storage[r] for r in res_set]
        if unmutable_result:
            return self.__storage.values()
        else:
            return list(self.__storage.values())

    def __len__(self):
        return self.__size

    def size(self,
             stage_id: Optional[int] = None,
             micro_batch_id: Optional[int] = None,
             name: Optional[str] = None,
             step_id: Optional[int] = None):
        assert sum([x is not None for x in [stage_id, micro_batch_id, name, step_id]]) <= 1, \
               "length method for instruction set can only take one constraint argument."
        if stage_id:
            return self.__stage_sizes[stage_id]
        if micro_batch_id:
            return self.__mbid_sizes[micro_batch_id]
        if name:
            return self.__name_sizes[name]
        if step_id:
            return self.__step_sizes[step_id]
        return self.__size
