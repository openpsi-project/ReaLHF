from enum import IntEnum
from typing import List, Optional
import dataclasses
import struct

from impl.model.backend.pipe_engine.dynamic_schedule import *
from impl.model.backend.pipe_engine.instruction import *

instuction_int_to_str = {
    0: "OPTIMIZER_STEP",
    1: "REDUCE_GRADS",
    2: "FORWARD_PASS",
    3: "BACKWARD_PASS",
    4: "SEND_ACTIVATION",
    5: "RECV_ACTIVATION",
    6: "SEND_GRAD",
    7: "RECV_GRAD",
    8: "SEND_NEXT_TOKENS",
    9: "RECV_NEXT_TOKENS",
    10: "END_SCHEDULE"
}


class MessageType(IntEnum):
    ISSUE_SCHEUDLE = 0
    TERMINATE = 1
    POST_INSTRUCTION = 2
    POLL_RESULT = 3
    NOOP = 4


class InstructionType(IntEnum):
    OPTIMIZER_STEP = 0
    REDUCE_GRADS = 1
    FORWARD_PASS = 2
    BACKWARD_PASS = 3
    SEND_ACTIVATION = 4
    RECV_ACTIVATION = 5
    SEND_GRAD = 6
    RECV_GRAD = 7
    SEND_NEXT_TOKENS = 8
    RECV_NEXT_TOKENS = 9
    END_SCHEDULE = 10


class ScheduleType(IntEnum):
    TRAINING = 0
    GENERATE = 1


class SignalCode(IntEnum):
    EXEC = 0
    END = 1
    HOLD = 2
    COMM_EXEC = 3


instruction_type_to_enum = {
    OptimizerStep: InstructionType.OPTIMIZER_STEP,
    ReduceGrads: InstructionType.REDUCE_GRADS,
    ForwardPass: InstructionType.FORWARD_PASS,
    BackwardPass: InstructionType.BACKWARD_PASS,
    SendActivation: InstructionType.SEND_ACTIVATION,
    RecvActivation: InstructionType.RECV_ACTIVATION,
    SendGrad: InstructionType.SEND_GRAD,
    RecvGrad: InstructionType.RECV_GRAD,
    SendNextTokens: InstructionType.SEND_NEXT_TOKENS,
    RecvNextTokens: InstructionType.RECV_NEXT_TOKENS,
    EndSchedule: InstructionType.END_SCHEDULE
}

enum_to_instruction_type = {v: k for k, v in instruction_type_to_enum.items()}

schedule_type_to_enum = {Train1F1BSchedule: ScheduleType.TRAINING, GenerationSchedule: ScheduleType.GENERATE}


@dataclasses.dataclass
class Instruction:
    type_: InstructionType
    stage_id: int
    micro_batch_id: int
    step_id: int
    schedule_id: int

    def is_comm(self):
        return self.type_ in [
            InstructionType.SEND_ACTIVATION, InstructionType.RECV_ACTIVATION, InstructionType.SEND_GRAD,
            InstructionType.RECV_GRAD, InstructionType.SEND_NEXT_TOKENS, InstructionType.RECV_NEXT_TOKENS,
            InstructionType.OPTIMIZER_STEP
        ]

    def __str__(self):
        return f"Instruction({instuction_int_to_str[self.type_]};{self.stage_id};" \
               f"{self.micro_batch_id};{self.step_id};{self.schedule_id})"

    def serialize(self):
        return struct.pack(">iiiii", self.type_, self.stage_id, self.micro_batch_id, self.step_id,
                           self.schedule_id)

    @staticmethod
    def serialize_none():
        return struct.pack(">iiiii", -1, 0, 0, 0, 0)

    @staticmethod
    def deserialize(data):
        args = struct.unpack(">iiiii", data)
        return Instruction(*args)


@dataclasses.dataclass
class Schedule:
    type_: ScheduleType
    num_micro_batches: int
    num_stages: int
    num_steps: int
    schedule_id: int

    def __str__(self):
        return f"Schedule({self.type_};{self.num_micro_batches};" \
               f"{self.num_stages};{self.num_steps};{self.schedule_id})"

    def serialize(self):
        return struct.pack(">iiiii", self.type_, self.num_micro_batches, self.num_stages, self.num_steps,
                           self.schedule_id)

    @staticmethod
    def serialize_none():
        return struct.pack(">iiiii", -1, 0, 0, 0, 0)

    @staticmethod
    def deserialize(data):
        args = struct.unpack(">iiiii", data)
        return Schedule(*args)


@dataclasses.dataclass
class Message:
    stage_id: int
    signal_code: SignalCode
    message_type: MessageType
    instruction: Instruction
    schedule: Schedule

    def __str__(self):
        return f"Message({self.stage_id};{self.signal_code};" \
               f"{self.message_type};{self.instruction};" \
               f"{self.schedule})"

    def serialize(self):
        inst_bytes = self.instruction.serialize() if self.instruction \
                     else Instruction.serialize_none()
        schedule_bytes = self.schedule.serialize() if self.schedule \
                         else Schedule.serialize_none()
        return struct.pack(">iii", self.stage_id, self.signal_code, self.message_type) + \
               inst_bytes + schedule_bytes

    @staticmethod
    def deserialize(data):
        stage_id, signal_code, message_type = struct.unpack(">iii", data[:12])
        inst_type = data[12:16]
        inst_bytes = data[12:32]
        if struct.unpack(">i", inst_type)[0] != -1:
            instruction = Instruction.deserialize(inst_bytes)
        else:
            instruction = None

        schedule_type = data[32:36]
        schedule_bytes = data[32:52]
        if struct.unpack(">i", schedule_type)[0] != -1:
            schedule = Schedule.deserialize(schedule_bytes)
        else:
            schedule = None

        return Message(stage_id, signal_code, message_type, instruction, schedule)


@dataclasses.dataclass
class MessageArray:
    n: int
    messages: List[Message]

    def serialize(self):
        return struct.pack(">i", self.n) + b"".join([m.serialize() for m in self.messages])

    @staticmethod
    def deserialize(data):
        n = struct.unpack(">i", data[:4])[0]
        messages = []
        for i in range(n):
            m = Message.deserialize(data[i * 52 + 4:i * 52 + 56])
            messages.append(m)
        return MessageArray(n, messages)


def schedule_to_message(s: DynamicPipeSchedule):
    try:
        type_ = schedule_type_to_enum[type(s)]
    except KeyError:
        raise NotImplementedError("Unknown schedule type for FastScheduleController")
    return Schedule(type_, s.num_micro_batches, s.num_stages, s.num_steps, s.schedule_id)


def instruction_to_message(i: Instruction):
    try:
        type_ = instruction_type_to_enum[type(i)]
    except KeyError:
        raise NotImplementedError("Unknown instruction type for FastScheduleController")
    return Instruction(type_, i.stage_id, i.micro_batch_id, i.step_id, i.schedule_id)


def message_to_instruction(i: Instruction):
    try:
        type_cls = enum_to_instruction_type[i.type_]
    except KeyError:
        raise NotImplementedError("Unknown instruction type for FastScheduleController")
    return type_cls(i.stage_id, i.micro_batch_id, step_id=i.step_id, schedule_id=i.schedule_id)
