from abc import ABC, abstractmethod
from collections import defaultdict
from enum import IntEnum
from typing import Dict, Tuple
import copy
import time

from impl.model.backend.pipe_engine.instruction import *

check_count = 0
check_times = 0


class DynamicPipeSchedule(ABC):
    """ Define an execution sequence of instructions of a stream pipeline engine 
    in the centralized EngineScheduleController.

    Each dynamic schedule should have its own state (tensors, other information ... )
    storage.

    Args:
        num_micro_batches (int): Number of micro-batches to split the batch into
        num_stages (int): Number of pipeline stages
        priority (int): Priority of the schedule, higher priority schedules are
                        scheduled first in the centralized controller.
    """

    def __init__(self,
                 num_micro_batches: int,
                 num_stages: int,
                 num_steps: Optional[int] = 0,
                 sched_id: int = 0):
        super().__init__()
        self.num_micro_batches = num_micro_batches
        self.num_stages = num_stages
        self.num_steps = num_steps
        # self.__all_instructions: InstructionSet = self.init_instructions()

        self.__not_ready: InstructionSet = InstructionSet()
        self.__executed: InstructionSet = InstructionSet()
        self.__inflight: InstructionSet = InstructionSet()
        self.__ready: InstructionSet = InstructionSet()

        self.__initialized = False
        self.__terminated = False  # terminate the schedule, used to avoid duplicated terminate
        self.__end_schedule_sent = False
        self.__sched_id = sched_id

    def __init_inst_set(self):
        # print("init inst set")
        self.__not_ready.add(self.init_instructions())

        # avoid execute instruction after stage executed EndSchedule
        self.__update_ready()

    @abstractmethod
    def init_instructions(self) -> List[PipeInstruction]:
        """Initialize all instructions to be executed in the schedule.
        
        This method should be implemented in every subclass of DynamicPipeSchedule.
        
        """
        raise NotImplementedError()

    def update_instructions(self) -> List[PipeInstruction]:
        """Update instructions.
        
        This method should be called when some stage has already executed all instructions 
        in the schedule and nothing to send.
        """
        return []

    def __update_ready(self):
        """ Update ready instructions from not_ready to ready
        """
        for inst in self.__not_ready.find():
            if self._is_update_ready(inst):
                self.__not_ready.remove(inst)
                self.__ready.add(inst)

    def exec(self, insts: List[PipeInstruction]):
        """Called by controller to execute a set of instructions.
        Steps: 
        1. Move executed instructions from ready to executed. 
        2. Update, move ready instructions from not_ready to ready.
        """
        if self.__terminated:
            raise RuntimeError("Cannot execute an already terminated schedule.")
        for inst in insts:
            if inst.name != "EndSchedule":
                self.__inflight.remove(inst)
                self.__executed.add(inst)
        self.__update_ready()

    def all_executed(self):
        return len(self.__ready) + len(self.__not_ready) + len(self.__inflight) == 0

    def ready(self) -> Dict[int, List[PipeInstruction]]:
        """ Called by controller, return a dict that contains mapping from stage_id to 
        ready instructions and enqueue all of them 
        """
        if self.__terminated:
            raise RuntimeError("Cannot get ready instructions from terminated schedule.")
        if not self.__initialized:
            self.__init_inst_set()
            self.__initialized = True

        r = dict()
        if self.all_executed() and not self.__end_schedule_sent:
            for i in range(self.num_stages):
                inst = EndSchedule(stage_id=i, micro_batch_id=0)
                r[i] = [inst]
            self.__end_schedule_sent = True
            return r

        for i in range(self.num_stages):
            r[i] = self.__ready.find(stage_id=i)
            for inst in r[i]:
                self.__ready.remove(inst)
                self.__inflight.add(inst)
        return r

    # def enqueue(self, insts: List[PipeInstruction]):
    #     """ mark instructions inflight
    #     """
    #     for inst in insts:
    #         self.__ready.remove(inst)
    #         self.__inflight.add(inst)

    def update(self, stage_id):
        """ Called by controller to find new instructions for some stage_id
        """
        if self.__terminated:
            raise RuntimeError("Cannot update an already terminated schedule.")
        not_ready = self.__not_ready.find(stage_id=stage_id)
        in_flight = self.__inflight.find(stage_id=stage_id)
        if len(not_ready) == 0 and len(in_flight) == 0:
            new_instructions = self.update_instructions()
            if len(new_instructions) > 0:
                self.__not_ready.add(new_instructions)
                self.__update_ready()
                return True
        return False

    # def post_one_ready(self, stage_id: Optional[int]) -> Tuple[PipeInstruction, bool]:
    #     """Called by controller to retrieve one ready instruction.
    #     If there is one ready instruction for stage id, move it from ready to inflight set
    #     and return this instruction.
    #     Otherwise, if there is no instruction ready for this stage or the schedule is terminated,
    #     return None
    #     """
    #     if not self.__initialized:
    #         self.__init_inst_set()
    #         self.__initialized = True
    #     if self.__stage_terminated[stage_id]:
    #         return None, False
    #         # raise RuntimeError("Cannot post ready instruction from an already terminated schedule.")

    #     # nothing to do for this schedule, terminate
    #     if len(self.__ready) + len(self.__not_ready) + len(self.__inflight) == 0 \
    #         and not self.__terminated:
    #         self.terminate()

    #     # check binded instructions
    #     bind_insts = self.__bind_insts[stage_id]
    #     if len(bind_insts) > 0:
    #         print(f"binded insts for stage {stage_id}: {bind_insts}")
    #         # there is a binded instruction queued for this stage
    #         inst = bind_insts.pop(0)
    #         if self.__ready.contain(inst):
    #             # assert self.__executed.contain(inst) or self.__inflight.contain(inst)
    #             self.__ready.remove(inst)
    #             self.__inflight.add(inst)
    #             print(f"Rank {stage_id} sched_id {self.__sched_id}: binded execute inst {inst}, left binded {bind_insts}")
    #             return inst, False
    #         else:
    #             if inst.name == "EndSchedule":
    #                 # only end schedule with endschedule instruction
    #                 self.__inflight.add(inst)
    #                 return inst, True
    #             elif not (self.__executed.contain(inst) or self.__inflight.contain(inst)):
    #                 raise RuntimeError(
    #                     f"Binded instruction {inst} for stage {stage_id} is not ready or does not exist.")
    #             else:
    #                 print(f"Rank {stage_id} sched_id {self.__sched_id}: binded instruction {inst} is already executed or in flight, "
    #                       f"left binded {bind_insts}")

    #     # no binded instruction, find another ready instruction
    #     insts = self.__ready.find(stage_id=stage_id)
    #     if len(insts) > 0:
    #         inst = insts[0]
    #         self.__ready.remove(inst)
    #         self.__inflight.add(inst)
    #         if inst.bind:
    #             other_stage_id = inst.bind.stage_id
    #             self.__bind_insts[other_stage_id].append(inst.bind)
    #             print(f"Rank {stage_id} sched_id {self.__sched_id}: bind inst {inst} + {inst.bind} for stage {other_stage_id}")
    #         # end = False
    #         print(f"Rank {stage_id} sched_id {self.__sched_id}: normal execute inst {inst}, binded list {self.__bind_insts[stage_id]}")
    #         return inst, False
    #     else:
    #         not_ready = self.__not_ready.find(stage_id=stage_id)
    #         in_flight = self.__inflight.find(stage_id=stage_id)
    #         if len(not_ready) == 0 and len(in_flight) == 0:
    #             new_instructions = self.update_instructions()
    #             if len(new_instructions) > 0:
    #                 self.__not_ready.add(new_instructions)
    #                 self.__update_ready()
    #                 return self.post_one_ready(stage_id)
    #     return None, False

    def _is_update_ready(self, inst: PipeInstruction):
        """ check if an instruction is ready but not put into ready set
        """
        global check_times, check_count
        assert not self.__ready.contain(inst) and \
               not self.__executed.contain(inst) and \
               not self.__inflight.contain(inst) and \
               self.__not_ready.contain(inst)

        update_ready = all([self.__executed.contain(dep) for dep in inst.deps])

        # print(f"inst {inst} deps {inst.deps} {update_ready}")
        # if not update_ready:
        #     print(f"executed {self.__executed.find()}")
        return update_ready

    def terminate(self):
        """ Called by controller to terminate the schedule, force execute end schedule instruction for all stages
        Move all instructions from ready to executed.
        """
        if self.__terminated:
            raise RuntimeError("Cannot terminate an already terminated schedule.")

        self.__terminated = True
        # for i in range(self.num_stages):
        #     self.__bind_insts[i].append(EndSchedule(stage_id=i, micro_batch_id=0))
        #     self.__terminated = True


class InferenceSchedule(DynamicPipeSchedule):
    """Schedule for inference a batch.
    
    Instruction types: 
    1. ForwardPass; 2. SendActivation; 3.RecvActivation;
    
    Dependency:
    1. ForwardPass(stage_id=S, micro_batch_id=M, step_id=0):
        (1) if M > 0, S = 0, to keep forward order:
            SendActivation(stage_id=0, micro_batch_id=M-1)
        (2) if S > 0:
            RecvActivation(stage_id=S, micro_batch_id=M)

    2. SendActivation(stage_id=S, micro_batch_id=M, step_id=0):
        (S < num_stages-1)
        (1) ForwardPass(stage_id=S, micro_batch_id=M, step_id=0)
        binded with RecvActivation(stage_id=S+1, micro_batch_id=M)
    
    3. RecvActivation(stage_id=S, micro_batch_id=M, step_id=0):
        (S > 0)
        (1) ForwardPass(stage_id=S-1, micro_batch_id=M, step_id=0)
        binded with SendActivation(stage_id=S-1, micro_batch_id=M)
    """

    def __post_init__(self):
        assert self.num_micro_batches >= self.num_stages, \
               "num_micro_batches must be >= num_stages for optimized performance."

    def init_instructions(self) -> List[PipeInstruction]:
        insts = []
        for s in range(self.num_stages):
            for m in range(self.num_micro_batches):
                fwd_deps = []
                if m > 0 and s == 0:
                    fwd_deps.append(SendActivation(stage_id=0, micro_batch_id=m - 1))
                if s > 0:
                    fwd_deps.append(RecvActivation(stage_id=s, micro_batch_id=m))
                insts.append(ForwardPass(stage_id=s, micro_batch_id=m, deps=fwd_deps))
                if s < self.num_stages - 1:
                    snd_deps = [ForwardPass(stage_id=s, micro_batch_id=m)]
                    snd_bind = [RecvActivation(stage_id=s + 1, micro_batch_id=m)]
                    # if m > 0:
                    #     snd_deps.append(SendActivation(stage_id=s, micro_batch_id=m - 1))
                    insts.append(SendActivation(stage_id=s, micro_batch_id=m, deps=snd_deps, bind=snd_bind))
                if s > 0:
                    rcv_deps = [ForwardPass(stage_id=s - 1, micro_batch_id=m)]
                    rcv_bind = [SendActivation(stage_id=s - 1, micro_batch_id=m)]
                    # if m > 0:
                    #     rcv_deps.append(RecvActivation(stage_id=s, micro_batch_id=m - 1))
                    insts.append(RecvActivation(stage_id=s, micro_batch_id=m, deps=rcv_deps, bind=rcv_bind))
        return insts


class GenerationSchedule(DynamicPipeSchedule):
    """Schedule for inference a batch.
    
    Instruction types: 
    1. ForwardPass; 2. SendActivation; 3.RecvActivation;
    4. SendNextTokens; 5.ReceiveNextTokens

    Dependency:
    1. ForwardPass(stage_id=S, micro_batch_id=M, step_id=T):
        (1) if M > 0, S = 0, to keep forward order:
            SendActivation(stage_id=0, micro_batch_id=M-1, step_id=T)
        (2) if S = 0, T > 0:
            RecvNextTokens(stage_id=0, micro_batch_id=M, step_id=T-1)
        (3) if S > 0:
            RecvActivation(stage_id=S, micro_batch_id=M, step_id=T)

    2. SendActivation(stage_id=S, micro_batch_id=M, step_id=T):
        (S < num_stages-1)
        (1) ForwardPass(stage_id=S, micro_batch_id=M, step_id=T)
        binded with RecvActivation(stage_id=S+1, micro_batch_id=M, step_id=T)

    3. RecvActivation(stage_id=S, micro_batch_id=M, step_id=T):
        (S > 0)
        (1) ForwardPass(stage_id=S-1, micro_batch_id=M, step_id=T)
        binded with SendActivation(stage_id=S-1, micro_batch_id=M, step_id=T)

    4. SendNextTokens(stage_id=S, micro_batch_id=M, step_id=T): 
        (S = num_stages - 1)
        (1) ForwardPass(stage_id=num_stages - 1, micro_batch_id=M, step_id=T)
        binded with RecvNextTokens(stage_id=0, micro_batch_id=M, step_id=T)
    
    5. RecvNextTokens(stage_id=S, micro_batch_id=M, step_id=T):
        (S = 0)
        (1) ForwardPass(stage_id=num_stages - 1, micro_batch_id=M, step_id=T)
        binded with SendNextTokens(stage_id=num_stages - 1, micro_batch_id=M, step_id=T)
    """

    def __init__(self, steps_per_update=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps_per_update = steps_per_update
        self.remaining_steps = self.num_steps

    def __post_init__(self):
        assert self.num_micro_batches >= self.num_stages, \
               "num_micro_batches must be >= num_stages for optimized performance."

    def init_instructions(self) -> List[PipeInstruction]:
        r = self.__n_step_instructions(self.steps_per_update)
        return r

    def update_instructions(self) -> List[PipeInstruction]:
        if self.remaining_steps == 0:
            return []
        else:
            r = self.__n_step_instructions(self.steps_per_update)
            return r

    def __n_step_instructions(self, n):
        insts = []
        import itertools
        n_t = min(n, self.remaining_steps)
        st = self.num_steps - self.remaining_steps
        et = st + n_t
        for m, s, t in itertools.product(range(self.num_micro_batches), range(self.num_stages), range(st,
                                                                                                      et)):
            fwd_deps = []
            if m > 0 and s == 0:
                fwd_deps.append(SendActivation(stage_id=0, micro_batch_id=m - 1, step_id=t))
            if s == 0 and t > 0:
                fwd_deps.append(RecvNextTokens(stage_id=0, micro_batch_id=m, step_id=t - 1))
            if s > 0:
                fwd_deps.append(RecvActivation(stage_id=s, micro_batch_id=m, step_id=t))
            insts.append(ForwardPass(stage_id=s, micro_batch_id=m, step_id=t, deps=fwd_deps))

            if s < self.num_stages - 1:
                snd_deps = [ForwardPass(stage_id=s, micro_batch_id=m, step_id=t)]
                snd_bind = [RecvActivation(stage_id=s + 1, micro_batch_id=m, step_id=t)]
                insts.append(
                    SendActivation(stage_id=s, micro_batch_id=m, step_id=t, deps=snd_deps, bind=snd_bind))

            if s > 0:
                rcv_deps = [ForwardPass(stage_id=s - 1, micro_batch_id=m, step_id=t)]
                rcv_bind = [SendActivation(stage_id=s - 1, micro_batch_id=m, step_id=t)]
                insts.append(
                    RecvActivation(stage_id=s, micro_batch_id=m, step_id=t, deps=rcv_deps, bind=rcv_bind))

            if s == self.num_stages - 1:
                snd_deps = [ForwardPass(stage_id=s, micro_batch_id=m, step_id=t)]
                snd_bind = [RecvNextTokens(stage_id=0, micro_batch_id=m, step_id=t)]
                insts.append(
                    SendNextTokens(stage_id=self.num_stages - 1,
                                   micro_batch_id=m,
                                   step_id=t,
                                   deps=snd_deps,
                                   bind=snd_bind))

            if s == 0:
                rcv_deps = [ForwardPass(stage_id=self.num_stages - 1, micro_batch_id=m, step_id=t)]
                rcv_bind = [SendNextTokens(stage_id=self.num_stages - 1, micro_batch_id=m, step_id=t)]
                insts.append(
                    RecvNextTokens(stage_id=0, micro_batch_id=m, step_id=t, deps=rcv_deps, bind=rcv_bind))

        self.remaining_steps -= n_t
        return insts


class Train1F1BSchedule(DynamicPipeSchedule):
    """Schedule for training a batch in 1F1B pipeline parallelism style.

    Instruction types: 
    1. ForwardPass; 2. SendActivation; 3.RecvActivation;
    4. BackwardPass; 5. SendGrad; 6. RecvGrad;
    7. ReduceGrad; 8. OptimizerStep;
    
    Dependency:
    1. ForwardPass(stage_id=S, micro_batch_id=M, step_id=0):
        (1) if M > 0, S = 0, to keep forward order:
            SendActivation(stage_id=0, micro_batch_id=M-1)
        (2) if S > 0:
            RecvActivation(stage_id=S, micro_batch_id=M)
        (3) if M >= num_stages, in 1F1B only "num_stages" micro batches is inflight:
            BackwardPass(stage_id=S, micro_batch_id=M-num_stages)
    
    2. SendActivation(stage_id=S, micro_batch_id=M, step_id=0):
        (S < num_stages-1)
        (1) ForwardPass(stage_id=S, micro_batch_id=M, step_id=0)

    3. RecvActivation(stage_id=S, micro_batch_id=M, step_id=0):
        (S > 0)
        (1) ForwardPass(stage_id=S-1, micro_batch_id=M, step_id=0)
    
    4. BackwardPass(stage_id=S, micro_batch_id=M, step_id=0):
        (1) if S == num_stages - 1:
            ForwardPass(stage_id=S, micro_batch_id=M, step_id=0)
        (2) if S < num_stages - 1:
            RecvGrad(stage_id=S, micro_batch_id=M)

    5. SendGrad(stage_id=S, micro_batch_id=M, step_id=0):
        (S > 0)
        (1) BackwardPass(stage_id=S, micro_batch_id=M, step_id=0)
    
    6. RecvGrad(stage_id=S, micro_batch_id=M, step_id=0):
        (S < num_stages - 1)
        (1) BackwardPass(stage_id=S+1, micro_batch_id=M, step_id=0)

    7. ReduceGrads(stage_id=S, micro_batch_id=0, step_id=0):
        (1) BackwardPass(stage_id=S, micro_batch_id=num_micro_batch-1, step_id=0)
    
    8. OptimizerStep(stage_id=S, micro_batch_id=0, step_id=0):
        (1) ReduceGrad(stage_id=S, micro_batch_id=0, step_id=0)    
    """

    def __post_init__(self):
        assert self.num_micro_batches >= self.num_stages, \
               "num_micro_batches must be >= num_stages for optimized performance."

    def _is_first_half(self, micro_batch_id):
        """ In 1F1B, the first half of micro-batches in the first stage 
        has no dependencies other than last micro batch
        """
        return 0 <= micro_batch_id < self.num_stages

    def init_instructions(self) -> List[PipeInstruction]:
        insts = []
        for s in range(self.num_stages):
            for m in range(self.num_micro_batches):
                # forward passes
                fwd_deps = []
                if m > 0 and s == 0:
                    fwd_deps.append(SendActivation(stage_id=0, micro_batch_id=m - 1))
                if s > 0:
                    fwd_deps.append(RecvActivation(stage_id=s, micro_batch_id=m))
                if m >= self.num_stages:
                    fwd_deps.append(BackwardPass(stage_id=s, micro_batch_id=m - self.num_stages))
                insts.append(ForwardPass(stage_id=s, micro_batch_id=m, deps=fwd_deps))

                if s < self.num_stages - 1:
                    snd_act_deps = [ForwardPass(stage_id=s, micro_batch_id=m)]
                    # if m > 0:
                    #     snd_act_deps.append(SendActivation(stage_id=s, micro_batch_id=m - 1))
                    snd_act_bind = [RecvActivation(stage_id=s + 1, micro_batch_id=m)]
                    insts.append(
                        SendActivation(stage_id=s, micro_batch_id=m, deps=snd_act_deps, bind=snd_act_bind))

                if s > 0:
                    rcv_act_deps = [ForwardPass(stage_id=s - 1, micro_batch_id=m)]
                    # if m > 0:
                    #     rcv_act_deps.append(RecvActivation(stage_id=s, micro_batch_id=m - 1))
                    rcv_act_bind = [SendActivation(stage_id=s - 1, micro_batch_id=m)]
                    insts.append(
                        RecvActivation(stage_id=s, micro_batch_id=m, deps=rcv_act_deps, bind=rcv_act_bind))

                # backward passes
                bwd_deps = []
                if s == self.num_stages - 1:
                    bwd_deps.append(ForwardPass(stage_id=s, micro_batch_id=m))
                if s < self.num_stages - 1:
                    bwd_deps.append(RecvGrad(stage_id=s, micro_batch_id=m))
                insts.append(BackwardPass(stage_id=s, micro_batch_id=m, deps=bwd_deps))

                if s > 0:
                    snd_grad_deps = [BackwardPass(stage_id=s, micro_batch_id=m)]
                    # if m > 0:
                    #     snd_grad_deps.append(SendGrad(stage_id=s, micro_batch_id=m - 1))
                    snd_grad_bind = [RecvGrad(stage_id=s - 1, micro_batch_id=m)]
                    insts.append(
                        SendGrad(stage_id=s, micro_batch_id=m, deps=snd_grad_deps, bind=snd_grad_bind))

                if s < self.num_stages - 1:
                    rcv_grad_deps = [BackwardPass(stage_id=s + 1, micro_batch_id=m)]
                    # if m > 0:
                    #     rcv_grad_deps.append(RecvGrad(stage_id=s, micro_batch_id=m - 1))
                    rcv_grad_bind = [SendGrad(stage_id=s + 1, micro_batch_id=m)]
                    insts.append(
                        RecvGrad(stage_id=s, micro_batch_id=m, deps=rcv_grad_deps, bind=rcv_grad_bind))

                if m == 0:
                    # reduce grad
                    reduce_grads_deps = [BackwardPass(stage_id=s, micro_batch_id=i) \
                                                for i in range(self.num_micro_batches)]
                    if s > 0:
                        reduce_grads_deps.extend([SendGrad(stage_id=s, micro_batch_id=i) \
                                                        for i in range(self.num_micro_batches)])
                    insts.append(ReduceGrads(stage_id=s, micro_batch_id=0, deps=reduce_grads_deps))

                    # optimizer step
                    optimize_deps = [
                        ReduceGrads(stage_id=i, micro_batch_id=0) for i in range(self.num_micro_batches)
                    ]
                    bind_stages = [i for i in range(self.num_stages)]
                    bind_stages.remove(s)
                    optimize_bind = [OptimizerStep(stage_id=s, micro_batch_id=0) for s in bind_stages]
                    insts.append(
                        OptimizerStep(stage_id=s, micro_batch_id=0, deps=optimize_deps, bind=optimize_bind))

        return insts
