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
        self.__not_ready.add(self.init_instructions())
        self.__stage_terminated = {stage_id: False for stage_id in range(self.num_stages)}

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
        ready_insts = []
        for inst in self.__not_ready.find(unmutable_result=True):
            if self._is_update_ready(inst):
                ready_insts.append(inst)

        self.__not_ready.remove(ready_insts)
        self.__ready.add(ready_insts)

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
        # self.__update_ready()

    def all_executed(self, stage=None):
        if stage is not None:
            return self.__ready.size(stage) + self.__not_ready.size(stage) + self.__inflight.size(stage) == 0
        # print(self.__ready.find(stage), self.__not_ready.find(stage), self.__inflight.find(stage))
        # print(f"all executed: {len(self.__ready)} {len(self.__not_ready)} {len(self.__inflight)}")
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

        for stage in range(self.num_stages):
            if self.all_executed(stage):
                self.update(stage)

        self.__update_ready()
        r = dict()
        if self.all_executed() and not self.__end_schedule_sent:
            for i in range(self.num_stages):
                if self.__stage_terminated[i]:
                    # print(f"stage {i} already terminated")
                    r[i] = []
                else:
                    # print(f"stage {i} send end schedule")
                    inst = EndSchedule(stage_id=i, micro_batch_id=0)
                    r[i] = [inst]
            self.__end_schedule_sent = True
            return r

        for i in range(self.num_stages):
            r[i] = self.__ready.find(stage_id=i)
            # print(f"not ready: {self.__not_ready.find(stage_id=i)}")
            # print(f"in flight: {self.__inflight.find(stage_id=i)}")
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

    def update(self, stage_id=None):
        """ Called by controller to find new instructions for some stage_id
        """
        if self.__terminated:
            raise RuntimeError("Cannot update an already terminated schedule.")
        # if self.__not_ready.size(stage_id) == 0 and self.__inflight.size(stage_id) == 0:
        new_instructions = self.update_instructions()
        if len(new_instructions) > 0:
            self.__not_ready.add(new_instructions)
            # self.__update_ready()
            return True
        return False

    def _is_update_ready(self, inst: PipeInstruction):
        """ check if an instruction is ready but not put into ready set
        """
        update_ready = all([self.__executed.contain(dep) for dep in inst.deps])
        # print(f"inst {inst} deps {inst.deps} update ready {update_ready}")
        return update_ready

    def terminate_stage(self, stage_id):
        if self.__stage_terminated[stage_id]:
            raise RuntimeError("Cannot terminate an already terminated stage.")

        ready_list = self.__ready.find(stage_id=stage_id)
        self.__ready.remove(ready_list)
        not_ready_list = self.__not_ready.find(stage_id=stage_id)
        self.__not_ready.remove(not_ready_list)
        inflight_list = self.__inflight.find(stage_id=stage_id)
        assert len(inflight_list) == 0, f"stage {stage_id} terminated with inflight instructions {inflight_list}, "\
                                        f"ready {ready_list}, not ready {not_ready_list}"

        self.__executed.add(ready_list + not_ready_list)

        self.__stage_terminated[stage_id] = True

    def terminate(self):
        """ Called by controller to terminate the schedule, force execute end schedule instruction for all stages
        Move all instructions from ready to executed.
        """
        if self.__terminated:
            raise RuntimeError("Cannot terminate an already terminated schedule.")

        assert all([self.__stage_terminated[stage_id] for stage_id in range(self.num_stages)]), \
               "Schedule terminate called before stage terminate"
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

    def __init__(self, steps_per_update=5, preserve_fwd_order=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.steps_per_update = steps_per_update
        self.remaining_steps = self.num_steps
        self.preserve_fwd_order = preserve_fwd_order

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
            # preserve forward order in generation, which is necessary for tensor parallel
            if self.preserve_fwd_order:
                if m > 0:
                    fwd_deps.append(ForwardPass(stage_id=s, micro_batch_id=m - 1, step_id=t))
                if m == 0 and t > 0:
                    fwd_deps.append(
                        ForwardPass(stage_id=s, micro_batch_id=self.num_micro_batches - 1, step_id=t - 1))

            if m > 0 and s == 0:
                fwd_deps.append(SendActivation(stage_id=s, micro_batch_id=m - 1, step_id=t))
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

            if s == self.num_stages - 1 and t < self.num_steps - 1:
                snd_deps = [ForwardPass(stage_id=self.num_stages - 1, micro_batch_id=m, step_id=t)]
                snd_bind = [RecvNextTokens(stage_id=0, micro_batch_id=m, step_id=t)]
                insts.append(
                    SendNextTokens(stage_id=self.num_stages - 1,
                                   micro_batch_id=m,
                                   step_id=t,
                                   deps=snd_deps,
                                   bind=snd_bind))

            if s == 0 and t < self.num_steps - 1:
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
                if m >= self.num_stages - s:
                    fwd_deps.append(BackwardPass(stage_id=s, micro_batch_id=m - self.num_stages + s))
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
                # if s == self.num_stages - 1:
                #     bwd_deps.append(ForwardPass(stage_id=s, micro_batch_id=m))
                if s < self.num_stages - 1:
                    bwd_deps.append(RecvGrad(stage_id=s, micro_batch_id=m))
                if m + self.num_stages - 1 - s < self.num_micro_batches:
                    bwd_deps.append(ForwardPass(stage_id=s, micro_batch_id=m + self.num_stages - 1 - s))
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
                        ReduceGrads(stage_id=i, micro_batch_id=0) for i in range(self.num_stages)
                    ]
                    bind_stages = [i for i in range(self.num_stages)]
                    bind_stages.remove(s)
                    optimize_bind = [OptimizerStep(stage_id=s, micro_batch_id=0) for s in bind_stages]
                    insts.append(
                        OptimizerStep(stage_id=s, micro_batch_id=0, deps=optimize_deps, bind=optimize_bind))

        return insts
