from __future__ import annotations  # python3.7+ feature to allow self-referencing type hints

from typing import List
import copy
import dataclasses
import logging
import random
import time

from base.monitor import mock_time_mark, summary_time_points

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
DATE_FORMAT = "%Y%m%d-%H:%M:%S"

INST_TIME_MAPPING = {
    "first_forward": 10 * 1e9,
    "other_forward": 2 * 1e9,
    "forward": 50 * 1e9,
    "backward": 100 * 1e9,
    "optim": 20 * 1e9,
    "wait_data": 20 * 10e9
}

global_executed = []


@dataclasses.dataclass
class Instruction:
    rank: int
    name: str
    mbid: int
    dependency: List[Instruction] = None
    start: int = None
    end: int = None
    stepid: int = 0

    def __eq__(self, other: Instruction):
        return self.rank == other.rank and self.name == other.name and \
               self.mbid == other.mbid and self.stepid == other.stepid

    def __str__(self):
        return f"Instruction({self.rank} {self.name} {self.mbid} {self.stepid})"


class InstructionExecutor:

    def __init__(self, rank, num_micro_batches):
        self.rank = rank
        self.num_mb = num_micro_batches
        self.to_execute: List[Instruction] = []  # instruction sequence to execute, dependency + isnt
        # dependency: (rank, inst, mbid)

    def register_instruction(self, inst: Instruction):
        self.to_execute.append(inst)

    def execute_one(self, now, async_wait=False):
        """ Execute one instruction if available

        Args:
            now (int): current time for this rank 
            async_wait (bool, optional): if True, wait_data instruction will not block. Defaults to False.
        
        Returns:
            now (int): current time for this rank after execution
        """
        if len(self.to_execute) == 0:
            # print(f"rank {self.rank} execute nothing")
            return now

        min_start_time = None  # jump to min_start_time if nothing to execute now,
        # min_start_time is the minimum start time of all instructions ready to execute in this rank
        ready = []
        ready_start_time = []
        min_inst = None
        for inst in self.to_execute:
            deps = inst.dependency
            start_time = 0
            go_next = False
            # print(f"check inst {inst} deps:")
            # time.sleep(0.5)
            for dep in deps:
                # print(f"dep:: {dep}")
                executed = False
                for done in global_executed:
                    if dep == done:
                        # print(f"dep done:: {done}")
                        start_time = max(start_time, done.end)
                        executed = True
                        break
                if not executed:
                    go_next = True

            if go_next:
                continue

            # find the lastest end time in global_executed for current rank
            for done in global_executed:
                if done.rank == self.rank:  # and done.name is not "wait_data":
                    if not async_wait or not done.name == "wait_data":
                        start_time = max(start_time, done.end)

            ready.append(inst)
            ready_start_time.append(start_time)

            if start_time > now:
                if min_start_time is None or start_time < min_start_time:
                    min_inst = inst
                min_start_time = start_time if min_start_time is None else min(min_start_time, start_time)
                continue

            # start_time = max(now, start_time)
            # all deps are executed

            self.to_execute.remove(inst)
            if self.rank == 0 and inst.name == "other_forward" and inst.mbid == 8 and inst.stepid == 1:
                print(f"mark :: {inst} {start_time}")
                print(f"deps ::")
                for d in inst.dependency:
                    print(d)

            if self.rank == 3 and inst.name == "first_forward" and inst.mbid == 8 and inst.stepid == 0:
                print(f"mark :: {inst} {start_time} {start_time + int(INST_TIME_MAPPING[inst.name])}")
                print(f"deps ::")
                for d in inst.dependency:
                    print(d)

            inst.start = start_time
            inst.end = start_time + int(INST_TIME_MAPPING[inst.name])
            # print("rank", self.rank, "execute", str(inst))
            global_executed.append(inst)
            mock_time_mark(inst.name + "_start", str(inst.rank), inst.start, 0)
            mock_time_mark(inst.name + "_end", str(inst.rank), inst.end, 0)

            if not inst.name == "wait_data":
                now = inst.end
            return now

        if min_start_time is not None:
            now = min_start_time
            print(f"rank {self.rank} execute nothing, next to execute {min_inst} at {now}")
            print("ready:")
            for r, st in zip(ready, ready_start_time):
                print(r, st)
        return now

    def is_empty(self):
        return len(self.to_execute) == 0


class Schedule:

    def __init__(self, num_pp, num_micro_batches, num_steps):
        self.num_pp = num_pp
        self.num_mb = num_micro_batches
        self.executors = [InstructionExecutor(i, num_micro_batches) for i in range(num_pp)]
        self.num_steps = num_steps

    def execute_sched(self, async_wait=False):
        # for executor in self.executors:
        #     print(executor.to_execute)
        nows = {rank: 0 for rank in range(self.num_pp)}
        while True:
            # for i in range(self.num_pp):
            #     print(f"rank {i} now {nows[i]}")
            all_empty = True
            for rank, executor in enumerate(self.executors):
                nows[rank] = executor.execute_one(nows[rank], async_wait=async_wait)
                if not executor.is_empty():
                    all_empty = False
            if all_empty:
                break


class Train1F1BSchedule(Schedule):

    def __init__(self, num_pp, num_micro_batches):
        assert num_micro_batches >= 2 * num_pp
        super().__init__(num_pp, num_micro_batches, 1)

        for rank, executor in enumerate(self.executors):
            forwards = []
            backwards = []
            optims = []
            for mbid in range(num_micro_batches):
                forward_deps = []
                backward_deps = []
                if mbid > 0:
                    forward_deps.append(Instruction(rank, "forward", mbid - 1))
                    backward_deps.append(Instruction(rank, "backward", mbid - 1))
                if rank > 0:
                    forward_deps.append(Instruction(rank - 1, "forward", mbid))
                if rank < self.num_pp - 1:
                    backward_deps.append(Instruction(rank + 1, "backward", mbid))

                if mbid > self.num_pp - 1:
                    # forward_deps.append(Instruction(rank, "forward", mbid-self.num_pp))
                    forward_deps.append(Instruction(rank, "backward", mbid - self.num_pp))

                backward_deps.append(Instruction(rank, "forward", mbid))

                forwards.append(Instruction(rank, "forward", mbid, forward_deps))
                backwards.append(Instruction(rank, "backward", mbid, backward_deps))
            optim_deps = [Instruction(0, "backward", mbid) for mbid in range(num_micro_batches)]
            optims.append(Instruction(rank, "optim", 0, optim_deps))

            # in 1f1b, backward priority is higher than forward
            for inst in backwards:
                executor.register_instruction(inst)
            for inst in forwards:
                executor.register_instruction(inst)
            for inst in optims:
                executor.register_instruction(inst)


class GenerateSchedule(Schedule):

    def __init__(self, num_pp, num_micro_batches, max_new_tokens):
        super().__init__(num_pp, num_micro_batches, max_new_tokens)

        for rank, executor in enumerate(self.executors):
            for mbid in range(num_micro_batches):
                deps = []
                if mbid > 0:
                    deps.append(Instruction(rank, "first_forward", mbid - 1, stepid=0))
                if rank > 0:
                    deps.append(Instruction(rank - 1, "first_forward", mbid, stepid=0))
                executor.register_instruction(Instruction(rank, "first_forward", mbid, deps, stepid=0))

            for stepid in range(1, max_new_tokens):
                for mbid in range(num_micro_batches):
                    deps = []
                    if rank == 0:
                        if stepid == 1:
                            deps.append(Instruction(self.num_pp - 1, "first_forward", mbid, stepid=0))
                        else:
                            deps.append(Instruction(self.num_pp - 1, "other_forward", mbid,
                                                    stepid=stepid - 1))
                    else:
                        deps.append(Instruction(rank - 1, "other_forward", mbid, stepid=stepid))
                    executor.register_instruction(
                        Instruction(rank, "other_forward", mbid, deps, stepid=stepid))


class TrivialRLHFActorSchedule(Schedule):

    def __init__(self, num_pp, num_micro_batches, max_new_tokens):
        super().__init__(num_pp, num_micro_batches, max_new_tokens)

        for rank, executor in enumerate(self.executors):
            for mbid in range(num_micro_batches):
                deps = []
                if mbid > 0:
                    deps.append(Instruction(rank, "first_forward", mbid - 1, stepid=0))
                if rank > 0:
                    deps.append(Instruction(rank - 1, "first_forward", mbid, stepid=0))
                executor.register_instruction(Instruction(rank, "first_forward", mbid, deps, stepid=0))

            for stepid in range(1, max_new_tokens):
                for mbid in range(num_micro_batches):
                    deps = []
                    if rank == 0:
                        if stepid == 1:
                            deps.append(Instruction(self.num_pp - 1, "first_forward", mbid, stepid=0))
                        else:
                            deps.append(Instruction(self.num_pp - 1, "other_forward", mbid,
                                                    stepid=stepid - 1))
                    else:
                        deps.append(Instruction(rank - 1, "other_forward", mbid, stepid=stepid))
                    executor.register_instruction(
                        Instruction(rank, "other_forward", mbid, deps, stepid=stepid))

            if rank == 0:
                for mbid in range(num_micro_batches):
                    wait_deps = [
                        Instruction(self.num_pp - 1, "other_forward", mbid, deps, stepid=max_new_tokens - 1)
                    ]
                    executor.register_instruction(Instruction(rank, "wait_data", mbid, wait_deps, stepid=0))

            forwards = []
            backwards = []
            optims = []

            for mbid in range(num_micro_batches):
                forward_deps = []
                backward_deps = []
                if rank == 0:
                    forward_deps.append(Instruction(rank, "wait_data", mbid))
                if mbid > 0:
                    forward_deps.append(Instruction(rank, "forward", mbid - 1))
                    backward_deps.append(Instruction(rank, "backward", mbid - 1))
                if rank > 0:
                    forward_deps.append(Instruction(rank - 1, "forward", mbid))
                if rank < self.num_pp - 1:
                    backward_deps.append(Instruction(rank + 1, "backward", mbid))

                if mbid > self.num_pp - 1:
                    # forward_deps.append(Instruction(rank, "forward", mbid-self.num_pp))
                    forward_deps.append(Instruction(rank, "backward", mbid - self.num_pp))

                backward_deps.append(Instruction(rank, "forward", mbid))

                forwards.append(Instruction(rank, "forward", mbid, forward_deps))
                backwards.append(Instruction(rank, "backward", mbid, backward_deps))
            optim_deps = [Instruction(0, "backward", mbid) for mbid in range(num_micro_batches)]
            optims.append(Instruction(rank, "optim", 0, optim_deps))

            # in 1f1b, backward priority is higher than forward
            for inst in backwards:
                executor.register_instruction(inst)
            for inst in forwards:
                executor.register_instruction(inst)
            for inst in optims:
                executor.register_instruction(inst)


class FilledRLHFActorSchedule(Schedule):

    def __init__(self, num_pp, num_micro_batches, max_new_tokens):
        super().__init__(num_pp, num_micro_batches, max_new_tokens)
        assert num_micro_batches >= 2 * num_pp
        assert num_micro_batches % 2 == 0

        half = num_micro_batches // 2

        # first large batch
        for rank, executor in enumerate(self.executors):
            for mbid in range(half):
                deps = []
                if mbid > 0:
                    deps.append(Instruction(rank, "first_forward", mbid - 1, stepid=0))
                if rank > 0:
                    deps.append(Instruction(rank - 1, "first_forward", mbid, stepid=0))
                executor.register_instruction(Instruction(rank, "first_forward", mbid, deps, stepid=0))

            for stepid in range(1, max_new_tokens):
                for mbid in range(half):
                    deps = []
                    if rank == 0:
                        if stepid == 1:
                            deps.append(Instruction(self.num_pp - 1, "first_forward", mbid, stepid=0))
                        else:
                            deps.append(Instruction(self.num_pp - 1, "other_forward", mbid,
                                                    stepid=stepid - 1))
                    else:
                        deps.append(Instruction(rank - 1, "other_forward", mbid, stepid=stepid))
                    executor.register_instruction(
                        Instruction(rank, "other_forward", mbid, deps, stepid=stepid))

            for mbid in range(half):
                wait_deps = [
                    Instruction(self.num_pp - 1, "other_forward", mbid, deps, stepid=max_new_tokens - 1)
                ]
                executor.register_instruction(Instruction(rank, "wait_data", mbid, wait_deps, stepid=0))

            forwards = []
            backwards = []
            optims = []
            for mbid in range(half):
                forward_deps = []
                backward_deps = []
                if rank == 0:
                    forward_deps.append(Instruction(rank, "wait_data", mbid))
                if mbid > 0:
                    forward_deps.append(Instruction(rank, "forward", mbid - 1))
                    backward_deps.append(Instruction(rank, "backward", mbid - 1))
                if rank > 0:
                    forward_deps.append(Instruction(rank - 1, "forward", mbid))
                if rank < self.num_pp - 1:
                    backward_deps.append(Instruction(rank + 1, "backward", mbid))

                if mbid > self.num_pp - 1:
                    # forward_deps.append(Instruction(rank, "forward", mbid-self.num_pp))
                    forward_deps.append(Instruction(rank, "backward", mbid - self.num_pp))

                backward_deps.append(Instruction(rank, "forward", mbid))

                forwards.append(Instruction(rank, "forward", mbid, forward_deps))
                backwards.append(Instruction(rank, "backward", mbid, backward_deps))
            # optim_deps = [Instruction(0, "backward", mbid) for mbid in range(half)]
            # optims.append(Instruction(rank, "optim", 0, optim_deps))

            # in 1f1b, backward priority is higher than forward
            for inst in backwards:
                executor.register_instruction(inst)
            for inst in forwards:
                executor.register_instruction(inst)
            for inst in optims:
                executor.register_instruction(inst)

        # second large batch
        for rank, executor in enumerate(self.executors):
            # prioritize other forward
            for mbid in range(half, num_micro_batches):
                deps = []
                if mbid == half:
                    deps.append(Instruction(rank, "other_forward", mbid - 1, stepid=max_new_tokens - 1))
                if mbid > half:
                    deps.append(Instruction(rank, "first_forward", mbid - 1, stepid=0))
                if rank > 0:
                    deps.append(Instruction(rank - 1, "first_forward", mbid, stepid=0))
                executor.register_instruction(Instruction(rank, "first_forward", mbid, deps, stepid=0))

            for stepid in range(1, max_new_tokens):
                for mbid in range(half, num_micro_batches):
                    deps = []
                    if rank == 0:
                        if stepid == 1:
                            deps.append(Instruction(self.num_pp - 1, "first_forward", mbid, stepid=0))
                        else:
                            deps.append(Instruction(self.num_pp - 1, "other_forward", mbid,
                                                    stepid=stepid - 1))
                    else:
                        deps.append(Instruction(rank - 1, "other_forward", mbid, stepid=stepid))
                    executor.register_instruction(
                        Instruction(rank, "other_forward", mbid, deps, stepid=stepid))
                    # if mbid == half and stepid == 1:
                    # print(f"other_forward {rank} {mbid} {stepid} deps::")
                    # for d in deps:
                    #     print(d)

            for mbid in range(half, num_micro_batches):
                wait_deps = [
                    Instruction(self.num_pp - 1, "other_forward", mbid, deps, stepid=max_new_tokens - 1)
                ]
                executor.register_instruction(Instruction(rank, "wait_data", mbid, wait_deps, stepid=0))

            forwards = []
            backwards = []
            optims = []
            for mbid in range(half, num_micro_batches):
                forward_deps = []
                backward_deps = []
                if rank == 0:
                    forward_deps.append(Instruction(rank, "wait_data", mbid))
                if mbid > half:
                    forward_deps.append(Instruction(rank, "forward", mbid - 1))
                    backward_deps.append(Instruction(rank, "backward", mbid - 1))
                if rank > 0:
                    forward_deps.append(Instruction(rank - 1, "forward", mbid))
                if rank < self.num_pp - 1:
                    backward_deps.append(Instruction(rank + 1, "backward", mbid))

                if mbid > half + self.num_pp - 1:
                    # forward_deps.append(Instruction(rank, "forward", mbid-self.num_pp))
                    forward_deps.append(Instruction(rank, "backward", mbid - self.num_pp))

                backward_deps.append(Instruction(rank, "forward", mbid))

                forwards.append(Instruction(rank, "forward", mbid, forward_deps))
                backwards.append(Instruction(rank, "backward", mbid, backward_deps))
            optim_deps = [Instruction(0, "backward", mbid) for mbid in range(num_micro_batches)]
            optims.append(Instruction(rank, "optim", 0, optim_deps))

            # in 1f1b, backward priority is higher than forward
            for inst in backwards:
                executor.register_instruction(inst)
            for inst in forwards:
                executor.register_instruction(inst)
            for inst in optims:
                executor.register_instruction(inst)


def execute_train_schedule(num_pp, num_micro_batches):
    logging.basicConfig(filename="/home/meizy/logs/simulated_train.log",
                        filemode="w",
                        format=LOG_FORMAT,
                        datefmt=DATE_FORMAT,
                        level="DEBUG")
    sched = Train1F1BSchedule(num_pp, num_micro_batches)
    sched.execute_sched()


def execute_generate_schedule(num_pp, num_micro_batches, max_new_tokens):
    logging.basicConfig(filename="/home/meizy/logs/simulated_generate.log",
                        filemode="w",
                        format=LOG_FORMAT,
                        datefmt=DATE_FORMAT,
                        level="DEBUG")
    sched = GenerateSchedule(num_pp, num_micro_batches, max_new_tokens)
    sched.execute_sched()


def execute_trivial_rlhf_schedule(num_pp, num_micro_batches, max_new_tokens):
    logging.basicConfig(filename="/home/meizy/logs/simulated_trivial_rlhf.log",
                        filemode="w",
                        format=LOG_FORMAT,
                        datefmt=DATE_FORMAT,
                        level="DEBUG")
    sched = TrivialRLHFActorSchedule(num_pp, num_micro_batches, max_new_tokens)
    sched.execute_sched(async_wait=True)


def execute_filled_rlhf_schedule(num_pp, num_micro_batches, max_new_tokens):
    logging.basicConfig(filename="/home/meizy/logs/simulated_filled_rlhf.log",
                        filemode="w",
                        format=LOG_FORMAT,
                        datefmt=DATE_FORMAT,
                        level="DEBUG")
    sched = FilledRLHFActorSchedule(num_pp, num_micro_batches, max_new_tokens)
    sched.execute_sched(async_wait=True)


def plot_simulated_pipeline(fn="train"):
    file_name = f"/home/meizy/logs/simulated_{fn}.log"
    identifiers = [str(i) for i in range(4)]
    fn = f"scripts/figs/simulated_{fn}.png"
    start_keys = [
        "first_forward_start",
        "other_forward_start",
        "forward_start",
        "backward_start",
        "optim_start"  # , "wait_data_start"
    ]
    end_keys = [
        "first_forward_end",
        "other_forward_end",
        "forward_end",
        "backward_end",
        "optim_end"  # , "wait_data_end"
    ]
    summary_time_points(file_name=file_name,
                        start_keys=start_keys,
                        end_keys=end_keys,
                        identifiers=identifiers,
                        step_range=None,
                        save_fig_path=fn,
                        figsize=(20, 4),
                        draw_boundary=False)


if __name__ == "__main__":
    # execute_train_schedule(4, 8)
    # plot_simulated_pipeline("train")
    # execute_generate_schedule(4, 4, 64)
    # plot_simulated_pipeline("generate")
    # execute_trivial_rlhf_schedule(4, 16, 64)
    # plot_simulated_pipeline("trivial_rlhf")
    execute_filled_rlhf_schedule(4, 16, 64)
    plot_simulated_pipeline("filled_rlhf")
