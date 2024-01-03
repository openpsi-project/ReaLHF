from collections import defaultdict
from typing import List, Union
import dataclasses
import multiprocessing
import queue
import threading
import time

import zmq

from base.monitor import get_tracer
from impl.model.backend.pipe_engine.dynamic_schedule import DynamicPipeSchedule
from impl.model.backend.pipe_engine.instruction import EndSchedule, PipeInstruction
import base.constants
import base.logging as logging
import base.name_resolve as name_resolve
import base.names as names
import base.network as network

logger = logging.getLogger("ScheduleController", "benchmark")


@dataclasses.dataclass
class PrioritizedSchedule:
    priority: int
    index: int
    schedule: DynamicPipeSchedule
    terminate_signal_count: int = 0

    def __lt__(self, other: 'PrioritizedSchedule'):
        return self.priority < other.priority


class EngineScheduleController:
    """ Centralized scheduler that schedules pipeline instruction for each engine.
    only exists on every engine that has pp_rank=0
    """

    def __init__(self, num_stages: int):
        self.num_stages = num_stages
        self.waiting_stages = list(range(self.num_stages))
        self.schedule_count = 0
        self.schedules: List[PrioritizedSchedule] = []  # prioritized schedule list
        self.__posted = 0
        self.__polled = 0

        self.__terminated = False
        self.__binded_schedule = None
        self.__terminate_queue = multiprocessing.Queue(1)
        self.__schedule_queue = multiprocessing.Queue(1)
        self.thread = multiprocessing.Process(target=self.run)
        # self.thread.start()

    def __init_sockets(self):
        address = network.gethostip()
        self.context = zmq.Context()

        self.instruction_socket = self.context.socket(zmq.PUB)
        instruction_port = network.find_free_port()
        self.instruction_socket.bind(f"tcp://*:{instruction_port}")

        self.signal_socket = self.context.socket(zmq.PULL)
        signal_port = network.find_free_port()
        self.signal_socket.bind(f"tcp://*:{signal_port}")

        logger.info(f"instruction binding to tcp://*:{instruction_port}")
        logger.info(f"signal binding to tcp://*:{signal_port}")

        expr_name = base.constants.experiment_name()
        trial_name = base.constants.trial_name()
        model_name = base.constants.model_name()
        dp_rank = base.constants.data_parallel_rank()
        mp_rank = base.constants.model_parallel_rank()

        name = names.model_controller(expr_name, trial_name, model_name, dp_rank, mp_rank)
        name_resolve.add(name, f"{address};{instruction_port};{signal_port}", keepalive_ttl=30)

        name = names.model_controller_barrier(expr_name, trial_name, model_name, dp_rank, mp_rank)
        while (len(name_resolve.get_subtree(name)) < self.num_stages):
            time.sleep(0.1)

    def __init_storage(self):
        self.__inst_queues = defaultdict(list)

    def start(self):
        self.thread.start()

    def stop(self):
        self.__terminate_queue.put(0)

    # @property
    # def posted(self):
    #     """ Number of instruction posted """
    #     return self.__posted

    # @property
    # def completed(self):
    #     """ Number of instruction completed """
    #     return self.__polled

    def issue_schedule(self, sched: DynamicPipeSchedule, priority: int):
        """Called by engine if an API (train_batch, eval_batch, generate ... ) is called, 
        issue a schedule to controller. 
        """
        self.__schedule_queue.put((sched, priority))

    def __issue_schedule(self, sched: DynamicPipeSchedule, priority: int):
        self.schedules.append(PrioritizedSchedule(priority, self.schedule_count, sched))
        self.schedules.sort(reverse=True)
        self.schedule_count += 1

    def __check_and_issue_schedule(self):
        try:
            sched, priority = self.__schedule_queue.get(block=False)
            self.__issue_schedule(sched, priority)
        except queue.Empty:
            pass

    def __check_terminate(self):
        try:
            self.__terminate_queue.get(block=False)
            self.__terminated = True
        except queue.Empty:
            pass

    def update_instruction_queues(self):
        """ core, manage instructions from all schedules, including following steps:
        1. fetch ready instructions from all schedules
        2. enqueue instructions to instruction queue with priority
        3. 
        """
        for prior_sched in self.schedules:
            sched = prior_sched.schedule
            ri = sched.ready()  # ready instructions
            for stage_id in range(self.num_stages):
                stage_ri = ri[stage_id]
                for inst in stage_ri:
                    inst: PipeInstruction
                    self.__inst_queues[stage_id].append((prior_sched, inst))
                    bind = inst.bind
                    if len(bind) > 0:
                        for b in bind:
                            bind_stage_ri = ri[b.stage_id]
                            self.__inst_queues[b.stage_id].append((prior_sched, b))
                            found = False
                            for bind_inst in bind_stage_ri:
                                if bind_inst == b:
                                    found = True
                                    bind_stage_ri.remove(bind_inst)
                            if not found:
                                raise RuntimeError(f"Binded instruction {bind} of {inst} not ready.")
                else:
                    sched.update(stage_id)

    def __post_one_instruction(self, stage: int, sched_index: int, inst: PipeInstruction, end: bool):
        """ core, post one instruction to one stage
        """
        msg = [
            int.to_bytes(stage, 4, byteorder="big"),
            int.to_bytes(sched_index, 4, byteorder="big"),
            int.to_bytes(int(end), 4, byteorder="big"),
            inst.encode()
        ]
        # print(f"posting msg {msg}")
        self.instruction_socket.send_multipart(msg)
        # print(f"posting {inst} done")
        self.waiting_stages.remove(stage)

    def post_instructions(self):
        posted = 0
        for stage in self.waiting_stages:
            stage_inst_squeue = self.__inst_queues[stage]
            if len(stage_inst_squeue) > 0:
                prior_sched, inst = stage_inst_squeue.pop(0)
                sched_index = prior_sched.index
                self.__post_one_instruction(stage, sched_index, inst, end=False)
                posted += 1
        return posted

    # def post_instructions(self):
    #     """Called by engine in every run step, check ready instructions and send **one** ready
    #     instruction with highest priority to all engines.

    #     Note: Binded instructions must be handled. Once one instruction is posted, its binded counterpart
    #     must be posted next for its corresponding stage.
    #     """
    #     instruction_posted = 0

    #     def check_schedule_and_post(prior_sched: PrioritizedSchedule, stage: int, add_bind: bool):
    #         """ Check ready instruction in schedule, if there is any ready, post it and return True,
    #         else return False.
    #         """
    #         index = prior_sched.index
    #         sched = prior_sched.schedule
    #         inst, end = sched.post_one_ready(stage)
    #         inst: PipeInstruction

    #         if inst:
    #             # print(f"posting {inst}")
    #             msg = [
    #                 int.to_bytes(stage, 4, byteorder="big"),
    #                 int.to_bytes(index, 4, byteorder="big"),
    #                 int.to_bytes(int(end), 4, byteorder="big"),
    #                 inst.encode()
    #             ]
    #             # print(f"posting msg {msg}")
    #             self.instruction_socket.send_multipart(msg)
    #             # print(f"posting {inst} done")
    #             self.waiting_stages.remove(stage)
    #             if inst.bind and add_bind:
    #                 self.__binded_schedule[inst.bind.stage_id] = prior_sched
    #             return True
    #         return False

    #     for stage in self.waiting_stages:
    #         if stage in self.__binded_schedule:
    #             prior_sched = self.__binded_schedule.pop(stage)
    #             posted = check_schedule_and_post(prior_sched, stage, add_bind=False)
    #             if posted:
    #                 instruction_posted += 1
    #                 continue

    #         for prior_sched in self.schedules:
    #             posted = check_schedule_and_post(prior_sched, stage, add_bind=True)
    #             if posted:
    #                 instruction_posted += 1
    #                 break

    #     return instruction_posted

    def poll_results(self):
        """Called by engine in every run step, check signals for instruction complete and 
        schedule stop. Check signal from all sources until nothing to receivce.

        Return: 
            int: number of instructions completed
        """
        completed = 0
        while True:
            try:
                stage_id, sched_id, signal_code, *insts = self.signal_socket.recv_multipart(flags=zmq.NOBLOCK)
                stage_id = int.from_bytes(stage_id, byteorder="big")
                sched_id = int.from_bytes(sched_id, byteorder="big")
                signal_code = int.from_bytes(signal_code, byteorder="big")

                assert stage_id not in self.waiting_stages
                self.waiting_stages.append(stage_id)
                sched = None
                this_prior_sched = None
                for prior_sched in self.schedules:
                    if prior_sched.index == sched_id:
                        sched = prior_sched.schedule
                        this_prior_sched = prior_sched
                if sched is None:
                    raise RuntimeError(f"Schedule with index {sched_id} not found.")

                insts = [PipeInstruction.decode(inst) for inst in insts]
                # print(f"Rank {stage_id}: sched {sched_id} stage {stage_id} executing {insts}")
                sched.exec(insts)
                # print(f"Rank {stage_id}: sched {sched_id} stage {stage_id} executed {insts}")

                completed += len(insts)
                if signal_code == 1:  # terminate
                    this_prior_sched.terminate_signal_count += 1
                    # print(f"{this_prior_sched.index} count = {this_prior_sched.terminate_signal_count}")
                    if this_prior_sched.terminate_signal_count == self.num_stages:
                        sched.terminate()
                        self.schedules.remove(this_prior_sched)
                        # print(f"removed sched {this_prior_sched.index}, schedules {len(self.schedules)}")
                elif signal_code == 0:  # normal instruction execute, nothing happens
                    pass
                else:
                    raise NotImplementedError(
                        f"Unknown signal code {signal_code} received when polling results.")
            except zmq.ZMQError:
                return completed

    def run(self):
        self.__init_sockets()
        self.__init_storage()
        # self.__binded_schedule = dict()
        # tracer = get_tracer(
        #         tracer_entries=int(2e6),
        #         # max_stack_depth=10,
        #         ignore_c_function=False,
        #         ignore_frozen=True,
        #         log_async=True,
        #         min_duration=10,
        #         output_file=f"/home/meizy/logs/viztracer/trace0.json")
        # tracer.start()

        while not self.__terminated:
            self.__check_and_issue_schedule()
            posted = self.post_instructions()
            # if posted > 0:
            #     print(f"schedule_controller.py: Posted {posted} instructions")
            # time.sleep(0.001)
            polled = self.poll_results()
            # if polled > 0:
            #     print(f"Polled {polled} results")
            self.update_instruction_queues()
            self.__posted += posted
            self.__polled += polled
            self.__check_terminate()
        # tracer.save()


class EngineScheduleClient:
    """ Client on every enigne instance to receive instruction from EngineScheduleController
    """

    def __init__(self, stage_id: int):
        self.stage_id = stage_id
        expr_name = base.constants.experiment_name()
        trial_name = base.constants.trial_name()
        model_name = base.constants.model_name()
        dp_rank = base.constants.data_parallel_rank()
        mp_rank = base.constants.model_parallel_rank()

        name = names.model_controller(expr_name, trial_name, model_name, dp_rank, mp_rank)
        address, instruction_port, signal_port = name_resolve.wait(name, timeout=30).split(";")
        # name_resolve.get(name).split(";")

        self.context = zmq.Context()
        self.instruction_socket = self.context.socket(zmq.SUB)
        self.instruction_socket.setsockopt(zmq.SUBSCRIBE, int.to_bytes(self.stage_id, 4, byteorder="big"))
        self.instruction_socket.connect(f"tcp://{address}:{instruction_port}")
        logger.info(
            f"EngineScheduleClient instruction socket connecting to tcp://{address}:{instruction_port}, "
            f"sub to {int.to_bytes(self.stage_id, 4, byteorder='big')}")

        self.signal_socket = self.context.socket(zmq.PUSH)
        self.signal_socket.connect(f"tcp://{address}:{signal_port}")
        logger.info(f"EngineScheduleClient signal socket connecting to tcp://{address}:{signal_port}")

        name = names.model_controller_barrier(expr_name, trial_name, model_name, dp_rank, mp_rank)
        name_resolve.add_subentry(name, self.stage_id)

        self.last_inst: PipeInstruction = None
        self.last_inst_sched: int = None

    def poll_instruction(self):
        """Called by engine in every run step, check instruction from controller.
        """
        try:
            stage_id, sched_index, end, encoded = self.instruction_socket.recv_multipart(flags=zmq.NOBLOCK)
            stage_id = int.from_bytes(stage_id, byteorder="big")
            sched_index = int.from_bytes(sched_index, byteorder="big")
            end = bool(int.from_bytes(end, byteorder="big"))
            assert stage_id == self.stage_id
            inst = PipeInstruction.decode(encoded)
            # print(f"Rank {stage_id}: stage {self.stage_id} received instruction {sched_index} {inst}")
            self.last_inst = inst
            self.last_inst_sched = sched_index
            # print(f"stage {self.stage_id} {sched_index} {inst} {end}")
            return sched_index, inst, end
        except zmq.ZMQError:
            return None, None, None

    def post_result(self, signal: int):
        assert self.last_inst is not None and self.last_inst_sched is not None
        msg = [
            int.to_bytes(self.stage_id, 4, byteorder="big"),
            int.to_bytes(self.last_inst_sched, 4, byteorder="big"),
            int.to_bytes(signal, 4, byteorder="big"),
            self.last_inst.encode()
        ]
        # print(
        #     f"Rank {self.stage_id}: posting result stage {self.stage_id} inst {self.last_inst} of sched {self.last_inst_sched}"
        # )
        self.signal_socket.send_multipart(msg)
        self.last_inst = None
        self.last_inst_sched = None
