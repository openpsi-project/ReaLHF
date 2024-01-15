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

    def __init__(self, num_stages: int, trace: bool = False):
        self.num_stages = num_stages
        self.waiting_stages = list(range(self.num_stages))
        self.schedule_count = 0
        self.schedules: List[PrioritizedSchedule] = []  # prioritized schedule list
        self.__posted = 0
        self.__polled = 0

        self.__experiment_name = None
        self.__trial_name = None

        self.__terminated = False
        self.__binded_schedule = None
        self.__terminate_queue = multiprocessing.Queue(1)
        self.__schedule_queue = multiprocessing.Queue(1)

        self.__tracer_save_queue = multiprocessing.Queue(1)
        multiprocessing.set_start_method("fork", force=True)
        self.thread = multiprocessing.Process(target=self.run)
        self._trace_controller = trace
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

        self.__experiment_name = expr_name = base.constants.experiment_name()
        self.__trial_name = trial_name = base.constants.trial_name()
        model_name = base.constants.model_name()
        dp_rank = base.constants.data_parallel_rank()
        mp_rank = base.constants.model_parallel_rank()

        name = names.model_controller(expr_name, trial_name, model_name, dp_rank, mp_rank)
        name_resolve.add(name, f"{address};{instruction_port};{signal_port}", keepalive_ttl=30)

        name = names.model_controller_barrier(expr_name, trial_name, model_name, dp_rank, mp_rank)
        while (len(name_resolve.get_subtree(name)) < self.num_stages):
            time.sleep(0.1)

        self.tracer = None

    def __init_storage(self):
        self.__inst_queues = defaultdict(list)

    def start(self):
        self.thread.start()

    def stop(self):
        self.__terminate_queue.put(0)

    def save_tracer(self):
        if self._trace_controller:
            self.__tracer_save_queue.put(0)

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
        prior_sched = PrioritizedSchedule(priority, self.schedule_count, sched)
        self.schedules.append(prior_sched)
        self.schedules.sort(reverse=True)
        self.schedule_count += 1
        return prior_sched

    def __check_and_issue_schedule(self) -> int:
        try:
            sched, priority = self.__schedule_queue.get(block=False)
            prior_sched = self.__issue_schedule(sched, priority)
            return prior_sched.index
        except queue.Empty:
            return None

    def __check_terminate(self):
        try:
            self.__terminate_queue.get(block=False)
            self.__terminated = True
        except queue.Empty:
            pass

    def __check_tracer_save(self):
        try:
            self.__tracer_save_queue.get(block=False)
            self.__save_tracer()
        except queue.Empty:
            pass

    def update_instruction_queues(self, to_update=None):
        """ core, manage instructions from all schedules, including following steps:
        1. fetch ready instructions from all schedules
        2. enqueue instructions to instruction queue with priority
        3. 
        """
        for prior_sched in self.schedules:
            sched = prior_sched.schedule
            if to_update is not None:
                if prior_sched.index not in to_update:
                    continue
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
                                raise RuntimeError(
                                    f"Binded instruction {b} of {inst} not ready. Current ready {ri}")
                # else:
                #     sched.update(stage_id)

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

    def poll_results(self):
        """Called by engine in every run step, check signals for instruction complete and 
        schedule stop. Check signal from all sources until nothing to receivce.

        Return: 
            int: number of instructions completed
        """
        completed = 0
        completed_scheds = set()
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
                completed_scheds.add(sched_id)
                if signal_code == 1:  # terminate
                    this_prior_sched.terminate_signal_count += 1
                    # print(f"terminate signal received {stage_id} {sched_id} {insts}")
                    sched.terminate_stage(stage_id)
                    # print(f"{this_prior_sched.index} count = {this_prior_sched.terminate_signal_count}")
                    if this_prior_sched.terminate_signal_count >= self.num_stages:
                        sched.terminate()
                        self.schedules.remove(this_prior_sched)
                        # print(f"removed sched {this_prior_sched.index}, schedules {len(self.schedules)}")
                elif signal_code == 0:  # normal instruction execute, nothing happens
                    pass
                else:
                    raise NotImplementedError(
                        f"Unknown signal code {signal_code} received when polling results.")
            except zmq.ZMQError:
                return completed, completed_scheds

    def init_tracer(self):
        import getpass
        import os

        import base.cluster
        if self._trace_controller:
            os.environ["DLLM_TRACE"] = "1"
        output_path = os.path.join(
            base.cluster.spec.fileroot,
            "logs",
            getpass.getuser(),
            self.__experiment_name,
            self.__trial_name,
            "trace_results",
            "controller.json",
        )
        self.tracer = get_tracer(
            # tracer_entries=int(5e6),
            # max_stack_depth=10,
            ignore_c_function=False,
            ignore_frozen=True,
            log_async=True,
            min_duration=15,
            output_file=output_path)
        self.tracer.start()

    def __save_tracer(self):
        if self.tracer is not None:
            logger.info("Saving tracer for controller")
            self.tracer.save()

    def run(self):
        self.__init_sockets()
        self.__init_storage()

        if self._trace_controller:
            self.init_tracer()  # TODO: trace

        while not self.__terminated:
            schedule_indices_to_update = set(
            )  # contains schedule indices that should be updated in this iteration
            schedule_indices_to_update.add(self.__check_and_issue_schedule())
            posted = self.post_instructions()
            # if posted > 0:
            #     print(f"schedule_controller.py: Posted {posted} instructions")
            # time.sleep(0.001)
            polled, completed_scheds = self.poll_results()
            schedule_indices_to_update.update(completed_scheds)
            # if polled > 0:
            #     print(f"Polled {polled} results")

            if len(schedule_indices_to_update) > 0:
                self.update_instruction_queues(schedule_indices_to_update)
            self.__posted += posted
            self.__polled += polled
            self.__check_terminate()
            if self._trace_controller:
                self.__check_tracer_save()  # TODO: trace


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
