# fast schedule and schedule controller implemented in C++
import os
import queue
import socket
import subprocess
import time

from impl.model.backend.pipe_engine.dynamic_schedule import DynamicPipeSchedule
from impl.model.backend.pipe_engine.instruction import PipeInstruction
import base.constants
import base.logging as logging
import base.name_resolve as name_resolve
import base.names as names
import base.network as network
import impl.model.backend.pipe_engine.message as message

logger = logging.getLogger("FastScheduleController", "benchmark")
SERVER_BINARY_PATH = os.path.join(os.path.dirname(__file__), "cpp_server/server")


class FastScheduleController:

    def __init__(self, num_stages):
        self.num_stages = num_stages
        self.__expr_name = base.constants.experiment_name()
        self.__trial_name = base.constants.trial_name()
        self.__model_name = base.constants.model_name()
        self.__dp_rank = base.constants.data_parallel_rank()
        self.__mp_rank = base.constants.model_parallel_rank()

        self.__port = network.find_free_port()
        self.__address = network.gethostip()

    def launch(self):
        name = names.model_controller(self.__expr_name, self.__trial_name, self.__model_name, self.__dp_rank,
                                      self.__mp_rank)

        name_resolve.add(name, f"{self.__address}:{self.__port}", keepalive_ttl=30)
        self.__process = subprocess.Popen(
            [SERVER_BINARY_PATH, str(self.__port), str(self.num_stages)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True)

        name = names.model_controller_barrier(self.__expr_name, self.__trial_name, self.__model_name,
                                              self.__dp_rank, self.__mp_rank)
        while (len(name_resolve.get_subtree(name)) < self.num_stages):
            time.sleep(0.1)

    def check(self):
        return_code = self.__process.poll()
        if return_code:
            raise subprocess.CalledProcessError(return_code, self.__process.args)
        else:
            logger.info("FastScheduleController stdout:")
            for line in self.__process.stdout:
                logger.info(line.strip().strip("\n"))
            logger.error("FastScheduleController stderr:")
            for line in self.__process.stderr:
                logger.error(line.strip().strip("\n"))

            if return_code == 0:
                logger.info(f"FastScheduleController terminated with returncode 0.")

            return return_code

    def stop(self):
        self.__process.terminate()
        final_output, final_error = self.__process.communicate()
        logger.info(f"FastScheduleController terminated with returncode {self.__process.returncode}:"
                    f"\nOutput: {final_output} \nError: {final_error}")
        return self.__process.returncode


class FastScheduleClient:

    def __init__(self, stage_id: int):
        self.stage_id = stage_id
        expr_name = base.constants.experiment_name()
        trial_name = base.constants.trial_name()
        model_name = base.constants.model_name()
        dp_rank = base.constants.data_parallel_rank()
        mp_rank = base.constants.model_parallel_rank()

        name = names.model_controller(expr_name, trial_name, model_name, dp_rank, mp_rank)
        address, port = name_resolve.wait(name, timeout=30).split(":")

        self.__socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__socket.connect((address, int(port)))
        logger.info(f"Engine ({stage_id}, {dp_rank}, {mp_rank}) Connected to {address}:{port}")

        name = names.model_controller_barrier(expr_name, trial_name, model_name, dp_rank, mp_rank)
        name_resolve.add_subentry(name, self.stage_id)

        self.post_message_buffer = queue.Queue(128)
        self.poll_message_buffer = queue.Queue(128)

    def issue_schedule(self, sched: DynamicPipeSchedule):
        if self.stage_id == 0:
            sched_msg = message.schedule_to_message(sched)
            msg = message.Message(self.stage_id, 0, message.MessageType.ISSUE_SCHEUDLE, None, sched_msg)
            self.post_message_buffer.put(msg)

    def post_result(self, inst: PipeInstruction, sched: DynamicPipeSchedule, signal_code: message.SignalCode):
        inst_msg = message.instruction_to_message(inst)
        sched_msg = message.schedule_to_message(sched)
        msg = message.Message(self.stage_id, signal_code, message.MessageType.POLL_RESULT, inst_msg,
                              sched_msg)
        self.post_message_buffer.put(msg)

    def post(self):
        if self.post_message_buffer:
            msg_array = message.MessageArray(len(self.post_message_buffer), self.post_message_buffer)
        else:
            msg_array = message.MessageArray(
                1, [message.Message(self.stage_id, 0, message.MessageType.NOOP, None, None)])
        self.__socket.sendall(msg_array.serialize())

    def poll(self):
        data = self.__socket.recv(1024)
        msg_array: message.MessageArray = message.MessageArray.deserialize(data)
        if msg_array.n == 1 and msg_array.messages[0].message_type == message.MessageType.NOOP:
            return
        for m in msg_array.messages:
            self.poll_message_buffer.put(m)

    def empty(self):
        return self.poll_message_buffer.empty()

    def pop(self):
        # pop one instruction to be executed
        msg: message.Message = self.poll_message_buffer.get()
        instruction = message.message_to_instruction(msg.instruction)
        schedule_id = msg.instruction.schedule_id
        return schedule_id, instruction
