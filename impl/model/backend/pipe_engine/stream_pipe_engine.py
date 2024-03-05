from collections import defaultdict
from types import MethodType
from typing import Callable, List, Optional, Tuple
import dataclasses
import gc
import os
import time

import torch
import transformers

from base.monitor import gpu_memory_mb
from impl.model.backend.pipe_engine.ds_pipe_engine import DeepSpeedPipelineEngine
from impl.model.backend.pipe_engine.dynamic_schedule import (DynamicPipeSchedule, GenerationSchedule,
                                                             InferenceSchedule, Train1F1BSchedule)
from impl.model.backend.pipe_engine.fast_schedule_controller import FastScheduleClient, FastScheduleController
from impl.model.backend.pipe_engine.message import SignalCode
from impl.model.backend.pipe_engine.schedule_controller import EngineScheduleClient, EngineScheduleController
from impl.model.nn.flash_mqat.flash_generate import GenerationConfig
from impl.model.parallelism.pipeline_parallel.tensor_storage import TensorBuffer
from impl.model.utils.data import PipeCacheData, PipeTransferData
import base.constants
import base.logging as logging

logger = logging.getLogger("StreamPipeEngine", "benchmark")


class EngineFuture:
    """A simple future used to retrive result of a StreamPipeEngine execution Result"""

    def __init__(self, req_id=None):
        self.__done = False
        self.__result = None
        self.__request_id = req_id

    def done(self):
        return self.__done

    def set_result(self, result):
        if self.__done:
            raise RuntimeError("Future result already set")
        self.__done = True
        self.__result = result

    def result(self):
        if not self.__done:
            raise RuntimeError("Future not done")
        return self.__result

    def request_id(self):
        return self.__request_id


class StreamPipeEngine(DeepSpeedPipelineEngine):

    def __init__(self, verbose=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__verbose = verbose

        assert base.constants.model_parallel_world_size() == 1, \
               "Currently stream pipe engine with tensor parallel has synchronization problem when multiple "\
               "schedules are issued. We have to keep the order between forward passes across tensor parallel ranks "\
               "or there will be deadlocks. Fixing it with global controller across tensor parallel ranks "\
               "and force the order between forward passes will probably cause severe performance issue."
        # TODO: FIX THIS
        self.pp_rank = base.constants.pipe_parallel_rank()

        self.engine_controller = None
        # self.engine_controller_started = False
        if self.pp_rank == 0:
            if self._use_fast_schedule_controller:
                logger.info("starting fast schedule controller")
                self.engine_controller = FastScheduleController(num_stages=self.num_stages)
                self.engine_controller.start()
            else:
                self.engine_controller = EngineScheduleController(num_stages=self.num_stages,
                                                                  trace=os.environ.get("DLLM_TRACE",
                                                                                       "0") == "1")
                self.engine_controller.start()

        if self._use_fast_schedule_controller:
            # if self.pp_rank == 0:
            #     self.engine_controller.check()
            logger.info("starting fast schedule client")
            self.engine_client = FastScheduleClient(stage_id=self.pp_rank)
        else:
            self.engine_client = EngineScheduleClient(stage_id=self.pp_rank)

        self.schedule_count = 0
        self.active_schedules = []
        self.finished_schedules = []
        self.schedule_mapping = dict()  # mapping from schedule index to corresponding schedule
        self.tensor_buffer_mapping = dict()  # mapping from schedule index to corresponding tensor buffer
        self.future_mapping = dict()  # mapping from schedule index to corresponding future
        self.num_micro_batches_mapping = dict()  # mapping from schedule index to num micro batches
        self.set_state_mapping = dict()
        self.result_collect_mapping = dict()

        # when a schedule is issued, self.generate_priority -= 1, self.train_priority += 1
        self.forward_priority = 0
        self.train_priority = 0
        self.generate_priority = 0

        self.train_sched_id = 0
        # debug
        self.run_count = 0
        self.executed = []

        self.instruction_buffer = None

        self.comm_handle_table = defaultdict(list)
        self.comm_instruction_table = defaultdict(list)

        self.controller_last_check = None
        self.sync_tensor = torch.tensor([0], device=self.device)

    def log_verbose(self, msg):
        if self.__verbose:
            logger.info(msg)

    def _forward_collect_result(self):
        logits = None
        if self.is_last_stage():
            logits_list = []
            for i in range(self.num_micro_batches):
                logits = self.tensor_buffer.get("logits", i, remove=True)
                # logger.info(f"mbid {i} before remove pad logits shape {logits.shape}")
                if self.sequence_parallel:
                    pad_size = self.tensor_buffer.get("pad_size", i, remove=True)
                    logits = logits[:-pad_size] if pad_size > 0 else logits
                    # logger.info(f"mbid {i} after remove pad {pad_size} logits shape {logits.shape}")
                logits_list.append(logits)
            logits = torch.cat(logits_list, dim=0)

        self._post_forward()
        return logits

    def _eval_batch_collect_result(self):
        avg_loss, avg_stats = None, None
        if self.is_last_stage():
            losses = []
            stats = []

            for mbid in range(self.num_micro_batches):
                loss = self.tensor_buffer.get("losses", mbid).detach()
                losses.append(loss)
                stats.append(self.tensor_buffer.get("stats", mbid))

            assert len(losses) > 0
            avg_loss = torch.stack(losses).mean()
            avg_stats = dict()
            for key in stats[0].keys():
                avg_stats[key] = torch.stack([stat[key] for stat in stats]).mean()

        self._post_eval_batch()
        return avg_loss, avg_stats

    def _train_batch_collect_result(self):
        avg_loss, avg_stats = None, None
        if self.is_last_stage():
            losses = []
            stats = []

            for mbid in range(self.num_micro_batches):
                loss = self.tensor_buffer.get("losses", mbid).detach()
                losses.append(loss)
                stats.append(self.tensor_buffer.get("stats", mbid))

            assert len(losses) > 0
            avg_loss = torch.stack(losses).mean()
            avg_stats = dict()
            for key in stats[0].keys():
                avg_stats[key] = torch.stack([stat[key] for stat in stats]).mean()

        self._post_train_batch()
        return avg_loss, avg_stats

    def _generate_collect_result(self):
        r = self._maybe_gather_generate_outputs()
        self._post_generate()
        return r

    def set_tensor_buffer(self, schedule_index: int):
        # print(f"setting tensor buffer {schedule_index}")
        self.tensor_buffer = self.tensor_buffer_mapping[schedule_index]

    def clear_tensor_buffer(self, schedule_index: int):
        self.tensor_buffer = None
        self.tensor_buffer_mapping[schedule_index] = None

    def set_num_micro_batches(self, num_micro_batches: int):
        self.num_micro_batches = num_micro_batches

    def start_schedule(self, sched: DynamicPipeSchedule, priority: int):
        sched_index = self.schedule_count
        sched.schedule_id = sched_index
        self.active_schedules.append(sched_index)
        self.schedule_mapping[sched_index] = sched
        self.tensor_buffer_mapping[sched_index] = TensorBuffer()
        # print(f"sched index {sched_index} tensor buffer init")

        if self.engine_controller is not None:
            if self._use_fast_schedule_controller:
                self.engine_client.issue_schedule(sched)
            else:
                self.engine_controller.issue_schedule(sched, priority)
            self.log_verbose(
                f"Issued schedule {sched_index} with priority {priority}, schedule type {sched.__class__.__name__}"
            )
        f = EngineFuture()
        self.future_mapping[sched_index] = f
        self.schedule_count += 1
        self.log_verbose(f"Rank {self.global_rank} started schedule {sched_index}.")
        return sched_index, f

    def end_schedule(self, schedule_index: int):
        self.tensor_buffer_mapping.pop(schedule_index)
        self.active_schedules.remove(schedule_index)
        self.finished_schedules.append(schedule_index)

        res = self.result_collect_mapping[schedule_index]()
        self.future_mapping[schedule_index].set_result(res)
        self.clear_tensor_buffer(schedule_index)
        self.future_mapping.pop(schedule_index)

    def forward(self,
                packed_input_ids: torch.Tensor,
                cu_seqlens: torch.Tensor,
                input_lens_for_partition: Optional[torch.Tensor] = None,
                num_micro_batches: Optional[int] = None) -> EngineFuture:
        num_micro_batches = num_micro_batches if num_micro_batches else self.num_stages
        sched = InferenceSchedule(num_micro_batches=num_micro_batches, num_stages=self.num_stages)
        self.set_num_micro_batches(num_micro_batches)
        sched_index, f = self.start_schedule(sched, self.forward_priority)
        self.set_state_mapping[sched_index] = self._set_forward_states
        self.result_collect_mapping[sched_index] = self._forward_collect_result
        self.num_micro_batches_mapping[
            sched_index] = num_micro_batches if num_micro_batches else self.num_stages
        self.set_tensor_buffer(sched_index)

        self._set_forward_states()
        self._prepare_input(packed_input_ids, cu_seqlens, input_lens_for_partition=input_lens_for_partition)
        self._pre_forward()

        return f

    def train_batch(self,
                    packed_input_ids: torch.Tensor,
                    cu_seqlens: torch.Tensor,
                    loss_fn: Callable,
                    input_lens_for_partition: Optional[torch.Tensor] = None,
                    num_micro_batches: Optional[int] = None,
                    **loss_fn_kwargs) -> EngineFuture:
        num_micro_batches = num_micro_batches if num_micro_batches else self.num_stages
        sched = Train1F1BSchedule(num_micro_batches=num_micro_batches,
                                  num_stages=self.num_stages,
                                  schedule_id=self.train_sched_id)
        self.train_sched_id += 1
        self.train_priority += 1
        sched_index, f = self.start_schedule(sched, self.train_priority)
        self.set_state_mapping[sched_index] = self._set_train_batch_states
        self.result_collect_mapping[sched_index] = self._train_batch_collect_result

        self.num_micro_batches_mapping[sched_index] = num_micro_batches
        self.set_num_micro_batches(num_micro_batches)

        self.set_tensor_buffer(sched_index)

        self._set_train_batch_states()
        self._prepare_input(packed_input_ids, cu_seqlens, input_lens_for_partition=input_lens_for_partition)
        self._loss_fn = loss_fn
        self._prepare_loss_input(**loss_fn_kwargs)
        self._pre_train_batch()

        return f

    @torch.no_grad()
    def generate(self,
                 packed_input_ids: torch.Tensor,
                 cu_seqlens: torch.Tensor,
                 tokenizer: transformers.PreTrainedTokenizerFast,
                 gconfig: GenerationConfig = dataclasses.field(default_factory=GenerationConfig),
                 num_micro_batches: Optional[int] = None) -> EngineFuture:
        # is_model_parallel = base.constants.model_parallel_world_size() > 1
        num_micro_batches = num_micro_batches if num_micro_batches else self.num_stages
        sched = GenerationSchedule(num_micro_batches=num_micro_batches,
                                   num_stages=self.num_stages,
                                   num_steps=gconfig.max_new_tokens,
                                   preserve_fwd_order=False)
        self.generate_priority -= 1
        sched_index, f = self.start_schedule(sched, self.generate_priority)
        # TODO: states: current_config, tokenizer, terminate_condition()
        self.set_state_mapping[sched_index] = self._set_generate_states
        self.result_collect_mapping[sched_index] = self._generate_collect_result

        num_micro_batches = num_micro_batches if num_micro_batches else self.num_stages
        self.num_micro_batches_mapping[sched_index] = num_micro_batches
        self.set_num_micro_batches(num_micro_batches)

        self.set_tensor_buffer(sched_index)

        self._set_generate_states()
        self._prepare_input(packed_input_ids, cu_seqlens)
        self.current_gconfig = gconfig
        self.tokenizer = tokenizer
        self._pre_generate()

        return f

    def eval_batch(self,
                   packed_input_ids: torch.Tensor,
                   cu_seqlens: torch.Tensor,
                   loss_fn: Callable,
                   input_lens_for_partition: Optional[torch.Tensor] = None,
                   num_micro_batches: Optional[int] = None,
                   **loss_fn_kwargs) -> EngineFuture:
        num_micro_batches = num_micro_batches if num_micro_batches else self.num_stages
        sched = InferenceSchedule(num_micro_batches=num_micro_batches, num_stages=self.num_stages)
        sched_index, f = self.start_schedule(sched, self.forward_priority)

        self.set_state_mapping[sched_index] = self._set_eval_batch_states
        self.result_collect_mapping[sched_index] = self._eval_batch_collect_result

        num_micro_batches = num_micro_batches if num_micro_batches else self.num_stages
        self.num_micro_batches_mapping[sched_index] = num_micro_batches
        self.set_num_micro_batches(num_micro_batches)

        self.set_tensor_buffer(sched_index)

        self._set_eval_batch_states()
        self._prepare_input(packed_input_ids, cu_seqlens, input_lens_for_partition=input_lens_for_partition)
        self._prepare_loss_input(**loss_fn_kwargs)
        self._loss_fn = loss_fn
        self._pre_eval_batch()

        return f

    def poll_one_step_py_controller(self):
        """ Run one step for stream pipe engine, including:
        1. poll instruction from engine controller;
        2. if instruction is not None, set tensor buffer, set state, set num micro batches, execute instruction;
        3. post result to engine controller;
        This method should be called by model worker.
        """
        assert not self._use_fast_schedule_controller

        if self.instruction_buffer is not None:
            sched_id, cmd, sched_end = self.instruction_buffer  # there is a chance when instruction from
            # schedule controller arrive before interface is executed on this engine,
            # in this case, store instruciton in buffer and wait for interface to be executed
        else:
            sched_id, cmd, sched_end = self.engine_client.poll_instruction()

        if sched_id is not None:
            if sched_id not in self.active_schedules + self.finished_schedules:
                self.instruction_buffer = (sched_id, cmd, sched_end)
                return

            signal_code = self.execute_instruction(cmd, sched_id)
            self.engine_client.post_result(cmd, sched_id, signal_code)

        finished = self.maybe_check_async_handles()
        for sched_id, instructions in finished.items():
            for instruction in instructions:
                self.engine_client.post_result(instruction, sched_id, SignalCode.COMM_EXEC)

    def poll_one_step_fast_controller(self):
        assert self._use_fast_schedule_controller
        self.engine_client: FastScheduleClient

        # check_interval = 10
        # if self.stage_id == 0:
        #     if self.controller_last_check is None:
        #         self.controller_last_check = time.time()
        #         self.engine_controller.check()
        #     elif time.time() - self.controller_last_check > check_interval:
        #         self.controller_last_check = time.time()
        #         self.engine_controller.check()

        self.engine_client.post()
        self.engine_client.poll()

        while not self.engine_client.empty():
            # execute all instructions stored in the poll message buffer
            sched_id, instruction = self.engine_client.pop()
            schedule = self.schedule_mapping[sched_id]

            signal_code = self.execute_instruction(instruction, sched_id)
            self.engine_client.post_result(instruction, schedule, signal_code)

        finished = self.maybe_check_async_handles()
        for sched_id, instructions in finished.items():
            for instruction in instructions:
                self.engine_client.post_result(instruction, self.schedule_mapping[sched_id],
                                               SignalCode.COMM_EXEC)

    def execute_instruction(self, instruction, sched_id):
        self.set_tensor_buffer(sched_id)
        # self.rank_print(f"setting tensor buffer for schedule {sched_id}")
        self.set_state_mapping[sched_id]()
        # self.rank_print(f"setting state for schedule {sched_id}")
        num_micro_batches = self.num_micro_batches_mapping[sched_id]
        self.set_num_micro_batches(num_micro_batches)

        if type(instruction) not in self._INSTRUCTION_MAP:
            raise RuntimeError(
                f'{self.__class__.__name__} does not understand instruction {repr(instruction)}')

        try:
            self._exec_instr = MethodType(self._INSTRUCTION_MAP[type(instruction)], self)
            self.log_verbose(f"Rank {self.global_rank}: START cmd {instruction} of sched {sched_id}")
            exec_end, comm_handle = self._exec_instr(*instruction.args)
            self.log_verbose(f"Rank {self.global_rank}: END cmd {instruction} of sched {sched_id}")
            # if comm_handle is None:
            torch.cuda.synchronize()
            # self.sync_tensor.to("cpu")
            # self.sync_tensor.to(self.device)
        except Exception as e:
            logger.error(f"Rank {self.global_rank} Exception {e} in cmd {instruction}")
            raise e

        signal_code = SignalCode.END if exec_end else SignalCode.EXEC
        # if signal_code == 1:
        #     logger.info(f"Rank {self.global_rank}: END sched {sched_id}, end cmd {cmd}")

        if self._async_instruction and not comm_handle is None:
            self.comm_handle_table[sched_id].append(comm_handle)
            self.comm_instruction_table[sched_id].append(instruction)
            signal_code = SignalCode.HOLD
            self.log_verbose(
                f"Rank {self.global_rank} sched id {sched_id} record handle {comm_handle} cmd {instruction}")

        if signal_code == SignalCode.END and sched_id in self.active_schedules:
            self.end_schedule(sched_id)

        return signal_code

    def maybe_check_async_handles(self):
        # check whether handle finished, if finished, pop. return a dict mapping schedule id to finished
        # comm instructions
        res = defaultdict(list)
        if self._async_instruction:
            # check handles
            for sched_id in self.active_schedules:
                handle_list = self.comm_handle_table[sched_id]
                inst_list = self.comm_instruction_table[sched_id]
                new_handle_list = []
                new_inst_list = []
                # print(f"sched_id {sched_id} length comm handle list {len(handle_list)} comm inst list {len(inst_list)}")
                for handle, cmd in zip(handle_list, inst_list):
                    # print(f"Rank {self.global_rank} handle {handle} cmd {cmd} check")
                    if handle.is_completed():
                        self.log_verbose(f"Rank {self.global_rank} handle {handle} cmd {cmd} completed, post")
                        # self.engine_client.post_result(cmd, sched_id, SignalCode.COMM_EXEC)
                        res[sched_id].append(cmd)
                    else:
                        # print(f"Rank {self.global_rank} handle {handle} cmd {cmd} NOT completed")
                        new_handle_list.append(handle)
                        new_inst_list.append(cmd)
                self.comm_handle_table[sched_id] = new_handle_list
                self.comm_instruction_table[sched_id] = new_inst_list
        return res

    def poll_one_step(self):
        return self.poll_one_step_py_controller() if not self._use_fast_schedule_controller \
               else self.poll_one_step_fast_controller()

    def print_executed(self):
        s = f"Rank {self.global_rank}: "
        for sched_id, cmd in self.executed:
            s += f"S{sched_id}::{cmd} - "
        print(s)

    def stop_controller(self):
        if self.engine_controller is not None:
            self.engine_controller.stop()

    def save_tracer(self):
        if self.engine_controller is not None:
            self.engine_controller.save_tracer()
