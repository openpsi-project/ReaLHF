from collections import defaultdict
from types import MethodType
from typing import Callable, List, Optional, Tuple
import dataclasses
import gc

import torch
import transformers

from base.monitor import gpu_memory_mb
from impl.model.backend.pipe_engine.ds_pipe_engine import DeepSpeedPipelineEngine
from impl.model.backend.pipe_engine.dynamic_schedule import (DynamicPipeSchedule, GenerationSchedule,
                                                             InferenceSchedule, Train1F1BSchedule)
from impl.model.backend.pipe_engine.schedule_controller import EngineScheduleClient, EngineScheduleController
from impl.model.nn.flash_mqat.flash_generate import GenerationConfig
from impl.model.utils.data import PipeCacheData, PipeTransferData
from impl.model.utils.tensor_storage import TensorBuffer
import base.constants
import base.logging as logging

logger = logging.getLogger("StreamPipeEngine", "benchmark")


class EngineFuture:
    """A simple future used to retrive result of a StreamPipeEngine execution Result"""

    def __init__(self):
        self.__done = False
        self.__result = None

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


class StreamPipeEngine(DeepSpeedPipelineEngine):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pp_rank = base.constants.pipe_parallel_rank()
        self.engine_controller = None
        # self.engine_controller_started = False
        if self.pp_rank == 0:
            self.engine_controller = EngineScheduleController(num_stages=self.num_stages)
            self.engine_controller.start()

        self.engine_client = EngineScheduleClient(stage_id=self.pp_rank)
        self.schedule_count = 0
        self.active_schedules = []
        self.finished_schedules = []
        self.tensor_buffer_mapping = dict()  # mapping from schedule index to corresponding tensor buffer
        self.future_mapping = dict()  # mapping from schedule index to corresponding future
        self.set_state_mapping = dict()
        self.result_collect_mapping = dict()

        # make these configurable
        self.forward_priority = 1
        self.train_priority = 2
        self.generate_priority = 0

        self.train_sched_id = 0
        # debug
        self.run_count = 0
        self.executed = []

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

    def start_schedule(self, sched: DynamicPipeSchedule, priority: int):
        sched_index = self.schedule_count
        self.active_schedules.append(sched_index)
        self.tensor_buffer_mapping[sched_index] = TensorBuffer()
        # print(f"sched index {sched_index} tensor buffer init")

        if self.engine_controller is not None:
            self.engine_controller.issue_schedule(sched, priority)
        f = EngineFuture()
        self.future_mapping[sched_index] = f
        self.schedule_count += 1
        return sched_index, f

    def end_schedule(self, schedule_index: int):
        self.tensor_buffer_mapping.pop(schedule_index)
        self.active_schedules.remove(schedule_index)
        self.finished_schedules.append(schedule_index)

    def forward(self,
                packed_input_ids: torch.Tensor,
                cu_seqlens: torch.Tensor,
                input_lens_for_partition: Optional[torch.Tensor] = None) -> EngineFuture:
        sched = InferenceSchedule(num_micro_batches=self.num_micro_batches, num_stages=self.num_stages)
        sched_index, f = self.start_schedule(sched, self.forward_priority)
        self.set_state_mapping[sched_index] = self._set_forward_states
        self.result_collect_mapping[sched_index] = self._forward_collect_result
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
                    **loss_fn_kwargs) -> EngineFuture:
        sched = Train1F1BSchedule(num_micro_batches=self.num_micro_batches,
                                  num_stages=self.num_stages,
                                  sched_id=self.train_sched_id)
        self.train_sched_id += 1
        sched_index, f = self.start_schedule(sched, self.train_priority)
        self.set_state_mapping[sched_index] = self._set_train_batch_states
        self.result_collect_mapping[sched_index] = self._train_batch_collect_result
        self.set_tensor_buffer(sched_index)

        self._set_train_batch_states()
        self._prepare_input(packed_input_ids, cu_seqlens, input_lens_for_partition=input_lens_for_partition)
        self._loss_fn = loss_fn
        self._prepare_loss_input(**loss_fn_kwargs)
        self._pre_train_batch()

        return f

    @torch.no_grad()
    def generate(
            self,
            packed_input_ids: torch.Tensor,
            cu_seqlens: torch.Tensor,
            tokenizer: transformers.PreTrainedTokenizerFast,
            gconfig: GenerationConfig = dataclasses.field(default_factory=GenerationConfig),
    ) -> EngineFuture:
        sched = GenerationSchedule(num_micro_batches=self.num_micro_batches,
                                   num_stages=self.num_stages,
                                   num_steps=gconfig.max_new_tokens,
                                   steps_per_update=5,
                                   sched_id=99)
        sched_index, f = self.start_schedule(sched, self.generate_priority)
        # TODO: states: current_config, tokenizer, terminate_condition()
        self.set_state_mapping[sched_index] = self._set_generate_states
        self.result_collect_mapping[sched_index] = self._generate_collect_result
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
                   **loss_fn_kwargs) -> EngineFuture:
        sched = InferenceSchedule(num_micro_batches=self.num_micro_batches, num_stages=self.num_stages)
        sched_index, f = self.start_schedule(sched, self.forward_priority)
        self.set_state_mapping[sched_index] = self._set_eval_batch_states
        self.result_collect_mapping[sched_index] = self._eval_batch_collect_result
        self.set_tensor_buffer(sched_index)

        self._set_eval_batch_states()
        self._prepare_input(packed_input_ids, cu_seqlens, input_lens_for_partition=input_lens_for_partition)
        self._prepare_loss_input(**loss_fn_kwargs)
        self._loss_fn = loss_fn
        self._pre_eval_batch()

        return f

    def run(self):
        # if not self.engine_controller_started and self.engine_controller is not None:
        #     self.engine_controller.start()
        #     self.engine_controller_started = True

        sched_id, cmd, sched_end = self.engine_client.poll_instruction()
        if sched_id is not None:
            # self.rank_print(
            #     f"received instruction sched id {sched_id} cmd {cmd} end {sched_end}"
            # )
            # sched_id, cmd, sched_end = r
            self.set_tensor_buffer(sched_id)
            # self.rank_print(f"setting tensor buffer for schedule {sched_id}")
            self.set_state_mapping[sched_id]()
            # self.rank_print(f"setting state for schedule {sched_id}")

            if type(cmd) not in self._INSTRUCTION_MAP:
                raise RuntimeError(f'{self.__class__.__name__} does not understand instruction {repr(cmd)}')

            try:
                self._exec_instr = MethodType(self._INSTRUCTION_MAP[type(cmd)], self)
                self.rank_print(f"START cmd {cmd} of sched {sched_id}")
                # self.executed.append((sched_id, cmd))
                # self.print_executed()
                exec_end = self._exec_instr(*cmd.args, **cmd.kwargs)
                self.rank_print(f"END cmd {cmd} of sched {sched_id}")
            except Exception as e:
                logger.error(f"Rank {self.global_rank} Exception {e} in cmd {cmd}")
                raise e

            signal_code = 1 if exec_end else 0
            self.engine_client.post_result(signal_code)

            if exec_end:
                # print(f"end sched id {sched_id}")
                self.end_schedule(sched_id)
                res = self.result_collect_mapping[sched_id]()
                self.future_mapping[sched_id].set_result(res)
                gpu_memory_mb(f"before clear tensor buffer {sched_id}")
                self.clear_tensor_buffer(sched_id)
                gpu_memory_mb(f"after clear tensor buffer {sched_id}")
                # gc.collect()
                # torch.cuda.empty_cache()
                # gc.collect()

        # self.run_count += 1
        # if self.run_count % 10000 == 0:
        #     logger.info("Rank {} run count {}".format(self.global_rank, self.run_count))

    def print_executed(self):
        s = f"Rank {self.global_rank}: "
        for sched_id, cmd in self.executed:
            s += f"S{sched_id}::{cmd} - "
        print(s)

    def stop_controller(self):
        if self.engine_controller is not None:
            self.engine_controller.stop()
