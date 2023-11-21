# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import MethodType
from typing import Callable, List, Optional, Tuple, Union
import dataclasses
import logging
import time

from deepspeed import comm as dist
from deepspeed.runtime.engine import DeepSpeedEngine, MEMORY_OPT_ALLREDUCE_SIZE
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.utils import logger
import torch

from base.dataparallel import PackedParallelDataBroker
from base.monitor import gpu_memory_mb, time_mark
from base.namedarray import NamedArray
from impl.model.backend.stream_pipe_engine.module import PipelineError, PipelineModule
from impl.model.backend.stream_pipe_engine.tensor_utils import (recv_grad, recv_pipe_transfer_data, send_grad,
                                                                send_pipe_transfer_data, TensorBuffer)
from impl.model.utils.data import PipeCacheData, PipeTransferData
import impl.model.backend.stream_pipe_engine.p2p as p2p
import impl.model.backend.stream_pipe_engine.schedule as schedule

logger = logging.getLogger("StreamPipeEngine")


def is_even(number):
    return number % 2 == 0


class StreamPipeEngine(DeepSpeedEngine):
    """ A training engine hybrid pipeline, data, and model parallel training.

    This engine is created by ``deepspeed.initialize()`` when a :class:`PipelineModule`
    is provided.
    """
    ID_TO_DTYPE = [
        torch.float32, torch.float64, torch.complex64, torch.complex128, torch.float16, torch.bfloat16,
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.bool
    ]
    DTYPE_TO_ID = {dtype: id_ for id_, dtype in enumerate(ID_TO_DTYPE)}

    def __init__(self, expr_config, num_micro_batches=None, *super_args, **super_kwargs):
        super().__init__(*super_args, **super_kwargs)
        assert isinstance(self.module, PipelineModule), "model must base PipelineModule"
        self.pipeline_parallelism = True  # deepspeed engine config
        self.config = self.module.config  # FlashMQATConfig in PipelineModule
        # for tensor buffer initialization
        self.hidden_dim = self.config.hidden_dim
        self.n_kv = self.config.n_kv_heads
        # self.bfloat16_enabled()
        # TODO: temporary for debugging, should be passed in from experiment configs!!
        # self.max_seq_len = 512
        # self.max_new_tokens = 512
        # self.max_mb_size = 8
        self.expr_name = expr_config.experiment_name
        self.trial_name = expr_config.trial_name
        self.model_name = expr_config.model_name

        assert self.zero_optimization_stage(
        ) < 2, "ZeRO-2 and ZeRO-3 are incompatible with pipeline parallelism"

        # We schedule the all-reduces, so disable it in super().backward()
        self.enable_backward_allreduce = False
        self.sched_count = 0

        # Set Grid and Communication Groups
        self.grid = self.module._grid

        self.global_rank = self.grid.get_global_rank()

        assert self.dp_world_size == self.grid.data_parallel_size
        self.num_stages = self.grid.pipe_parallel_size
        self.stage_id = self.grid.get_stage_id()
        self.dp_id = self.grid.data_parallel_id
        self.mp_id = 0

        self.micro_batch_size = self.train_micro_batch_size_per_gpu()
        self.num_micro_batches = num_micro_batches if num_micro_batches else 2 * self.num_stages
        self.num_layers = self.module.num_layers  # number of leyers in current pipeline stage
        self._force_grad_boundary = False

        # PipelineEngine needs to handle data loading specially due to only the first
        # and last stages loading inputs/labels. We construct a sampler that uses

        self.is_pipe_parallel = self.grid.pipe_parallel_size > 1
        self.is_data_parallel = self.grid.data_parallel_size > 1
        self.is_model_parallel = self.grid.model_parallel_size > 1

        model_parameters = filter(lambda p: p.requires_grad, self.module.parameters())
        num_params = sum([p.numel() for p in model_parameters])
        unique_params = num_params
        # Subtract tied parameters if we don't own them
        if self.module.tied_comms:
            tied_params = 0
            for key, d in self.module.tied_comms.items():
                if self.global_rank != min(d['ranks']):
                    tied_params += sum(p.numel() for p in d['module'].parameters())
            unique_params -= tied_params

        params_tensor = torch.LongTensor(data=[num_params, unique_params]).to(self.device)
        dist.all_reduce(params_tensor, group=self.grid.get_model_parallel_group())
        params_tensor = params_tensor.tolist()
        total_params = params_tensor[0]
        unique_params = params_tensor[1]

        if self.grid.data_parallel_id == 0:
            logger.info(f'RANK={self.global_rank} '
                        f'STAGE={self.stage_id}/{self.num_stages} \n'
                        f'LAYERS={self.module._local_stop - self.module._local_start} '
                        f'[{self.module._local_start}, {self.module._local_stop}) \n'
                        f'STAGE_PARAMS={num_params} ({num_params/1e6:0.3f}M) '
                        f'TOTAL_PARAMS={total_params} ({total_params/1e6:0.3f}M) '
                        f'UNIQUE_PARAMS={unique_params} ({unique_params/1e6:0.3f}M) \n'
                        f"MICRO_BATCHES={self.num_micro_batches} "
                        f"MICRO_BATCH_SIZE={self.micro_batch_size} \n")

        # initialize peer-2-peer communication and allreduce groups
        if self.is_pipe_parallel:
            p2p.init_process_groups(self.grid)

        #stores the loss for the current micro batch being processed
        self.loss = torch.tensor(0.0).to(self.device)
        # stats for microbatches in this batch
        self.stats = []

        # loss related
        self._compute_loss = False
        self._loss_fn = None
        self._loss_inputs = []
        self._input_cache = []

        # optimizer lr scheduler variables
        self.version_steps = 0

        self.tensor_buffer = TensorBuffer()
        self.if_activation_inited = False
        self.if_grad_inited = False
        self.if_generate_inited = False

        self.prev_stage = (self.stage_id - 1) % self.num_stages
        self.next_stage = (self.stage_id + 1) % self.num_stages
        self._initialize_p2p()

        self.sched_controller = None
        self.sched_client = None
        self.instruction_queue = []
        self.__init_schedule()

    def __init_schedule(self):
        """ Initialize EngineScheduleClient, and EngineScheduleController if pp_rank = 0;
        """
        if self.stage_id == 0:
            self.sched_controller = schedule.EngineScheduleController(self.expr_name, self.trial_name,
                                                                      self.model_name, self.dp_id, self.mp_id)
        self.sched_client = schedule.EngineScheduleClient(self.expr_name, self.trial_name, self.model_name,
                                                          self.stage_id, self.dp_id, self.mp_id)

    def set_loss_fn(self, fn):
        self._loss_fn = fn

    def set_version_steps(self, version_steps):
        # version_steps = batch id (not micro batch !!)
        self.version_steps = version_steps

    def _initialize_p2p(self):
        # check connectivity
        buf = torch.zeros(1, dtype=torch.int32).cuda()
        if is_even(self.stage_id):
            if not self.is_last_stage():
                p2p.send(buf, self.next_stage)
            if not self.is_first_stage():
                p2p.recv(buf, self.prev_stage)
        else:
            if not self.is_first_stage():
                p2p.recv(buf, self.prev_stage)
            if not self.is_last_stage():
                p2p.send(buf, self.next_stage)

        if self.is_first_stage():
            p2p.recv(buf, self.prev_stage)
        if self.is_last_stage():
            p2p.send(buf, self.next_stage)

    def _prepare_input(self,
                       packed_input_ids: torch.Tensor,
                       cu_seqlens: torch.Tensor,
                       generate: bool = False):
        """ Prepare input for train or inference
        split all input tensors into micro batches for pipeline parallel

        Args:
            packed_input_ids (torch.Tensor): packed input ids of shape [total_seq_len]
            cu_seqlens (torch.Tensor): cu_seqlens of shape [batch_size]
        """
        data = NamedArray(packed_input_ids=packed_input_ids, cu_seqlens=cu_seqlens)
        splitted = PackedParallelDataBroker.scatter_to(data, self.num_micro_batches)
        self._input_cache = splitted
        mb_seq_lens = []

        def input_to_pipe_model_input(input: NamedArray):
            max_seqlen = torch.tensor(int(max(input.cu_seqlens[1:] - input.cu_seqlens[:-1]))).cuda()
            store_kvcache = torch.tensor(1).cuda() if generate else torch.tensor(0).cuda()

            cu_seqlens = input.cu_seqlens.to(self.device)
            packed_input_ids = input.packed_input_ids.to(self.device)

            x = PipeTransferData(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, store_kvcache=store_kvcache)
            if self.is_first_stage():
                ys = [PipeCacheData(input_ids=packed_input_ids)
                      ] + [PipeCacheData() for _ in range(self.num_layers - 1)]
            else:
                ys = [PipeCacheData() for _ in range(self.num_layers)]
            mb_seq_lens.append(packed_input_ids.shape[0])
            return (x, ys)

        batches = [input_to_pipe_model_input(x) for x in splitted]
        batch_lengths = []
        for b in batches:
            cu_seqlens = b[0].cu_seqlens
            batch_lengths.append(cu_seqlens.shape[0] - 1)
        # batch_lengths = [b[1][0].input_ids.shape[0] for b in batches]
        logger.debug("self._prepare_input:: batch_lengths: {}".format(batch_lengths))

        # pre allocate receive buffers
        for i in range(self.num_micro_batches):
            dtype = torch.half if not self.bfloat16_enabled() else torch.bfloat16
            activation_shape = (mb_seq_lens[i], self.hidden_dim)
            self.tensor_buffer.alloc("activation",
                                     i,
                                     activation_shape,
                                     dtype,
                                     self.device,
                                     require_grads=True)
            self.tensor_buffer.alloc("grad", i, activation_shape, dtype, self.device)
            others_cache = dict(cu_seqlens=batches[i][0].cu_seqlens,
                                max_seqlen=batches[i][0].max_seqlen,
                                store_kvcache=batches[i][0].store_kvcache)
            self.tensor_buffer.put_non_tensor("pipe_transfer_infos", i, others_cache)

        return iter(batches)

    def _prepare_loss_input(self, **loss_kwargs):
        data = NamedArray(**loss_kwargs)
        splitted = PackedParallelDataBroker.scatter_to(data, self.num_micro_batches)
        self._loss_inputs = splitted

    def eval(self):
        self.module.eval()

    def train(self):
        self.module.train()

    def train_batch(self, packed_input_ids: torch.Tensor, cu_seqlens: torch.Tensor, loss_fn: Callable,
                    **loss_fn_kwargs):
        # self.__init_train_batch_tensor_buffer()
        if not torch._C.is_grad_enabled():
            raise RuntimeError(f'train_batch() requires gradients enabled. Use eval_batch() instead.')

        time_mark("Prepare_start", str(self.global_rank), self.sched_count)
        data_iter = self._prepare_input(packed_input_ids, cu_seqlens)
        self.set_loss_fn(loss_fn)
        self._prepare_loss_input(**loss_fn_kwargs)
        self.set_dataiterator(data_iter)
        time_mark("Prepare_end", str(self.global_rank), self.sched_count)

        self.total_loss = None
        self._compute_loss = True

        # Do the work
        sched = schedule.TrainSchedule(micro_batches=self.num_micro_batches,
                                       stages=self.num_stages,
                                       stage_id=self.stage_id)
        self._exec_schedule(sched)

        return self.stats

    def is_first_stage(self):
        """True if this process is in the first stage in the pipeline."""
        return self.stage_id == 0

    def is_last_stage(self):
        """True if this process is in the last stage in the pipeline."""
        return self.stage_id == self.num_stages - 1

    def set_dataiterator(self, iterator):
        """ Store an iterator to sample for training data. """
        if self.is_first_stage() or self.is_last_stage():
            self.training_dataloader = None
            self.data_iterator = iterator

    def is_gradient_accumulation_boundary(self):
        """True if the engine is executing a gradient reduction or optimizer step instruction.

        This is overridden from :class:`DeepSpeedEngine` to force reductions
        and steps when the pipeline engine is instructed to do so.

        Returns:
            bool: whether reductions and optimizer steps should occur.
        """
        return self._force_grad_boundary

    def _next_batch(self):
        # If using 3D parallelism, only some first-stage ranks may do IO
        batch = None
        if self.data_iterator is not None:
            batch = next(self.data_iterator)
        return batch

    def _exec_forward_pass(self, stage_id: int, micro_batch_id: int, step_id: int):
        x = self.tensor_buffer.get_non_tensor("batch_input_x", micro_batch_id, remove=True)
        try:
            ys = self.tensor_buffer.get_non_tensor("batch_input_ys", micro_batch_id, remove=False)
        except KeyError:
            ys = [PipeCacheData() for _ in range(self.num_layers)]

        self._zero_grads(x)
        self._zero_grads(ys)

        x, ys = super().forward(x, ys)

        if self.is_last_stage():
            if self._compute_loss:  # 1f1b only
                logits = x.pp_input
                logger.info(f"logits shape {logits.shape}")
                loss_kwargs = self._loss_inputs.pop(0)
                input_cache = self._input_cache.pop(0)
                packed_input_ids = input_cache.packed_input_ids
                cu_seqlens = input_cache.cu_seqlens
                assert self._loss_fn is not None, "loss function is not set, please use engine.set_loss_fn(fn)"

                self.loss, stats = self._loss_fn(logits, packed_input_ids, cu_seqlens, **loss_kwargs)
                # print(f"calculate loss {logits.shape} {packed_input_ids.shape} {cu_seqlens.shape}")
                self.stats.append(stats)
                if self.total_loss is None:
                    self.total_loss = torch.zeros_like(self.loss)
                self.total_loss += self.loss.detach()
            else:
                self.fwd_outputs.append(x)

        self.tensor_buffer.put_non_tensor("batch_output_x", micro_batch_id, x)  # send activation

    def _exec_backward_pass(self, stage_id: int, micro_batch_id: int, step_id: int):
        assert self.optimizer is not None, "must provide optimizer during " \
                                           "init in order to use backward"
        # The last stage just runs backward on the loss using DeepSpeed's typical
        # mechanisms.
        if self.is_last_stage():
            super().backward(self.loss)
            return

        if self.bfloat16_enabled() and not self.is_last_stage():
            # manually call because we don't call optimizer.backward()
            self.optimizer.clear_lp_grads()

        grad = self.tensor_buffer.get("grad", micro_batch_id, remove=True)
        output_x = self.tensor_buffer.get_non_tensor("batch_output_x", micro_batch_id, remove=True)
        output_tensor = output_x.pp_input
        torch.autograd.backward(tensors=output_tensor, grad_tensors=grad)

    def _exec_load_micro_batch(self, stage_id: int, micro_batch_id: int, step_id: int):
        assert micro_batch_id >= 0
        if self.is_first_stage():
            x, ys = self._next_batch()
            self.tensor_buffer.put_non_tensor("batch_input_x", micro_batch_id, x)
            self.tensor_buffer.put_non_tensor("batch_input_ys", micro_batch_id, ys)

    def _exec_send_activations(self, stage_id: int, micro_batch_id: int, step_id: int):
        x = self.tensor_buffer.get_non_tensor("batch_output_x", micro_batch_id)
        send_pipe_transfer_data(x, self.next_stage)

    def _exec_recv_activations(self, stage_id: int, micro_batch_id: int, step_id: int):
        buf = self.tensor_buffer.get("activation", micro_batch_id, remove=False)
        others = self.tensor_buffer.get_non_tensor("pipe_transfer_infos", micro_batch_id, remove=False)
        x = recv_pipe_transfer_data(buf, self.prev_stage, others)
        self.tensor_buffer.put_non_tensor("batch_input_x", micro_batch_id, x)

    def _exec_send_grads(self, stage_id: int, micro_batch_id: int, step_id: int):
        # x: PipeTransferData = self.tensor_buffer.get_non_tensor("batch_input_x", micro_batch_id, remove=True)
        # activation = x.pp_input
        activation = self.tensor_buffer.get("activation", micro_batch_id, remove=False)
        assert activation.grad is not None
        send_grad(activation.grad, self.prev_stage)

    def _exec_recv_grads(self, stage_id: int, micro_batch_id: int, step_id: int):
        buf = self.tensor_buffer.get("grad", micro_batch_id)
        recv_grad(buf, self.next_stage)

    def _exec_optimizer_step(self, stage_id: int, micro_batch_id: int, step_id: int):
        self._force_grad_boundary = True
        self._take_model_step(lr_kwargs={'epoch': self.version_steps})
        self._force_grad_boundary = False

    def _exec_reduce_grads(self, stage_id: int, micro_batch_id: int, step_id: int):
        self._force_grad_boundary = True
        if self.bfloat16_enabled():
            if self.zero_optimization_stage() < ZeroStageEnum.gradients:
                self._bf16_reduce_grads()
            else:
                raise NotImplementedError("PP+BF16 only work for ZeRO Stage 1")
        else:
            self.allreduce_gradients(bucket_size=MEMORY_OPT_ALLREDUCE_SIZE)
        self._force_grad_boundary = False

    def _bf16_reduce_grads(self):
        # Make our own list of gradients from the optimizer's FP32 grads
        grads = []
        self.buffered_allreduce_fallback(grads=self.optimizer.get_grads_for_reduction(),
                                         elements_per_buffer=MEMORY_OPT_ALLREDUCE_SIZE)

    def _zero_grads(self, inputs):
        if isinstance(inputs, torch.Tensor):
            if inputs.grad is not None:
                inputs.grad.data.zero_()
        elif isinstance(inputs, tuple):
            for t in inputs:
                if t.grad is not None:
                    t.grad.data.zero_()
        elif dataclasses.is_dataclass(inputs):
            for f in dataclasses.fields(inputs):
                self._zero_grads(getattr(inputs, f.name))
        else:
            # do nothing for non tensor
            pass

    def init_stream_generate(self, packed_input_ids: torch.Tensor, cu_seqlens: torch.Tensor):
        pass

    def forward(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``eval_batch()``. """
        raise PipelineError("Only eval_batch() is accessible in pipeline mode.")

    def backward(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def step(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    # A map of PipeInstruction types to methods. Each method will be executed with the
    # kwargs provided to the PipeInstruction from the scheduler.
    _INSTRUCTION_MAP = {
        schedule.OptimizerStep: _exec_optimizer_step,
        schedule.ReduceGrads: _exec_reduce_grads,
        schedule.LoadMicroBatch: _exec_load_micro_batch,
        schedule.ForwardPass: _exec_forward_pass,
        schedule.BackwardPass: _exec_backward_pass,
        schedule.SendActivation: _exec_send_activations,
        schedule.RecvActivation: _exec_recv_activations,
        schedule.SendGrad: _exec_send_grads,
        schedule.RecvGrad: _exec_recv_grads,
        # schedule.
    }

    def _exec_schedule(self, pipe_schedule, terminate_condition=None):
        """ Execute schedules
        Args:
            pipe_schedule: an instance of schedule
            terminate_condition: a callable that returns boolean value indicating if 
                                 the pipeline execution should terminate
        """
        self.fwd_outputs = []

        # For each step in the schedule
        self.step_count = 0
        for step_cmds in pipe_schedule:
            if terminate_condition is not None:
                terminate_tensor = torch.tensor(0, dtype=torch.int32, device=self.device)
                if terminate_condition():
                    terminate_tensor = torch.tensor(1, dtype=torch.int32, device=self.device)
                    logger.debug(f"rank {self.global_rank} reach terminate condition")
                dist.all_reduce(terminate_tensor)
                logger.debug(f"rank {self.global_rank} terminate_tensor {terminate_tensor}")
                if terminate_tensor.item() > 0:
                    logger.debug(f"{self.global_rank} terminate")
                    break
            # For each instruction in the step
            step_id, micro_batch_id, step_cmds = step_cmds
            logger.info(
                f"rank {self.global_rank} step {self.step_count}, st {step_id} mb {micro_batch_id} step_cmds: {step_cmds}"
            )
            for cmd in step_cmds:
                logger.info(f"rank {self.global_rank} exec cmd: {cmd}")
                if type(cmd) not in self._INSTRUCTION_MAP:
                    raise RuntimeError(
                        f'{self.__class__.__name__} does not understand instruction {repr(cmd)}')

                # Equivalent to: self._exec_forward_pass(buffer_id=0)
                try:
                    cmd_type_string = str(type(cmd)).split('\'')[1].split(".")[-1]
                    time_mark(name=f"{cmd_type_string}_start",
                              identifier=str(self.global_rank),
                              step=self.sched_count)
                    self._exec_instr = MethodType(self._INSTRUCTION_MAP[type(cmd)], self)
                    self._exec_instr(*cmd.args, **cmd.kwargs)
                    time_mark(name=f"{cmd_type_string}_end",
                              identifier=str(self.global_rank),
                              step=self.sched_count)
                except Exception as e:
                    logger.error(f"Rank {self.global_rank} step {self.step_count}, Exception in cmd {cmd}")
                    raise e
            self.step_count += 1

        self.sched_count += 1

    def run(self):
        while True:
            pass
