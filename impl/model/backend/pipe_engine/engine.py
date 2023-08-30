# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import MethodType
from typing import Optional, Union
import logging

from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.activation_checkpointing import checkpointing as ds_checkpointing
from deepspeed.runtime.dataloader import RepeatingLoader
from deepspeed.runtime.engine import DeepSpeedEngine, MEMORY_OPT_ALLREDUCE_SIZE
from deepspeed.runtime.utils import PartitionedTensor
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.utils import logger
from deepspeed.utils.timer import ThroughputTimer
import torch

from . import p2p, schedule
from .module import PipelineError, PipelineModule
from impl.model.utils.spec import TransformerData

TARGET_ID = -2
LOG_STAGE = -2
DATA_PARALLEL_ID = -2

logger = logging.getLogger("CustomizedPipelineEngine")


def is_even(number):
    return number % 2 == 0


mem_alloced = 0
mem_cached = 0


def _tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


class PipelineEngine(DeepSpeedEngine):
    """ A training engine hybrid pipeline, data, and model parallel training.

    This engine is created by ``deepspeed.initialize()`` when a :class:`PipelineModule`
    is provided.
    """
    ID_TO_DTYPE = [
        torch.float32, torch.float64, torch.complex64, torch.complex128, torch.float16, torch.bfloat16,
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.bool
    ]
    DTYPE_TO_ID = {dtype: id_ for id_, dtype in enumerate(ID_TO_DTYPE)}

    def __init__(self, *super_args, **super_kwargs):
        super().__init__(*super_args, **super_kwargs)
        assert isinstance(self.module, PipelineModule), "model must base PipelineModule"

        assert self.zero_optimization_stage(
        ) < 2, "ZeRO-2 and ZeRO-3 are incompatible with pipeline parallelism"

        # We schedule the all-reduces, so disable it in super().backward()
        self.enable_backward_allreduce = False
        # does not support boolean tesnor
        self.has_bool_tensors = False
        self.eval_return_logits = False
        self.outputs = None

        # used to disable the pipeline all-reduce when used with 1-bit Adam/1-bit LAMB
        self.pipeline_enable_backward_allreduce = True

        if self.elasticity_enabled():
            if not self.is_elastic_model_parallel_supported():
                assert not self.elasticity_enabled(), "Elasticity is not currently supported" \
                " with pipeline parallelism."

        # pipeline step for logging
        self.log_batch_step_id = -1

        self.micro_batch_size = self.train_micro_batch_size_per_gpu()
        self.micro_batches = self.gradient_accumulation_steps()
        self.generation_micro_batches = self.num_stages
        # TODO: make this configurable, should be larger than self.num_stages for best performance
        # assert self.train_batch_size() == \
        #     self.micro_batch_size * self.micro_batches * self.grid.data_parallel_size

        # Set Grid and Communication Groups
        self.grid = self.module._grid
        if self.grid.get_global_rank() == 0:
            logger.info(f'CONFIG: micro_batches={self.micro_batches} '
                        f'micro_batch_size={self.micro_batch_size}')

        self.global_rank = self.grid.get_global_rank()

        assert self.dp_world_size == self.grid.data_parallel_size
        #  Set Stage Inf
        self.num_stages = self.grid.pipe_parallel_size
        self.stage_id = self.grid.get_stage_id()

        # only for generation
        self.data_iterator = None  # TODO: set data iterator when calling methods
        #       each batch is a iterator, output microbatches

        self.batch_fn = None

        self._force_grad_boundary = False

        # PipelineEngine needs to handle data loading specially due to only the first
        # and last stages loading inputs/labels. We construct a sampler that uses

        self.is_pipe_parallel = self.grid.pipe_parallel_size > 1
        self.is_data_parallel = self.grid.data_parallel_size > 1
        self.is_model_parallel = self.grid.model_parallel_size > 1

        # Partition input/output buffers
        # XXX temporarily disable while I revert some partition hacks.
        self.is_pipe_partitioned = self.is_model_parallel
        self.is_grad_partitioned = self.is_model_parallel

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
                        f'STAGE={self.stage_id} '
                        f'LAYERS={self.module._local_stop - self.module._local_start} '
                        f'[{self.module._local_start}, {self.module._local_stop}) '
                        f'STAGE_PARAMS={num_params} ({num_params/1e6:0.3f}M) '
                        f'TOTAL_PARAMS={total_params} ({total_params/1e6:0.3f}M) '
                        f'UNIQUE_PARAMS={unique_params} ({unique_params/1e6:0.3f}M)')

        # initialize peer-2-peer communication and allreduce groups
        if self.is_pipe_parallel:
            p2p.init_process_groups(self.grid)

        # Pipeline buffers
        self.num_pipe_buffers = 0
        self.pipe_buffers = {
            'inputs': [],  # batch input and received activations
            'labels': [],  # labels from batch input
            'outputs': [],  # activations
            'output_tensors': [],  # tensor object to preserve backward graph
        }
        self.pipe_recv_buf = None
        self.grad_layer = None

        self.meta_buffer = None

        self.first_output_send = True
        self.first_gradient_send = True

        #stores the loss for the current micro batch being processed
        self.loss = torch.tensor(0.0).to(self.device)

        #stores the loss for the entire batch
        self.total_loss = None
        self.agg_loss = torch.tensor(0.0, requires_grad=False).to(self.device)
        self.dp_group_loss = torch.tensor(0.0, requires_grad=False).to(self.device)

        if self._config.pipeline['activation_checkpoint_interval'] > 0:
            self.module.activation_checkpoint_interval = self._config.pipeline[
                'activation_checkpoint_interval']

        self.module.checkpoint_parallel_write_pipeline = self._config.checkpoint_parallel_write_pipeline

        self.has_attention_mask = False  # avoid boolean tensor in forward

        self._normal_mode()

        # XXX look into timer reporting timing
        # Initialize some timers because of early weirdness.
        if self.wall_clock_breakdown():
            self.timers('forward_microstep').start()
            self.timers('forward_microstep').stop()
            self.timers('backward_microstep').start()
            self.timers('backward_microstep').stop()
            self.timers('backward_inner_microstep').start()
            self.timers('backward_inner_microstep').stop()
            self.timers('backward_allreduce_microstep').start()
            self.timers('backward_allreduce_microstep').stop()
            self.timers('backward_allreduce').start()
            self.timers('backward_allreduce').stop()
            self.timers('step_microstep').start()
            self.timers('step_microstep').stop()

        # generation related
        self.generate_configured = False
        self.max_tokens = None
        self.max_new_tokens = None
        self.eos_token_id = None
        self.pad_token_id = None

        self.mini_batch_finished = dict()  # mini_batch_id -> bool
        self.unfinished_sequences = dict()

        # for recv tensor meta
        self.prev_tensor_meta = None

    def _generate_mode(self):
        self.prev_stage = self.prev_stage % self.num_stages
        self.next_stage = self.next_stage % self.num_stages
        self._initialize_p2p()

    def _normal_mode(self):
        # for generate and inference
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1
        self._initialize_p2p()

    def _initialize_p2p(self):
        if is_even(self.stage_id):
            if not self.is_last_stage():
                p2p.send(self.loss, self.next_stage)
            if not self.is_first_stage():
                p2p.recv(self.loss, self.prev_stage)
        else:
            if not self.is_first_stage():
                p2p.recv(self.loss, self.prev_stage)
            if not self.is_last_stage():
                p2p.send(self.loss, self.next_stage)

    def set_has_attention_mask(self, value):
        assert isinstance(value, bool)
        self.has_attention_mask = value

    def _exec_reduce_tied_grads(self):
        # We need to run this first to write to self.averaged_gradients;
        # since this class turns `enable_backward_allreduce` off,
        # `self.overlapping_partition_gradients_reduce_epilogue()` defined in the DeepSpeedEngine
        # never actually runs. I suspect this is because of efficiency problems; get_flat_partition in
        # stage2.py might do something expensive; someone will have to look into that later. But
        # in the meantime, this fixes ZeRO2 + Pipelining enough to run a demo. Further profiling
        # needed to decide if it actually breaks everything.
        # (see https://github.com/EleutherAI/gpt-neox/issues/62#issuecomment-761471944)
        if self.zero_optimization_partition_gradients():
            self.optimizer.overlapping_partition_gradients_reduce_epilogue()

        weight_group_list = self.module.get_tied_weights_and_groups()
        for weight, group in weight_group_list:
            grad = weight._hp_grad if self.bfloat16_enabled() else weight.grad
            dist.all_reduce(grad, group=group)

    def _exec_reduce_grads(self):
        self._force_grad_boundary = True
        if self.pipeline_enable_backward_allreduce:
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

    def _reserve_pipe_buffers(self, num_buffers):
        """Ensure that each pipeline buffer has at least ``num_buffers`` slots.

        This method only reserves slots and does not allocate tensors.

        Args:
            num_buffers (int): The number of buffers to reserve.
        """
        if self.num_pipe_buffers >= num_buffers:
            return

        num_added = num_buffers - self.num_pipe_buffers
        for key in self.pipe_buffers:
            self.pipe_buffers[key].extend([None] * num_added)
        self.num_pipe_buffers = num_buffers

    def reset_activation_shape(self):
        """Reset the buffers when the shape of activation and gradient change.
        For example, for curriculum learning that changes the seqlen of each
        sample, we need to call this whenever the seqlen is going to change.
        """
        self.first_output_send = True
        self.pipe_recv_buf = None
        self.grad_layer = None
        self.meta_buffer = None

    def train_batch(self, data_iter=None):
        """Progress the pipeline to train the next batch of data. The engine will ingest
        ``self.train_batch_size()`` total samples collectively across all workers.


        An iterator that over training data should be provided as an argument
        unless ``deepspeed.initialize()`` was provided a training set. In that event,
        the training data will automatically be read.


        .. warning::
            A total of ``self.gradient_accumulation_steps()`` entries will be pulled
            from ``data_iter`` by each pipeline. There must be sufficient
            data left in ``data_iter`` or else a ``StopIteration`` will halt training.

            DeepSpeed provides a convenience class :class:`deepspeed.utils.RepeatingLoader`
            that wraps data loaders to automatically restart upon a ``StopIteration``.

        Args:
            data_iter (Iterator, optional): Iterator of training data.

        Returns:
            The arithmetic mean of the losses computed this batch.
        """
        if not torch._C.is_grad_enabled():
            raise RuntimeError(f'train_batch() requires gradients enabled. Use eval_batch() instead.')

        # Curriculum learning could change activation shape
        if self.curriculum_enabled_legacy():
            new_difficulty = self.curriculum_scheduler_legacy.update_difficulty( \
                self.global_steps + 1)
            if self.global_steps == 0 or self.curriculum_scheduler_legacy.first_step:
                self.reset_activation_shape()
                self.curriculum_scheduler_legacy.first_step = False
            elif new_difficulty != self.curriculum_scheduler_legacy.get_difficulty( \
                self.global_steps):
                self.reset_activation_shape()

        if data_iter:
            self.set_dataiterator(data_iter)

        self.module.train()
        self.total_loss = None
        self._compute_loss = True

        # Do the work
        self.timers('train_batch').start()
        sched = schedule.TrainSchedule(micro_batches=self.micro_batches,
                                       stages=self.num_stages,
                                       stage_id=self.stage_id)
        self._exec_schedule(sched)
        self.agg_train_loss = self._aggregate_total_loss()

        self.timers('train_batch').stop()

        if self.global_steps % self.steps_per_print() == 0:
            if self.global_rank == 0:
                elapsed = self.timers('train_batch').elapsed(reset=True) / 1000.0
                iter_time = elapsed / self.steps_per_print()
                tput = self.train_batch_size() / iter_time
                print(f'steps: {self.global_steps} '
                      f'loss: {self.agg_train_loss:0.4f} '
                      f'iter time (s): {iter_time:0.3f} '
                      f'samples/sec: {tput:0.3f}')

        # Monitoring
        if self.global_rank == 0 and self.monitor.enabled:
            self.summary_events = [(f'Train/Samples/train_loss', self.agg_train_loss.mean().item(),
                                    self.global_samples)]
            self.monitor.write_events(self.summary_events)

        if self.wall_clock_breakdown() and self.global_steps % self.steps_per_print() == 0:
            self.timers.log(['pipe_send_output', 'pipe_send_grad', 'pipe_recv_input', 'pipe_recv_grad'])

        # TODO: should return precisely what loss returned and allow others to be queried?
        return self.agg_train_loss

    def _exec_prepare_generate_input(self, buffer_id):
        # preprocess model input for generation on only first stage
        inputs = self.pipe_buffers['inputs'][buffer_id]
        # inputs = tuple(t.clone() for t in self.pipe_buffers['inputs'][buffer_id])

        x = TransformerData.from_tuple(inputs)

        attention_mask = x.raw_attention_mask.clone()
        input_ids = x.raw_input_ids.clone()

        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        if x.generation_id is not None:
            # generation_id is assigned in model preprocessing, if no generation id, its the first pass forward of generation
            input_ids = input_ids[:, -1:]
            position_ids = position_ids[:, -1].unsqueeze(-1)

        x.input_ids = input_ids
        # print("input_ids after processing", x.input_ids, x.input_ids.shape)
        x.attention_mask = attention_mask
        x.position_ids = position_ids

        inputs = x.to_tuple()
        self.pipe_buffers['inputs'][buffer_id] = inputs

    def _exec_postprocess_generate_output(self, buffer_id):
        # post process model output for generation on only last stage
        outputs = self.pipe_buffers['outputs'][buffer_id]
        # outputs = tuple(t.clone() for t in self.pipe_buffers['outputs'][buffer_id])

        x = TransformerData.from_tuple(outputs)
        logits = x.logits
        input_ids = x.raw_input_ids
        attention_mask = x.raw_attention_mask

        if self.max_tokens is None:
            # self.generate_mini_batch_size = input_ids.shape[0]
            self.max_tokens = self.max_new_tokens + input_ids.shape[-1]

        if x.generation_id not in self.mini_batch_finished:
            self.mini_batch_finished[x.generation_id] = False
            self.unfinished_sequences[x.generation_id] = torch.ones(input_ids.shape[0],
                                                                    dtype=torch.long,
                                                                    device=torch.cuda.current_device())
        unfinished_sequences = self.unfinished_sequences[x.generation_id]

        # without postprocessing logits
        next_token_logits = next_token_scores = logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)

        # finished sentences should have their next token be a padding token
        if self.eos_token_id is not None:
            if self.pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + self.pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        # TODO: fix post process input ids
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
        if input_ids.shape[-1] >= self.max_tokens:
            # TODO: terminate
            logger.info(f"rank {self.global_rank} terminate {x.generation_id}")
            self.mini_batch_finished[x.generation_id] = True

        # update unfinished sequences
        unfinished_sequences = next_tokens.ne(self.eos_token_id).long() * unfinished_sequences

        if unfinished_sequences.max() == 0:
            # TODO: terminate
            logger.info(f"rank {self.global_rank} terminate {x.generation_id}")
            self.mini_batch_finished[x.generation_id] = True

        x.raw_input_ids = input_ids
        x.raw_attention_mask = attention_mask
        self.unfinished_sequences[x.generation_id] = unfinished_sequences
        outputs = x.to_tuple()
        self.pipe_buffers['outputs'][buffer_id] = outputs
        self.generate_outputs.append(input_ids)

    def _generate_term_condition(self):
        num_batches = self.micro_batches
        this_peer_terminate = False  # only will be true on last stage

        if len(self.mini_batch_finished) < num_batches:
            this_peer_terminate = False
        else:
            this_peer_terminate = all(self.mini_batch_finished.values())

        this_peer_terminate = torch.tensor(this_peer_terminate,
                                           dtype=torch.long,
                                           device=torch.cuda.current_device())
        dist.all_reduce(this_peer_terminate)
        return this_peer_terminate == 1

    def _split_data_into_micro_batches(self, data_dict, splits):
        # return a data iterator yield micro batches
        batches = []
        new_data_dict = {}
        for key, data in data_dict.items():
            new_data_dict[key] = torch.split(data, splits, dim=0)

        splited_dicts = [{key: data[i] for key, data in new_data_dict.items()} for i in range(splits)]
        for d in splited_dicts:
            batch = TransformerData()
            for k, v in d.items():
                try:
                    setattr(batch, k, v)
                except AttributeError:
                    logger.error(f"key {k} not found in TransformerData")
            batches.append(batch.to_tuple())

        return iter(batches)

    def is_first_stage(self):
        """True if this process is in the first stage in the pipeline."""
        return self.stage_id == 0

    def is_last_stage(self):
        """True if this process is in the last stage in the pipeline."""
        return self.stage_id == self.num_stages - 1

    def _reduce_outputs(self, outputs, reduce='avg', reduce_dp=True):
        if reduce is None:
            return outputs

        if reduce.lower() == 'avg':
            # first sum over all microbatches
            if torch.is_tensor(outputs[0]):
                reduced = sum(outputs)
            else:
                assert isinstance(outputs, (list, tuple))
                reduced = [torch.zeros_like(o) for o in outputs[0]]
                for idx, out in outputs:
                    reduced[idx] += out

            # Average over the microbatches
            reduced = self._scale_loss_by_gas(reduced)

            # Average over DP groups
            if reduce_dp and self.is_data_parallel:
                if torch.is_tensor(reduced):
                    dist.all_reduce(reduced, group=self.mpu.get_data_parallel_group())
                    reduced /= self.dp_world_size
                else:
                    for idx in range(len(reduced)):
                        dist.all_reduce(reduced[idx], group=self.mpu.get_data_parallel_group())
                        reduced[idx] /= self.dp_world_size

            return reduced
        else:
            raise NotImplementedError(f'reduction type {reduce} not supported.')

    def _bcast_pipe_scalar(self, data, src_rank=None, dtype=torch.float32):
        # Default to last stage (e.g., for broadcasting loss)
        if src_rank is None:
            src_rank = self.grid.stage_to_global(self.num_stages - 1)
        assert src_rank in self.grid.pp_group

        if self.global_rank == src_rank:
            result = data.clone().detach().type(dtype).to(self.device)
        else:
            result = torch.Tensor([0.]).type(dtype).to(self.device)

        dist.broadcast(tensor=result, src=src_rank, group=self.mpu.get_pipe_parallel_group())

        return result

    def _aggregate_total_loss(self):
        # Scale loss, average among DP ranks, and bcast loss to the rest of my DP group
        if self.is_last_stage():
            loss = self._scale_loss_by_gas(self.total_loss)
            self.dp_group_loss = loss.clone().detach()

            ## Average loss across all data-parallel groups
            agg_loss = self.dp_group_loss.clone().detach()
            #print(f'RANK={self.global_rank} bcast SENDER src={self.global_rank} group={self.grid.pp_group}', flush=True)
            if self.is_data_parallel:
                dist.all_reduce(agg_loss, group=self.mpu.get_data_parallel_group())
                agg_loss /= self.dp_world_size

            assert self.global_rank in self.grid.pp_group
            losses = torch.Tensor([self.dp_group_loss, agg_loss]).to(self.device)
            dist.broadcast(tensor=losses, src=self.global_rank, group=self.mpu.get_pipe_parallel_group())

        else:
            # Get loss from last stage
            src_rank = self.grid.stage_to_global(self.num_stages - 1)
            assert src_rank in self.grid.pp_group
            losses = torch.Tensor([0., 0.]).to(self.device)
            dist.broadcast(tensor=losses, src=src_rank, group=self.grid.get_pipe_parallel_group())
            self.dp_group_loss = losses[0].clone().detach()
            agg_loss = losses[1].clone().detach()

        return agg_loss

    def set_dataloader(self, loader):
        """"""
        if self.is_first_stage() or self.is_last_stage():
            self.training_dataloader = loader
            self.data_iterator = iter(self.training_dataloader)

    def set_dataiterator(self, iterator):
        """ Store an iterator to sample for training data. """
        if self.is_first_stage() or self.is_last_stage():
            self.training_dataloader = None
            self.data_iterator = iterator

    def set_batch_fn(self, fn):
        """Execute a post-processing function on input data.

        Args:
            fn (function): The function to run.
        """
        self.batch_fn = fn

    def is_gradient_accumulation_boundary(self):
        """True if the engine is executing a gradient reduction or optimizer step instruction.

        This is overridden from :class:`DeepSpeedEngine` to force reductions
        and steps when the pipeline engine is instructed to do so.

        Returns:
            bool: whether reductions and optimizer steps should occur.
        """
        return self._force_grad_boundary

    def log_for_device(self, *msg):
        if LOG_STAGE == self.stage_id or LOG_STAGE == -1:
            if DATA_PARALLEL_ID == self.grid.data_parallel_id or DATA_PARALLEL_ID == -1:
                print(
                    f'RANK={dist.get_rank()} '
                    f'PIPE-ID={self.stage_id} '
                    f'DATA-ID={self.grid.data_parallel_id} '
                    f'MBATCH-ID={self.microbatch_id} '
                    f'STEP-ID={self.log_batch_step_id} '
                    '::',
                    *msg,
                    flush=True)

    def tput_log(self, *msg):
        if self.global_rank == 0 and self.global_steps % self.steps_per_print() == 0:
            print(*msg)

    def _next_batch(self):
        # If using 3D parallelism, only some first-stage ranks may do IO
        batch = None
        if self.data_iterator is not None:
            batch = next(self.data_iterator)

        # Any post-processing, like broadcasting across a slice-parallel group.
        if self.batch_fn:
            batch = self.batch_fn(batch)

        return batch

    def _exec_forward_pass(self, buffer_id):
        self.tput_timer.start()
        self.mem_status('BEFORE FWD', reset_max=True)
        if isinstance(buffer_id, tuple):
            src_buffer_id, dst_buffer_id = buffer_id
        elif isinstance(buffer_id, int):
            src_buffer_id = dst_buffer_id = buffer_id
        else:
            raise ValueError("buffer_id must be int or tuple of ints")

        if isinstance(self.pipe_buffers['inputs'][src_buffer_id], tuple):
            inputs = tuple(t.clone() for t in self.pipe_buffers['inputs'][src_buffer_id])
        else:
            inputs = self.pipe_buffers['inputs'][src_buffer_id].clone()

        # collect the partitioned input from the previous stage
        if self.is_pipe_partitioned and not self.is_first_stage():
            part_input = PartitionedTensor.from_meta(meta=inputs[0],
                                                     local_part=inputs[1],
                                                     group=self.grid.get_slice_parallel_group())

            inputs = (part_input.full(), *inputs[2:])
            inputs[0].requires_grad = True
            # skip mask
            #inputs[1].requires_grad = True
            part_input = None
            inputs = inputs[0] if len(inputs) == 1 else inputs
            self.pipe_buffers['inputs'][src_buffer_id] = inputs

        # Zero out the gradients each time we use the tensor because only the data in
        # tensor changes across batches
        self._zero_grads(inputs)

        outputs = super().forward(inputs)

        # Reset activation checkpointing buffers.
        # Need to call this between evaluation iterations
        if not self.module.training:
            ds_checkpointing.reset()

        # not pipeline partitioned
        # Partition the outputs if we are not the last stage
        if self.is_pipe_partitioned and not self.is_last_stage():
            if isinstance(outputs, tuple):
                first_output = outputs[0]
                # TODO: Improve pipe partitioning to pass multiple tensors that require grads
                assert all([torch.is_tensor(elt) and elt.requires_grad is False for elt in outputs[1:]])
                outputs_tail = outputs[1:]
            elif torch.is_tensor(outputs):
                first_output = outputs
                outputs_tail = []
            else:
                raise ValueError("expecting a tensor or a tuple of tensors")
            part = PartitionedTensor(tensor=first_output, group=self.grid.get_slice_parallel_group())
            # Clear the large output data, but save the computation graph
            first_output.data = torch.zeros(1)
            self.pipe_buffers['output_tensors'][dst_buffer_id] = first_output
            # Inject the partitioned tensor into the output before sending
            outputs = (part.to_meta(), part.data(), *outputs_tail)
            part = None

        self.pipe_buffers['outputs'][dst_buffer_id] = outputs

        # func generate() will not get outputs here
        # clear previous forward outputs all micro batches are finished
        if self.is_last_stage() and len(self.fwd_outputs) == self.micro_batches:
            self.fwd_outputs.clear()
        # temporarily store last forward outputs in fwd_outputs
        if self.is_last_stage():
            self.fwd_outputs.append([o.detach() for o in outputs])
        # loss: compute loss whenever lalels are present in data
        # loss is stored in fwd_outputs

        # Optionally compute loss on the last device
        # if self.is_last_stage():
        #     if self._compute_loss and self.module.loss_fn is not None:
        #         labels = self.pipe_buffers['labels'][buffer_id]
        #         self.loss = self.module.loss_fn(outputs, labels)
        #     else:
        #         # Some models just return loss from forward()
        #         self.loss = outputs
        #     if self.eval_return_logits:
        #         self.outputs = outputs
        #     if isinstance(self.loss, torch.Tensor):
        #         self.fwd_outputs.append(self.loss.detach())

        #         if self.total_loss is None:
        #             self.total_loss = torch.zeros_like(self.loss)
        #         self.total_loss += self.loss.detach()
        #     else:
        #         self.fwd_outputs.append([l.detach() for l in self.loss])
        #         if self.total_loss is None:
        #             self.total_loss = [torch.zeros_like(l) for l in self.loss]
        #         for idx, l in enumerate(self.loss):
        #             self.total_loss[idx] += l.detach()

    # def _exec_clear_fwd_output(self, buffer_id):
    #     self.fwd_outputs = []

    def _exec_backward_pass(self, buffer_id):
        assert self.optimizer is not None, "must provide optimizer during " \
                                           "init in order to use backward"

        self.mem_status('BEFORE BWD', reset_max=True)

        # The last stage just runs backward on the loss using DeepSpeed's typical
        # mechanisms.
        if self.is_last_stage():
            super().backward(self.loss)
            self.mem_status('AFTER BWD')
            return

        outputs = self.pipe_buffers['outputs'][buffer_id]

        if self.wall_clock_breakdown():
            self.timers('backward_microstep').start()
            self.timers('backward').start()
            self.timers('backward_inner_microstep').start()
            self.timers('backward_inner').start()

        # Reconstruct if we previously partitioned the output. We must be
        # careful to also restore the computational graph of the tensors we partitioned.
        # TODO: temporarily no model parallel partition
        # if self.is_pipe_partitioned:
        #     if self.is_grad_partitioned:
        #         part_output = PartitionedTensor.from_meta(meta=outputs[0],
        #                                                   local_part=outputs[1],
        #                                                   group=self.grid.get_slice_parallel_group())
        #         self.pipe_buffers['output_tensors'][buffer_id].data = part_output.full()
        #         outputs = (self.pipe_buffers['output_tensors'][buffer_id], *outputs[2:])
        #     else:
        #         # Already restored from partition
        #         self.pipe_buffers['output_tensors'][buffer_id].data = outputs[0]
        #         outputs = (self.pipe_buffers['output_tensors'][buffer_id], *outputs[1:])

        grad_tensors = self.grad_layer

        # if self.is_grad_partitioned:
        #     #print(f'RANK={self.global_rank} BEFORE-BWD restoring grad={self.grad_layer[0].size()} {self.grad_layer[1].size()}')
        #     part_grad = PartitionedTensor.from_meta(meta=self.grad_layer[0],
        #                                             local_part=self.grad_layer[1],
        #                                             group=self.grid.get_slice_parallel_group())
        #     grad_tensors = (part_grad.full(), *grad_tensors[2:])
        #     part_grad = None
        #     #print(f'RANK={self.global_rank} BEFORE-BWD restored grad={self.grad_layer[0].size()} {self.grad_layer[1].size()}')

        if self.bfloat16_enabled() and not self.is_last_stage():
            # manually call because we don't call optimizer.backward()
            self.optimizer.clear_lp_grads()

        # This handles either a single tensor or tuple of tensors.
        if isinstance(outputs, tuple):
            out_tensors = [t for t in outputs if t.is_floating_point()]
            assert len(out_tensors) == len(grad_tensors)
            torch.autograd.backward(tensors=out_tensors, grad_tensors=grad_tensors)
        else:
            torch.autograd.backward(tensors=(outputs,), grad_tensors=(grad_tensors,))

        if self.bfloat16_enabled() and not self.is_last_stage():
            # manually call because we don't call optimizer.backward()
            self.optimizer.update_hp_grads(clear_lp_grads=False)

        # Free up the memory from the output of forward()
        self.pipe_buffers['output_tensors'][buffer_id] = None
        self.pipe_buffers['outputs'][buffer_id] = None
        grad_tensors = None

        if self.wall_clock_breakdown():
            self.timers('backward_inner').stop()
            self.timers('backward_inner_microstep').stop()
            self.timers('backward').stop()
            self.timers('backward_microstep').stop()

        self.mem_status('AFTER BWD')

    def _exec_load_micro_batch(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('batch_input').start()

        batch = self._next_batch()
        # batch should be a tuple in all conditions
        # all stages should be the same
        assert isinstance(batch, (tuple, list))
        # Assume list or tuple
        loaded = []
        for x in batch:
            assert torch.is_tensor(x)
            mine = x.clone().detach().to(self.device)
            mine.requires_grad = mine.is_floating_point()
            loaded.append(mine)
        loaded = tuple(loaded)
        self.pipe_buffers['inputs'][buffer_id] = loaded

        # if self.is_first_stage():
        #     loaded = None
        #     if torch.is_tensor(batch[0]):
        #         loaded = batch[0].clone().to(self.device).detach()
        #         loaded.requires_grad = loaded.is_floating_point()
        #     else:
        #         assert isinstance(batch[0], (tuple, list))
        #         # Assume list or tuple
        #         loaded = []
        #         for x in batch[0]:
        #             assert torch.is_tensor(x)
        #             mine = x.clone().detach().to(self.device)
        #             mine.requires_grad = mine.is_floating_point()
        #             loaded.append(mine)
        #         loaded = tuple(loaded)

        #     self.pipe_buffers['inputs'][buffer_id] = loaded

        # if self.is_last_stage():
        #     loaded = batch[1]
        #     if torch.is_tensor(batch[1]):
        #         loaded = batch[1].to(self.device)
        #     elif isinstance(batch[1], tuple):
        #         loaded = []
        #         for x in batch[1]:
        #             assert torch.is_tensor(x)
        #             x = x.to(self.device).detach()
        #             loaded.append(x)
        #         loaded = tuple(loaded)

        #     self.pipe_buffers['labels'][buffer_id] = loaded

        if self.wall_clock_breakdown():
            self.timers('batch_input').stop()

    def _send_tensor_meta(self, buffer, recv_stage):
        """ Communicate metadata about upcoming p2p transfers.

        Metadata is communicated in this order:
            * type (0: tensor, 1: list)
            * num_tensors if type=list
            foreach tensor in buffer:
                * ndims
                * shape
        """
        send_bytes = 0
        if isinstance(buffer, torch.Tensor):
            type_tensor = torch.LongTensor(data=[0]).to(self.device)
            p2p.send(type_tensor, recv_stage)
            send_shape = torch.LongTensor(data=buffer.size()).to(self.device)
            send_ndims = torch.LongTensor(data=[len(buffer.size())]).to(self.device)
            p2p.send(send_ndims, recv_stage)
            p2p.send(send_shape, recv_stage)
            send_bytes += _tensor_bytes(buffer)
        elif isinstance(buffer, list):
            assert (False)
            type_tensor = torch.LongTensor(data=[1]).to(self.device)
            p2p.send(type_tensor, recv_stage)
            count_tensor = torch.LongTensor(data=[len(buffer)]).to(self.device)
            p2p.send(count_tensor, recv_stage)
            for tensor in buffer:
                assert isinstance(tensor, torch.Tensor)
                send_shape = torch.LongTensor(data=tensor.size()).to(self.device)
                send_ndims = torch.LongTensor(data=[len(tensor.size())]).to(self.device)
                p2p.send(send_ndims, recv_stage)
                p2p.send(send_shape, recv_stage)
                send_bytes += _tensor_bytes(tensor)
        elif isinstance(buffer, tuple):
            type_tensor = torch.LongTensor(data=[2]).to(self.device)
            p2p.send(type_tensor, recv_stage)
            count_tensor = torch.LongTensor(data=[len(buffer)]).to(self.device)
            p2p.send(count_tensor, recv_stage)
            for idx, tensor in enumerate(buffer):
                assert isinstance(tensor, torch.Tensor)
                send_shape = torch.LongTensor(data=tensor.size()).to(self.device)
                send_ndims = torch.LongTensor(data=[len(tensor.size())]).to(self.device)
                send_dtype = torch.LongTensor(data=[self.DTYPE_TO_ID[tensor.dtype]]).to(self.device)
                p2p.send(send_dtype, recv_stage)
                p2p.send(send_ndims, recv_stage)
                p2p.send(send_shape, recv_stage)
                # Useful for performance debugging.
                '''
                new_bytes = _tensor_bytes(tensor)
                send_bytes += _tensor_bytes(tensor)
                # Useful for performance debugging.
                if self.grid.data_parallel_id == 0:
                    print(
                        f'STAGE={self.stage_id} pipe-send-volume[{idx}]: shape={send_shape} {new_bytes/1024**2:0.2f}MB'
                    )
                '''
        else:
            raise NotImplementedError(f'Could not send meta type {type(buffer)}')

        # Useful for performance debugging.
        '''
        if self.grid.data_parallel_id == 0:
            print(f'STAGE={self.stage_id} pipe-send-volume: {send_bytes/1024**2:0.2f}MB')
        '''

    def _recv_tensor_meta(self, send_stage):
        """Receive metadata about upcoming p2p transfers and return allocated buffers.

        Metadata is communicated in this order:
            * type (0: tensor, 1: list)
            * num_tensors if type=list
            foreach tensor in buffer:
                * ndims
                * shape

        Returns:
            Allocated buffer for receiving from send_stage.
        """
        type_tensor = torch.LongTensor(data=[0]).to(self.device)
        p2p.recv(type_tensor, send_stage)
        recv_type = type_tensor.item()
        recv_shapes_and_dtypes = None

        # A single tensor will be sent.
        if recv_type == 0:
            recv_ndims = torch.LongTensor(data=[0]).to(self.device)
            p2p.recv(recv_ndims, send_stage)
            recv_ndims = recv_ndims.item()
            recv_shape = torch.LongTensor([1] * recv_ndims).to(self.device)
            p2p.recv(recv_shape, send_stage)
            recv_shape = recv_shape.tolist()
            return self._allocate_buffer(recv_shape, num_buffers=1)[0]
        # List or tuple of tensors
        elif recv_type == 1 or recv_type == 2:
            count_tensor = torch.LongTensor(data=[0]).to(self.device)
            p2p.recv(count_tensor, send_stage)
            num_tensors = count_tensor.item()
            recv_shapes_and_dtypes = []
            for idx in range(num_tensors):
                recv_dtype = torch.LongTensor(data=[0]).to(self.device)
                p2p.recv(recv_dtype, send_stage)
                recv_dtype = self.ID_TO_DTYPE[recv_dtype.item()]
                recv_ndims = torch.LongTensor(data=[0]).to(self.device)
                p2p.recv(recv_ndims, send_stage)
                recv_ndims = recv_ndims.item()
                recv_shape = torch.LongTensor([1] * recv_ndims).to(self.device)
                p2p.recv(recv_shape, send_stage)
                recv_shapes_and_dtypes.append((recv_shape.tolist(), recv_dtype))

            # if self.prev_tensor_meta != recv_shapes_and_dtypes:
            #     self.prev_tensor_meta = recv_shapes_and_dtypes
            #     buffers = self._allocate_buffers(recv_shapes_and_dtypes, num_buffers=1)[0]
            # else:
            #     return None
            # Convert to tuples if requested.
            buffers = self._allocate_buffers(recv_shapes_and_dtypes, num_buffers=1)[0]
            if recv_type == 2:
                buffers = tuple(buffers)
            return buffers

        else:
            raise NotImplementedError(f'Could not receive type {type(recv_type)}')

    def _exec_send_activations(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('pipe_send_output').start()

        outputs = self.pipe_buffers['outputs'][buffer_id]

        # NCCL does not like to send torch.BoolTensor types, so cast the mask to half().
        # We could do char, but with half() we can eventually flatten with other fp16
        # messages (TODO)
        # if self.has_attention_mask or self.has_bool_tensors:
        #     outputs = list(outputs)
        #     outputs[-1] = outputs[-1].half()
        #     outputs = tuple(outputs)

        # if self.first_output_send:
        #     self.first_output_send = False

        self._send_tensor_meta(outputs, self.next_stage)

        if isinstance(outputs, torch.Tensor):
            p2p.send(outputs, self.next_stage)
        elif isinstance(outputs, tuple):
            for idx, buffer in enumerate(outputs):
                p2p.send(buffer, self.next_stage)
        else:
            raise NotImplementedError('Could not send output of type '
                                      f'{type(outputs)}')

        # Restore the boolean tensor
        # if self.has_attention_mask or self.has_bool_tensors:
        #     outputs = list(outputs)
        #     outputs[-1] = outputs[-1].bool()
        #     outputs = tuple(outputs)

        if self.wall_clock_breakdown():
            self.timers('pipe_send_output').stop()

    def _exec_send_grads(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('pipe_send_grad').start()

        inputs = self.pipe_buffers['inputs'][buffer_id]

        # Partition the gradient
        if self.is_grad_partitioned:
            if isinstance(inputs, tuple):
                first_input = inputs[0]
                assert all([torch.is_tensor(elt) for elt in inputs[1:]])
                inputs_grad_tail = [elt.grad for elt in inputs[1:] if elt.grad is not None]
            elif torch.is_tensor(inputs):
                first_input = inputs
                inputs_grad_tail = []
            else:
                raise ValueError("expecting a tensor or a tuple of tensors")
            assert torch.is_tensor(first_input)
            part = PartitionedTensor(tensor=first_input.grad, group=self.grid.get_slice_parallel_group())

            inputs = (part.to_meta(), part.data(), *inputs_grad_tail)

        # XXX Terrible hack
        # Drop the attention mask from the input buffer here. It does not have
        # a grad that needs to be communicated. We free the buffer immediately
        # after, so no need to restore it. The receiver also has a hack that skips
        # the recv. This is because NCCL does not let us send torch.BoolTensor :-(.
        if self.has_attention_mask or self.has_bool_tensors:
            inputs = list(inputs)
            inputs.pop()
            inputs = tuple(inputs)

        if isinstance(inputs, torch.Tensor):
            assert inputs.grad is not None
            p2p.send(inputs.grad, self.prev_stage)
        else:
            # XXX terrible hacky branch
            if self.is_grad_partitioned:
                # First two sends are partitioned gradient
                p2p.send(inputs[0], self.prev_stage)
                p2p.send(inputs[1], self.prev_stage)
            else:
                for idx, buffer in enumerate(inputs):
                    # Skip tensors that will not produce a grad
                    if not buffer.is_floating_point():
                        assert buffer.grad is None
                        continue
                    assert buffer.grad is not None
                    p2p.send(buffer.grad, self.prev_stage)

        # We can free up the input buffer now
        self.pipe_buffers['inputs'][buffer_id] = None

        if self.wall_clock_breakdown():
            self.timers('pipe_send_grad').stop()

    def _exec_recv_activations(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('pipe_recv_input').start()

        recvd = None

        # Allocate the buffer if necessary
        self.pipe_recv_buf = self._recv_tensor_meta(self.prev_stage)

        # pipe_recv_buf = self._recv_tensor_meta(self.prev_stage)
        # if pipe_recv_buf is not None:
        #     self.pipe_recv_buf = pipe_recv_buf

        if isinstance(self.pipe_recv_buf, torch.Tensor):
            p2p.recv(self.pipe_recv_buf, self.prev_stage)
            recvd = self.pipe_recv_buf.clone().detach()
            recvd.requires_grad = recvd.is_floating_point()
        else:
            assert isinstance(self.pipe_recv_buf, tuple)
            recvd = [None] * len(self.pipe_recv_buf)
            for idx, buffer in enumerate(self.pipe_recv_buf):
                assert torch.is_tensor(buffer)
                # XXX hardcode meta type
                if self.is_pipe_partitioned and idx == 0 and buffer.dtype != torch.long:
                    if self.meta_buffer is None:
                        self.meta_buffer = torch.zeros(buffer.size(), dtype=torch.long, device=self.device)
                    buffer = self.meta_buffer

                p2p.recv(buffer, self.prev_stage)
                recvd[idx] = buffer.clone().detach()

            # NCCL does not like to send torch.BoolTensor types, so un-cast the
            # attention mask
            if self.has_attention_mask or self.has_bool_tensors:
                recvd[-1] = recvd[-1].bool()

            recvd = tuple(recvd)

            for buffer in recvd:
                buffer.requires_grad = buffer.is_floating_point()

        self.pipe_buffers['inputs'][buffer_id] = recvd

        if self.wall_clock_breakdown():
            self.timers('pipe_recv_input').stop()

    def _exec_recv_grads(self, buffer_id):
        if self.wall_clock_breakdown():
            self.timers('pipe_recv_grad').start()

        outputs = self.pipe_buffers['outputs'][buffer_id]

        # XXX these shapes are hardcoded for Megatron
        # Restore partitioned output if it was partitioned and we are sending full gradients
        # if self.is_pipe_partitioned and not self.is_grad_partitioned:
        #     part_output = PartitionedTensor.from_meta(meta=outputs[0],
        #                                               local_part=outputs[1],
        #                                               group=self.grid.get_slice_parallel_group())
        #     outputs[0].data = part_output.full()
        #     outputs = (outputs[0], *outputs[2:])
        #     # save for backward
        #     self.pipe_buffers['outputs'][buffer_id] = outputs

        # Allocate gradient if necessary
        if self.grad_layer is None:
            if isinstance(outputs, torch.Tensor):
                s = list(outputs.size())
                self.grad_layer = self._allocate_buffer(s, dtype=outputs.dtype, num_buffers=1)[0]
            else:
                # XXX This is a HACK
                # When we exchange activations/gradients, the two pipe stages
                # need to issue the send/recv with the same buffer sizes or
                # else there is a deadlock. The is_floating_point() filter is
                # used to avoid sending gradients for tensors that do not
                # produce gradients. When TP>1, we partition the first
                # activations/gradients across TP ranks to save communication
                # volume and memory. That partitioned tensor is represented as
                # two tensors: a 1/TPth chunk of the original data and also a
                # small LongTensor storing the metadata used to reconstruct on
                # the other side. When combined, the floating point filter also
                # filtered out the metadata tensor. This quick (hacky) fix just
                # branches on is_grad_partitioned so we don't filter out the
                # metadata tensor.
                if self.is_grad_partitioned:
                    sizes_and_dtypes = [(list(t.size()), t.dtype) for t in outputs[:2]] + [
                        (list(t.size()), t.dtype) for t in outputs[2:] if t.is_floating_point()
                    ]
                else:
                    sizes_and_dtypes = [(list(t.size()), t.dtype) for t in outputs if t.is_floating_point()]
                self.grad_layer = self._allocate_buffers(sizes_and_dtypes, num_buffers=1)[0]

        if isinstance(self.grad_layer, torch.Tensor):
            p2p.recv(self.grad_layer, self.next_stage)
        else:
            assert isinstance(outputs, tuple)
            for idx, buffer in enumerate(self.grad_layer):
                # XXX GPT-2 hack
                if self.is_grad_partitioned and idx == 0 and buffer.dtype != torch.long:
                    buffer.data = torch.zeros(buffer.size(), dtype=torch.long, device=self.device)
                p2p.recv(buffer, self.next_stage)

        if self.wall_clock_breakdown():
            self.timers('pipe_recv_grad').stop()

    def _exec_optimizer_step(self, lr_kwargs=None):
        if self.wall_clock_breakdown():
            self.timers('step_microstep').start()
            self.timers('step').start()
        self.mem_status('BEFORE STEP', reset_max=True)

        self._force_grad_boundary = True
        self._take_model_step(lr_kwargs)
        self._force_grad_boundary = False

        self.mem_status('AFTER STEP')

        if self.global_rank == 0 and self.monitor.enabled:
            self.summary_events = [(f'Train/Samples/lr', self.get_lr()[0], self.global_samples)]
            if self.fp16_enabled() and hasattr(self.optimizer, 'cur_scale'):
                self.summary_events.append(
                    (f'Train/Samples/loss_scale', self.optimizer.cur_scale, self.global_samples))
            self.monitor.write_events(self.summary_events)

        if self.wall_clock_breakdown():
            self.timers('step_microstep').stop()
            self.timers('step').stop()
            if self.global_steps % self.steps_per_print() == 0:
                self.timers.log([
                    'batch_input', 'forward_microstep', 'backward_microstep', 'backward_inner_microstep',
                    'backward_allreduce_microstep', 'backward_tied_allreduce_microstep', 'step_microstep'
                ])
            if self.global_steps % self.steps_per_print() == 0:
                self.timers.log(['forward', 'backward', 'backward_inner', 'backward_allreduce', 'step'])

    def _zero_grads(self, inputs):
        if isinstance(inputs, torch.Tensor):
            if inputs.grad is not None:
                inputs.grad.data.zero_()
        else:
            for t in inputs:
                if t.grad is not None:
                    t.grad.data.zero_()

    def _allocate_zeros(self, shape, **kwargs):
        """ Allocate a tensor of zeros on the engine's device.

        Arguments:
            shape: the shape of the tensor to allocate
            kwargs: passed to torch.zeros()

        Returns:
            A tensor from torch.zeros() allocated on self.device.
        """
        if "dtype" not in kwargs:
            if self.fp16_enabled():
                kwargs["dtype"] = torch.half
            if self.bfloat16_enabled():
                kwargs["dtype"] = torch.bfloat16

        return torch.zeros(shape, device=self.device, **kwargs)

    def _allocate_buffer(self, shape, num_buffers=-1, **kwargs):
        buffers = []
        if num_buffers == -1:
            num_buffers = self.num_pipe_buffers
        for count in range(num_buffers):
            buffers.append(self._allocate_zeros(shape, **kwargs))
        return buffers

    def _allocate_buffers(self, shapes_and_dtypes, requires_grad=False, num_buffers=-1):
        buffers = []
        if num_buffers == -1:
            num_buffers = self.num_pipe_buffers
        for count in range(num_buffers):
            buffer = []
            for shape, dtype in shapes_and_dtypes:
                buffer.append(self._allocate_zeros(shape, dtype=dtype, requires_grad=requires_grad))
            buffers.append(buffer)
        return buffers

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 512,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None  # support one eos token only
        # TODO: add other fields to match huggingface generation
    ):
        """ Generate sequence """
        self._generate_mode()

        if not self.generate_configured:
            self.max_new_tokens = max_new_tokens
            self.pad_token_id = pad_token_id
            self.eos_token_id = eos_token_id

        self.module.eval()

        # split data into micro batches
        data_dict = dict(raw_input_ids=input_ids, raw_attention_mask=attention_mask)
        data_iter = self._split_data_into_micro_batches(data_dict, self.generation_micro_batches)

        # Use the provided data iterator
        self.set_dataiterator(data_iter)

        # Do the work
        sched = schedule.GenerateInferenceSchedule(micro_batches=self.micro_batches,
                                                   stages=self.num_stages,
                                                   stage_id=self.stage_id,
                                                   max_new_tokens=self.max_new_tokens)

        dist.barrier()
        # prevent dead-lock with multiple evals sequence

        with torch.no_grad():
            self._exec_schedule(sched, terminate_condition=self._generate_term_condition)

        def recursive_del_attr(name):
            sub_names = name.split('.')
            module = self.module
            for sub_name in sub_names:
                attr = getattr(module, sub_name)
                if torch.is_tensor(attr):
                    delattr(module, sub_name)
                    break
                else:
                    module = attr
            torch.cuda.empty_cache()

        names = []
        # TODO: consider async generate requests
        # delete kv_cache after generate finish
        for buf in self.module.named_buffers(recurse=True):
            if "kv_cache" in buf[0]:
                names.append(buf[0])

        for name in names:
            recursive_del_attr(name)

        # output should be seq
        output = self.fwd_outputs
        self._normal_mode()
        # TODO: data parallel, other micro batches

        if len(output) > 0:
            raw_input_ids = [TransformerData.from_tuple(o).raw_input_ids for o in output]
            return torch.cat(raw_input_ids, dim=0)
        else:
            return None

    def forward(self, input_ids, attention_mask):
        self.module.eval()

        # split data into micro batches
        data_dict = dict(raw_input_ids=input_ids, raw_attention_mask=attention_mask)
        data_iter = self._split_data_into_micro_batches(data_dict, self.generation_micro_batches)

        # Use the provided data iterator
        self.set_dataiterator(data_iter)

        # Do the work
        sched = schedule.InferenceSchedule(micro_batches=self.micro_batches,
                                           stages=self.num_stages,
                                           stage_id=self.stage_id)

        # prevent dead-lock with multiple evals sequence
        dist.barrier()

        self._exec_schedule(sched)
        output = self.fwd_outputs

        # Reset any buffers that may have been populated during the forward passes.
        # ds_checkpointing.reset()
        if len(output) > 0:
            logits = [TransformerData.from_tuple(o).logits for o in output]
            # loss = torch.cat(raw_input_ids, dim=0)
            return torch.cat(logits, dim=0)
        else:
            return None

    def backward(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def step(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``. """
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def module_state_dict(self):
        """Override hack to save a pipe model and return the directory path of the save.

        This method should only be called by DeepSpeed's ``save_checkpoint()``. The
        recommended way of saving a ``PipelineModule`` outside of ``save_checkpoint()``
        is ``save_state_dict()``.

        Returns:
            None
        """
        assert isinstance(self.module, PipelineModule)
        assert self._curr_ckpt_path is not None, \
            "PipelineEngine expects module_state_dict() to be called from save_checkpoint()"

        self.module.save_state_dict(self._curr_ckpt_path, checkpoint_engine=self.checkpoint_engine)
        return None

    def load_module_state_dict(self, checkpoint, strict=True, custom_load_fn=None):
        """Override hack to instead use a directory path.

        This is important because pipeline models checkpoint by layer instead of rank.

        If ``state_dict`` is not ``None`` or a ``str``, we revert to ``super()`` expecting a ``dict``.

        Args:
            state_dict (str, None): unused
            strict (bool, optional): Strict state loading. Defaults to True.
        """
        assert custom_load_fn is None, "custom_load_fn not supported w. pipeline parallelism"
        state_dict = checkpoint['module']
        if (state_dict is not None) and (not isinstance(state_dict, str)):
            super().load_module_state_dict(state_dict, strict)
            return

        self.module.load_state_dir(load_dir=self._curr_ckpt_path,
                                   strict=strict,
                                   checkpoint_engine=self.checkpoint_engine)

    # A map of PipeInstruction types to methods. Each method will be executed with the
    # kwargs provided to the PipeInstruction from the scheduler.
    _INSTRUCTION_MAP = {
        schedule.OptimizerStep: _exec_optimizer_step,
        schedule.ReduceGrads: _exec_reduce_grads,
        schedule.ReduceTiedGrads: _exec_reduce_tied_grads,
        schedule.LoadMicroBatch: _exec_load_micro_batch,
        schedule.ForwardPass: _exec_forward_pass,
        schedule.BackwardPass: _exec_backward_pass,
        schedule.SendActivation: _exec_send_activations,
        schedule.RecvActivation: _exec_recv_activations,
        schedule.SendGrad: _exec_send_grads,
        schedule.RecvGrad: _exec_recv_grads,
        schedule.PrepareGenerateInput: _exec_prepare_generate_input,
        schedule.PostprocessGenerateOutput: _exec_postprocess_generate_output,
        # schedule.ClearFwdOutput: _exec_clear_fwd_output
    }

    def _exec_schedule(self, pipe_schedule, terminate_condition=None):
        # TODO: add terminate condition for generate
        # Reserve and reset buffers.
        self._reserve_pipe_buffers(pipe_schedule.num_pipe_buffers())
        self.fwd_outputs = []
        self.generate_outputs = []

        # For each step in the schedule
        step_count = 0
        for step_cmds in pipe_schedule:
            if terminate_condition is not None:
                if terminate_condition():
                    break
            # For each instruction in the step
            # print(f"rank {self.global_rank} step {step_count}, step_cmds: {step_cmds}")
            for cmd in step_cmds:
                # print(f"rank {self.global_rank} exec cmd: {cmd}")
                if type(cmd) not in self._INSTRUCTION_MAP:
                    raise RuntimeError(
                        f'{self.__class__.__name__} does not understand instruction {repr(cmd)}')

                # Equivalent to: self._exec_forward_pass(buffer_id=0)
                try:
                    self._exec_instr = MethodType(self._INSTRUCTION_MAP[type(cmd)], self)
                    self._exec_instr(**cmd.kwargs)
                except Exception as e:
                    print(f"Rank {self.global_rank} step {step_count}, Exception in cmd {cmd}")
                    raise e
            step_count += 1
