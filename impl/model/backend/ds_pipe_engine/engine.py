# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import MethodType
from typing import Optional, Union
import logging
import time

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
from base.dataparallel import PackedParallelDataRouter
from base.namedarray import NamedArray
from impl.model.utils.data import (data_list_to_tensor_tuple, DuckGenerationOutput, DuckModelOutput,
                                   PipeCacheData, PipeTransferData, tensor_tuple_to_data_list)

TARGET_ID = -2
LOG_STAGE = -2
DATA_PARALLEL_ID = -2

logger = logging.getLogger("DeepSpeedPipelineEngine")


def is_even(number):
    return number % 2 == 0


def _tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


class DeepSpeedPipelineEngine(DeepSpeedEngine):
    """ A training engine hybrid pipeline, data, and model parallel training.

    This engine is created by ``deepspeed.initialize()`` when a :class:`PipelineModule`
    is provided.
    """
    ID_TO_DTYPE = [
        torch.float32, torch.float64, torch.complex64, torch.complex128, torch.float16, torch.bfloat16,
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.bool
    ]
    DTYPE_TO_ID = {dtype: id_ for id_, dtype in enumerate(ID_TO_DTYPE)}

    def __init__(self, num_micro_batches=None, *super_args, **super_kwargs):
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

        # pipeline step for logging
        self.log_batch_step_id = -1

        # Set Grid and Communication Groups
        self.grid = self.module._grid

        self.global_rank = self.grid.get_global_rank()

        assert self.dp_world_size == self.grid.data_parallel_size
        self.num_stages = self.grid.pipe_parallel_size
        self.stage_id = self.grid.get_stage_id()

        self.micro_batch_size = self.train_micro_batch_size_per_gpu()
        self.num_micro_batches = num_micro_batches if num_micro_batches else self.num_stages
        self.num_layers = self.module.num_layers  # number of leyers in current pipeline stage

        if self.grid.get_global_rank() == 0:
            logger.info(f'CONFIG: micro_batches={self.num_micro_batches} '
                        f'micro_batch_size={self.micro_batch_size} '
                        f'num_layers={self.num_layers} '
                        f'num_stages={self.num_stages} ')

        self.batch_fn = None

        self._force_grad_boundary = False
        self.pipeline_enable_backward_allreduce = True

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

        # if self._config.pipeline['activation_checkpoint_interval'] > 0:
        #     self.module.activation_checkpoint_interval = self._config.pipeline[
        #         'activation_checkpoint_interval']

        # self.module.checkpoint_parallel_write_pipeline = self._config.checkpoint_parallel_write_pipeline

        self._normal_mode()

        # loss related
        self._compute_loss = False
        self._loss_fn = None
        # store original inputs for each micro_batch to calculate loss
        self.next_batch_micro_batch_id = 0
        self.original_input = []  # fifo queue for original input, only used in last stage

        self.pipe_cache_data = data_list_to_tensor_tuple([PipeCacheData() for _ in range(self.num_layers)])

    def set_loss_fn(self, fn):
        self._loss_fn = fn

    def _normal_mode(self):
        # for train and one step inference
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

    def _prepare_input(self, packed_input_ids: torch.Tensor, cu_seqlens: torch.Tensor,
                       prompt_mask: torch.Tensor):
        """ Prepare input for train or inference
        split all input tensors into micro batches for pipeline parallel
        """
        data = NamedArray(
            packed_input_ids=packed_input_ids,
            cu_seqlens=cu_seqlens,
            prompt_mask=prompt_mask,
        )
        splitted = PackedParallelDataRouter.scatter_to(data, self.num_micro_batches)

        def input_to_pipe_model_input(input: NamedArray):
            max_seqlen = int(max(input.cu_seqlens[1:] - input.cu_seqlens[:-1]))
            x = PipeTransferData(cu_seqlens=input.cu_seqlens, max_seqlen=max_seqlen)
            ys = [PipeCacheData(input_ids=input.packed_input_ids)
                  ] + [PipeCacheData() for _ in range(self.num_layers - 1)]
            return (x, ys)

        return iter([input_to_pipe_model_input(x) for x in splitted])

    def eval(self):
        self.module.eval()

    def train(self):
        self.module.train()

    def forward(self, packed_input_ids: torch.Tensor, cu_seqlens: torch.Tensor, prompt_mask: torch.Tensor):
        self._compute_loss = False
        # Use the provided data iterator
        data_iter = self._prepare_input(packed_input_ids, cu_seqlens, prompt_mask)
        self.set_dataiterator(data_iter)

        # Do the work
        sched = schedule.InferenceSchedule(micro_batches=self.num_micro_batches,
                                           stages=self.num_stages,
                                           stage_id=self.stage_id)

        # prevent dead-lock with multiple evals sequence
        dist.barrier()

        self._exec_schedule(sched)
        output = self.fwd_outputs

        # Reset any buffers that may have been populated during the forward passes.
        # ds_checkpointing.reset()
        if len(output) > 0:
            logits = [PipeTransferData.from_tuple(o).pp_output for o in output]
            # logits = [TransformerData.from_tuple(o).logits for o in output]
            # loss = torch.cat(raw_input_ids, dim=0)
            return DuckModelOutput(logits=torch.cat(logits, dim=0))
        else:
            return None

    def train_batch(self, packed_input_ids: torch.Tensor, cu_seqlens: torch.Tensor,
                    prompt_mask: torch.Tensor):
        """
        """
        if not torch._C.is_grad_enabled():
            raise RuntimeError(f'train_batch() requires gradients enabled. Use eval_batch() instead.')

        data_iter = self._prepare_input(packed_input_ids, cu_seqlens, prompt_mask)
        self.set_dataiterator(data_iter)

        self.total_loss = None
        self._compute_loss = True

        # Do the work
        st = time.time()
        sched = schedule.TrainSchedule(micro_batches=self.num_micro_batches,
                                       stages=self.num_stages,
                                       stage_id=self.stage_id)
        self._exec_schedule(sched)
        self.agg_train_loss = self._aggregate_total_loss()
        train_time = time.time() - st

        if self.global_steps % self.steps_per_print() == 0:
            if self.global_rank == 0:
                elapsed = train_time
                iter_time = elapsed / self.steps_per_print()
                tput = self.train_batch_size() / iter_time
                print(f'steps: {self.global_steps} '
                      f'loss: {self.agg_train_loss:0.4f} '
                      f'iter time (s): {iter_time:0.3f} '
                      f'samples/sec: {tput:0.3f}')

        return self.agg_train_loss

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

    def _next_batch(self):
        # If using 3D parallelism, only some first-stage ranks may do IO
        batch = None
        if self.data_iterator is not None:
            batch = next(self.data_iterator)

        # Any post-processing, like broadcasting across a slice-parallel group.
        # if self.batch_fn:
        #     batch = self.batch_fn(batch)

        return batch

    def _exec_forward_pass(self, buffer_id):
        if isinstance(buffer_id, tuple):
            src_buffer_id, dst_buffer_id = buffer_id
        elif isinstance(buffer_id, int):
            src_buffer_id = dst_buffer_id = buffer_id
        else:
            raise ValueError("buffer_id must be int or tuple of ints")

        assert isinstance(self.pipe_buffers['inputs'][src_buffer_id], tuple)
        inputs = tuple(t.clone() for t in self.pipe_buffers['inputs'][src_buffer_id])
        # TODO: this is only a temp solution to get the pipeline running, fix afterwards
        inputs += self.pipe_cache_data

        self._zero_grads(inputs)

        outputs = super().forward(inputs)

        self.pipe_buffers['outputs'][dst_buffer_id] = outputs

        # if self.is_last_stage() and len(self.fwd_outputs) == self.micro_batches:
        #     self.fwd_outputs.clear()

        if self.is_last_stage():
            if self._compute_loss:  # 1f1b only
                o: PipeTransferData = tensor_tuple_to_data_list(outputs)[0]
                x, y0 = self.original_input.pop(0)
                # compute loss, currently hard coded
                logits = o.pp_output
                packed_input_ids = y0.input_ids
                cu_seqlens = x.cu_seqlens
                loss_mask = torch.ones_like(packed_input_ids,
                                            dtype=torch.float16,
                                            device=torch.cuda.current_device())
                assert self._loss_fn is not None, "loss function is not set, please use engine.set_loss_fn(fn)"
                self.loss = self._loss_fn(logits, packed_input_ids, cu_seqlens, loss_mask)
            else:
                self.fwd_outputs.append(outputs)

    def _exec_backward_pass(self, buffer_id):
        assert self.optimizer is not None, "must provide optimizer during " \
                                           "init in order to use backward"

        # The last stage just runs backward on the loss using DeepSpeed's typical
        # mechanisms.
        if self.is_last_stage():
            super().backward(self.loss)
            return

        outputs = self.pipe_buffers['outputs'][buffer_id]
        grad_tensors = self.grad_layer

        if self.bfloat16_enabled() and not self.is_last_stage():
            # manually call because we don't call optimizer.backward()
            self.optimizer.clear_lp_grads()

        # This handles either a single tensor or tuple of tensors.
        assert isinstance(outputs, tuple)
        out_tensors = [t for t in outputs if t.is_floating_point()]
        assert len(out_tensors) == len(grad_tensors)
        torch.autograd.backward(tensors=out_tensors, grad_tensors=grad_tensors)

        if self.bfloat16_enabled() and not self.is_last_stage():
            # manually call because we don't call optimizer.backward()
            self.optimizer.update_hp_grads(clear_lp_grads=False)

        # Free up the memory from the output of forward()
        self.pipe_buffers['output_tensors'][buffer_id] = None
        self.pipe_buffers['outputs'][buffer_id] = None
        grad_tensors = None

    def _exec_load_micro_batch(self, buffer_id):
        if self.is_first_stage():
            batch, caches = self._next_batch()
            batch = data_list_to_tensor_tuple([batch])  # PipeTransferData as input
            caches = data_list_to_tensor_tuple(caches)
            self.pipe_cache_data = caches
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

        if self._compute_loss:
            # prepare for loss computation, send original inputs to last stage
            if self.is_first_stage():
                # TODO: this is currently hard coded for specific loss computation
                original_inputs = batch + caches
                self._send_tensor_meta(original_inputs, self.num_stages - 1)
                for idx, buffer in enumerate(original_inputs):
                    # logger.info(f"DEBUG:: sending buffer {buffer}, device: {buffer.device}")
                    p2p.send(buffer, self.num_stages - 1)
            elif self.is_last_stage():
                # on last stage recv and store original inputs
                input_buffer = self._recv_tensor_meta(0)
                assert isinstance(input_buffer, tuple)
                recvd = [None] * len(input_buffer)
                for idx, buffer in enumerate(input_buffer):
                    assert torch.is_tensor(buffer)
                    p2p.recv(buffer, 0)
                    recvd[idx] = buffer.clone().detach()

                recvd = tuple(recvd)

                for buffer in recvd:
                    buffer.requires_grad = buffer.is_floating_point()

                original_inputs = tensor_tuple_to_data_list(recvd)
                logger.debug(f"recvd original inputs: {original_inputs[0]}, {original_inputs[1]}")
                self.original_input.append((original_inputs[0], original_inputs[1]))

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

            buffers = self._allocate_buffers(recv_shapes_and_dtypes, num_buffers=1)[0]
            if recv_type == 2:
                buffers = tuple(buffers)
            return buffers
        else:
            raise NotImplementedError(f'Could not receive type {type(recv_type)}')

    def _exec_send_activations(self, buffer_id):
        outputs = self.pipe_buffers['outputs'][buffer_id]

        self._send_tensor_meta(outputs, self.next_stage)

        if isinstance(outputs, torch.Tensor):
            p2p.send(outputs, self.next_stage)
        elif isinstance(outputs, tuple):
            for idx, buffer in enumerate(outputs):
                # logger.info(f"DEBUG:: sending buffer {buffer}, device: {buffer.device}")
                p2p.send(buffer, self.next_stage)
        else:
            raise NotImplementedError('Could not send output of type '
                                      f'{type(outputs)}')

    def _exec_send_grads(self, buffer_id):
        inputs = self.pipe_buffers['inputs'][buffer_id]
        if isinstance(inputs, torch.Tensor):
            assert inputs.grad is not None
            p2p.send(inputs.grad, self.prev_stage)
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

    def _exec_recv_activations(self, buffer_id):
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
                p2p.recv(buffer, self.prev_stage)
                recvd[idx] = buffer.clone().detach()

            recvd = tuple(recvd)

            for buffer in recvd:
                buffer.requires_grad = buffer.is_floating_point()

        self.pipe_buffers['inputs'][buffer_id] = recvd

    def _exec_recv_grads(self, buffer_id):
        outputs = self.pipe_buffers['outputs'][buffer_id]

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
                sizes_and_dtypes = [(list(t.size()), t.dtype) for t in outputs if t.is_floating_point()]
                self.grad_layer = self._allocate_buffers(sizes_and_dtypes, num_buffers=1)[0]

        if isinstance(self.grad_layer, torch.Tensor):
            p2p.recv(self.grad_layer, self.next_stage)
        else:
            assert isinstance(outputs, tuple)
            for idx, buffer in enumerate(self.grad_layer):
                p2p.recv(buffer, self.next_stage)

    def _exec_optimizer_step(self, lr_kwargs=None):
        self._force_grad_boundary = True
        self._take_model_step(lr_kwargs)
        self._force_grad_boundary = False

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
            logger.info(f"rank {self.global_rank} step {step_count}, step_cmds: {step_cmds}")
            for cmd in step_cmds:
                logger.info(f"rank {self.global_rank} exec cmd: {cmd}")
                if type(cmd) not in self._INSTRUCTION_MAP:
                    raise RuntimeError(
                        f'{self.__class__.__name__} does not understand instruction {repr(cmd)}')

                # Equivalent to: self._exec_forward_pass(buffer_id=0)
                try:
                    self._exec_instr = MethodType(self._INSTRUCTION_MAP[type(cmd)], self)
                    self._exec_instr(**cmd.kwargs)
                except Exception as e:
                    logger.error(f"Rank {self.global_rank} step {step_count}, Exception in cmd {cmd}")
                    raise e
            step_count += 1
