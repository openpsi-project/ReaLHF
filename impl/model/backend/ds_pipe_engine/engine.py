# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import MethodType
from typing import Callable, List, Optional, Tuple, Union
import dataclasses
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
import transformers

from . import p2p, schedule
from .module import PipelineError, PipelineModule
from base.dataparallel import PackedParallelDataBroker
from base.monitor import gpu_memory_mb, time_mark
from base.namedarray import NamedArray
from impl.model.nn.flash_mqat import GenerationConfig, genstep
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

        self.sched_count = 0

        # pipeline step for logging
        self.log_batch_step_id = -1

        # Set Grid and Communication Groups
        self.grid = self.module._grid

        self.global_rank = self.grid.get_global_rank()

        assert self.dp_world_size == self.grid.data_parallel_size
        self.num_stages = self.grid.pipe_parallel_size
        self.stage_id = self.grid.get_stage_id()
        self.dp_id = self.grid.data_parallel_id

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
        # stats for microbatches in this batch
        self.stats = []

        #stores the loss for the entire batch # TODO: deprecate this
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
        self._loss_inputs = []
        self._input_cache = []

        # store original inputs for each micro_batch to calculate loss
        self.next_batch_micro_batch_id = 0
        self.pipe_cache_data = {
            i: [PipeCacheData() for _ in range(self.num_layers)]
            for i in range(self.num_micro_batches)
        }

        # for generation
        self.kv_cache_reserved = []
        self.tokenizer = None
        self.gconfig = None
        self.next_tokens_cache = {}  # only for first stage
        self.next_tokens_to_send = None  # only for last stage
        self.generated_idx = {}
        self.terminate = {}
        self.unfinished_sequences = {}  # micro batch to unfinished seqs
        self.last_logits = None
        self.gen_token_ph = {}
        self.gen_logprob_ph = {}
        self.gen_logits_mask_ph = {}
        # self.batch_size = None
        self.batch_lengths = []
        self.prompt_logits = []
        self.generate_mode = False

        # optimizer lr scheduler variables
        self.version_steps = 0

    def set_loss_fn(self, fn):
        self._loss_fn = fn

    def set_version_steps(self, version_steps):
        # version_steps = batch id (not micro batch !!)
        self.version_steps = version_steps

    def _normal_mode(self):
        # for train and one step inference
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1
        self._initialize_p2p()
        self.generate_mode = False
        self.kv_cache_reserved = []
        self._loss_inputs = []
        self._input_cache = []
        self.stats = []
        self.pipe_cache_data = {
            i: [PipeCacheData() for _ in range(self.num_layers)]
            for i in range(self.num_micro_batches)
        }

    def _generate_mode(self, tokenizer, gconfig):
        self.prev_stage = (self.stage_id - 1) % self.num_stages
        self.next_stage = (self.stage_id + 1) % self.num_stages
        self._initialize_p2p()
        self.tokenizer = tokenizer
        self.gconfig = gconfig
        self.terminate = {i: False for i in range(self.num_micro_batches)}
        # self.batch_lengths = []
        self.pipe_cache_data = {
            i: [PipeCacheData() for _ in range(self.num_layers)]
            for i in range(self.num_micro_batches)
        }
        self.stats = []
        self.unfinished_sequences = {
            i: torch.ones(self.batch_lengths[i], dtype=torch.long, device=self.device)
            for i in range(self.num_micro_batches)
        }
        self.generated_idx = {i: 0 for i in range(self.num_micro_batches)}
        self.gen_token_ph = {i: [] for i in range(self.num_micro_batches)}
        self.gen_logprob_ph = {i: [] for i in range(self.num_micro_batches)}
        self.gen_logits_mask_ph = {i: [] for i in range(self.num_micro_batches)}
        self.generate_mode = True
        self.kv_cache_reserved = []
        self.prompt_logits = []

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

    def _prepare_input(self, packed_input_ids: torch.Tensor, cu_seqlens: torch.Tensor):
        """ Prepare input for train or inference
        split all input tensors into micro batches for pipeline parallel

        Args:
            packed_input_ids (torch.Tensor): packed input ids of shape [total_seq_len]
            cu_seqlens (torch.Tensor): cu_seqlens of shape [batch_size]
        """
        data = NamedArray(packed_input_ids=packed_input_ids, cu_seqlens=cu_seqlens)
        splitted = PackedParallelDataBroker.scatter_to(data, self.num_micro_batches)
        self._input_cache = splitted

        def input_to_pipe_model_input(input: NamedArray):
            max_seqlen = int(max(input.cu_seqlens[1:] - input.cu_seqlens[:-1]))
            x = PipeTransferData(cu_seqlens=input.cu_seqlens, max_seqlen=max_seqlen)
            if self.is_first_stage():
                ys = [PipeCacheData(input_ids=input.packed_input_ids)
                      ] + [PipeCacheData() for _ in range(self.num_layers - 1)]
            else:
                ys = [PipeCacheData() for _ in range(self.num_layers)]
            return (x, ys)

        batches = [input_to_pipe_model_input(x) for x in splitted]
        batch_lengths = []
        for i, b in enumerate(batches):
            cu_seqlens = b[0].cu_seqlens
            batch_lengths.append(cu_seqlens.shape[0] - 1)
        # batch_lengths = [b[1][0].input_ids.shape[0] for b in batches]
        # logger.debug("self._prepare_input:: batch_lengths: {}".format(batch_lengths))
        self.batch_lengths = batch_lengths
        return iter(batches)

    def _prepare_loss_input(self, **loss_kwargs):
        data = NamedArray(**loss_kwargs)
        splitted = PackedParallelDataBroker.scatter_to(data, self.num_micro_batches)
        self._loss_inputs = splitted

    def eval(self):
        self.module.eval()

    def train(self):
        self.module.train()

    def forward(self, packed_input_ids: torch.Tensor, cu_seqlens: torch.Tensor):
        self._normal_mode()
        self._compute_loss = False
        # Use the provided data iterator
        data_iter = self._prepare_input(packed_input_ids, cu_seqlens)
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
            logits = [o.pp_input for o in output]
            # logits = [TransformerData.from_tuple(o).logits for o in output]
            # loss = torch.cat(raw_input_ids, dim=0)
            return DuckModelOutput(logits=torch.cat(logits, dim=0))
        else:
            return None

    def train_batch(self, packed_input_ids: torch.Tensor, cu_seqlens: torch.Tensor, loss_fn: Callable,
                    **loss_fn_kwargs):
        self._normal_mode()
        if not torch._C.is_grad_enabled():
            raise RuntimeError(f'train_batch() requires gradients enabled. Use eval_batch() instead.')

        # print(f"in train_batch packed_input_ids shape {packed_input_ids.shape}")
        data_iter = self._prepare_input(packed_input_ids, cu_seqlens)
        self.set_loss_fn(loss_fn)
        self._prepare_loss_input(**loss_fn_kwargs)
        self.set_dataiterator(data_iter)

        self.total_loss = None
        self._compute_loss = True

        # Do the work
        sched = schedule.TrainSchedule(micro_batches=self.num_micro_batches,
                                       stages=self.num_stages,
                                       stage_id=self.stage_id)
        self._exec_schedule(sched)

        return self.stats

    @torch.no_grad()
    def generate(
        self,
        tokenizer: transformers.PreTrainedTokenizerFast,
        packed_input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        gconfig: GenerationConfig = dataclasses.field(default_factory=GenerationConfig),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[PipeCacheData]]:
        self._compute_loss = False
        data_iter = self._prepare_input(packed_input_ids, cu_seqlens)
        self._generate_mode(tokenizer=tokenizer, gconfig=gconfig)
        self.set_dataiterator(data_iter)

        logger.debug(f"GenerateSchedule:: \n"
                     f"micro_batches={self.num_micro_batches} \n"
                     f"stages={self.num_stages} \n"
                     f"stage_id={self.stage_id} \n"
                     f"max_new_tokens={gconfig.max_new_tokens} \n")
        sched = schedule.GenerateSchedule(micro_batches=self.num_micro_batches,
                                          stages=self.num_stages,
                                          stage_id=self.stage_id,
                                          max_new_tokens=gconfig.max_new_tokens)

        def terminate_condition():
            return all(list(self.terminate.values()))

        self._exec_schedule(sched, terminate_condition)
        logger.debug(f"rank {self.global_rank} schedule complete")
        if self.is_last_stage():
            all_gen_tokens = []
            all_log_probs = []
            all_logits_mask = []
            vocab_size = -1
            for i in range(self.num_micro_batches):
                gen_tokens = torch.stack(self.gen_token_ph[i], -1)
                log_probs = torch.stack(self.gen_logprob_ph[i], -1)
                if all([m is None for m in self.gen_logits_mask_ph[i]]):
                    logits_mask = None
                else:
                    mm = next(m for m in self.gen_logits_mask_ph[i] if m is not None)
                    self.gen_logits_mask_ph[i] = [
                        torch.ones_like(mm) if m is None else m for m in self.gen_logits_mask_ph[i]
                    ]
                    logits_mask = torch.stack(self.gen_logits_mask_ph[i], -2)
                all_gen_tokens.append(gen_tokens)
                all_log_probs.append(log_probs)
                all_logits_mask.append(logits_mask)
                if torch.is_tensor(gen_tokens) and torch.is_tensor(log_probs) and \
                    torch.is_tensor(logits_mask):
                    vocab_size = logits_mask.shape[-1]
                    logger.info(
                        f"generation on microbatch {i}: {gen_tokens.shape} {log_probs.shape} {logits_mask.shape}"
                    )

            # if sequence is terminated, there might be situations where tensors in all_gen_tokens have difference shapes
            gen_tokens_lengths = [t.shape[-1] for t in all_gen_tokens]
            max_gen_tokens_length = max(gen_tokens_lengths)
            for i in range(len(all_gen_tokens)):
                assert all_gen_tokens[i].shape == log_probs[i].shape
                if all_gen_tokens[i].shape[-1] < max_gen_tokens_length:
                    device = all_gen_tokens[i].device
                    # if t.shape[-1] < max_gen_tokens_length, pad it with zeros
                    pad_shape = all_gen_tokens[i].shape[:-1] + (max_gen_tokens_length -
                                                                all_gen_tokens[i].shape[-1],)
                    all_gen_tokens[i] = torch.cat([
                        all_gen_tokens[i],
                        torch.full(pad_shape, self.tokenizer.pad_token_id, device=device)
                    ],
                                                  dim=1)
                    # hack for log_probs and logits_mask, check if correct
                    all_log_probs[i] = torch.cat(
                        [all_log_probs[i], torch.zeros(pad_shape, device=device)], dim=1)
                    if all_logits_mask[i] is not None:
                        all_logits_mask[i] = torch.cat(
                            [all_logits_mask[i],
                             torch.ones((*pad_shape, vocab_size), device=device)], dim=1)

            gen_tokens = torch.cat(all_gen_tokens, dim=0)
            log_probs = torch.cat(all_log_probs, dim=0)
            if all([m is None for m in all_logits_mask]):
                logits_mask = None
            else:
                mm = next(m for m in all_logits_mask if m is not None)
                all_logits_mask = [torch.ones_like(mm) if m is None else m for m in all_logits_mask]
                logits_mask = torch.cat(all_logits_mask, dim=0)
            # self._normal_mode()
            prompt_logits = torch.cat(self.prompt_logits, dim=0)
            return gen_tokens, log_probs, logits_mask, None, prompt_logits
        else:
            return None

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
            # logger.debug("self._aggregate_total_loss:: total_loss: {}, loss: {}, gas: {}".format(
            #     self.total_loss, loss, self.gradient_accumulation_steps()))
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

    def _exec_forward_pass(self, buffer_id, micro_batch_id):
        assert micro_batch_id >= 0
        if isinstance(buffer_id, tuple):
            src_buffer_id, dst_buffer_id = buffer_id
        elif isinstance(buffer_id, int):
            src_buffer_id = dst_buffer_id = buffer_id
        else:
            raise ValueError("buffer_id must be int or tuple of ints")

        ys = self.pipe_cache_data[micro_batch_id]
        assert isinstance(self.pipe_buffers['inputs'][src_buffer_id], tuple)
        # inputs = tuple(t.clone() for t in self.pipe_buffers['inputs'][src_buffer_id])
        inputs = self.pipe_buffers['inputs'][src_buffer_id]
        # TODO: this is only a temp solution to get the pipeline running, fix afterwards
        inputs += data_list_to_tensor_tuple(ys)

        # if not self.generate_mode:
        #     for i, y in enumerate(ys):
        #         logger.info(f"rank {self.global_rank} layer {i} k_cache {y.k_cache}")
        self._zero_grads(inputs)

        x, ys = super().forward(inputs)

        self.pipe_cache_data[micro_batch_id] = ys
        self.pipe_buffers['outputs'][dst_buffer_id] = data_list_to_tensor_tuple([x])

        if self.generate_mode:
            logits = x.pp_input.squeeze(dim=1)
            # if kv cache is not reserved for this micro batch
            if micro_batch_id not in self.kv_cache_reserved:
                # store prompt logits
                self.prompt_logits.append(logits)
                # reserve kv cache
                cu_seqlens = x.cu_seqlens
                logits = logits[cu_seqlens[1:] - 1]
                input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
                max_seq_len = int(max(input_lens))

                if self.is_first_stage():
                    ys[0].cache_seqlens = input_lens.clone().to(dtype=torch.int32)
                    ys = ys[1:]
                if self.is_last_stage():
                    ys = ys[:-1]

                bs = len(input_lens)
                for y in ys:
                    assert y.k_cache is not None and y.v_cache is not None and y.cache_seqlens is not None
                    kvcache_seqlen = max(max_seq_len + self.gconfig.max_new_tokens, 256)
                    k_cache = torch.zeros((bs, kvcache_seqlen, *y.k_cache.shape[1:]),
                                          dtype=y.k_cache.dtype,
                                          device=self.device)
                    v_cache = torch.zeros((bs, kvcache_seqlen, *y.v_cache.shape[1:]),
                                          dtype=y.v_cache.dtype,
                                          device=self.device)
                    for i in range(bs):
                        k_cache[i, :input_lens[i]] = y.k_cache[cu_seqlens[i]:cu_seqlens[i + 1]]
                        v_cache[i, :input_lens[i]] = y.v_cache[cu_seqlens[i]:cu_seqlens[i + 1]]
                    y.k_cache = k_cache
                    # logger.debug("in _exec_forward_pass():: rank {} in reserve y k_cache: {}"\
                    #              .format(self.global_rank, i, y.k_cache))
                    y.v_cache = v_cache
                    y.cache_seqlens = input_lens.clone().to(dtype=torch.int32)

                # for i, y in enumerate(ys):
                #     logger.debug("in _exec_forward_pass():: rank {} mbid {} in reserve ys[{}] cache_seqlens: {}, cu_seqlens: {}, input_lens: {}"\
                #                  .format(self.global_rank, micro_batch_id, i, y.cache_seqlens, x.cu_seqlens, input_lens))
                self.kv_cache_reserved.append(micro_batch_id)
            else:
                # else, only increase cache_seqlens
                if self.is_last_stage():
                    ys = ys[:-1]
                for y in ys:
                    y.cache_seqlens += 1

            if self.is_last_stage():
                next_tokens, logprob, logits_mask, terminate, unfinished_sequences = genstep(
                    logits, self.tokenizer, self.unfinished_sequences[micro_batch_id],
                    self.generated_idx[micro_batch_id], self.gconfig)
                self.terminate[micro_batch_id] = terminate
                self.unfinished_sequences[micro_batch_id] = unfinished_sequences
                self.generated_idx[micro_batch_id] += 1
                assert next_tokens is not None and logprob is not None
                self.gen_token_ph[micro_batch_id].append(next_tokens)
                self.gen_logprob_ph[micro_batch_id].append(logprob)
                self.gen_logits_mask_ph[micro_batch_id].append(logits_mask)
                self.next_tokens_to_send = next_tokens
        else:
            if self.is_last_stage():
                if self._compute_loss:  # 1f1b only
                    # compute loss, currently hard coded
                    logits = x.pp_input
                    loss_kwargs = self._loss_inputs.pop(0)
                    input_cache = self._input_cache.pop(0)
                    packed_input_ids = input_cache.packed_input_ids
                    cu_seqlens = input_cache.cu_seqlens
                    assert self._loss_fn is not None, "loss function is not set, please use engine.set_loss_fn(fn)"
                    self.loss, stats = self._loss_fn(logits, packed_input_ids, cu_seqlens, **loss_kwargs)
                    self.stats.append(stats)
                    if self.total_loss is None:
                        self.total_loss = torch.zeros_like(self.loss)
                    self.total_loss += self.loss.detach()
                else:
                    self.fwd_outputs.append(x)

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
        self.grad_layer = None

    def _exec_load_micro_batch(self, buffer_id, micro_batch_id):
        assert micro_batch_id >= 0
        if self.is_first_stage():
            batch, caches = self._next_batch()
            batch = data_list_to_tensor_tuple([batch])  # PipeTransferData as input
            # caches = data_list_to_tensor_tuple(caches)
            self.pipe_cache_data[micro_batch_id] = caches
            # batch should be a tuple in all conditions
            # all stages should be the same
            assert isinstance(batch, (tuple, list))
            # Assume list or tuple
            loaded = []
            for x in batch:
                assert torch.is_tensor(x)
                # mine = x.clone().detach().to(self.device) # TODO: check if this is necessary
                mine = x.to(self.device)
                mine.requires_grad = mine.is_floating_point()
                loaded.append(mine)
            loaded = tuple(loaded)
            self.pipe_buffers['inputs'][buffer_id] = loaded

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
                p2p.send(buffer, self.next_stage)
        else:
            raise NotImplementedError('Could not send output of type '
                                      f'{type(outputs)}')

    def _exec_recv_activations(self, buffer_id):
        recvd = None

        # Allocate the buffer if necessary
        self.pipe_recv_buf = self._recv_tensor_meta(self.prev_stage)

        # pipe_recv_buf = self._recv_tensor_meta(self.prev_stage)
        # if pipe_recv_buf is not None:
        #     self.pipe_recv_buf = pipe_recv_buf

        if isinstance(self.pipe_recv_buf, torch.Tensor):
            p2p.recv(self.pipe_recv_buf, self.prev_stage)
            recvd = self.pipe_recv_buf.clone().detach()  # TODO: check if this is necessary
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

    def _exec_send_next_tokens(self, buffer_id):
        """ When generating, send next tokens from the last stage to the first stage.
        Here buffer_id is the next token id.
        """
        assert self.is_last_stage(), "_exec_send_next_tokens() should be only executed on the last stage"
        self._send_tensor_meta((self.next_tokens_to_send,), self.next_stage)
        p2p.send(self.next_tokens_to_send, self.next_stage)

    def _exec_recv_next_tokens(self, buffer_id):
        """ When generating, recv next tokens from the last stage on the first stage
        Here buffer_id is the micro_batch_id.
        """
        assert self.is_first_stage(), "_exec_recv_next_tokens() should be only executed on the first stage"
        recv_buf = self._recv_tensor_meta(self.prev_stage)
        p2p.recv(recv_buf[0], self.prev_stage)
        recvd = recv_buf[0].clone().detach()
        self.next_tokens_cache[buffer_id] = recvd

    def _exec_load_next_tokens(self, buffer_id, micro_batch_id):
        """ When continuing to generate tokens, load the previous next_tokens from the cache
        Here buffer_id is the micro_batch_id.
        """
        assert micro_batch_id >= 0
        assert self.is_first_stage(), "_exec_load_next_tokens() should be only executed on the first stage"
        assert buffer_id in self.next_tokens_cache, f"next tokens cache of micro batch id {buffer_id} is empty"
        x = PipeTransferData()
        ys = self.pipe_cache_data[micro_batch_id]
        ys[0].input_ids = self.next_tokens_cache[micro_batch_id].unsqueeze(-1)
        ys[0].position_ids = None
        t = data_list_to_tensor_tuple([x])
        self.pipe_buffers['inputs'][buffer_id] = t

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
                assert buffer.grad is not None, f"buffer {idx} does not have a grad, tensor: {buffer}"
                p2p.send(buffer.grad, self.prev_stage)

        # We can free up the input buffer now
        self.pipe_buffers['inputs'][buffer_id] = None

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

    def _exec_optimizer_step(self):
        self._force_grad_boundary = True
        self._take_model_step(lr_kwargs={'epoch': self.version_steps})
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
        schedule.SendNextTokens: _exec_send_next_tokens,
        schedule.RecvNextTokens: _exec_recv_next_tokens,
        schedule.LoadNextTokens: _exec_load_next_tokens,
        # schedule.
    }

    def _exec_schedule(self, pipe_schedule, terminate_condition=None):
        """ Execute schedules
        Args:
            pipe_schedule: an instance of schedule
            terminate_condition: a callable that returns boolean value indicating if 
                                 the pipeline execution should terminate
        """
        # Reserve and reset buffers.
        self._reserve_pipe_buffers(pipe_schedule.num_pipe_buffers())
        self.fwd_outputs = []
        self.generate_outputs = []

        # For each step in the schedule
        self.step_count = 0
        for step_cmds in pipe_schedule:
            if terminate_condition is not None:
                terminate_tensor = torch.tensor(0, dtype=torch.int32, device=self.device)
                if terminate_condition():
                    terminate_tensor = torch.tensor(1, dtype=torch.int32, device=self.device)
                    logger.info(f"rank {self.global_rank} reach terminate condition")
                dist.all_reduce(terminate_tensor)
                if terminate_tensor.item() > 0:
                    logger.info(f"{self.global_rank} terminate")
                    break
            # For each instruction in the step
            step_id, micro_batch_id, step_cmds = step_cmds
            logger.debug(
                f"rank {self.global_rank} step {self.step_count}, st {step_id} mb {micro_batch_id} step_cmds: {step_cmds}"
            )
            for cmd in step_cmds:
                logger.debug(f"rank {self.global_rank} exec cmd: {cmd}")
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
