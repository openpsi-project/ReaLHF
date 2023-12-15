# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import MethodType
from typing import Callable, List, Optional, Tuple, Union
import dataclasses

from deepspeed import comm as dist
from deepspeed.runtime.bf16_optimizer import BF16_Optimizer
from deepspeed.runtime.engine import DeepSpeedEngine, MEMORY_OPT_ALLREDUCE_SIZE
from deepspeed.runtime.zero.config import ZeroStageEnum
import torch
import transformers

from base.dataparallel import PackedParallelDataBroker
from base.monitor import time_mark
from base.namedarray import NamedArray
from base.topology import PipelineParallelGrid
from impl.model.nn.flash_mqat.flash_generate import GenerationConfig, genstep
from impl.model.utils.data import PipeCacheData, PipeTransferData
from impl.model.utils.pipeline_module import PipelineError, PipelineModule
from impl.model.utils.tensor_storage import recv_grad, send_grad, TensorBuffer
import base.constants
import base.logging as logging
import impl.model.backend.pipe_engine.static_schedule as schedule
import impl.model.utils.p2p as p2p

logger = logging.getLogger("DeepSpeedPipelineEngine", "benchmark")


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

        # deepspeed enigne attributes
        self.pipeline_parallelism = True
        # We schedule the all-reduces, so disable it in super().backward()
        self.enable_backward_allreduce = False
        # see method is_gradient_accumulation_boundary()
        self._force_grad_boundary = False
        self.using_bf16_optimizer = type(self.optimizer) == BF16_Optimizer

        # configs for data shape
        self.config = self.module.config  # FlashMQATConfig in PipelineModule
        # for tensor buffer initialization
        self.hidden_dim = self.config.hidden_dim
        self.head_dim = self.config.head_dim
        self.n_kv = self.config.n_kv_heads

        # logging
        self.sched_count = 0

        # parallelism constants
        self.grid: PipelineParallelGrid = base.constants.grid()
        assert self.dp_world_size == self.grid.data_parallel_size

        self.global_rank = self.grid.get_global_rank()
        self.num_stages = self.grid.get_pipe_parallel_world_size()
        self.stage_id = self.grid.get_stage_id()
        self.dp_id = self.grid.get_data_parallel_id()
        self.num_micro_batches = num_micro_batches if num_micro_batches else self.num_stages
        # num_micro_batches is configurable, default value: 2 * num_stages
        self.num_layers = self.module.num_layers  # number of leyers in current pipeline stage

        # PipelineEngine needs to handle data loading specially due to only the first
        # and last stages loading inputs/labels. We construct a sampler that uses

        self.is_pipe_parallel = self.grid.pipe_parallel_size > 1
        assert self.is_pipe_parallel, "Must use pipeline parallelism with PipelineModule"
        self.is_data_parallel = self.grid.data_parallel_size > 1
        self.is_model_parallel = self.grid.model_parallel_size > 1

        self.prev_stage = (self.stage_id - 1) % self.num_stages
        self.next_stage = (self.stage_id + 1) % self.num_stages

        # initialize communication
        self._initialize_comm()

        # storages
        self.tensor_buffer = TensorBuffer()

        # TODO: add activation checkpoints
        # schedule execution states
        self.tokenizer = None
        self.current_gconfig = None

        # loss related
        self._compute_loss = False
        self._loss_fn = None  # set when train_batch is called

        self.kv_cache_reserved = []

        # optimizer lr scheduler variables
        self.version_steps = None

        self._post_init_logging()

    def _post_init_logging(self):
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

        if self.global_rank == 0:
            logger.info(f'CONFIG: num_micro_batches={self.num_micro_batches} '
                        f'num_layers(this stage)={self.num_layers} '
                        f'pp_size={self.num_stages} '
                        f'dp_size={self.grid.get_data_parallel_world_size()} '
                        f'mp_size={self.grid.get_model_parallel_world_size()} '
                        f'bf16={self.bfloat16_enabled()} ')
        if self.dp_id == 0:
            logger.info(f'rank={self.global_rank} '
                        f'stage={self.stage_id} '
                        f'layers={self.module._local_stop - self.module._local_start} '
                        f'[{self.module._local_start}, {self.module._local_stop}) '
                        f'stage_params={num_params} ({num_params/1e6:0.3f}M) '
                        f'total_params={total_params} ({total_params/1e6:0.3f}M) '
                        f'unique_params={unique_params} ({unique_params/1e6:0.3f}M)')

    def is_first_stage(self):
        """True if this process is in the first stage in the pipeline."""
        return self.stage_id == 0

    def is_last_stage(self):
        """True if this process is in the last stage in the pipeline."""
        return self.stage_id == self.num_stages - 1

    def is_gradient_accumulation_boundary(self):
        """True if the engine is executing a gradient reduction or optimizer step instruction.

        This is overridden from :class:`DeepSpeedEngine` to force reductions
        and steps when the pipeline engine is instructed to do so.

        Returns:
            bool: whether reductions and optimizer steps should occur.
        """
        return self._force_grad_boundary

    def gradient_checkpointing_enable(self):
        self.module.gradient_checkpointing_enable()

    def _prepare_input(self,
                       packed_input_ids: torch.Tensor,
                       cu_seqlens: torch.Tensor,
                       input_lens_for_partition: Optional[torch.Tensor] = None,
                       generate_mode: bool = False):
        """ Prepare input for train or inference
        split all input tensors into micro batches for pipeline parallel

        Args:
            packed_input_ids (torch.Tensor): packed input ids of shape [total_seq_len]
            cu_seqlens (torch.Tensor): cu_seqlens of shape [batch_size]
        """
        if input_lens_for_partition is not None:
            pair_input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            data = NamedArray(packed_input_ids=packed_input_ids,
                              pair_input_lens=pair_input_lens,
                              input_lens=input_lens_for_partition)
        else:
            data = NamedArray(packed_input_ids=packed_input_ids, cu_seqlens=cu_seqlens)
        splitted = PackedParallelDataBroker.scatter_to(data, self.num_micro_batches)
        if input_lens_for_partition is not None:
            splitted = [
                NamedArray(packed_input_ids=x['packed_input_ids'],
                           cu_seqlens=torch.nn.functional.pad(x['pair_input_lens'].cumsum(0), (1, 0)))
                for x in splitted
            ]
        if not generate_mode:
            for mbid, x in enumerate(splitted):
                self.tensor_buffer.put("input_cache", mbid, x)
        mb_seq_lens = []

        def input_to_pipe_model_input(input: NamedArray):
            max_seqlen = torch.tensor(int(max(input.cu_seqlens[1:] - input.cu_seqlens[:-1]))).cuda()
            store_kv_cache = generate_mode

            cu_seqlens = input.cu_seqlens.to(self.device)
            packed_input_ids = input.packed_input_ids.to(self.device)

            x = PipeTransferData(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, store_kv_cache=store_kv_cache)
            if self.is_first_stage():
                ys = [PipeCacheData(input_ids=packed_input_ids)
                      ] + [PipeCacheData() for _ in range(self.num_layers - 1)]
            else:
                ys = [PipeCacheData() for _ in range(self.num_layers)]
            mb_seq_lens.append(packed_input_ids.shape[0])
            return (x, ys)

        batches = [input_to_pipe_model_input(x) for x in splitted]
        for mbid, batch in enumerate(batches):
            x, ys = batch
            self.tensor_buffer.put("batch_input_x", mbid, x)
            self.tensor_buffer.put("batch_input_ys", mbid, ys)
            self.tensor_buffer.put("batch_lengths", mbid, x.cu_seqlens.shape[0] - 1)

        # pre allocate receive buffers and pre store other information
        for mbid, batch in enumerate(batches):
            if not generate_mode:
                dtype = torch.half if not self.bfloat16_enabled() else torch.bfloat16
                activation_shape = (mb_seq_lens[mbid], self.hidden_dim)
                self.tensor_buffer.alloc("activation",
                                         mbid,
                                         activation_shape,
                                         dtype,
                                         self.device,
                                         require_grads=True)
                self.tensor_buffer.alloc("grad", mbid, activation_shape, dtype, self.device)
            others_cache = dict(cu_seqlens=batch[0].cu_seqlens,
                                max_seqlen=batch[0].max_seqlen,
                                store_kv_cache=batch[0].store_kv_cache)
            self.tensor_buffer.put("pipe_transfer_infos", mbid, others_cache)

    def _prepare_loss_input(self, **loss_kwargs):
        data = NamedArray(**loss_kwargs)
        splitted = PackedParallelDataBroker.scatter_to(data, self.num_micro_batches)
        for mbid, x in enumerate(splitted):
            self.tensor_buffer.put("loss_inputs", mbid, x)

    def _pre_generate(self):
        self.kv_cache_reserved = []  # ids of micro batches that have reserved kv cache
        self._compute_loss = False
        self._generating = True

        for mbid in range(self.num_micro_batches):
            self.tensor_buffer.put("kv_cache_reserved", mbid, False)
            self.tensor_buffer.put("terminate", mbid, False)
            self.tensor_buffer.put("generated_idx", mbid, 0)
            batch_length = self.tensor_buffer.get("batch_lengths", mbid)
            self.tensor_buffer.put("unfinished_sequences", mbid,
                                   torch.ones(batch_length, dtype=torch.long, device=self.device))
            self.tensor_buffer.put("gen_token_ph", mbid, [])
            self.tensor_buffer.put("gen_logprob_ph", mbid, [])
            self.tensor_buffer.put("gen_logits_mask_ph", mbid, [])

    def _post_generate(self):
        # reset states
        self._generating = False

        # clear tensors
        self.tensor_buffer.remove("next_tokens_cache")
        self.tensor_buffer.remove("next_tokens_to_send")
        self.tensor_buffer.remove("generated_idx")
        self.tensor_buffer.remove("terminate")
        self.tensor_buffer.remove("unfinished_sequences")
        self.tensor_buffer.remove("gen_token_ph")
        self.tensor_buffer.remove("gen_logprob_ph")
        self.tensor_buffer.remove("gen_logits_mask_ph")
        self.tensor_buffer.remove("batch_lengths")
        self.tensor_buffer.remove("prompt_logits")
        self.tensor_buffer.remove("kv_cache_reserved")
        self.tensor_buffer.remove("batch_input_x")
        self.tensor_buffer.remove("batch_input_ys")
        self.tensor_buffer.remove("input_cache")

    def _pre_eval_batch(self):
        self._compute_loss = True
        self._generating = False

    def _post_eval_batch(self):
        self.tensor_buffer.remove("batch_input_x")
        self.tensor_buffer.remove("batch_input_ys")
        self.tensor_buffer.remove("batch_lengths")
        self.tensor_buffer.remove("loss_inputs")
        self.tensor_buffer.remove("input_cache")
        self.tensor_buffer.remove("losses")
        self.tensor_buffer.remove("stats")

    def _pre_forward(self):
        self._compute_loss = False
        self._generating = False

    def _post_forward(self):
        self._post_eval_batch()
        self.tensor_buffer.remove("logits")

    def _pre_train_batch(self):
        self._compute_loss = True
        self._generating = False

    def _post_train_batch(self):
        self._post_eval_batch()

    def set_version_steps(self, version_steps):
        self.version_steps = version_steps

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def _initialize_comm(self):
        p2p.init_process_groups(self.grid)
        # check connectivity
        buf = torch.zeros(1, dtype=torch.int32).cuda()
        if self.stage_id % 2 == 0:
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

    def eval(self):
        self.module.eval()

    def train(self):
        self.module.train()

    def forward(self,
                packed_input_ids: torch.Tensor,
                cu_seqlens: torch.Tensor,
                input_lens_for_partition: Optional[torch.Tensor] = None):
        # forward one step and return packed logits
        self._prepare_input(packed_input_ids,
                            cu_seqlens,
                            generate_mode=False,
                            input_lens_for_partition=input_lens_for_partition)
        self._pre_forward()
        sched = schedule.InferenceSchedule(micro_batches=self.num_micro_batches,
                                           stages=self.num_stages,
                                           stage_id=self.stage_id)
        self._exec_schedule(sched)

        logits = None
        if self.is_last_stage():
            logits_list = []
            for i in range(self.num_micro_batches):
                logits = self.tensor_buffer.get("logits", i, remove=True)
                logits_list.append(logits)
            logits = torch.cat(logits_list, dim=0)

        self._post_forward()
        return logits

    def eval_batch(self,
                   packed_input_ids: torch.Tensor,
                   cu_seqlens: torch.Tensor,
                   loss_fn: Callable,
                   input_lens_for_partition: Optional[torch.Tensor] = None,
                   **loss_fn_kwargs):
        self._prepare_input(packed_input_ids,
                            cu_seqlens,
                            generate_mode=False,
                            input_lens_for_partition=input_lens_for_partition)
        self._loss_fn = loss_fn
        self._prepare_loss_input(**loss_fn_kwargs)
        self._pre_eval_batch()

        # Do the work
        sched = schedule.InferenceSchedule(micro_batches=self.num_micro_batches,
                                           stages=self.num_stages,
                                           stage_id=self.stage_id)

        # prevent dead-lock with multiple evals sequence
        # dist.barrier()

        self._exec_schedule(sched)

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

    def train_batch(self,
                    packed_input_ids: torch.Tensor,
                    cu_seqlens: torch.Tensor,
                    loss_fn: Callable,
                    input_lens_for_partition: Optional[torch.Tensor] = None,
                    **loss_fn_kwargs):
        if not torch._C.is_grad_enabled():
            raise RuntimeError(f'train_batch() requires gradients enabled. Use eval_batch() instead.')

        self._prepare_input(packed_input_ids,
                            cu_seqlens,
                            generate_mode=False,
                            input_lens_for_partition=input_lens_for_partition)
        self._loss_fn = loss_fn
        self._prepare_loss_input(**loss_fn_kwargs)
        self._pre_train_batch()

        # Do the work
        sched = schedule.TrainSchedule(micro_batches=self.num_micro_batches,
                                       stages=self.num_stages,
                                       stage_id=self.stage_id)
        self._exec_schedule(sched)

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

    @torch.no_grad()
    def generate(
        self,
        packed_input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        tokenizer: transformers.PreTrainedTokenizerFast,
        gconfig: GenerationConfig = dataclasses.field(default_factory=GenerationConfig),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[PipeCacheData]]:
        self._prepare_input(packed_input_ids, cu_seqlens, generate_mode=True)
        self.current_gconfig = gconfig
        self.tokenizer = tokenizer
        self._pre_generate()

        sched = schedule.GenerateSchedule(micro_batches=self.num_micro_batches,
                                          stages=self.num_stages,
                                          stage_id=self.stage_id,
                                          max_new_tokens=gconfig.max_new_tokens)

        def terminate_condition():
            return all([self.tensor_buffer.get("terminate", mbid) for mbid in range(self.num_micro_batches)])

        self._exec_schedule(sched, terminate_condition)
        r = self.__maybe_gather_generate_outputs()
        self._post_generate()
        return r

    def __maybe_gather_generate_outputs(self):
        if not self.is_last_stage():
            return None

        all_gen_tokens = []
        all_log_probs = []
        all_logits_mask = []
        vocab_size = None
        for mbid in range(self.num_micro_batches):
            gen_token_ph = self.tensor_buffer.get("gen_token_ph", mbid)
            gen_logprob_ph = self.tensor_buffer.get("gen_logprob_ph", mbid)
            gen_logits_mask_ph = self.tensor_buffer.get("gen_logits_mask_ph", mbid)
            gen_tokens = torch.stack(gen_token_ph, -1)
            log_probs = torch.stack(gen_logprob_ph, -1)
            if all([m is None for m in gen_logits_mask_ph]):
                logits_mask = None
            else:
                mm = next(m for m in gen_logits_mask_ph if m is not None)
                gen_logits_mask_ph = [torch.ones_like(mm) if m is None else m for m in gen_logits_mask_ph]
                logits_mask = torch.stack(gen_logits_mask_ph, -2)
            all_gen_tokens.append(gen_tokens)
            all_log_probs.append(log_probs)
            all_logits_mask.append(logits_mask)
            if torch.is_tensor(gen_tokens) and torch.is_tensor(log_probs) and \
                torch.is_tensor(logits_mask):
                vocab_size = logits_mask.shape[-1]
                logger.debug(
                    f"generation on microbatch {mbid}: {gen_tokens.shape} {log_probs.shape} {logits_mask.shape}"
                )

        # if sequence is terminated, there might be situations where tensors in all_gen_tokens have difference shapes
        gen_tokens_lengths = [t.shape[-1] for t in all_gen_tokens]
        max_gen_tokens_length = max(gen_tokens_lengths)
        for i in range(len(all_gen_tokens)):
            assert all_gen_tokens[i].shape == all_log_probs[i].shape
            if all_gen_tokens[i].shape[-1] < max_gen_tokens_length:
                # if t.shape[-1] < max_gen_tokens_length, pad it with zeros
                pad_shape = all_gen_tokens[i].shape[:-1] + (max_gen_tokens_length -
                                                            all_gen_tokens[i].shape[-1],)
                all_gen_tokens[i] = torch.cat([
                    all_gen_tokens[i],
                    torch.full(pad_shape, self.tokenizer.pad_token_id, device=self.device)
                ],
                                              dim=1)
                # hack for log_probs and logits_mask, check if correct
                all_log_probs[i] = torch.cat(
                    [all_log_probs[i], torch.zeros(pad_shape, device=self.device)], dim=1)
                if all_logits_mask[i] is not None:
                    all_logits_mask[i] = torch.cat(
                        [all_logits_mask[i],
                         torch.ones((*pad_shape, vocab_size), device=self.device)], dim=1)

        gen_tokens = torch.cat(all_gen_tokens, dim=0)
        log_probs = torch.cat(all_log_probs, dim=0)
        if all([m is None for m in all_logits_mask]):
            logits_mask = None
        else:
            mm = next(m for m in all_logits_mask if m is not None)
            all_logits_mask = [torch.ones_like(mm) if m is None else m for m in all_logits_mask]
            logits_mask = torch.cat(all_logits_mask, dim=0)

        prompt_logits = [
            self.tensor_buffer.get("prompt_logits", mbid) for mbid in range(self.num_micro_batches)
        ]
        prompt_logits = torch.cat(prompt_logits, dim=0)
        return gen_tokens, log_probs, logits_mask, None, prompt_logits

    def _exec_reduce_tied_grads(self, stage_id: int, micro_batch_id: int, step_id: int):
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

    def _exec_reduce_grads(self, stage_id: int, micro_batch_id: int, step_id: int):
        self._force_grad_boundary = True
        if self.using_bf16_optimizer:
            if self.zero_optimization_stage() < ZeroStageEnum.gradients:
                self._bf16_reduce_grads()
            else:
                raise NotImplementedError("PP+BF16 only work for ZeRO Stage 1")
        else:
            self.allreduce_gradients(bucket_size=MEMORY_OPT_ALLREDUCE_SIZE)
        self._force_grad_boundary = False

    def _bf16_reduce_grads(self):
        # Make our own list of gradients from the optimizer's FP32 grads
        self.buffered_allreduce_fallback(grads=self.optimizer.get_grads_for_reduction(),
                                         elements_per_buffer=MEMORY_OPT_ALLREDUCE_SIZE)

    def _exec_forward_pass(self, stage_id: int, micro_batch_id: int, step_id: int):
        x = self.tensor_buffer.get("batch_input_x", micro_batch_id, remove=True)
        ys = self.tensor_buffer.get("batch_input_ys", micro_batch_id, remove=False)

        self._zero_grads(x)
        self._zero_grads(ys)

        x, ys = super().forward(x, ys)  # ys will be modified inplace in tensor buffer

        logits = self.__maybe_init_kv_cache(x, ys, micro_batch_id)
        self.__maybe_increase_cache_seqlens(x, ys, micro_batch_id, logits=logits)
        self.__maybe_genstep(x, ys, micro_batch_id, logits=logits)
        self.__maybe_calculate_loss(x, micro_batch_id)
        self.__maybe_store_logits(x, micro_batch_id)
        self.tensor_buffer.put("batch_output_x", micro_batch_id, x)  # send activation

    def __maybe_init_kv_cache(self, x: PipeTransferData, ys: List[PipeCacheData], mbid: int):
        if not self._generating:
            return None
        if self.tensor_buffer.get("kv_cache_reserved", mbid):
            return None

        logits = x.pp_input.squeeze(dim=1)
        # store prompt logits
        self.tensor_buffer.put("prompt_logits", mbid, logits)
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
            kvcache_seqlen = max(max_seq_len + self.current_gconfig.max_new_tokens,
                                 self.hidden_dim // self.head_dim + 10)
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
            y.v_cache = v_cache
            y.cache_seqlens = input_lens.clone().to(dtype=torch.int32)

        self.tensor_buffer.put("kv_cache_reserved", mbid, True)
        return logits  # if generating first token, return logits to pass to genstep

    def __maybe_increase_cache_seqlens(self,
                                       x: PipeTransferData,
                                       ys: List[PipeCacheData],
                                       mbid: int,
                                       logits=None):
        if not self._generating:
            return
        if logits is not None:  # Do not increase cache seqlens for the first token step
            return

        if self.is_last_stage():
            ys = ys[:-1]
        for y in ys:
            y.cache_seqlens += 1

    def __maybe_genstep(self, x: PipeTransferData, ys: List[PipeCacheData], mbid: int, logits=None):
        if not (self._generating and self.is_last_stage()):
            return

        if logits is None:
            logits = x.pp_input.squeeze(dim=1)
        unfinished_sequences = self.tensor_buffer.get("unfinished_sequences", mbid)
        generated_idx = self.tensor_buffer.get("generated_idx", mbid)

        next_tokens, logprob, logits_mask, terminate, unfinished_sequences = genstep(
            logits, self.tokenizer, unfinished_sequences, generated_idx, self.current_gconfig)

        self.tensor_buffer.put("terminate", mbid, terminate)
        self.tensor_buffer.put("unfinished_sequences", mbid, unfinished_sequences)
        self.tensor_buffer.put("generated_idx", mbid, generated_idx + 1)
        assert next_tokens is not None and logprob is not None
        self.tensor_buffer.get("gen_token_ph", mbid).append(next_tokens)
        self.tensor_buffer.get("gen_logprob_ph", mbid).append(logprob)
        self.tensor_buffer.get("gen_logits_mask_ph", mbid).append(logits_mask)
        self.tensor_buffer.put("next_tokens_to_send", mbid, next_tokens)

    def __maybe_calculate_loss(self, x: PipeTransferData, mbid: int):
        if self.is_last_stage() and self._compute_loss:
            model_output = x.pp_input
            loss_kwargs = self.tensor_buffer.get("loss_inputs", mbid, remove=True)
            input_cache = self.tensor_buffer.get("input_cache", mbid, remove=True)
            packed_input_ids = input_cache.packed_input_ids
            cu_seqlens = input_cache.cu_seqlens
            assert self._loss_fn is not None, "loss function is not set, please use engine.set_loss_fn(fn)"
            loss, stats = self._loss_fn(model_output, packed_input_ids, cu_seqlens, **loss_kwargs)
            loss = loss / self.num_micro_batches
            self.tensor_buffer.put("losses", mbid, loss)
            self.tensor_buffer.put("stats", mbid, stats)

    def __maybe_store_logits(self, x: PipeTransferData, mbid: int):
        if self.is_last_stage() and not self._compute_loss:
            logits = x.pp_input
            self.tensor_buffer.put("logits", mbid, logits)

    def _exec_backward_pass(self, stage_id: int, micro_batch_id: int, step_id: int):
        assert self.optimizer is not None, "must provide optimizer during " \
                                           "init in order to use backward"
        # The last stage just runs backward on the loss using DeepSpeed's typical
        # mechanisms.

        if self.is_last_stage():
            loss = self.tensor_buffer.get("losses", micro_batch_id, remove=False)
            super().backward(loss)
            return

        if self.using_bf16_optimizer and not self.is_last_stage():
            # manually call because we don't call optimizer.backward()
            self.optimizer.clear_lp_grads()

        grad = self.tensor_buffer.get("grad", micro_batch_id, remove=True)
        output_x = self.tensor_buffer.get("batch_output_x", micro_batch_id, remove=True)
        output_tensor = output_x.pp_input
        torch.autograd.backward(tensors=output_tensor, grad_tensors=grad)

        if self.using_bf16_optimizer and not self.is_last_stage():
            # manually call because we don't call optimizer.backward()
            self.optimizer.update_hp_grads(clear_lp_grads=False)

    def _exec_send_activations(self, stage_id: int, micro_batch_id: int, step_id: int):
        assert not self.is_last_stage()
        x: PipeTransferData = self.tensor_buffer.get("batch_output_x", micro_batch_id)
        # send_pipe_transfer_data(x, self.next_stage)
        if self._generating:
            p2p.send_tensor_meta(x.pp_input, self.next_stage)
        p2p.send(x.pp_input, self.next_stage)

    def _exec_recv_activations(self, stage_id: int, micro_batch_id: int, step_id: int):
        assert not self.is_first_stage()
        if self._generating:
            buf = p2p.recv_tensor_meta(self.prev_stage)
        else:
            buf = self.tensor_buffer.get("activation", micro_batch_id, remove=False)
        others = self.tensor_buffer.get("pipe_transfer_infos", micro_batch_id, remove=False)
        p2p.recv(buf, self.prev_stage)
        x = PipeTransferData(pp_input=buf, **others)
        # x = recv_pipe_transfer_data(buf, self.prev_stage, others)
        self.tensor_buffer.put("batch_input_x", micro_batch_id, x)

    def _exec_send_grads(self, stage_id: int, micro_batch_id: int, step_id: int):
        # x: PipeTransferData = self.tensor_buffer.get("batch_input_x", micro_batch_id, remove=True)
        # activation = x.pp_input
        assert not self.is_first_stage()
        activation = self.tensor_buffer.get("activation", micro_batch_id, remove=False)
        assert activation.grad is not None
        send_grad(activation.grad, self.prev_stage)

    def _exec_recv_grads(self, stage_id: int, micro_batch_id: int, step_id: int):
        assert not self.is_last_stage()
        buf = self.tensor_buffer.get("grad", micro_batch_id)
        recv_grad(buf, self.next_stage)

    def _exec_send_next_tokens(self, stage_id: int, micro_batch_id: int, step_id: int):
        """ When generating, send next tokens from the last stage to the first stage.
        """
        assert self.is_last_stage(), "_exec_send_next_tokens() should be only executed on the last stage"
        next_tokens_to_send = self.tensor_buffer.get("next_tokens_to_send", micro_batch_id, remove=True)
        p2p.send_tensor_meta(next_tokens_to_send, self.next_stage)
        p2p.send(next_tokens_to_send, self.next_stage)

    def _exec_recv_next_tokens(self, stage_id: int, micro_batch_id: int, step_id: int):
        """ When generating, recv next tokens from the last stage on the first stage
        Construct next forward input
        """
        assert self.is_first_stage(), "_exec_recv_next_tokens() should be only executed on the first stage"
        recv_buf = p2p.recv_tensor_meta(self.prev_stage)
        p2p.recv(recv_buf, self.prev_stage)

        # recvd = recv_buf.clone().detach()
        # self.tensor_buffer.put("next_tokens_cache", micro_batch_id, recvd)
        x = PipeTransferData(store_kv_cache=True)
        # others_cache = dict(store_kv_cache=True)
        # self.tensor_buffer.put("pipe_transfer_infos", micro_batch_id, others_cache)
        self.tensor_buffer.put("batch_input_x", micro_batch_id, x)
        ys = self.tensor_buffer.get("batch_input_ys", micro_batch_id, remove=False)
        ys[0].input_ids = recv_buf.unsqueeze(-1)
        ys[0].position_ids = None

    def _exec_optimizer_step(self, stage_id: int, micro_batch_id: int, step_id: int):
        self._force_grad_boundary = True
        lr_kwargs = None
        if self.version_steps is not None:
            lr_kwargs = {'epoch': self.version_steps}
        self._take_model_step(lr_kwargs=lr_kwargs)

        # sync loss scale across pipeline stages
        loss_scale = self.optimizer.loss_scale
        total_scale_cuda = torch.FloatTensor([float(loss_scale)]).to(self.device)
        dist.all_reduce(total_scale_cuda, op=dist.ReduceOp.MIN, group=self.grid.get_model_parallel_group())
        # all_loss_scale = total_scale_cuda[0].item()
        logger.info(
            f"loss scale: {total_scale_cuda}, group: { torch.distributed.get_process_group_ranks(self.mpu.get_model_parallel_group())}"
        )
        self.optimizer.loss_scaler.cur_scale = min(total_scale_cuda[0].item(), 8192)

        self._force_grad_boundary = False

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
        schedule.ForwardPass: _exec_forward_pass,
        schedule.BackwardPass: _exec_backward_pass,
        schedule.SendActivation: _exec_send_activations,
        schedule.RecvActivation: _exec_recv_activations,
        schedule.SendGrad: _exec_send_grads,
        schedule.RecvGrad: _exec_recv_grads,
        schedule.SendNextTokens: _exec_send_next_tokens,
        schedule.RecvNextTokens: _exec_recv_next_tokens,
    }

    def _exec_schedule(self, pipe_schedule, terminate_condition=None):
        """ Execute schedules
        Args:
            pipe_schedule: an instance of schedule
            terminate_condition: a callable that returns boolean value indicating if 
                                 the pipeline execution should terminate
        """
        # For each step in the schedule
        self.step_count = 0
        for step_cmds in pipe_schedule:
            if terminate_condition is not None:
                terminate_tensor = torch.tensor(0, dtype=torch.int32, device=self.device)
                if terminate_condition():
                    terminate_tensor = torch.tensor(1, dtype=torch.int32, device=self.device)
                # all reduce terminate tensor from all ranks
                dist.all_reduce(terminate_tensor)
                if terminate_tensor.item() >= self.grid.get_data_parallel_world_size():
                    break
            # For each instruction in the step
            step_id, micro_batch_id, step_cmds = step_cmds
            # logger.info(
            #     f"rank {self.global_rank} step {self.step_count}, st {step_id} mb {micro_batch_id} step_cmds: {step_cmds}"
            # )
            for cmd in step_cmds:
                # logger.info(f"rank {self.global_rank} exec cmd: {cmd}")
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
                # logger.info(f"rank {self.global_rank} complete cmd: {cmd}")
            self.step_count += 1
        self.sched_count += 1
