# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from types import MethodType
from typing import Callable, Dict, List, Optional, Tuple, Union
import contextlib
import dataclasses
import gc

from deepspeed import comm as dist
from deepspeed.runtime.bf16_optimizer import BF16_Optimizer
from deepspeed.runtime.engine import DeepSpeedEngine, MEMORY_OPT_ALLREDUCE_SIZE
from deepspeed.runtime.zero.config import ZeroStageEnum
import numpy as np
import torch
import transformers

from base.dataparallel import PackedParallelDataBroker
from base.monitor import time_mark
from base.namedarray import NamedArray
from base.topology import ParallelGrid
from impl.model.nn.flash_mqat.flash_generate import GenerationConfig, genstep
from impl.model.nn.flash_mqat.flash_mqat_api import FlashMQATModel
from impl.model.parallelism.pipeline_parallel.tensor_storage import TensorBuffer
from impl.model.utils.data import PipeCacheData, PipeTransferData
from impl.model.utils.tensor import pad_sequence_parallel_input
import base.constants
import base.logging as logging
import impl.model.backend.pipe_engine.static_schedule as schedule
import impl.model.parallelism.pipeline_parallel.p2p as p2p

logger = logging.getLogger("DeepSpeedPipelineEngine", "benchmark")


class PipelineError(Exception):
    pass


class DeepSpeedPipelineEngine(DeepSpeedEngine):
    """A training engine hybrid pipeline, data, and model parallel training."""

    def __init__(
        self,
        sequence_parallel=False,
        enable_async_p2p_communication=False,
        enable_async_instruction=False,
        *super_args,
        **super_kwargs,
    ):
        super().__init__(*super_args, **super_kwargs)
        assert (
            self.zero_optimization_stage() < 2
        ), "ZeRO-2 and ZeRO-3 are incompatible with pipeline parallelism"

        self.module: FlashMQATModel

        # deepspeed enigne attributes
        self.pipeline_parallelism = True
        # We schedule the all-reduces, so disable it in super().backward()
        self.enable_backward_allreduce = False
        # see method is_gradient_accumulation_boundary()
        self._force_grad_boundary = False

        # configs for data shape
        self.config = self.module.config
        # for tensor buffer initialization
        self.hidden_dim = self.config.hidden_dim
        self.head_dim = self.config.head_dim
        self.n_kv = self.config.n_kv_heads
        if self.bfloat16_enabled():
            assert isinstance(self.optimizer, BF16_Optimizer)
        self.dtype = torch.half if not self.bfloat16_enabled() else torch.bfloat16
        self.sequence_parallel = sequence_parallel
        # tensor model parallel option, whether to enable sequence parallel

        # logging
        self.sched_count = 0

        # parallelism constants
        self.grid: ParallelGrid = base.constants.grid()
        assert self.dp_world_size == self.grid.data_parallel_size

        self.global_rank = self.grid.get_global_rank()
        self.num_stages = self.grid.get_pipe_parallel_world_size()
        assert self.num_stages > 1
        self.stage_id = self.grid.get_stage_id()
        self.dp_id = self.grid.get_data_parallel_id()

        # num_micro_batches is configurable, default value: num_stages * 2
        self.default_num_micro_batches = self.num_stages * 2
        self.num_layers = self.module.num_layers  # number of leyers in current pipeline stage

        # PipelineEngine needs to handle data loading specially due to only the first
        # and last stages loading inputs/labels. We construct a sampler that uses

        self.is_pipe_parallel = self.grid.pipe_parallel_size > 1
        assert self.is_pipe_parallel
        self.is_data_parallel = self.grid.data_parallel_size > 1
        self.is_model_parallel = self.grid.model_parallel_size > 1

        self.prev_stage = (self.stage_id - 1) % self.num_stages
        self.next_stage = (self.stage_id + 1) % self.num_stages

        # initialize communication
        self._initialize_comm()

        # storages
        self.tensor_buffer = TensorBuffer()

        # schedule execution states
        self.tokenizer = None
        self.current_gconfig = None

        # loss related
        self._compute_loss = False
        self._loss_fn = None  # set when train_batch is called

        self.kv_cache_reserved = []

        # optimizer lr scheduler variables
        self.version_steps = None

        # engine mode
        self._generate_mode = False  # for generate()
        self._train_mode = False  # for train_batch()
        self._inference_mode = True  # for evaluate() and forward()

        self._first_token = False

        self._gd_graph = None

        self._sa_prompt_graph_size = self._sa_decoding_graph_size = -1
        self._ra_prompt_graph_size = self._ra_decoding_graph_size = -1

        self._async_p2p = enable_async_p2p_communication
        self._async_instruction = enable_async_instruction and self._async_p2p

        self._post_init_logging()

    def _post_init_logging(self):
        model_parameters = filter(lambda p: p.requires_grad, self.module.parameters())
        num_params = sum([p.numel() for p in model_parameters])
        unique_params = num_params
        # Subtract tied parameters if we don't own them
        # if self.module.tied_comms:
        #     tied_params = 0
        #     for key, d in self.module.tied_comms.items():
        #         if self.global_rank != min(d['ranks']):
        #             tied_params += sum(p.numel() for p in d['module'].parameters())
        #     unique_params -= tied_params

        params_tensor = torch.LongTensor(data=[num_params, unique_params]).to(self.device)
        dist.all_reduce(params_tensor, group=self.grid.get_model_parallel_group())
        params_tensor = params_tensor.tolist()
        total_params = params_tensor[0]
        unique_params = params_tensor[1]

        if self.global_rank == 0:
            logger.info(
                f"CONFIG: default_num_micro_batches={self.default_num_micro_batches} "
                f"num_layers(this stage)={self.num_layers} "
                f"pp_size={self.num_stages} "
                f"dp_size={self.grid.get_data_parallel_world_size()} "
                f"mp_size={self.grid.get_model_parallel_world_size()} "
                f"bf16={self.bfloat16_enabled()} "
            )
        if self.dp_id == 0:
            logger.info(
                f"rank={self.global_rank} "
                f"stage={self.stage_id} "
                f"layers={self.module.num_layers} "
                f"[{self.module.layer_idx_start}, {self.module.layer_idx_end}) "
                f"stage_params={num_params} ({num_params/1e6:0.3f}M) "
                f"total_params={total_params} ({total_params/1e6:0.3f}M) "
                f"unique_params={unique_params} ({unique_params/1e6:0.3f}M)"
            )

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

    def gradient_checkpointing_enable(self, attn: Optional[bool] = False, mlp: Optional[bool] = False):
        self.module.gradient_checkpointing_enable(attn, mlp)

    def _prepare_input(
        self,
        packed_input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        input_lens_for_partition: Optional[torch.Tensor] = None,
    ):
        """Prepare input for train or inference
        split all input tensors into micro batches for pipeline parallel

        Args:
            packed_input_ids (torch.Tensor): packed input ids of shape [total_seq_len]
            cu_seqlens (torch.Tensor): cu_seqlens of shape [batch_size]
        """
        if input_lens_for_partition is not None:
            pair_input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            data = NamedArray(
                packed_input_ids=packed_input_ids,
                pair_input_lens=pair_input_lens,
                input_lens=input_lens_for_partition,
            )
            n_seqs = input_lens_for_partition.shape[0]
        else:
            data = NamedArray(packed_input_ids=packed_input_ids, cu_seqlens=cu_seqlens)
            n_seqs = cu_seqlens.shape[0] - 1
        n_mbs = self.num_micro_batches
        splitted = PackedParallelDataBroker.scatter_to(data, n_mbs, min_size=n_seqs // n_mbs)
        if input_lens_for_partition is not None:
            splitted = [
                NamedArray(
                    packed_input_ids=x["packed_input_ids"],
                    cu_seqlens=torch.nn.functional.pad(x["pair_input_lens"].cumsum(0), (1, 0)),
                )
                for x in splitted
            ]
        if self._compute_loss:
            for mbid, x in enumerate(splitted):
                self.tensor_buffer.put("input_cache", mbid, x)
        mb_seq_lens = []

        def input_to_pipe_model_input(input: NamedArray, mbid: int):
            max_seqlen = int(max(input.cu_seqlens[1:] - input.cu_seqlens[:-1]))
            store_kv_cache = self._generate_mode

            cu_seqlens = input.cu_seqlens
            packed_input_ids = input.packed_input_ids

            # sequence parallel input padding
            if self.sequence_parallel:
                assert not self._generate_mode
                packed_input_ids, cu_seqlens, max_seqlen, pad_size = pad_sequence_parallel_input(
                    packed_input_ids, cu_seqlens, max_seqlen
                )
                self.tensor_buffer.put("pad_size", mbid, pad_size)
            x = PipeTransferData(
                cu_seqlens=cu_seqlens.int(), max_seqlen=int(max_seqlen), store_kv_cache=store_kv_cache
            )
            if self.is_first_stage():
                ys = [PipeCacheData(input_ids=packed_input_ids)] + [
                    PipeCacheData() for _ in range(self.num_layers - 1)
                ]
            else:
                ys = [PipeCacheData() for _ in range(self.num_layers)]
            total_len = (
                packed_input_ids.shape[0]
                if not self.sequence_parallel
                else packed_input_ids.shape[0] // base.constants.model_parallel_world_size()
            )
            mb_seq_lens.append(total_len)
            return (x, ys)

        batches = [input_to_pipe_model_input(x, i) for i, x in enumerate(splitted)]
        for mbid, batch in enumerate(batches):
            x, ys = batch
            self.tensor_buffer.put("batch_input_x", mbid, x)
            self.tensor_buffer.put("batch_input_ys", mbid, ys)
            self.tensor_buffer.put("batch_lengths", mbid, x.cu_seqlens.shape[0] - 1)
            self.tensor_buffer.put("mb_seq_lens", mbid, mb_seq_lens[mbid])

        # pre allocate receive buffers and pre store other information
        for mbid, batch in enumerate(batches):
            others_cache = dict(
                cu_seqlens=batch[0].cu_seqlens.int(),
                max_seqlen=int(batch[0].max_seqlen),
                store_kv_cache=batch[0].store_kv_cache,
            )
            self.tensor_buffer.put("pipe_transfer_infos", mbid, others_cache)

    def _prepare_loss_input(self, n_seqs: int, **loss_kwargs):
        data = NamedArray(**loss_kwargs)
        splitted = PackedParallelDataBroker.scatter_to(
            data, self.num_micro_batches, min_size=n_seqs // self.num_micro_batches
        )
        for mbid, x in enumerate(splitted):
            self.tensor_buffer.put("loss_inputs", mbid, x)

    def _set_generate_states(self):
        self._generate_mode = True
        self._train_mode = False
        self._inference_mode = False
        self._compute_loss = False
        self._first_token = True

    def _pre_generate(self):
        self.kv_cache_reserved = []  # ids of micro batches that have reserved kv cache
        self._compute_loss = False
        self._generate_mode = True

        for mbid in range(self.num_micro_batches):
            self.tensor_buffer.put("kv_cache_reserved", mbid, False)
            self.tensor_buffer.put("terminate", mbid, torch.tensor(0, dtype=torch.bool, device=self.device))
            self.tensor_buffer.put("generated_idx", mbid, 0)
            batch_length = self.tensor_buffer.get("batch_lengths", mbid)
            self.tensor_buffer.put(
                "unfinished_sequences", mbid, torch.ones(batch_length, dtype=torch.long, device=self.device)
            )
            self.tensor_buffer.put("gen_token_ph", mbid, [])
            self.tensor_buffer.put("gen_logprob_ph", mbid, [])
            self.tensor_buffer.put("gen_logits_mask_ph", mbid, [])
            self.tensor_buffer.put("first_token", mbid, True)

    def _post_generate(self):
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
        self.tensor_buffer.remove("first_token")
        self._clear_p2p_caches()

    def _set_eval_batch_states(self):
        self._compute_loss = True
        self._generate_mode = False
        self._train_mode = False
        self._inference_mode = True

    def _pre_eval_batch(self):
        pass

    def _post_eval_batch(self):
        self.tensor_buffer.remove("batch_input_x")
        self.tensor_buffer.remove("batch_input_ys")
        self.tensor_buffer.remove("batch_lengths")
        self.tensor_buffer.remove("loss_inputs")
        self.tensor_buffer.remove("input_cache")
        self.tensor_buffer.remove("losses")
        self.tensor_buffer.remove("stats")
        self._clear_p2p_caches()

    def _clear_p2p_caches(self):
        self.tensor_buffer.remove("recv_next_tokens_buf")
        self.tensor_buffer.remove("recv_act_buf")
        self.tensor_buffer.remove("recv_next_tokens_handle")
        self.tensor_buffer.remove("recv_act_handle")
        self.tensor_buffer.remove("recv_grad_handle")
        self.tensor_buffer.remove("send_act_handle")
        self.tensor_buffer.remove("send_next_tokens_handle")
        self.tensor_buffer.remove("send_grad_handle")

    def _set_forward_states(self):
        self._compute_loss = False
        self._generate_mode = False
        self._train_mode = False
        self._inference_mode = True

    def _pre_forward(self):
        pass

    def _post_forward(self):
        self._post_eval_batch()
        self.tensor_buffer.remove("logits")
        self._clear_p2p_caches()

    def _set_train_batch_states(self):
        self._compute_loss = True
        self._generate_mode = False
        self._train_mode = True
        self._inference_mode = False

    def _pre_train_batch(self):
        pass

    def _post_train_batch(self):
        self._post_eval_batch()

    def set_version_steps(self, version_steps):
        self.version_steps = version_steps

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def _initialize_comm(self):
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

    def forward(
        self,
        packed_input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        input_lens_for_partition: Optional[torch.Tensor] = None,
        num_micro_batches: Optional[int] = None,
    ):
        self.num_micro_batches = num_micro_batches if num_micro_batches else self.default_num_micro_batches
        self._set_forward_states()
        # forward one step and return packed logits
        self._prepare_input(packed_input_ids, cu_seqlens, input_lens_for_partition=input_lens_for_partition)
        self._pre_forward()
        sched = schedule.InferenceSchedule(
            micro_batches=self.num_micro_batches, stages=self.num_stages, stage_id=self.stage_id
        )
        self._exec_schedule(sched)

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

    def eval_batch(
        self,
        packed_input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        loss_fn: Callable,
        input_lens_for_partition: Optional[torch.Tensor] = None,
        num_micro_batches: Optional[int] = None,
        **loss_fn_kwargs,
    ):
        self.num_micro_batches = num_micro_batches if num_micro_batches else self.default_num_micro_batches
        self._set_eval_batch_states()
        self._prepare_input(packed_input_ids, cu_seqlens, input_lens_for_partition=input_lens_for_partition)
        if input_lens_for_partition is not None:
            n_seqs = input_lens_for_partition.shape[0]
        else:
            n_seqs = cu_seqlens.shape[0] - 1
        self._loss_fn = loss_fn
        self._prepare_loss_input(n_seqs=n_seqs, **loss_fn_kwargs)
        self._pre_eval_batch()

        # Do the work
        sched = schedule.InferenceSchedule(
            micro_batches=self.num_micro_batches, stages=self.num_stages, stage_id=self.stage_id
        )

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

    def train_batch(
        self,
        packed_input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        loss_fn: Callable,
        input_lens_for_partition: Optional[torch.Tensor] = None,
        num_micro_batches: Optional[int] = None,
        **loss_fn_kwargs,
    ):
        if not torch._C.is_grad_enabled():
            raise RuntimeError(f"train_batch() requires gradients enabled. Use eval_batch() instead.")

        self.num_micro_batches = num_micro_batches if num_micro_batches else self.default_num_micro_batches
        self._set_train_batch_states()
        self._prepare_input(packed_input_ids, cu_seqlens, input_lens_for_partition=input_lens_for_partition)
        self._loss_fn = loss_fn
        if input_lens_for_partition is not None:
            n_seqs = input_lens_for_partition.shape[0]
        else:
            n_seqs = cu_seqlens.shape[0] - 1
        self._prepare_loss_input(n_seqs=n_seqs, **loss_fn_kwargs)
        self._pre_train_batch()

        # Do the work
        sched = schedule.TrainSchedule(
            micro_batches=self.num_micro_batches, stages=self.num_stages, stage_id=self.stage_id
        )
        self._exec_schedule(sched)

        avg_loss, avg_stats = None, None
        if self.is_last_stage():
            losses = []
            stats = []

            for mbid in range(self.num_micro_batches):
                loss = self.tensor_buffer.get("losses", mbid)  # .detach()
                losses.append(loss)
                stats.append(self.tensor_buffer.get("stats", mbid))

            assert len(losses) > 0
            avg_loss = torch.stack(losses).mean()
            avg_stats = dict()
            for key in stats[0].keys():
                avg_stats[key] = torch.stack([stat[key] for stat in stats]).mean()

        self._post_eval_batch()
        return avg_loss, avg_stats

    @contextlib.contextmanager
    def sequence_parallel_disable(self):
        a = self.sequence_parallel
        self.sequence_parallel = False
        yield
        self.sequence_parallel = a

    def capture_generate_decoding_steps(self, ys: List[PipeCacheData]):
        torch.cuda.synchronize()
        if self.is_first_stage():
            assert all(y.k_cache is not None for y in ys[1:]), self.stage_id
            max_batch_size, seqlen = ys[1].k_cache.shape[:2]
            cache_seqlens = ys[1].cache_seqlens
        elif self.is_last_stage():
            assert all(y.k_cache is not None for y in ys[:-1]), self.stage_id
            max_batch_size, seqlen = ys[0].k_cache.shape[:2]
            cache_seqlens = ys[0].cache_seqlens
        else:
            assert all(y.k_cache is not None for y in ys), self.stage_id
            max_batch_size, seqlen = ys[0].k_cache.shape[:2]
            cache_seqlens = ys[0].cache_seqlens

        self._gd_input_buffers = dict(
            input_ids=torch.ones(max_batch_size, 1, dtype=torch.long, device=self.device),
            position_ids=cache_seqlens.clone().long()[:, None],
            k_caches=[y.k_cache for y in ys],
            v_caches=[y.v_cache for y in ys],
            hidden_states=torch.randn(
                max_batch_size, 1, self.module.config.hidden_dim, device=self.device, dtype=self.module.dtype
            ),
            # NOTE: here cache_seqlens should be the real cache_seqlens,
            # otherwise k/v cache will be changed in-place during capturing
            cache_seqlens=cache_seqlens.clone(),
            max_seqlen=None,
            cu_seqlens=None,
        )
        # Run the model once without capturing the graph.
        # This is to make sure that the captured graph does not include the
        # kernel launches for initial benchmarking (e.g., Triton autotune).
        self.module._forward(**self._gd_input_buffers)
        torch.cuda.synchronize()

        self._gd_graph = torch.cuda.CUDAGraph()
        self._gd_graph_bs = max_batch_size
        self._gd_graph_seqlen = seqlen
        with torch.cuda.graph(self._gd_graph):
            output = self.module._forward(**self._gd_input_buffers)
        torch.cuda.synchronize()

        self._gd_output_buffers = dict(output=output)
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def generate(
        self,
        packed_input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        tokenizer: transformers.PreTrainedTokenizerFast,
        gconfig: GenerationConfig = dataclasses.field(default_factory=GenerationConfig),
        num_micro_batches: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[PipeCacheData]]:
        with self.sequence_parallel_disable():
            self.num_micro_batches = (
                num_micro_batches if num_micro_batches else self.default_num_micro_batches
            )
            self._set_generate_states()
            self._prepare_input(packed_input_ids, cu_seqlens)
            # for elegant generation termination
            gconfig.max_new_tokens += self.num_stages - 1
            self.current_gconfig = gconfig
            self.tokenizer = tokenizer
            self._pre_generate()

            sched = schedule.GenerateSchedule(
                micro_batches=self.num_micro_batches,
                stages=self.num_stages,
                stage_id=self.stage_id,
                max_new_tokens=gconfig.max_new_tokens,
            )

            def terminate_condition():
                return all(
                    [self.tensor_buffer.get("terminate", mbid) for mbid in range(self.num_micro_batches)]
                )

            with self.module.gradient_checkpointing_disable(), self.module.sequence_parallel_disable():
                self._exec_schedule(sched, terminate_condition)
            r = self._maybe_gather_generate_outputs()
            self._post_generate()
        return r

    def _maybe_gather_generate_outputs(self):
        if not self.is_last_stage():
            return None

        all_gen_tokens = []
        all_log_probs = []
        all_logits_mask = []
        vocab_size = None
        for mbid in range(self.num_micro_batches):
            gen_token_ph = self.tensor_buffer.get("gen_token_ph", mbid, remove=True)
            gen_logprob_ph = self.tensor_buffer.get("gen_logprob_ph", mbid, remove=True)
            gen_logits_mask_ph = self.tensor_buffer.get("gen_logits_mask_ph", mbid, remove=True)

            gen_tokens = torch.stack(gen_token_ph, -1)
            log_probs = torch.stack(gen_logprob_ph, -1)
            if all([m is None for m in gen_logits_mask_ph]):
                logits_mask = None
            else:
                mm = next(m for m in gen_logits_mask_ph if m is not None)
                gen_logits_mask_ph = [torch.ones_like(mm) if m is None else m for m in gen_logits_mask_ph]
                logits_mask = torch.stack(gen_logits_mask_ph, -2)

            # logger.info(f"gen_tokens shape {gen_tokens.shape} "
            #             f"log_probs shape {log_probs.shape} ")
            if self.sequence_parallel:
                pad_seq_size = self.tensor_buffer.get("pad_seq_size", mbid, remove=True)
                gen_tokens = gen_tokens[:-pad_seq_size] if pad_seq_size > 0 else gen_tokens
                log_probs = log_probs[:-pad_seq_size] if pad_seq_size > 0 else log_probs
                if torch.is_tensor(logits_mask):
                    logits_mask = logits_mask[:-pad_seq_size] if pad_seq_size > 0 else logits_mask

            all_gen_tokens.append(gen_tokens)
            all_log_probs.append(log_probs)
            all_logits_mask.append(logits_mask)
            if torch.is_tensor(gen_tokens) and torch.is_tensor(log_probs) and torch.is_tensor(logits_mask):
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
                pad_shape = all_gen_tokens[i].shape[:-1] + (
                    max_gen_tokens_length - all_gen_tokens[i].shape[-1],
                )
                all_gen_tokens[i] = torch.cat(
                    [
                        all_gen_tokens[i],
                        torch.full(pad_shape, self.tokenizer.pad_token_id, device=self.device),
                    ],
                    dim=1,
                )
                # hack for log_probs and logits_mask, check if correct
                all_log_probs[i] = torch.cat(
                    [all_log_probs[i], torch.zeros(pad_shape, device=self.device)], dim=1
                )
                if all_logits_mask[i] is not None:
                    all_logits_mask[i] = torch.cat(
                        [all_logits_mask[i], torch.ones((*pad_shape, vocab_size), device=self.device)], dim=1
                    )

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

    # def _exec_reduce_tied_grads(self, stage_id: int, micro_batch_id: int, step_id: int):
    #     # We need to run this first to write to self.averaged_gradients;
    #     # since this class turns `enable_backward_allreduce` off,
    #     # `self.overlapping_partition_gradients_reduce_epilogue()` defined in the DeepSpeedEngine
    #     # never actually runs. I suspect this is because of efficiency problems; get_flat_partition in
    #     # stage2.py might do something expensive; someone will have to look into that later. But
    #     # in the meantime, this fixes ZeRO2 + Pipelining enough to run a demo. Further profiling
    #     # needed to decide if it actually breaks everything.
    #     # (see https://github.com/EleutherAI/gpt-neox/issues/62#issuecomment-761471944)
    #     if self.zero_optimization_partition_gradients():
    #         self.optimizer.overlapping_partition_gradients_reduce_epilogue()

    #     weight_group_list = self.module.get_tied_weights_and_groups()
    #     for weight, group in weight_group_list:
    #         grad = weight._hp_grad if self.bfloat16_enabled() else weight.grad
    #         dist.all_reduce(grad, group=group)

    def _exec_reduce_grads(self, stage_id: int, micro_batch_id: int, step_id: int):
        assert self._train_mode, "_exec_reduce_grads() should only be executed in train mode"
        self._force_grad_boundary = True
        self.set_gradient_accumulation_boundary(True)
        if self.bfloat16_enabled():
            if self.zero_optimization_stage() < ZeroStageEnum.gradients:
                self._bf16_reduce_grads()
            else:
                raise NotImplementedError("PP+BF16 only work for ZeRO Stage 1")
        else:
            self.allreduce_gradients(bucket_size=MEMORY_OPT_ALLREDUCE_SIZE)
        self._force_grad_boundary = False

        return False, None

    def _bf16_reduce_grads(self):
        # Make our own list of gradients from the optimizer's FP32 grads
        self.buffered_allreduce_fallback(
            grads=self.optimizer.get_grads_for_reduction(), elements_per_buffer=MEMORY_OPT_ALLREDUCE_SIZE
        )

    def _exec_forward_pass(self, stage_id: int, micro_batch_id: int, step_id: int):
        if self._generate_mode and self.is_first_stage():
            buf = self.tensor_buffer.get(
                "recv_next_tokens_buf", micro_batch_id, remove=True, raise_error=False
            )
            handle = self.tensor_buffer.get(
                "recv_next_tokens_handle", micro_batch_id, remove=True, raise_error=False
            )
        else:
            buf = self.tensor_buffer.get("recv_act_buf", micro_batch_id, remove=True, raise_error=False)
            handle = self.tensor_buffer.get("recv_act_handle", micro_batch_id, remove=True, raise_error=False)

        if handle is not None:
            handle.wait()

        ys = self.tensor_buffer.get("batch_input_ys", micro_batch_id, remove=False)

        if buf is not None:
            if self._generate_mode and self.is_first_stage():
                x = self.tensor_buffer.get("batch_input_x", micro_batch_id, remove=True)
                ys = self.tensor_buffer.get("batch_input_ys", micro_batch_id, remove=False)
                ys[0].input_ids = (
                    buf  # .unsqueeze(-1) # sequence parallel forward only accept one dim input_ids
                )
                ys[0].position_ids = None
            else:
                others = self.tensor_buffer.get("pipe_transfer_infos", micro_batch_id, remove=False)
                x = PipeTransferData(pp_input=buf, **others)
                self.tensor_buffer.put("batch_input_x", micro_batch_id, x)
        else:
            x = self.tensor_buffer.get("batch_input_x", micro_batch_id, remove=True)

        self._zero_grads(x)
        self._zero_grads(ys)

        # TODO: since pytorch all-reduce has bug during capturing and vllm all-reduce does not support >8 GPUs,
        # we defer the implementation of CUDAGraph generation in the future
        # if self._generate_mode and self.tensor_buffer.get("kv_cache_reserved", micro_batch_id):
        #     kvcache_seqlen = max(
        #         base.constants.dataset_max_seqlen() + self.current_gconfig.max_new_tokens,
        #         self.hidden_dim // self.head_dim + 10,
        #     )
        #     with torch.no_grad():
        #         bs = self.tensor_buffer.get("batch_lengths", micro_batch_id)
        #         if self._gd_graph is None:
        #             self.capture_generate_decoding_steps(ys)
        #         assert self._gd_graph_bs >= bs, (self._gd_graph_bs, bs)
        #         assert self._gd_graph_seqlen >= kvcache_seqlen, (self._gd_graph_seqlen, kvcache_seqlen)
        #         first_y = ys[1] if self.is_first_stage() else ys[0]
        #         if ys[0].input_ids is not None:
        #             self._gd_input_buffers["input_ids"][:bs].copy_(ys[0].input_ids, non_blocking=True)
        #         if x.pp_input is not None:
        #             self._gd_input_buffers["hidden_states"][:bs].copy_(x.pp_input, non_blocking=True)
        #         self._gd_input_buffers["position_ids"][:bs].copy_(first_y.cache_seqlens.unsqueeze(-1),
        #                                                           non_blocking=True)
        #         self._gd_input_buffers["cache_seqlens"][:bs].copy_(first_y.cache_seqlens, non_blocking=True)
        #         self._gd_graph.replay()
        #     x.pp_output = self._gd_output_buffers["output"][:bs]
        # else:
        x, ys = super().forward(x, ys)  # ys will be modified inplace in tensor buffer

        # logger.info(f"rank {self.global_rank} mbid {micro_batch_id} step {step_id} x.pp_output shape {x.pp_output.shape}")
        is_first_step = self.__maybe_init_kv_cache(x, ys, micro_batch_id)
        self.__maybe_increase_cache_seqlens(x, ys, micro_batch_id, is_first_step)
        end = self.__maybe_genstep(x, ys, micro_batch_id, is_first_step)
        self.__maybe_calculate_loss(x, micro_batch_id)
        self.__maybe_store_logits(x, micro_batch_id)
        self.tensor_buffer.put("batch_output_x", micro_batch_id, x)  # send activation
        # if end:
        #     logger.info(f"rank {self.global_rank} stage_id {stage_id} mbid {micro_batch_id} step {step_id} forward pass end")
        return end, None

    def __maybe_init_kv_cache(self, x: PipeTransferData, ys: List[PipeCacheData], mbid: int):
        if not self._generate_mode:
            return False
        if self.tensor_buffer.get("kv_cache_reserved", mbid):
            return False

        logits = x.pp_output.squeeze(dim=1)
        # store prompt logits
        self.tensor_buffer.put("prompt_logits", mbid, logits)
        # reserve kv cache
        cu_seqlens = x.cu_seqlens
        max_seq_len = self.tensor_buffer.get("pipe_transfer_infos", mbid)["max_seqlen"]
        input_lens = cu_seqlens[1:] - cu_seqlens[:-1]

        layer_indices = range(self.module.layer_idx_start, self.module.layer_idx_end)
        gd_input_buffer_kv_cache_indices = range(len(layer_indices))
        assert len(ys) == len(layer_indices)
        if self.is_first_stage():
            ys[0].cache_seqlens = input_lens.clone().to(dtype=torch.int32)
            ys = ys[1:]
            layer_indices = layer_indices[1:]
            gd_input_buffer_kv_cache_indices = gd_input_buffer_kv_cache_indices[1:]
        if self.is_last_stage():
            ys = ys[:-1]
            layer_indices = layer_indices[:-1]
            gd_input_buffer_kv_cache_indices = gd_input_buffer_kv_cache_indices[:-1]

        bs = input_lens.shape[0]
        for y, layer_idx, bk_idx in zip(ys, layer_indices, gd_input_buffer_kv_cache_indices):
            assert y.k_cache is not None and y.v_cache is not None and y.cache_seqlens is not None
            kvcache_seqlen = max(
                base.constants.dataset_max_seqlen() + self.current_gconfig.max_new_tokens,
                self.hidden_dim // self.head_dim + 10,
            )
            if (
                self._gd_graph is not None
                and self._gd_graph_bs >= bs
                and self._gd_graph_seqlen >= kvcache_seqlen
            ):
                if self._gd_graph_bs < bs or self._gd_graph_seqlen < kvcache_seqlen:
                    raise RuntimeError(
                        f"CUDAGraph batch size {self._gd_graph_bs} or seqlen {self._gd_graph_seqlen} "
                        f"is smaller than the data batch size {bs} or seqlen {kvcache_seqlen}. "
                        "Have you correctly set the `max_seqlen` constant and set a `min_n_seqs_per_dp` in RPC config?"
                    )
                k_cache = self._gd_input_buffers["k_caches"][bk_idx][:bs, :kvcache_seqlen]
                v_cache = self._gd_input_buffers["v_caches"][bk_idx][:bs, :kvcache_seqlen]
            else:
                k_cache = torch.zeros(
                    (bs, kvcache_seqlen, *y.k_cache.shape[1:]),
                    dtype=y.k_cache.dtype,
                    device=y.k_cache.device,
                )
                v_cache = torch.zeros(
                    (bs, kvcache_seqlen, *y.v_cache.shape[1:]),
                    dtype=y.v_cache.dtype,
                    device=y.v_cache.device,
                )
            indices = (
                torch.arange(kvcache_seqlen, device=self.device, dtype=torch.long)[None, :]
                < input_lens[:, None]
            )
            k_cache[indices] = y.k_cache
            v_cache[indices] = y.v_cache
            y.k_cache = k_cache
            y.v_cache = v_cache
            y.cache_seqlens = input_lens.clone().to(dtype=torch.int32)

        self.tensor_buffer.put("kv_cache_reserved", mbid, True)
        return True  # if generating first token, return logits to pass to genstep

    def __maybe_increase_cache_seqlens(
        self, x: PipeTransferData, ys: List[PipeCacheData], mbid: int, is_first_step: bool
    ):
        if not self._generate_mode:
            return
        if is_first_step:  # Do not increase cache seqlens for the first token step
            return

        if self.is_last_stage():
            ys = ys[:-1]
        for y in ys:
            y.cache_seqlens += 1

    def __maybe_genstep(self, x: PipeTransferData, ys: List[PipeCacheData], mbid: int, is_first_step: bool):
        if not (self._generate_mode and self.is_last_stage()):
            return False

        logits = x.pp_output
        if is_first_step:
            logits = logits[x.cu_seqlens[1:] - 1]
        logits = logits.squeeze(dim=1)

        unfinished_sequences = self.tensor_buffer.get("unfinished_sequences", mbid)
        generated_idx = self.tensor_buffer.get("generated_idx", mbid)

        next_tokens, logprob, logits_mask, terminate, unfinished_sequences = genstep(
            logits, self.tokenizer, unfinished_sequences, generated_idx, self.current_gconfig
        )

        if isinstance(terminate, bool):
            terminate = torch.tensor(terminate, device=self.device, dtype=torch.bool)
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
            model_output = x.pp_output
            if self.sequence_parallel:
                pad_size = self.tensor_buffer.get("pad_size", mbid, remove=True)
                model_output = model_output[:-pad_size] if pad_size > 0 else model_output
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
            logits = x.pp_output
            self.tensor_buffer.put("logits", mbid, logits)

    def _exec_backward_pass(self, stage_id: int, micro_batch_id: int, step_id: int):
        assert self.optimizer is not None, "must provide optimizer during " "init in order to use backward"
        # The last stage just runs backward on the loss using DeepSpeed's typical
        # mechanisms.
        output_x = self.tensor_buffer.get("batch_output_x", micro_batch_id, remove=True)
        self.enable_backward_allreduce = False
        self.set_gradient_accumulation_boundary(False)

        if self.is_last_stage():
            loss = self.tensor_buffer.get("losses", micro_batch_id, remove=True)
            super().backward(loss)
            self.tensor_buffer.put("losses", micro_batch_id, loss.detach().clone())
            return False, None

        if self.bfloat16_enabled() and not self.is_last_stage():
            # manually call because we don't call optimizer.backward()
            self.optimizer.clear_lp_grads()

        if not self.is_last_stage() and self.zero_optimization():
            # manually call because we don't call optimizer.backward()
            self.optimizer.micro_step_id += 1

            if self.optimizer.contiguous_gradients:
                self.optimizer.ipg_buffer = []
                buf_0 = torch.empty(
                    int(self.optimizer.reduce_bucket_size), dtype=self.dtype, device=self.device
                )
                self.optimizer.ipg_buffer.append(buf_0)

                # Use double buffers to avoid data access conflict when overlap_comm is enabled.
                if self.optimizer.overlap_comm:
                    buf_1 = torch.empty(
                        int(self.optimizer.reduce_bucket_size), dtype=self.dtype, device=self.device
                    )
                    self.optimizer.ipg_buffer.append(buf_1)
                self.optimizer.ipg_index = 0

        handle = self.tensor_buffer.get("recv_grad_handle", micro_batch_id, remove=True, raise_error=False)
        if handle is not None:
            handle.wait()
        grad = self.tensor_buffer.get("grad", micro_batch_id, remove=True)

        output_tensor = output_x.pp_output
        torch.autograd.backward(tensors=output_tensor, grad_tensors=grad)

        if not self.is_last_stage() and self.zero_optimization():
            # manually call because we don't call optimizer.backward()
            # Only for Stage 1, Mode 2
            if self.optimizer.use_grad_accum_attribute:
                self.optimizer.fill_grad_accum_attribute()

        if self.bfloat16_enabled() and not self.is_last_stage():
            # manually call because we don't call optimizer.backward()
            self.optimizer.update_hp_grads(clear_lp_grads=False)
        return False, None

    def _capture_send_activation(self, shape, dtype, is_decoding_phase: bool):
        sa_graph_size = int(np.prod(shape))
        sa_input_buffer = dict(
            x=torch.empty(sa_graph_size, dtype=dtype, device=self.device),
            terminate=torch.empty((), dtype=torch.bool, device=self.device),
        )

        torch.cuda.synchronize()
        p2p.send(sa_input_buffer["x"], self.next_stage)
        p2p.send(sa_input_buffer["terminate"], self.next_stage)

        sa_graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(sa_graph):
            p2p.send(sa_input_buffer["x"], self.next_stage)
            p2p.send(sa_input_buffer["terminate"], self.next_stage)
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        if not is_decoding_phase:
            self._sa_prompt_graph = sa_graph
            self._sa_prompt_input_buffer = sa_input_buffer
            self._sa_prompt_graph_size = sa_graph_size
        else:
            self._sa_decoding_graph = sa_graph
            self._sa_decoding_input_buffer = sa_input_buffer
            self._sa_decoding_graph_size = sa_graph_size

    def _exec_send_activations(self, stage_id: int, micro_batch_id: int, step_id: int):
        assert not self.is_last_stage()
        x: PipeTransferData = self.tensor_buffer.get(
            "batch_output_x", micro_batch_id, remove=not self._train_mode
        )
        # NOTE: enable the following optimization will sometimes report CUDA error: operation not permitted when stream is capturing
        # The performance improvement is about 5%~10%, so we disable it for now.

        # if type(self) == DeepSpeedPipelineEngine and self._generate_mode:
        #     # An ad-hoc implementation, just used for normal pipeline engine generate decoding phase.
        #     length = int(np.prod(activation.shape))
        #     is_decoding_phase = not self.tensor_buffer.get("first_token", micro_batch_id, remove=False)
        #     if (is_decoding_phase
        #             and length > self._sa_decoding_graph_size) or (not is_decoding_phase
        #                                                            and length > self._sa_prompt_graph_size):
        #         with torch.no_grad():
        #             self._capture_send_activation(activation.shape, activation.dtype, is_decoding_phase)
        #     graph = self._sa_decoding_graph if is_decoding_phase else self._sa_prompt_graph
        #     input_buffer = self._sa_decoding_input_buffer if is_decoding_phase else self._sa_prompt_input_buffer
        #     input_buffer['x'][:activation.numel()] = activation.view(-1)
        #     input_buffer['terminate'].copy_(self.tensor_buffer.get("terminate", micro_batch_id))
        #     graph.replay()

        #     self.tensor_buffer.put("first_token", micro_batch_id, False)
        #     return False, None

        handle = p2p.send(x.pp_output, self.next_stage, async_op=self._async_p2p)
        if self._async_p2p:
            self.tensor_buffer.put("send_act_handle", micro_batch_id, handle)
        if self._generate_mode:
            self.tensor_buffer.put("first_token", micro_batch_id, False)

        if type(self) == DeepSpeedPipelineEngine and self._generate_mode:
            if self._async_p2p:
                handle.wait()
            terminate = self.tensor_buffer.get("terminate", micro_batch_id)
            p2p.send(terminate, self.next_stage)
        # self.rank_print(f"send tensor DONE {micro_batch_id}, {x.pp_output.shape}")
        return False, handle

    def _capture_recv_activation(self, shape, dtype, is_decoding_phase):
        ra_graph_size = int(np.prod(shape))
        ra_output_buffer = dict(
            x=torch.empty(ra_graph_size, dtype=dtype, device=self.device),
            terminate=torch.empty((), dtype=torch.bool, device=self.device),
        )

        torch.cuda.synchronize()
        p2p.recv(ra_output_buffer["x"], self.prev_stage)
        p2p.recv(ra_output_buffer["terminate"], self.prev_stage)

        ra_graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(ra_graph):
            p2p.recv(ra_output_buffer["x"], self.prev_stage)
            p2p.recv(ra_output_buffer["terminate"], self.prev_stage)
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        if not is_decoding_phase:
            self._ra_prompt_graph = ra_graph
            self._ra_prompt_output_buffer = ra_output_buffer
            self._ra_prompt_graph_size = ra_graph_size
        else:
            self._ra_decoding_graph = ra_graph
            self._ra_decoding_output_buffer = ra_output_buffer
            self._ra_decoding_graph_size = ra_graph_size

    def _exec_recv_activations(self, stage_id: int, micro_batch_id: int, step_id: int):
        assert not self.is_first_stage()

        mb_seq_len = self.tensor_buffer.get("mb_seq_lens", micro_batch_id, remove=False)
        act_shape = (mb_seq_len, self.hidden_dim)

        # if type(self) == DeepSpeedPipelineEngine and self._generate_mode:
        #     ft = self.tensor_buffer.get("first_token", micro_batch_id, remove=False)
        #     if not ft:
        #         batch_length = self.tensor_buffer.get("batch_lengths", micro_batch_id, remove=False)
        #         batch_length = batch_length // base.constants.model_parallel_world_size() \
        #                        if self.sequence_parallel else batch_length
        #         act_shape = (batch_length, 1, self.hidden_dim)
        #     length = int(np.prod(act_shape))
        #     if (ft and length > self._ra_prompt_graph_size) or (not ft
        #                                                         and length > self._ra_decoding_graph_size):
        #         with torch.no_grad():
        #             self._capture_recv_activation(act_shape, self.dtype, not ft)
        #     graph = self._ra_prompt_graph if ft else self._ra_decoding_graph
        #     output_buffer = self._ra_prompt_output_buffer if ft else self._ra_decoding_output_buffer
        #     graph.replay()
        #     x = output_buffer['x'][:int(np.prod(act_shape))].view(act_shape).clone()
        #     terminate = output_buffer['terminate'].clone()
        #     self.tensor_buffer.put("recv_act_buf", micro_batch_id, x)
        #     self.tensor_buffer.put("terminate", micro_batch_id, terminate)
        #     return False, None

        if self._train_mode:
            buf = self.tensor_buffer.alloc(
                "activation", micro_batch_id, act_shape, self.dtype, self.device, require_grads=True
            )
        elif self._inference_mode:
            buf = torch.empty(act_shape, dtype=self.dtype, device=self.device, requires_grad=False)
        elif self._generate_mode:
            ft = self.tensor_buffer.get("first_token", micro_batch_id, remove=False)
            if ft:
                buf = torch.empty(act_shape, dtype=self.dtype, device=self.device, requires_grad=False)
            else:
                batch_length = self.tensor_buffer.get("batch_lengths", micro_batch_id, remove=False)
                batch_length = (
                    batch_length // base.constants.model_parallel_world_size()
                    if self.sequence_parallel
                    else batch_length
                )
                act_shape = (batch_length, 1, self.hidden_dim)
                buf = torch.empty(act_shape, dtype=self.dtype, device=self.device, requires_grad=False)

        recv_handle = p2p.recv(buf, self.prev_stage, async_op=self._async_p2p)
        if self._async_p2p:
            self.tensor_buffer.put("recv_act_handle", micro_batch_id, recv_handle)
        self.tensor_buffer.put("recv_act_buf", micro_batch_id, buf)

        if type(self) == DeepSpeedPipelineEngine and self._generate_mode:
            if self._async_p2p:
                recv_handle.wait()
            terminate = torch.empty((), dtype=torch.bool, device=self.device)
            p2p.recv(terminate, self.prev_stage)
            self.tensor_buffer.put("terminate", micro_batch_id, terminate)

        # others = self.tensor_buffer.get("pipe_transfer_infos", micro_batch_id, remove=False)
        # x = PipeTransferData(pp_input=buf, **others)
        # self.tensor_buffer.put("batch_input_x", micro_batch_id, x)
        return False, recv_handle

    def _exec_send_grads(self, stage_id: int, micro_batch_id: int, step_id: int):
        assert self._train_mode, "_exec_send_grads() should be only executed in train mode."
        assert not self.is_first_stage()
        activation = self.tensor_buffer.get("activation", micro_batch_id, remove=True)
        assert activation.grad is not None
        handle = p2p.send(activation.grad, self.prev_stage, async_op=self._async_p2p)
        if self._async_p2p:
            self.tensor_buffer.put("send_grad_handle", micro_batch_id, handle)
        return False, handle

    def _exec_recv_grads(self, stage_id: int, micro_batch_id: int, step_id: int):
        assert self._train_mode, "_exec_recv_grads() should be only executed in train mode."
        assert not self.is_last_stage()
        mb_seq_len = self.tensor_buffer.get("mb_seq_lens", micro_batch_id, remove=False)
        grad_shape = (mb_seq_len, self.hidden_dim)
        buf = self.tensor_buffer.alloc("grad", micro_batch_id, grad_shape, self.dtype, self.device)
        handle = p2p.recv(buf, self.next_stage, async_op=self._async_p2p)
        if self._async_p2p:
            self.tensor_buffer.put("recv_grad_handle", micro_batch_id, handle)
        return False, handle

    def _capture_send_next_token(self, batch_length):
        self._sn_graph_size = batch_length
        self._sn_input_buffer = dict(
            next_tokens=torch.empty((batch_length, 1), dtype=torch.long, device=self.device),
            terminate=torch.empty((), dtype=torch.bool, device=self.device),
        )

        p2p.send(self._sn_input_buffer["next_tokens"], self.next_stage)
        p2p.send(self._sn_input_buffer["terminate"], self.next_stage)

        self._sn_graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(self._sn_graph):
            p2p.send(self._sn_input_buffer["next_tokens"], self.next_stage)
            p2p.send(self._sn_input_buffer["terminate"], self.next_stage)
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    def _exec_send_next_tokens(self, stage_id: int, micro_batch_id: int, step_id: int):
        """When generating, send next tokens from the last stage to the first stage."""
        assert self._generate_mode, "_exec_send_next_tokens() should be only executed in generate mode."
        assert self.is_last_stage(), "_exec_send_next_tokens() should be only executed on the last stage"
        next_tokens_to_send = self.tensor_buffer.get("next_tokens_to_send", micro_batch_id, remove=True)
        # if type(self) == DeepSpeedPipelineEngine and self._generate_mode:
        #     if not hasattr(self, "_sn_graph") or next_tokens_to_send.shape[0] > self._sn_graph_size:
        #         with torch.no_grad():
        #             self._capture_send_next_token(next_tokens_to_send.shape[0])
        #     self._sn_input_buffer['next_tokens'][:next_tokens_to_send.
        #                                          shape[0]] = next_tokens_to_send.unsqueeze(-1)
        #     self._sn_input_buffer['terminate'].copy_(self.tensor_buffer.get("terminate", micro_batch_id))
        #     self._sn_graph.replay()
        #     self.tensor_buffer.put("first_token", micro_batch_id, False)
        #     return False, None

        # p2p.send_tensor_meta(next_tokens_to_send, self.next_stage)
        handle = p2p.send(next_tokens_to_send, self.next_stage, async_op=self._async_p2p)
        if self._async_p2p:
            self.tensor_buffer.put("send_next_tokens_handle", micro_batch_id, handle)

        if type(self) == DeepSpeedPipelineEngine:
            if self._async_p2p:
                handle.wait()
            p2p.send(self.tensor_buffer.get("terminate", micro_batch_id), self.next_stage)
        self.tensor_buffer.put("first_token", micro_batch_id, False)
        return False, handle

    def _capture_recv_next_token(self, batch_length):
        self._rn_graph_size = batch_length
        self._rn_output_buffer = dict(
            next_tokens=torch.empty((batch_length, 1), dtype=torch.long, device=self.device),
            terminate=torch.empty((), dtype=torch.bool, device=self.device),
        )

        p2p.recv(self._rn_output_buffer["next_tokens"], self.prev_stage)
        p2p.recv(self._rn_output_buffer["terminate"], self.prev_stage)

        self._rn_graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(self._rn_graph):
            p2p.recv(self._rn_output_buffer["next_tokens"], self.prev_stage)
            p2p.recv(self._rn_output_buffer["terminate"], self.prev_stage)
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

    def _exec_recv_next_tokens(self, stage_id: int, micro_batch_id: int, step_id: int):
        """When generating, recv next tokens from the last stage on the first stage
        Construct next forward input
        """
        assert self._generate_mode, "_exec_recv_next_tokens() should be only executed in generate mode."
        assert self.is_first_stage(), "_exec_recv_next_tokens() should be only executed on the first stage"
        # recv_buf = p2p.recv_tensor_meta(self.prev_stage)
        batch_length = self.tensor_buffer.get("batch_lengths", micro_batch_id, remove=False)
        # if type(self) == DeepSpeedPipelineEngine and self._generate_mode:
        #     if not hasattr(self, "_rn_graph") or batch_length > self._rn_graph_size:
        #         with torch.no_grad():
        #             self._capture_recv_next_token(batch_length)
        #     self._rn_graph.replay()
        #     recv_buf = self._rn_output_buffer['next_tokens'][:batch_length].clone()
        #     terminate = self._rn_output_buffer['terminate'].clone()
        #     self.tensor_buffer.put("recv_next_tokens_buf", micro_batch_id, recv_buf)
        #     x = PipeTransferData(store_kv_cache=True)
        #     self.tensor_buffer.put("batch_input_x", micro_batch_id, x)
        #     self.tensor_buffer.put("terminate", micro_batch_id, terminate)
        #     return False, None

        recv_buf = torch.empty((batch_length, 1), dtype=torch.long, device=self.device)
        handle = p2p.recv(recv_buf, self.prev_stage, async_op=self._async_p2p)
        if self._async_p2p:
            self.tensor_buffer.put("recv_next_tokens_handle", micro_batch_id, handle)
        self.tensor_buffer.put("recv_next_tokens_buf", micro_batch_id, recv_buf)

        x = PipeTransferData(store_kv_cache=True)
        self.tensor_buffer.put("batch_input_x", micro_batch_id, x)

        if type(self) == DeepSpeedPipelineEngine:
            if self._async_p2p:
                handle.wait()
            terminate = torch.empty((), dtype=torch.bool, device=self.device)
            p2p.recv(terminate, self.prev_stage)
            self.tensor_buffer.put("terminate", micro_batch_id, terminate)

        return False, handle

    def _exec_optimizer_step(self, stage_id: int, micro_batch_id: int, step_id: int):
        assert self._train_mode, "_exec_optimizer_step() should be only executed in train mode."
        self._force_grad_boundary = True
        lr_kwargs = None
        if self.version_steps is not None:
            lr_kwargs = {"epoch": self.version_steps}
        # self.rank_print("before take model step")
        self._take_model_step(lr_kwargs=lr_kwargs)
        # self.rank_print("after take model step")

        # sync loss scale across pipeline stages
        if not self.bfloat16_enabled():
            loss_scale = self.optimizer.loss_scale
            total_scale_cuda = torch.FloatTensor([float(loss_scale)]).to(self.device)
            dist.all_reduce(
                total_scale_cuda, op=dist.ReduceOp.MIN, group=self.grid.get_model_parallel_group()
            )
            # all_loss_scale = total_scale_cuda[0].item()
            logger.info(
                f"loss scale: {total_scale_cuda}, group: { torch.distributed.get_process_group_ranks(self.mpu.get_model_parallel_group())}"
            )
            self.optimizer.loss_scaler.cur_scale = min(total_scale_cuda[0].item(), 8192)

        # self.rank_print("after sync loss scale")

        self._force_grad_boundary = False
        return False, None

    def _exec_end_schedule(self, stage_id: int, micro_batch_id: int, step_id: int):
        """Used in StreamPipeEngine to force end the schedule. Do nothing."""
        # logger.info(f"rank {self.global_rank} stage {stage_id} execute EndSchedule")
        return True, None

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
        """Disabled for pipeline parallel training. See ``train_batch()``."""
        raise PipelineError("Only train_batch() is accessible in pipeline mode.")

    def step(self, *args, **kwargs):
        """Disabled for pipeline parallel training. See ``train_batch()``."""
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
        schedule.EndSchedule: _exec_end_schedule,
    }

    def _exec_schedule(self, pipe_schedule, terminate_condition=None):
        """Execute schedules
        Args:
            pipe_schedule: an instance of schedule
            terminate_condition: a callable that returns boolean value indicating if
                                 the pipeline execution should terminate
        """
        # For each step in the schedule
        self.step_count = 0
        will_break = False
        if self.is_last_stage():
            burn_out_steps = self.num_stages - 1
        elif self.stage_id == self.num_stages - 2:
            burn_out_steps = 0
        else:
            burn_out_steps = 1
        for step_cmds in pipe_schedule:
            # For each instruction in the step
            step_id, micro_batch_id, step_cmds = step_cmds
            # logger.info(
            #     f"rank {self.global_rank} step {self.step_count}, st {step_id} mb {micro_batch_id} step_cmds: {step_cmds}"
            # )
            for cmd in step_cmds:
                # logger.info(f"rank {self.global_rank} exec cmd: {cmd}")
                if type(cmd) not in self._INSTRUCTION_MAP:
                    raise RuntimeError(
                        f"{self.__class__.__name__} does not understand instruction {repr(cmd)}"
                    )

                if will_break:
                    if (
                        self.is_last_stage()
                        and burn_out_steps < self.num_stages - 1
                        and type(cmd) != schedule.RecvActivation
                    ):
                        # logger.info(f"rank {self.global_rank} skip cmd: {cmd}")
                        continue
                    elif not self.is_last_stage() and type(cmd) != schedule.SendActivation:
                        # logger.info(f"rank {self.global_rank} skip cmd: {cmd}")
                        continue

                # Equivalent to: self._exec_forward_pass(buffer_id=0)
                try:
                    cmd_type_string = str(type(cmd)).split("'")[1].split(".")[-1]
                    time_mark(
                        name=f"{cmd_type_string}_start",
                        identifier=str(self.global_rank),
                        step=self.sched_count,
                    )
                    self._exec_instr = MethodType(self._INSTRUCTION_MAP[type(cmd)], self)
                    self._exec_instr(*cmd.args)
                    time_mark(
                        name=f"{cmd_type_string}_end", identifier=str(self.global_rank), step=self.sched_count
                    )
                except Exception as e:
                    logger.error(f"Rank {self.global_rank} step {self.step_count}, Exception in cmd {cmd}")
                    raise e
                # logger.info(f"rank {self.global_rank} complete cmd: {cmd}")
            self.step_count += 1

            # exec_forward_pass() and __maybe_genstep() will put the "terminate" signal into tensor_buffer, such that
            # terminate_condition() will return True for the last stage.
            if will_break:
                burn_out_steps -= 1
            if terminate_condition is not None and terminate_condition():
                will_break = True
            if will_break and burn_out_steps <= 0:
                # logger.info(f"rank {self.global_rank} break the exec_schedule loop")
                break
        self.sched_count += 1

    def rank_print(self, *args, **kwargs):
        print(f"Rank {self.global_rank}: ", *args, **kwargs)
