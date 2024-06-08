from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import dataclasses

from deepspeed.runtime.engine import DeepSpeedEngine
from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel as MegatronDDP
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer as MegatronDistOptim
import torch
import torch.distributed as dist
import transformers

from reallm.api.core import data_api
from reallm.impl.model.nn.real_llm_api import ReaLModel
from reallm.impl.model.nn.real_llm_base import PipeCacheData, PipeTransferData
from reallm.impl.model.nn.real_llm_generate import (_gather_gen_output_from_list,
                                                    _gather_minibatch_gen_outputs, GenerationConfig)
from reallm.impl.model.parallelism.pipeline_parallel.instruction_impl import (_exec_pipe_schedule,
                                                                              _prepare_input, PipeGenInstrSet,
                                                                              PipeInferenceInstrSet,
                                                                              PipeTrainForwardCommInstrSet,
                                                                              PipeTrainInstrSet)
from reallm.impl.model.parallelism.pipeline_parallel.tensor_storage import TensorBuffer
import reallm.base.constants as constants
import reallm.impl.model.parallelism.pipeline_parallel.static_schedule as schedule


@dataclasses.dataclass
class PipelineRunner:
    module: ReaLModel

    @property
    def default_train_mbs(self):
        return constants.pipe_parallel_world_size() * 2

    @property
    def default_inf_mbs(self):
        return constants.pipe_parallel_world_size()

    def eval(self, *args, **kwargs):
        return self.module.eval(*args, **kwargs)

    def train(self, *args, **kwargs):
        return self.module.train(*args, **kwargs)

    def forward(
        self,
        seqlens_cpu: List[int],
        packed_input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        input_lens_for_partition: Optional[torch.Tensor] = None,
        num_micro_batches: Optional[int] = None,
    ):
        """Run one forward step over a batch of tokens and return the logits."""

        if num_micro_batches is None:
            num_micro_batches = self.default_inf_mbs

        tensor_buffer = TensorBuffer()

        _prepare_input(
            module=self.module,
            tensor_buffer=tensor_buffer,
            num_micro_batches=num_micro_batches,
            seqlens_cpu=seqlens_cpu,
            packed_input_ids=packed_input_ids,
            cu_seqlens=cu_seqlens,
            store_kv_cache=False,
            input_lens_for_partition=input_lens_for_partition,
        )

        sched = schedule.InferenceSchedule(
            micro_batches=num_micro_batches,
            stages=constants.pipe_parallel_world_size(),
            stage_id=constants.pipe_parallel_rank(),
        )
        _exec_pipe_schedule(
            self.module,
            tensor_buffer,
            instr_map=PipeInferenceInstrSet.INSTRUCTION_MAP,
            pipe_schedule=sched,
        )

        logits = None
        if constants.is_last_pipe_stage():
            logits_list = []
            for i in range(num_micro_batches):
                logits = tensor_buffer.get("logits", i, remove=True)
                # logger.info(f"mbid {i} before remove pad logits shape {logits.shape}")
                if constants.sequence_parallel():
                    pad_size = tensor_buffer.get("pad_size", i, remove=True)
                    logits = logits[:-pad_size] if pad_size > 0 else logits
                    # logger.info(f"mbid {i} after remove pad {pad_size} logits shape {logits.shape}")
                logits_list.append(logits)
            logits = torch.cat(logits_list, dim=0)

        return logits

    @torch.no_grad()
    def generate(
        self,
        seqlens_cpu: List[int],
        packed_input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        tokenizer: transformers.PreTrainedTokenizerFast,
        gconfig: GenerationConfig = dataclasses.field(default_factory=GenerationConfig),
        num_micro_batches: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[PipeCacheData]]:
        if constants.sequence_parallel():
            raise NotImplementedError("Sequence parallel is not supported for generation")

        if num_micro_batches is None:
            num_micro_batches = self.default_inf_mbs

        tensor_buffer = TensorBuffer()

        _prepare_input(
            module=self.module,
            tensor_buffer=tensor_buffer,
            num_micro_batches=num_micro_batches,
            seqlens_cpu=seqlens_cpu,
            packed_input_ids=packed_input_ids,
            cu_seqlens=cu_seqlens,
            store_kv_cache=True,
        )

        # for elegant generation termination
        gconfig.max_new_tokens += constants.pipe_parallel_world_size() - 1

        for mbid in range(num_micro_batches):
            tensor_buffer.put("kv_cache_reserved", mbid, False)
            tensor_buffer.put(
                "terminate",
                mbid,
                torch.tensor(0, dtype=torch.bool, device=self.module.device),
            )
            tensor_buffer.put("generated_idx", mbid, 0)
            batch_length = tensor_buffer.get("batch_lengths", mbid)
            tensor_buffer.put(
                "unfinished_sequences",
                mbid,
                torch.ones(batch_length, dtype=torch.long, device=self.module.device),
            )
            tensor_buffer.put("gen_token_ph", mbid, [])
            tensor_buffer.put("gen_logprob_ph", mbid, [])
            tensor_buffer.put("gen_logits_mask_ph", mbid, [])
            tensor_buffer.put("first_token", mbid, True)
            tensor_buffer.put("tokenizer", mbid, tokenizer)
            tensor_buffer.put("gconfig", mbid, gconfig)

        sched = schedule.GenerateSchedule(
            micro_batches=num_micro_batches,
            stages=constants.pipe_parallel_world_size(),
            stage_id=constants.pipe_parallel_rank(),
            max_new_tokens=gconfig.max_new_tokens,
        )

        def terminate_condition():
            return all([tensor_buffer.get("terminate", mbid) for mbid in range(num_micro_batches)])

        _exec_pipe_schedule(
            self.module,
            tensor_buffer,
            instr_map=PipeGenInstrSet.INSTRUCTION_MAP,
            pipe_schedule=sched,
            terminate_condition=terminate_condition,
        )

        if not constants.is_last_pipe_stage():
            return None

        # Gather generation outputs, including generated tokens, logprobs, and logits_mask.
        generate_output = []
        for mbid in range(num_micro_batches):
            generate_output += [
                _gather_gen_output_from_list(
                    gen_token_ph=tensor_buffer.get("gen_token_ph", mbid, remove=True),
                    gen_logprob_ph=tensor_buffer.get("gen_logprob_ph", mbid, remove=True),
                    gen_logits_mask_ph=tensor_buffer.get("gen_logits_mask_ph", mbid, remove=True),
                )
            ]

        gen_tokens, log_probs, logits_mask = _gather_minibatch_gen_outputs(*list(zip(*generate_output)))

        return gen_tokens, log_probs, logits_mask, None, None

    def train_batch(
        self,
        engine: Any,
        seqlens_cpu: List[int],
        packed_input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        loss_fn: Callable,
        version_steps: int,
        input_lens_for_partition: Optional[torch.Tensor] = None,
        num_micro_batches: Optional[int] = None,
        **loss_fn_kwargs,
    ):
        if not torch._C.is_grad_enabled():
            raise RuntimeError(f"train_batch() requires gradients enabled. Use eval_batch() instead.")

        if num_micro_batches is None:
            num_micro_batches = self.default_train_mbs

        tensor_buffer = TensorBuffer()
        for i in range(num_micro_batches):
            tensor_buffer.put("num_micro_batches", i, num_micro_batches)
            tensor_buffer.put("version_steps", i, version_steps)
            tensor_buffer.put("loss_fn", i, loss_fn)

        _prepare_input(
            module=self.module,
            tensor_buffer=tensor_buffer,
            num_micro_batches=num_micro_batches,
            seqlens_cpu=seqlens_cpu,
            packed_input_ids=packed_input_ids,
            cu_seqlens=cu_seqlens,
            store_kv_cache=False,
            loss_fn=loss_fn,
            input_lens_for_partition=input_lens_for_partition,
            **loss_fn_kwargs,
        )

        if isinstance(engine, DeepSpeedEngine):
            instr_map = PipeTrainInstrSet(backend="deepspeed", ds_engine=engine).INSTRUCTION_MAP
        elif isinstance(engine, MegatronDDP):
            # FIXME:
            instr_map = PipeTrainInstrSet(backend="megatron", megatron_ddp=engine,
                                          megatron_dist_optim=None).INSTRUCTION_MAP
        else:
            raise NotImplementedError(f"Unknown backend type for training: {type(engine)}")

        sched = schedule.TrainSchedule(
            micro_batches=num_micro_batches,
            stages=constants.pipe_parallel_world_size(),
            stage_id=constants.pipe_parallel_rank(),
        )
        _exec_pipe_schedule(
            module=self.module,
            tensor_buffer=tensor_buffer,
            instr_map=instr_map,
            pipe_schedule=sched,
        )

        agg_stats = None
        if constants.is_last_pipe_stage():
            stats = []
            for mbid in range(num_micro_batches):
                stats.append(tensor_buffer.get("stats", mbid))
            agg_stats = dict()
            for key in stats[0].keys():
                agg_stats[key] = torch.stack([stat[key] for stat in stats]).sum()

        return agg_stats

    @torch.no_grad()
    def eval_batch(
        self,
        seqlens_cpu: List[int],
        packed_input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        loss_fn: Callable,
        input_lens_for_partition: Optional[torch.Tensor] = None,
        num_micro_batches: Optional[int] = None,
        **loss_fn_kwargs,
    ):
        if num_micro_batches is None:
            num_micro_batches = self.default_train_mbs

        tensor_buffer = TensorBuffer()
        for i in range(num_micro_batches):
            tensor_buffer.put("num_micro_batches", i, num_micro_batches)
            tensor_buffer.put("loss_fn", i, loss_fn)

        _prepare_input(
            module=self.module,
            tensor_buffer=tensor_buffer,
            num_micro_batches=num_micro_batches,
            seqlens_cpu=seqlens_cpu,
            packed_input_ids=packed_input_ids,
            cu_seqlens=cu_seqlens,
            store_kv_cache=False,
            loss_fn=loss_fn,
            input_lens_for_partition=input_lens_for_partition,
            **loss_fn_kwargs,
        )

        sched = schedule.InferenceSchedule(
            micro_batches=num_micro_batches,
            stages=constants.pipe_parallel_world_size(),
            stage_id=constants.pipe_parallel_rank(),
        )
        _exec_pipe_schedule(
            module=self.module,
            tensor_buffer=tensor_buffer,
            instr_map=PipeTrainForwardCommInstrSet.INSTRUCTION_MAP,
            pipe_schedule=sched,
        )

        agg_stats = None
        if constants.is_last_pipe_stage():
            stats = []
            for mbid in range(num_micro_batches):
                stats.append(tensor_buffer.get("stats", mbid))
            agg_stats = dict()
            for key in stats[0].keys():
                agg_stats[key] = torch.stack([stat[key] for stat in stats]).sum()
        return agg_stats
