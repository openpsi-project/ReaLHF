from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import contextlib
import dataclasses
import functools
import gc

from deepspeed.runtime.engine import DeepSpeedEngine, MEMORY_OPT_ALLREDUCE_SIZE
from deepspeed.runtime.zero.config import ZeroStageEnum
from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel as MegatronDDP
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer as MegatronDistOptim
import torch
import torch.distributed as dist

from reallm.api.core import data_api
from reallm.base.monitor import cuda_tmark, cuda_tmarked, CUDATimeMarkType
from reallm.base.namedarray import NamedArray
from reallm.impl.model.nn.real_llm_api import ReaLModel
from reallm.impl.model.nn.real_llm_base import PipeCacheData, PipeTransferData
from reallm.impl.model.nn.real_llm_generate import genstep, init_kv_cache
from reallm.impl.model.parallelism.pipeline_parallel.instruction import PipeInstruction
from reallm.impl.model.parallelism.pipeline_parallel.static_schedule import PipeSchedule
from reallm.impl.model.parallelism.pipeline_parallel.tensor_storage import TensorBuffer
from reallm.impl.model.utils.padding import pad_sequence_parallel_input
import reallm.base.constants as constants
import reallm.base.logging as logging
import reallm.impl.model.parallelism.pipeline_parallel.p2p as p2p
import reallm.impl.model.parallelism.pipeline_parallel.static_schedule as schedule

logger = logging.getLogger("Pipeline Instruction", "benchmark")


class PipelineError(Exception):
    pass


# FIXME: remove seqlens_cpu here
def _prepare_input(
    module: ReaLModel,
    tensor_buffer: TensorBuffer,
    num_micro_batches: int,
    seqlens_cpu: List[int],
    packed_input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    store_kv_cache: bool,
    loss_fn: Optional[Callable] = None,
    input_lens_for_partition: Optional[torch.Tensor] = None,
    **loss_kwargs,
):
    """Prepare input for train or inference
    split all input tensors into micro batches for pipeline parallel

    Args:
        packed_input_ids (torch.Tensor): packed input ids of shape [total_seq_len]
        cu_seqlens (torch.Tensor): cu_seqlens of shape [batch_size]
    """
    if input_lens_for_partition is not None:
        n_groups = input_lens_for_partition.shape[0]
        group_input_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).view(n_groups, -1)
        data = NamedArray(
            packed_input_ids=packed_input_ids,
            group_input_lens=group_input_lens,
            input_lens=input_lens_for_partition,
        )
        n_seqs = input_lens_for_partition.shape[0]
    else:
        data = NamedArray(packed_input_ids=packed_input_ids, cu_seqlens=cu_seqlens)
        n_seqs = cu_seqlens.shape[0] - 1
    data.register_metadata(seqlens=seqlens_cpu)

    # split into sequences
    n_mbs = num_micro_batches
    assert n_seqs >= n_mbs
    splitted, partitions = data_api.split_sequences(data,
                                                    n_mbs,
                                                    min_size=n_seqs // n_mbs,
                                                    return_partitions=True)
    batch_seqlens = [seqlens_cpu[start:end] for start, end in partitions]

    # Sanity check to ensure that the order of splitted sequences
    # is the same across pipeline parallel ranks.
    _batch_seqlen = torch.tensor([sum(x) for x in batch_seqlens], device=cu_seqlens.device, dtype=torch.long)
    _batch_seqlen_all_gathered = [
        torch.zeros_like(_batch_seqlen) for _ in range(constants.pipe_parallel_world_size())
    ]
    _batch_seqlen_all_gathered[constants.pipe_parallel_rank()] = _batch_seqlen
    dist.all_gather(
        _batch_seqlen_all_gathered,
        _batch_seqlen,
        group=constants.pipe_parallel_group(),
    )
    for i in range(constants.pipe_parallel_world_size()):
        if not torch.allclose(_batch_seqlen_all_gathered[i], _batch_seqlen):
            raise PipelineError(
                "Partitioned seqlens are not equal across pipeline parallel ranks. "
                f"Current rank (dp={constants.data_parallel_rank()},"
                f"tp={constants.model_parallel_rank()},pp={constants.pipe_parallel_rank()}), "
                f"gathered batch seqlens={_batch_seqlen_all_gathered}, "
                f"Have you ensured that the order of dataset across ranks is the same?",)

    if input_lens_for_partition is not None:
        splitted = [
            NamedArray(
                packed_input_ids=x["packed_input_ids"],
                cu_seqlens=torch.nn.functional.pad(x["group_input_lens"].view(-1).cumsum(0), (1, 0)),
            ) for x in splitted
        ]

    if loss_fn is not None:
        for mbid, x in enumerate(splitted):
            tensor_buffer.put("input_cache", mbid, x)

        loss_input_data = NamedArray(**loss_kwargs)
        loss_input_data.register_metadata(seqlens=seqlens_cpu)
        splitted_loss_input = data_api.split_sequences(
            loss_input_data,
            num_micro_batches,
            min_size=n_seqs // num_micro_batches,
        )
        for mbid, x in enumerate(splitted_loss_input):
            tensor_buffer.put("loss_inputs", mbid, x)

    mb_seq_lens = []

    def input_to_pipe_model_input(input: NamedArray, mbid: int):
        max_seqlen = int(max(batch_seqlens[mbid]))

        cu_seqlens = input.cu_seqlens
        packed_input_ids = input.packed_input_ids

        # sequence parallel input padding
        if constants.sequence_parallel():
            packed_input_ids, cu_seqlens, max_seqlen, pad_size = (pad_sequence_parallel_input(
                packed_input_ids, cu_seqlens, max_seqlen))
            tensor_buffer.put("pad_size", mbid, pad_size)
        x = PipeTransferData(
            cu_seqlens=cu_seqlens.int(),
            max_seqlen=int(max_seqlen),
            store_kv_cache=store_kv_cache,
        )
        if constants.is_first_pipe_stage():
            ys = [PipeCacheData(packed_input_ids=packed_input_ids)
                  ] + [PipeCacheData() for _ in range(module.num_layers - 1)]
        else:
            ys = [PipeCacheData() for _ in range(module.num_layers)]
        total_len = (packed_input_ids.shape[0] if not constants.sequence_parallel() else
                     packed_input_ids.shape[0] // constants.model_parallel_world_size())
        mb_seq_lens.append(total_len)
        return (x, ys)

    batches = [input_to_pipe_model_input(x, i) for i, x in enumerate(splitted)]
    for mbid, batch in enumerate(batches):
        x, ys = batch
        tensor_buffer.put("batch_input_x", mbid, x)
        tensor_buffer.put("batch_input_ys", mbid, ys)
        tensor_buffer.put("batch_lengths", mbid, x.cu_seqlens.shape[0] - 1)
        tensor_buffer.put("mb_seq_lens", mbid, mb_seq_lens[mbid])

    # pre allocate receive buffers and pre store other information
    for mbid, batch in enumerate(batches):
        others_cache = dict(
            cu_seqlens=batch[0].cu_seqlens.int(),
            max_seqlen=int(batch[0].max_seqlen),
            store_kv_cache=batch[0].store_kv_cache,
        )
        tensor_buffer.put("pipe_transfer_infos", mbid, others_cache)


def _exec_pipe_schedule(
    module: ReaLModel,
    tensor_buffer: TensorBuffer,
    instr_map: Dict[PipeInstruction, Callable],
    pipe_schedule: PipeSchedule,
    terminate_condition: Optional[Callable] = None,
):
    """Execute schedules
    Args:
        module: The model to execute the schedule on.
        tensor_buffer: A temporary buffer that stores necessary information during running.
        instr_map: A map of PipeInstruction types to methods. Each method will be executed with the
            kwargs provided to the PipeInstruction from the scheduler.
        pipe_schedule: an instance of schedule
        terminate_condition: a callable that returns boolean value indicating if
                                the pipeline execution should terminate
    """
    step_count = 0
    is_last_stage = constants.is_last_pipe_stage()
    num_stages = constants.pipe_parallel_world_size()
    stage_id = constants.pipe_parallel_rank()
    global_rank = dist.get_rank()
    parllelism_rank = constants.parallelism_rank()

    # A termination machanism to avoid all-reduce at each step.
    # If the schedule is about to terminate (i.e., will_break is True),
    # the last stage will send this message to the previous stages with
    # one more pipeline round (last -> 0 -> 1 -> .. -> last-1).
    will_break = False
    if is_last_stage:
        burn_out_steps = num_stages - 1
    elif stage_id == num_stages - 2:
        burn_out_steps = 0
    else:
        burn_out_steps = 1

    # For each step in the schedule
    for step_cmds in pipe_schedule:
        # For each instruction in the step
        step_id, micro_batch_id, step_cmds = step_cmds
        for cmd in step_cmds:
            if type(cmd) not in instr_map:
                raise RuntimeError(
                    f"Pipeline instruction executor does not understand instruction {repr(cmd)}")

            if will_break:
                # With the termination mechanism, skip communication instructions
                # because its peer stages have been terminated
                if (is_last_stage and burn_out_steps < num_stages - 1
                        and type(cmd) != schedule.RecvActivation):
                    continue
                elif not is_last_stage and type(cmd) != schedule.SendActivation:
                    continue

            try:
                instr_map[type(cmd)](module, tensor_buffer, *cmd.args)
            except Exception as e:
                logger.error(f"Model name {constants.model_name()} rank {parllelism_rank}"
                             f" (global rank {global_rank}) step {step_count}, "
                             f"Exception in cmd: {cmd}")
                raise e
        step_count += 1

        if will_break:
            burn_out_steps -= 1
        if terminate_condition is not None and terminate_condition():
            will_break = True
        if will_break and burn_out_steps <= 0:
            break


def _zero_grads(inputs):
    if isinstance(inputs, torch.Tensor):
        if inputs.grad is not None:
            inputs.grad.data.zero_()
    elif isinstance(inputs, tuple):
        for t in inputs:
            if t.grad is not None:
                t.grad.data.zero_()
    elif dataclasses.is_dataclass(inputs):
        for f in dataclasses.fields(inputs):
            _zero_grads(getattr(inputs, f.name))
    else:
        # do nothing for non tensor
        pass


class PipeInferenceInstrSet:

    def _exec_forward_pass(
        module: ReaLModel,
        tensor_buffer: TensorBuffer,
        stage_id: int,
        micro_batch_id: int,
        step_id: int,
    ):
        buf = tensor_buffer.get("recv_act_buf", micro_batch_id, remove=True, raise_error=False)
        ys = tensor_buffer.get("batch_input_ys", micro_batch_id, remove=False)

        if buf is not None:
            others = tensor_buffer.get("pipe_transfer_infos", micro_batch_id, remove=False)
            x = PipeTransferData(pp_input=buf, **others)
            tensor_buffer.put("batch_input_x", micro_batch_id, x)
        else:
            x = tensor_buffer.get("batch_input_x", micro_batch_id, remove=True)

        _zero_grads(x)
        _zero_grads(ys)
        x, ys = module.forward(x, ys)

        tensor_buffer.put("batch_output_x", micro_batch_id, x)  # Used by send_activation

        if constants.is_last_pipe_stage():
            logits = x.pp_output
            tensor_buffer.put("logits", micro_batch_id, logits)

    def _exec_send_activations(
        module: ReaLModel,
        tensor_buffer: TensorBuffer,
        stage_id: int,
        micro_batch_id: int,
        step_id: int,
    ):
        assert stage_id != constants.pipe_parallel_world_size() - 1
        x: PipeTransferData = tensor_buffer.get(
            "batch_output_x",
            micro_batch_id,
            remove=True,
        )
        p2p.send(x.pp_output, constants.next_pipe_stage(), async_op=False)

    def _exec_recv_activations(
        module: ReaLModel,
        tensor_buffer: TensorBuffer,
        stage_id: int,
        micro_batch_id: int,
        step_id: int,
    ):
        assert not constants.is_first_pipe_stage()

        device = module.device
        dtype = module.dtype
        hidden_dim = module.config.hidden_dim

        mb_seq_len = tensor_buffer.get("mb_seq_lens", micro_batch_id, remove=False)
        act_shape = (mb_seq_len, hidden_dim)
        buf = torch.empty(act_shape, dtype=dtype, device=device, requires_grad=False)

        p2p.recv(buf, constants.prev_pipe_stage(), async_op=False)
        tensor_buffer.put("recv_act_buf", micro_batch_id, buf)

    INSTRUCTION_MAP = {
        schedule.ForwardPass: _exec_forward_pass,
        schedule.SendActivation: _exec_send_activations,
        schedule.RecvActivation: _exec_recv_activations,
    }


class PipeGenInstrSet:

    def _exec_forward_pass(
        module: ReaLModel,
        tensor_buffer: TensorBuffer,
        stage_id: int,
        micro_batch_id: int,
        step_id: int,
    ):
        is_first_stage = constants.is_first_pipe_stage()
        if is_first_stage:
            buf = tensor_buffer.get(
                "recv_next_tokens_buf",
                micro_batch_id,
                remove=True,
                raise_error=False,
            )
        else:
            buf = tensor_buffer.get(
                "recv_act_buf",
                micro_batch_id,
                remove=True,
                raise_error=False,
            )

        ys = tensor_buffer.get("batch_input_ys", micro_batch_id, remove=False)

        if buf is not None:
            if is_first_stage:
                x = tensor_buffer.get("batch_input_x", micro_batch_id, remove=True)
                ys = tensor_buffer.get("batch_input_ys", micro_batch_id, remove=False)
                ys[0].packed_input_ids = (
                    buf  # sequence parallel forward only accept one dim input_ids
                )
                ys[0].packed_position_ids = None
            else:
                others = tensor_buffer.get("pipe_transfer_infos", micro_batch_id, remove=False)
                x = PipeTransferData(pp_input=buf, **others)
                tensor_buffer.put("batch_input_x", micro_batch_id, x)
        else:
            x = tensor_buffer.get("batch_input_x", micro_batch_id, remove=True)

        _zero_grads(x)
        _zero_grads(ys)
        x, ys = module.forward(x, ys)

        tensor_buffer.put("batch_output_x", micro_batch_id, x)

        # Init KV cache.
        is_prefill_phase = False
        if (tensor_buffer.get("kv_cache_reserved", micro_batch_id, raise_error=False) is None):
            # KV cache is attached to x and ys.
            init_kv_cache(x, ys)
            is_prefill_phase = True
            tensor_buffer.put("kv_cache_reserved", micro_batch_id, True)

        # Increase cache_seqlens in the decoding phase.
        if not is_prefill_phase:
            for y in ys:
                y.cache_seqlens += 1

        # Perform a decoding step.
        if constants.is_last_pipe_stage():
            # Gather logits of the final token
            logits = x.pp_output
            if is_prefill_phase:
                logits = logits[x.cu_seqlens[1:] - 1]

            unfinished_sequences = tensor_buffer.get("unfinished_sequences", micro_batch_id)
            generated_idx = tensor_buffer.get("generated_idx", micro_batch_id)

            tokenizer = tensor_buffer.get("tokenizer", micro_batch_id)
            gconfig = tensor_buffer.get("gconfig", micro_batch_id)
            next_tokens, logprob, logits_mask, terminate, unfinished_sequences = (genstep(
                logits,
                tokenizer,
                unfinished_sequences,
                generated_idx,
                gconfig,
            ))

            if isinstance(terminate, bool):
                terminate = torch.tensor(terminate, device=logits.device, dtype=torch.bool)

            tensor_buffer.put("terminate", micro_batch_id, terminate)
            tensor_buffer.put("unfinished_sequences", micro_batch_id, unfinished_sequences)
            tensor_buffer.put("generated_idx", micro_batch_id, generated_idx + 1)
            assert next_tokens is not None and logprob is not None
            tensor_buffer.get("gen_token_ph", micro_batch_id).append(next_tokens)
            tensor_buffer.get("gen_logprob_ph", micro_batch_id).append(logprob)
            tensor_buffer.get("gen_logits_mask_ph", micro_batch_id).append(logits_mask)
            tensor_buffer.put("next_tokens_to_send", micro_batch_id, next_tokens)

    def _exec_send_activations(
        module: ReaLModel,
        tensor_buffer: TensorBuffer,
        stage_id: int,
        micro_batch_id: int,
        step_id: int,
    ):
        PipeInferenceInstrSet._exec_send_activations(module, tensor_buffer, stage_id, micro_batch_id, step_id)
        tensor_buffer.put("first_token", micro_batch_id, False)
        terminate = tensor_buffer.get("terminate", micro_batch_id)
        p2p.send(terminate, constants.next_pipe_stage())

    def _exec_recv_activations(
        module: ReaLModel,
        tensor_buffer: TensorBuffer,
        stage_id: int,
        micro_batch_id: int,
        step_id: int,
    ):
        assert not constants.is_first_pipe_stage()

        device = module.device
        dtype = module.dtype
        hidden_dim = module.config.hidden_dim

        mb_seq_len = tensor_buffer.get("mb_seq_lens", micro_batch_id, remove=False)
        act_shape = (mb_seq_len, hidden_dim)

        ft = tensor_buffer.get("first_token", micro_batch_id, remove=False)
        if ft:
            buf = torch.empty(act_shape, dtype=dtype, device=device, requires_grad=False)
        else:
            batch_length = tensor_buffer.get("batch_lengths", micro_batch_id, remove=False)
            batch_length = (batch_length // constants.model_parallel_world_size()
                            if constants.sequence_parallel() else batch_length)
            act_shape = (batch_length, hidden_dim)
            buf = torch.empty(act_shape, dtype=dtype, device=device, requires_grad=False)

        prev_stage = constants.prev_pipe_stage()
        p2p.recv(buf, prev_stage, async_op=False)
        tensor_buffer.put("recv_act_buf", micro_batch_id, buf)

        terminate = torch.empty((), dtype=torch.bool, device=device)
        p2p.recv(terminate, prev_stage)
        tensor_buffer.put("terminate", micro_batch_id, terminate)

    def _exec_send_next_tokens(
        module: ReaLModel,
        tensor_buffer: TensorBuffer,
        stage_id: int,
        micro_batch_id: int,
        step_id: int,
    ):
        """When generating, send next tokens from the last stage to the first stage."""
        assert constants.is_last_pipe_stage()
        next_stage = constants.next_pipe_stage()
        next_tokens_to_send = tensor_buffer.get("next_tokens_to_send", micro_batch_id, remove=True)
        p2p.send(next_tokens_to_send, next_stage, async_op=False)
        p2p.send(tensor_buffer.get("terminate", micro_batch_id), next_stage)
        tensor_buffer.put("first_token", micro_batch_id, False)

    def _exec_recv_next_tokens(
        module: ReaLModel,
        tensor_buffer: TensorBuffer,
        stage_id: int,
        micro_batch_id: int,
        step_id: int,
    ):
        """When generating, recv next tokens from the last stage on the first stage
        Construct next forward input
        """
        assert constants.is_first_pipe_stage()
        batch_length = tensor_buffer.get("batch_lengths", micro_batch_id, remove=False)

        device = tensor_buffer.get("device", micro_batch_id)
        prev_stage = constants.prev_pipe_stage()

        recv_buf = torch.empty((batch_length,), dtype=torch.long, device=device)
        p2p.recv(recv_buf, prev_stage, async_op=False)
        tensor_buffer.put("recv_next_tokens_buf", micro_batch_id, recv_buf)

        x = PipeTransferData(
            store_kv_cache=True,
            cu_seqlens=torch.arange(batch_length + 1, dtype=torch.int32, device=device),
            max_seqlen=1,
        )
        tensor_buffer.put("batch_input_x", micro_batch_id, x)

        terminate = torch.empty((), dtype=torch.bool, device=device)
        p2p.recv(terminate, prev_stage)
        tensor_buffer.put("terminate", micro_batch_id, terminate)

    INSTRUCTION_MAP = {
        schedule.ForwardPass: _exec_forward_pass,
        schedule.SendActivation: _exec_send_activations,
        schedule.RecvActivation: _exec_recv_activations,
        schedule.SendNextTokens: _exec_send_next_tokens,
        schedule.RecvNextTokens: _exec_recv_next_tokens,
    }


class PipeTrainForwardCommInstrSet:

    def _exec_forward_pass(
        module: ReaLModel,
        tensor_buffer: TensorBuffer,
        stage_id: int,
        micro_batch_id: int,
        step_id: int,
    ):
        PipeInferenceInstrSet._exec_forward_pass(module, tensor_buffer, stage_id, micro_batch_id, step_id)

        loss_fn = tensor_buffer.get("loss_fn", micro_batch_id)
        if loss_fn is not None and constants.is_last_pipe_stage():
            model_output = tensor_buffer.get("batch_output_x", micro_batch_id).pp_output
            if constants.sequence_parallel():
                pad_size = tensor_buffer.get("pad_size", micro_batch_id, remove=True)
                model_output = (model_output[:-pad_size] if pad_size > 0 else model_output)
            loss_kwargs = tensor_buffer.get("loss_inputs", micro_batch_id, remove=True)
            input_cache = tensor_buffer.get("input_cache", micro_batch_id, remove=True)
            packed_input_ids = input_cache.packed_input_ids
            cu_seqlens = input_cache.cu_seqlens

            loss, stats = loss_fn(model_output, packed_input_ids, cu_seqlens, **loss_kwargs)
            loss = loss / tensor_buffer.get("num_micro_batches", micro_batch_id)
            tensor_buffer.put("losses", micro_batch_id, loss)
            tensor_buffer.put("stats", micro_batch_id, stats)

    def _exec_send_activations(
        module: ReaLModel,
        tensor_buffer: TensorBuffer,
        stage_id: int,
        micro_batch_id: int,
        step_id: int,
    ):
        assert stage_id != constants.pipe_parallel_world_size() - 1
        # NOTE: This is different from inference, we remain batch_output_x for backward.
        x: PipeTransferData = tensor_buffer.get("batch_output_x", micro_batch_id)
        p2p.send(x.pp_output, constants.next_pipe_stage(), async_op=False)

    def _exec_recv_activations(
        module: ReaLModel,
        tensor_buffer: TensorBuffer,
        stage_id: int,
        micro_batch_id: int,
        step_id: int,
    ):
        assert not constants.is_first_pipe_stage()

        device = module.device
        dtype = module.dtype
        hidden_dim = module.config.hidden_dim

        mb_seq_len = tensor_buffer.get("mb_seq_lens", micro_batch_id, remove=False)
        act_shape = (mb_seq_len, hidden_dim)

        buf = tensor_buffer.alloc(
            "activation",
            micro_batch_id,
            act_shape,
            dtype,
            device,
            require_grads=True,
        )

        p2p.recv(buf, constants.prev_pipe_stage(), async_op=False)
        tensor_buffer.put("recv_act_buf", micro_batch_id, buf)

    def _exec_send_grads(
        module: ReaLModel,
        tensor_buffer: TensorBuffer,
        stage_id: int,
        micro_batch_id: int,
        step_id: int,
    ):
        assert not constants.is_first_pipe_stage()
        activation = tensor_buffer.get("activation", micro_batch_id, remove=True)
        assert activation.grad is not None
        p2p.send(activation.grad, constants.prev_pipe_stage(), async_op=False)

    def _exec_recv_grads(
        module: ReaLModel,
        tensor_buffer: TensorBuffer,
        stage_id: int,
        micro_batch_id: int,
        step_id: int,
    ):
        assert not constants.is_last_pipe_stage()
        device = module.device
        dtype = module.dtype
        hidden_dim = module.config.hidden_dim
        mb_seq_len = tensor_buffer.get("mb_seq_lens", micro_batch_id, remove=False)
        grad_shape = (mb_seq_len, hidden_dim)
        buf = tensor_buffer.alloc("grad", micro_batch_id, grad_shape, dtype, device)
        p2p.recv(buf, constants.next_pipe_stage(), async_op=False)

    INSTRUCTION_MAP = {
        schedule.ForwardPass: _exec_forward_pass,
        schedule.SendActivation: _exec_send_activations,
        schedule.RecvActivation: _exec_recv_activations,
        schedule.SendGrad: _exec_send_grads,
        schedule.RecvGrad: _exec_recv_grads,
    }


@dataclasses.dataclass
class PipeTrainBackwardReduceInstrSetForDeepSpeed:
    ds_engine: DeepSpeedEngine

    def __post_init__(self):
        self.ds_engine.pipeline_parallelism = True

    @cuda_tmark("bwd", CUDATimeMarkType.backward)
    def _exec_backward_pass(
        self,
        module: ReaLModel,
        tensor_buffer: TensorBuffer,
        stage_id: int,
        micro_batch_id: int,
        step_id: int,
    ):
        assert self.ds_engine is not None
        assert self.ds_engine.optimizer is not None, ("must provide optimizer during "
                                                      "init in order to use backward")
        # The last stage just runs backward on the loss using DeepSpeed's typical
        # mechanisms.
        output_x = tensor_buffer.get("batch_output_x", micro_batch_id, remove=True)

        # We schedule the all-reduces, so disable it in super().backward()
        self.ds_engine.enable_backward_allreduce = False
        self.ds_engine.set_gradient_accumulation_boundary(False)

        is_last_stage = constants.is_last_pipe_stage()
        if is_last_stage:
            loss = tensor_buffer.get("losses", micro_batch_id, remove=True)
            self.ds_engine.backward(loss)
            tensor_buffer.put("losses", micro_batch_id, loss.detach().clone())
            return False, None

        if self.ds_engine.bfloat16_enabled() and not is_last_stage:
            # manually call because we don't call optimizer.backward()
            self.ds_engine.optimizer.clear_lp_grads()

        if not is_last_stage and self.ds_engine.zero_optimization():
            # manually call because we don't call optimizer.backward()
            self.ds_engine.optimizer.micro_step_id += 1

            if self.ds_engine.optimizer.contiguous_gradients:
                self.ds_engine.optimizer.ipg_buffer = []
                buf_0 = torch.empty(
                    int(self.ds_engine.optimizer.reduce_bucket_size),
                    dtype=module.dtype,
                    device=module.device,
                )
                self.ds_engine.optimizer.ipg_buffer.append(buf_0)

                # Use double buffers to avoid data access conflict when overlap_comm is enabled.
                if self.ds_engine.optimizer.overlap_comm:
                    buf_1 = torch.empty(
                        int(self.ds_engine.optimizer.reduce_bucket_size),
                        dtype=module.dtype,
                        device=module.device,
                    )
                    self.ds_engine.optimizer.ipg_buffer.append(buf_1)
                self.ds_engine.optimizer.ipg_index = 0

        handle = tensor_buffer.get("recv_grad_handle", micro_batch_id, remove=True, raise_error=False)
        if handle is not None:
            handle.wait()
        grad = tensor_buffer.get("grad", micro_batch_id, remove=True)

        output_tensor = output_x.pp_output
        torch.autograd.backward(tensors=output_tensor, grad_tensors=grad)

        if not is_last_stage and self.ds_engine.zero_optimization():
            # manually call because we don't call optimizer.backward()
            # Only for Stage 1, Mode 2
            if self.ds_engine.optimizer.use_grad_accum_attribute:
                self.ds_engine.optimizer.fill_grad_accum_attribute()

        if self.ds_engine.bfloat16_enabled() and not is_last_stage:
            # manually call because we don't call optimizer.backward()
            self.ds_engine.optimizer.update_hp_grads(clear_lp_grads=False)

    def _exec_reduce_grads(
        self,
        module: ReaLModel,
        tensor_buffer: TensorBuffer,
        stage_id: int,
        micro_batch_id: int,
        step_id: int,
    ):

        self.ds_engine.set_gradient_accumulation_boundary(True)
        if self.ds_engine.bfloat16_enabled():
            if self.ds_engine.zero_optimization_stage() < ZeroStageEnum.gradients:
                # Make our own list of gradients from the optimizer's FP32 grads
                self.ds_engine.buffered_allreduce_fallback(
                    grads=self.ds_engine.optimizer.get_grads_for_reduction(),
                    elements_per_buffer=MEMORY_OPT_ALLREDUCE_SIZE,
                )
            else:
                raise NotImplementedError("PP+BF16 only work for ZeRO Stage 1")
        else:
            self.ds_engine.allreduce_gradients(bucket_size=MEMORY_OPT_ALLREDUCE_SIZE)

    def _exec_optimizer_step(
        self,
        module: ReaLModel,
        tensor_buffer: TensorBuffer,
        stage_id: int,
        micro_batch_id: int,
        step_id: int,
    ):
        self.ds_engine.set_gradient_accumulation_boundary(True)
        version_steps = tensor_buffer.get("version_steps", 0)
        lr_kwargs = {"epoch": version_steps}
        self.ds_engine._take_model_step(lr_kwargs=lr_kwargs)

        # sync loss scale across pipeline stages
        if not self.ds_engine.bfloat16_enabled():
            loss_scale = self.ds_engine.optimizer.loss_scale
            total_scale_cuda = torch.FloatTensor([float(loss_scale)]).to(module.device)
            dist.all_reduce(
                total_scale_cuda,
                op=dist.ReduceOp.MIN,
                group=constants.grid().get_model_parallel_group(),
            )
            # all_loss_scale = total_scale_cuda[0].item()
            logger.info(
                f"loss scale: {total_scale_cuda}, "
                f"group: {dist.get_process_group_ranks(self.ds_engine.mpu.get_model_parallel_group())}")
            self.ds_engine.optimizer.loss_scaler.cur_scale = min(total_scale_cuda[0].item(), 8192)

    @property
    def INSTRUCTION_MAP(self):
        return {
            schedule.OptimizerStep: self._exec_optimizer_step,
            schedule.ReduceGrads: self._exec_reduce_grads,
            schedule.BackwardPass: self._exec_backward_pass,
        }


@dataclasses.dataclass
class PipeTrainBackwardReduceInstrSetForMegatron:
    megatron_ddp: MegatronDDP
    megatron_dist_optim: MegatronDistOptim

    @property
    def INSTRUCTION_MAP(self):
        return {
            schedule.OptimizerStep: self._exec_optimizer_step,
            schedule.ReduceGrads: self._exec_reduce_grads,
            schedule.BackwardPass: self._exec_backward_pass,
        }


@dataclasses.dataclass
class PipeTrainInstrSet:
    backend: str = dataclasses.field(metadata={"choices": ["deepspeed", "megatron"]})
    ds_engine: Optional[DeepSpeedEngine] = None
    megatron_ddp: Optional[MegatronDDP] = None
    megatron_dist_optim: Optional[MegatronDistOptim] = None

    @property
    def INSTRUCTION_MAP(self):
        if self.backend == "deepspeed":
            return {
                **PipeTrainForwardCommInstrSet.INSTRUCTION_MAP,
                **PipeTrainBackwardReduceInstrSetForDeepSpeed(self.ds_engine).INSTRUCTION_MAP,
            }
        elif self.backend == "megatron":
            return {
                **PipeTrainForwardCommInstrSet.INSTRUCTION_MAP,
                **PipeTrainBackwardReduceInstrSetForMegatron(self.megatron_ddp, self.megatron_dist_optim).INSTRUCTION_MAP,
            }
        else:
            raise NotImplementedError()
