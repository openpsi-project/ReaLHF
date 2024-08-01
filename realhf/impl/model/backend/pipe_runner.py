import copy
import dataclasses
import os
from typing import *

import torch
import torch.distributed as dist
import transformers

import realhf.base.constants as constants
import realhf.base.logging as logging
import realhf.impl.model.parallelism.pipeline_parallel.p2p as p2p
import realhf.impl.model.parallelism.pipeline_parallel.static_schedule as schedule
import realhf.impl.model.utils.cuda_graph as cuda_graph
from realhf.api.core.data_api import SequenceSample
from realhf.api.core.model_api import GenerationHyperparameters
from realhf.impl.model.nn.real_llm_api import ReaLModel
from realhf.impl.model.nn.real_llm_base import PipeCacheData, PipeTransferData
from realhf.impl.model.nn.real_llm_generate import (
    _gather_gen_output_from_list,
    _gather_minibatch_gen_outputs,
    genstep,
    maybe_capture_cudagraph,
    prepare_generate_inputs,
)
from realhf.impl.model.parallelism.pipeline_parallel.instruction import PipeInstruction
from realhf.impl.model.parallelism.pipeline_parallel.static_schedule import PipeSchedule
from realhf.impl.model.parallelism.pipeline_parallel.tensor_storage import TensorBuffer
from realhf.impl.model.utils.padding import pad_sequence_parallel_input

logger = logging.getLogger("Pipeline Runner", "benchmark")


class PipelineError(Exception):
    pass


def _split_and_prefill_pipe_input(
    module: ReaLModel,
    input_: SequenceSample,
    tensor_buffer: TensorBuffer,
    num_micro_batches: int,
    store_kv_cache: bool,
    loss_fn: Optional[Callable] = None,
):
    """Prepare input for pipelined generate, train, or inference.

    Basically, splitting all input tensors into micro batches for
    pipeline parallel.
    """
    n_mbs = num_micro_batches

    # Split sequence into several mini-batches.
    partition_min_size = input_.bs // n_mbs
    splitted = input_.split(n_mbs, min_size=partition_min_size)

    batch_seqlens = [torch.cat(s.seqlens["packed_input_ids"]) for s in splitted]
    assert all(all(x > 0 for x in sls) for sls in batch_seqlens)

    # Sanity check to ensure that the order of splitted sequences
    # is the same across pipeline parallel ranks.
    _batch_seqlen = torch.tensor(
        [sum(x) for x in batch_seqlens],
        device=module.device,
        dtype=torch.long,
    )
    _batch_seqlen_all_gathered = [
        torch.zeros_like(_batch_seqlen)
        for _ in range(constants.pipe_parallel_world_size())
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
                f"Have you ensured that the order of dataset across ranks is the same?",
            )

    mb_seq_lens = []

    # Store partitioned inputs into tensor buffer for later use.
    def input_to_pipe_model_input(input: SequenceSample, mbid: int):
        max_seqlen = int(max(batch_seqlens[mbid]))

        cu_seqlens = torch.nn.functional.pad(
            batch_seqlens[mbid].cuda().cumsum(0), (1, 0)
        ).int()
        packed_input_ids = input.data["packed_input_ids"]

        # sequence parallel input padding
        if constants.sequence_parallel():
            packed_input_ids, cu_seqlens, max_seqlen, pad_size = (
                pad_sequence_parallel_input(packed_input_ids, cu_seqlens, max_seqlen)
            )
            tensor_buffer.put("pad_size", mbid, pad_size)
        x = PipeTransferData(
            cu_seqlens=cu_seqlens.int(),
            max_seqlen=int(max_seqlen),
            store_kv_cache=store_kv_cache,
        )
        if constants.is_first_pipe_stage():
            ys = [PipeCacheData(packed_input_ids=packed_input_ids)] + [
                PipeCacheData() for _ in range(module.num_layers - 1)
            ]
        else:
            ys = [PipeCacheData() for _ in range(module.num_layers)]
        total_len = (
            packed_input_ids.shape[0]
            if not constants.sequence_parallel()
            else packed_input_ids.shape[0] // constants.model_parallel_world_size()
        )
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

    if loss_fn is not None:
        for mbid, x1 in enumerate(splitted):
            tensor_buffer.put("input_cache", mbid, x1)


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
                    f"Pipeline instruction executor does not understand instruction {repr(cmd)}"
                )

            if will_break:
                # With the termination mechanism, skip communication instructions
                # because its peer stages have been terminated
                if (
                    is_last_stage
                    and burn_out_steps < num_stages - 1
                    and type(cmd) != schedule.RecvActivation
                ):
                    continue
                elif not is_last_stage and type(cmd) != schedule.SendActivation:
                    continue

            try:
                instr_map[type(cmd)](module, tensor_buffer, *cmd.args)
            except Exception as e:
                logger.error(
                    f"Model name {constants.model_name()} rank {parllelism_rank}"
                    f" (global rank {global_rank}) step {step_count}, "
                    f"Exception in cmd: {cmd}"
                )
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
        buf = tensor_buffer.get(
            "recv_act_buf", micro_batch_id, remove=True, raise_error=False
        )
        ys = tensor_buffer.get("batch_input_ys", micro_batch_id, remove=False)

        if buf is not None:
            others = tensor_buffer.get(
                "pipe_transfer_infos", micro_batch_id, remove=False
            )
            x = PipeTransferData(pp_input=buf, **others)
            tensor_buffer.put("batch_input_x", micro_batch_id, x)
        else:
            x = tensor_buffer.get("batch_input_x", micro_batch_id, remove=True)

        _zero_grads(x)
        _zero_grads(ys)
        x, ys = module.forward(x, ys)

        tensor_buffer.put(
            "batch_output_x", micro_batch_id, x
        )  # Used by send_activation

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
        cuda_graph_name = f"decoding_{micro_batch_id}"
        graph = cuda_graph.get_graph(cuda_graph_name)

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

        others = None
        if buf is not None:
            if is_first_stage:
                x = tensor_buffer.get("batch_input_x", micro_batch_id, remove=True)
                ys = tensor_buffer.get("batch_input_ys", micro_batch_id, remove=False)
                ys[0].packed_input_ids = (
                    buf  # sequence parallel forward only accept one dim input_ids
                )
                ys[0].packed_position_ids = None
            else:
                others = tensor_buffer.get(
                    "pipe_transfer_infos", micro_batch_id, remove=False
                )
                x = PipeTransferData(pp_input=buf, **others)
                tensor_buffer.put("batch_input_x", micro_batch_id, x)
        else:
            x = tensor_buffer.get("batch_input_x", micro_batch_id, remove=True)

        if graph is None or step_id == 0:
            x, ys = module.forward(x, ys)
        else:
            # only replay decoding phase
            bs = ys[0].cache_seqlens.shape[0]
            if is_first_stage:
                cuda_graph.input_buffer_handle(cuda_graph_name, "input_ids")[:bs].copy_(
                    ys[0].packed_input_ids, non_blocking=True
                )
            if not is_first_stage:
                cuda_graph.input_buffer_handle(cuda_graph_name, "hidden_states").copy_(
                    x.pp_input, non_blocking=True
                )
                cuda_graph.input_buffer_handle(cuda_graph_name, "cu_seqlens").copy_(
                    x.cu_seqlens, non_blocking=True
                )
            cuda_graph.input_buffer_handle(cuda_graph_name, "position_ids")[:bs].copy_(
                ys[0].cache_seqlens, non_blocking=True
            )
            cuda_graph.input_buffer_handle(cuda_graph_name, "cache_seqlens")[:bs].copy_(
                ys[0].cache_seqlens, non_blocking=True
            )

            graph.replay()
            x.pp_output = cuda_graph.output_buffer_handle(cuda_graph_name, "output")

        tensor_buffer.put("batch_output_x", micro_batch_id, x)

        tokenizer = tensor_buffer.get("tokenizer", micro_batch_id)
        gconfig = tensor_buffer.get("gconfig", micro_batch_id)

        # Init KV cache.
        is_prefill_phase = False
        if not tensor_buffer.get("kv_cache_reserved", micro_batch_id):
            # KV cache is attached to x and ys.
            assert constants.pipe_parallel_world_size() >= 2
            x, ys = prepare_generate_inputs(module, gconfig, x, ys, cuda_graph_name)
            if gconfig.use_cuda_graph:
                graph, _, _ = maybe_capture_cudagraph(
                    module,
                    x,
                    ys,
                    cuda_graph_name,
                    force_recapture=gconfig.force_cudagraph_recapture,
                )
            is_prefill_phase = True
            tensor_buffer.put("kv_cache_reserved", micro_batch_id, True)

        # Increase cache_seqlens in the decoding phase.
        if not is_prefill_phase:
            ys[0].cache_seqlens += 1  # global handle

        # Perform a decoding step.
        if constants.is_last_pipe_stage():
            # Gather logits of the final token
            logits = x.pp_output
            if is_prefill_phase:
                logits = logits[x.cu_seqlens[1:] - 1]

            unfinished_sequences = tensor_buffer.get(
                "unfinished_sequences", micro_batch_id
            )
            generated_idx = tensor_buffer.get("generated_idx", micro_batch_id)

            (
                next_tokens,
                logprob,
                logits_mask,
                terminate,
                unfinished_sequences,
            ) = genstep(
                logits,
                tokenizer,
                unfinished_sequences,
                generated_idx,
                gconfig,
            )

            if isinstance(terminate, bool):
                terminate = torch.tensor(
                    terminate, device=logits.device, dtype=torch.bool
                )

            tensor_buffer.put("terminate", micro_batch_id, terminate)
            tensor_buffer.put(
                "unfinished_sequences", micro_batch_id, unfinished_sequences
            )
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
        PipeInferenceInstrSet._exec_send_activations(
            module, tensor_buffer, stage_id, micro_batch_id, step_id
        )
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
            buf = torch.empty(
                act_shape, dtype=dtype, device=device, requires_grad=False
            )
        else:
            batch_length = tensor_buffer.get(
                "batch_lengths", micro_batch_id, remove=False
            )
            batch_length = (
                batch_length // constants.model_parallel_world_size()
                if constants.sequence_parallel()
                else batch_length
            )
            act_shape = (batch_length, hidden_dim)
            buf = torch.empty(
                act_shape, dtype=dtype, device=device, requires_grad=False
            )

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
        """When generating, send next tokens from the last stage to the first
        stage."""
        assert constants.is_last_pipe_stage()
        next_stage = constants.next_pipe_stage()
        next_tokens_to_send = tensor_buffer.get(
            "next_tokens_to_send", micro_batch_id, remove=True
        )
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
        """When generating, recv next tokens from the last stage on the first
        stage Construct next forward input."""
        assert constants.is_first_pipe_stage()
        batch_length = tensor_buffer.get("batch_lengths", micro_batch_id, remove=False)

        device = module.device
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
        PipeInferenceInstrSet._exec_forward_pass(
            module, tensor_buffer, stage_id, micro_batch_id, step_id
        )

        loss_fn = tensor_buffer.get("loss_fn", micro_batch_id)
        if loss_fn is not None and constants.is_last_pipe_stage():
            model_output = tensor_buffer.get("batch_output_x", micro_batch_id).pp_output
            if constants.sequence_parallel():
                pad_size = tensor_buffer.get("pad_size", micro_batch_id, remove=True)
                model_output = (
                    model_output[:-pad_size] if pad_size > 0 else model_output
                )
            input_cache: SequenceSample = tensor_buffer.get(
                "input_cache", micro_batch_id, remove=True
            )
            loss, stats = loss_fn(model_output, input_cache)
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
class PipeTrainInstrSet:

    def _exec_optimizer_step(self, *args, **kwargs):
        raise NotImplementedError()

    def _exec_reduce_grads(self, *args, **kwargs):
        raise NotImplementedError()

    def _exec_backward_pass(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def INSTRUCTION_MAP(self):
        return {
            **PipeTrainForwardCommInstrSet.INSTRUCTION_MAP,
            schedule.OptimizerStep: self._exec_optimizer_step,
            schedule.ReduceGrads: self._exec_reduce_grads,
            schedule.BackwardPass: self._exec_backward_pass,
        }


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
        input_: SequenceSample,
        num_micro_batches: Optional[int] = None,
    ):
        """Run one forward step over a batch of tokens and return the
        logits."""

        if num_micro_batches is None:
            num_micro_batches = self.default_inf_mbs

        tensor_buffer = TensorBuffer()

        _split_and_prefill_pipe_input(
            module=self.module,
            tensor_buffer=tensor_buffer,
            num_micro_batches=num_micro_batches,
            input_=input_,
            store_kv_cache=False,
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
                if constants.sequence_parallel():
                    pad_size = tensor_buffer.get("pad_size", i, remove=True)
                    logits = logits[:-pad_size] if pad_size > 0 else logits
                logits_list.append(logits)
            logits = torch.cat(logits_list, dim=0)

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_: SequenceSample,
        tokenizer: transformers.PreTrainedTokenizerFast,
        gconfig: GenerationHyperparameters = dataclasses.field(
            default_factory=GenerationHyperparameters
        ),
        num_micro_batches: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[PipeCacheData]]:
        if constants.sequence_parallel():
            raise NotImplementedError(
                "Sequence parallel is not supported for generation"
            )

        if num_micro_batches is None:
            num_micro_batches = self.default_inf_mbs

        tensor_buffer = TensorBuffer()

        _split_and_prefill_pipe_input(
            module=self.module,
            tensor_buffer=tensor_buffer,
            num_micro_batches=num_micro_batches,
            input_=input_,
            store_kv_cache=True,
        )

        # for elegant generation termination
        gconfig = copy.deepcopy(gconfig)
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
            term = all(
                [
                    tensor_buffer.get("terminate", mbid)
                    for mbid in range(num_micro_batches)
                ]
            )
            return term

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
                    gen_logprob_ph=tensor_buffer.get(
                        "gen_logprob_ph", mbid, remove=True
                    ),
                    gen_logits_mask_ph=tensor_buffer.get(
                        "gen_logits_mask_ph", mbid, remove=True
                    ),
                )
            ]

        gen_tokens, log_probs, logits_mask = _gather_minibatch_gen_outputs(
            *list(zip(*generate_output))
        )

        if gconfig.use_cuda_graph and gconfig.force_cudagraph_recapture:
            for micro_batch_id in range(num_micro_batches):
                cuda_graph.destroy(f"decoding_{micro_batch_id}")

        return gen_tokens, log_probs, logits_mask, None, None

    def train_batch(
        self,
        instr_set: PipeTrainInstrSet,
        input_: SequenceSample,
        loss_fn: Callable,
        version_steps: int,
        num_micro_batches: Optional[int] = None,
    ):
        # TODO: return whether update success
        if not torch._C.is_grad_enabled():
            raise RuntimeError(
                f"train_batch() requires gradients enabled. Use eval_batch() instead."
            )

        if num_micro_batches is None:
            num_micro_batches = self.default_train_mbs

        tensor_buffer = TensorBuffer()
        for i in range(num_micro_batches):
            tensor_buffer.put("num_micro_batches", i, num_micro_batches)
            tensor_buffer.put("version_steps", i, version_steps)
            tensor_buffer.put("loss_fn", i, loss_fn)

        _split_and_prefill_pipe_input(
            module=self.module,
            tensor_buffer=tensor_buffer,
            num_micro_batches=num_micro_batches,
            input_=input_,
            store_kv_cache=False,
            loss_fn=loss_fn,
        )

        sched = schedule.TrainSchedule(
            micro_batches=num_micro_batches,
            stages=constants.pipe_parallel_world_size(),
            stage_id=constants.pipe_parallel_rank(),
        )
        _exec_pipe_schedule(
            module=self.module,
            tensor_buffer=tensor_buffer,
            instr_map=instr_set.INSTRUCTION_MAP,
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
        input_: SequenceSample,
        loss_fn: Callable,
        num_micro_batches: Optional[int] = None,
    ):
        if num_micro_batches is None:
            num_micro_batches = self.default_train_mbs

        tensor_buffer = TensorBuffer()
        for i in range(num_micro_batches):
            tensor_buffer.put("num_micro_batches", i, num_micro_batches)
            tensor_buffer.put("loss_fn", i, loss_fn)

        _split_and_prefill_pipe_input(
            module=self.module,
            tensor_buffer=tensor_buffer,
            num_micro_batches=num_micro_batches,
            input_=input_,
            store_kv_cache=False,
            loss_fn=loss_fn,
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
