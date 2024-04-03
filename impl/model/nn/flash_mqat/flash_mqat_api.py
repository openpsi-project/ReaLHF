from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import contextlib
import copy
import dataclasses
import functools
import gc
import json
import os
import time

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers

from api.config.config_base import ModelName
from api.config.config_flash_model import FlashMQATConfig
from impl.model.nn.flash_mqat.flash_generate import generate, GenerationConfig
from impl.model.nn.flash_mqat.flash_mqat_base import (flash_model_embed_param_count,
                                                      flash_model_embedding_param_keys,
                                                      flash_model_head_param_count,
                                                      flash_model_head_param_keys,
                                                      flash_model_tblock_param_count,
                                                      flash_model_tblock_param_keys, FlashMQATBlock,
                                                      OutputHead, SequenceParallelActorHead,
                                                      SequenceParallelCriticHead, VocabPositionEmbedding)
from impl.model.nn.flash_mqat.flash_mqat_parallel import (get_flash_model_param_shape, intervals_partition_fn,
                                                          mp_merge_flash_mqat_state_dict,
                                                          mp_partition_flash_mqat_state_dict,
                                                          mp_partition_key, partition_pipeline_layers,
                                                          pipeline_repartition_strategy, shape_partition_fn)
from impl.model.parallelism.model_parallel.modules import (ColumnParallelLinear, ParallelEmbedding,
                                                           RowParallelLinear)
from impl.model.utils.data import DuckGenerationOutput, DuckModelOutput, PipeCacheData, PipeTransferData
from impl.model.utils.save_load import get_ckpt_spec, load_from_disk, save_to_disk
import api.config.config_system
import api.huggingface
import api.model
import base.constants
import base.gpu_utils as gpu_utils
import base.logging as logging
import base.topology

try:
    from flash_attn.bert_padding import pad_input, unpad_input
except ModuleNotFoundError:
    pass
import base.logging as logging

logger = logging.getLogger("FlashMQAT Interface")

MAX_PYTORCH_N_INTERVALS = 1024
CUDA_INTERVAL_OP_CHUNK_SIZE = 2048


@dataclasses.dataclass
class FlashMQATParallelismHelper:
    embedding_param_names: Callable[[FlashMQATConfig], List[str]]
    tblock_param_names: Callable[[FlashMQATConfig, int], List[str]]
    head_param_names: Callable[[FlashMQATConfig], List[str]]


@dataclasses.dataclass
class FlashMQATConvertHelper:
    config_converter: Callable[[transformers.PretrainedConfig], FlashMQATConfig]
    state_dict_converter: Optional[Callable[[Dict, FlashMQATConfig], Dict]]
    state_dict_converter_to_hf: Optional[Callable[[Dict, FlashMQATConfig], Dict]] = None


@dataclasses.dataclass
class ReparallelizeSenderStep:
    rank: int
    sender_mp_portion_id: int
    receiver_mp_portion_id: int
    param_keys: List[str]
    param_intervals: torch.Tensor
    param_intervals_cpu: List[Tuple[int, int]]
    max_interval_size: int
    param_size: int
    group: torch.distributed.ProcessGroup
    dst_ranks: List[int]
    remove: bool = False


@dataclasses.dataclass
class ReparallelizeReceiverStep:
    rank: int
    sender_mp_portion_id: int
    receiver_mp_portion_id: int
    sender_param_intervals: torch.Tensor
    sender_param_intervals_cpu: List[Tuple[int, int]]
    sender_max_interval_size: int
    receiver_param_intervals: torch.Tensor
    receiver_param_intervals_cpu: List[Tuple[int, int]]
    receiver_max_interval_size: int
    param_size: int
    param_keys: List[str]
    param_dtype: torch.dtype
    src: int
    group: torch.distributed.ProcessGroup


def _is_integer_list_contiguous(l: List[int]) -> bool:
    return np.all(np.array(l) == np.arange(len(l)) + l[0])


def _are_intervals_contiguous(l: List[Tuple[int, int]]) -> bool:
    l = sorted(l, key=lambda x: x[0])
    res = True
    for i in range(len(l) - 1):
        res &= l[i][1] == l[i + 1][0]
    return res


def slice_intervals(
    tensor: torch.Tensor,
    intervals: torch.IntTensor,
    intervals_cpu: List[Tuple[int, int]],
    max_interval_size: int,
    output_size: int,
) -> torch.Tensor:
    assert len(tensor.shape) == 1
    if len(intervals_cpu) == 1:
        return tensor[intervals_cpu[0][0]:intervals_cpu[0][1]]
    elif len(intervals_cpu) <= MAX_PYTORCH_N_INTERVALS:
        return torch.cat([tensor[start:end] for start, end in intervals_cpu])
    try:
        import interval_op_cuda

        interval_sizes = intervals[:, 1] - intervals[:, 0]
        offsets = torch.nn.functional.pad(interval_sizes.cumsum(0)[:-1], (1, 0), value=0)
        return interval_op_cuda.slice_intervals_cuda_half(
            tensor,
            intervals,
            interval_sizes,
            offsets,
            max_interval_size,
            output_size,
        )
    except ModuleNotFoundError:
        logger.warning("interval_op_cuda not found, falling back to PyTorch implementation.")
        return torch.cat([tensor[start:end] for start, end in intervals])


def set_intervals(
    src: torch.Tensor,
    dst: torch.Tensor,
    intervals: torch.IntTensor,
    intervals_cpu: List[Tuple[int, int]],
    max_interval_size: int,
):
    assert len(dst.shape) == len(src.shape) == 1
    if len(intervals_cpu) <= MAX_PYTORCH_N_INTERVALS:
        offset = 0
        for i, j in intervals_cpu:
            dst[i:j] = src[offset:offset + j - i]
            offset += j - i
        assert offset == src.shape[0]
    try:
        import interval_op_cuda

        interval_sizes = intervals[:, 1] - intervals[:, 0]
        offsets = torch.nn.functional.pad(interval_sizes.cumsum(0)[:-1], (1, 0), value=0)
        interval_op_cuda.set_intervals_cuda_half(
            src,
            dst,
            intervals,
            interval_sizes,
            offsets,
            max_interval_size,
        )
    except ModuleNotFoundError:
        logger.warning("interval_op_cuda not found, falling back to PyTorch implementation.")
        offset = 0
        for i, j in intervals_cpu:
            dst[i:j] = src[offset:offset + j - i]
            offset += j - i
        assert offset == src.shape[0]


def recursive_getattr(obj, attr_string):
    attrs = attr_string.split(".")
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def _keys_from_layer_indices(config: FlashMQATConfig, layer_indices: List[int]) -> List[str]:
    # assert _is_integer_list_contiguous(layer_indices)
    sd_keys = []
    for layer_idx in layer_indices:
        if layer_idx == 0:
            sd_keys += flash_model_embedding_param_keys(config)
        elif layer_idx == config.n_layers + 1:
            sd_keys += flash_model_head_param_keys(config)
        else:
            sd_keys += flash_model_tblock_param_keys(config, layer_idx - 1)
    return sd_keys


@dataclasses.dataclass
class ContiguousParamSpec:
    start_idx: int
    end_idx: int
    shape: torch.Size


_FLAT_PARAM_INDICES_CACHE = {}


def _param_intervals_from_keys(
    model_name: ModelName,
    config: FlashMQATConfig,
    param_spec: Dict[str, ContiguousParamSpec],
    mp_size: int,
    sd_keys: List[str],
    portion_size: int,
    portion_rank: int,
) -> List[int]:
    if portion_size == 1:
        start, end = None, None
        for k in sd_keys:
            if start is None or param_spec[k].start_idx < start:
                start = param_spec[k].start_idx
            if end is None or param_spec[k].end_idx > end:
                end = param_spec[k].end_idx
        return [(start, end)]

    intervals = []
    for k in sd_keys:
        if (model_name, k.split('.',
                                1)[1], mp_size, portion_rank, portion_size) not in _FLAT_PARAM_INDICES_CACHE:
            zero_start_intervals = mp_partition_key(
                k,
                get_flash_model_param_shape(k, config, mp_size),
                portion_rank,
                portion_size,
                config,
                partition_fn=intervals_partition_fn,
            )
            _FLAT_PARAM_INDICES_CACHE[(model_name, k.split('.', 1)[1], mp_size, portion_rank,
                                       portion_size)] = zero_start_intervals
        else:
            zero_start_intervals = _FLAT_PARAM_INDICES_CACHE[(model_name, k.split('.', 1)[1], mp_size,
                                                              portion_rank, portion_size)]
        intervals += (zero_start_intervals + param_spec[k].start_idx).tolist()
    # assert len(set([x[0] for x in intervals])) == len(intervals)
    intervals = sorted(intervals, key=lambda x: x[0])
    return intervals


def _param_size_from_keys(
    config: FlashMQATConfig,
    src_mp_size: int,
    sd_keys: List[str],
    src2dst_tp_size: int,
    src2dst_tp_rank: int,
) -> Tuple[List[int], int]:
    param_size = 0
    for k in sd_keys:
        new_shape = mp_partition_key(
            k,
            get_flash_model_param_shape(k, config, src_mp_size),
            src2dst_tp_rank,
            src2dst_tp_size,
            config,
            partition_fn=shape_partition_fn,
        )
        param_size += int(np.prod(new_shape))
    return param_size


@dataclasses.dataclass
class ReparallelizeTraget:
    comm_plan: List[Union[ReparallelizeSenderStep, ReparallelizeReceiverStep]]
    to_param_spec: Dict[str, ContiguousParamSpec]
    to_param_size: int
    to_layers_handle: nn.ModuleList
    to_layer_start_idx: int
    to_layer_end_idx: int


@contextlib.contextmanager
def _disable_sequence_parallel_of_module(l: nn.Module):
    _states = []
    for _, m in l.named_modules():
        if isinstance(m, (RowParallelLinear, ColumnParallelLinear, VocabPositionEmbedding)):
            _states.append(m.sequence_parallel)
            m.sequence_parallel_enable(False)
    yield
    for _, m in l.named_modules():
        if isinstance(m, (RowParallelLinear, ColumnParallelLinear, VocabPositionEmbedding)):
            m.sequence_parallel_enable(_states.pop(0))
    assert len(_states) == 0


def _build_param_spec(layer_indices: List[int], config: FlashMQATConfig,
                      mp_size: int) -> Tuple[Dict[str, ContiguousParamSpec], int]:
    if len(layer_indices) == 0:
        return {}, 0
    param_spec = {}
    param_size = 0
    for layer_idx in layer_indices:
        sd_keys = []
        if layer_idx == 0:
            sd_keys += flash_model_embedding_param_keys(config)
        elif layer_idx == config.n_layers + 1:
            sd_keys += flash_model_head_param_keys(config)
        else:
            sd_keys += flash_model_tblock_param_keys(config, layer_idx - 1)

        for k in sd_keys:
            shape = get_flash_model_param_shape(k, config, mp_size)
            param_spec[k] = ContiguousParamSpec(param_size, param_size + int(np.prod(shape)), shape)
            param_size += int(np.prod(shape))
    return param_spec, param_size


def map_param_to_contigous_memory(
    layers: nn.ModuleList,
    param_spec: Dict[str, ContiguousParamSpec],
    contiguous_param: torch.Tensor,
    layer_idx_offset: int,
):
    for local_layer_idx, l in enumerate(layers):
        layer_idx = local_layer_idx + layer_idx_offset
        for k, v in l.named_parameters():
            spec = param_spec[f"{layer_idx}.{k}"]
            old_param_data = v.data
            recursive_getattr(l, k).data = contiguous_param[spec.start_idx:spec.end_idx].view(spec.shape)
            # This is for reward model. We should initialize the reward head instead of letting it be all-zero.
            if old_param_data.shape == spec.shape:
                v.data.copy_(old_param_data)
            else:
                assert old_param_data.shape == torch.Size([0]), (old_param_data.shape, spec.shape)


def _split_intervals(intervals, K):
    result = []
    for start, end in intervals:
        if end - start <= K:
            result.append((start, end))
        else:
            # Calculate how many chunks are needed
            num_chunks = (end - start) // K
            remainder = (end - start) % K

            # Generate the chunks
            for i in range(num_chunks):
                result.append((start, start + K))
                start += K
            if remainder:
                result.append((start, start + remainder))

    return result


def _derive_reparallelize_comm_plan(
    from_model_name: ModelName,
    to_model_name: ModelName,
    from_topo: base.topology.PipeModelDataParallelTopology,
    to_topo: base.topology.PipeModelDataParallelTopology,
    from_model_config: FlashMQATConfig,
    to_model_config: FlashMQATConfig,
    pg_info: gpu_utils.NCCLProcessGroupInfo,
    dtype: Optional[torch.dtype] = torch.float16,
) -> List[ReparallelizeReceiverStep | ReparallelizeSenderStep]:
    src_mp_size = from_topo.get_dim("model")
    dst_mp_size = to_topo.get_dim("model")
    assert src_mp_size % dst_mp_size == 0 or dst_mp_size % src_mp_size == 0
    for k, v in dataclasses.asdict(to_model_config).items():
        if k not in [
                "is_critic",
                "sequence_parallel",
                "gradient_accumulation_fusion",
                "ckpt_attn",
                "ckpt_mlp",
        ] and v != getattr(from_model_config, k):
            raise ValueError(
                f"Can't load a checkpoint with different config (key `{k}`, "
                f"value in checkpoint is `{v}`, current value is `{getattr(from_model_config, k)}`).")
    if (from_model_config.n_kv_heads % src_mp_size == 0) != (from_model_config.n_kv_heads % dst_mp_size == 0):
        raise ValueError("Whether to partition kv heads should remain the same.")

    from_layer_mapping = partition_pipeline_layers(
        from_model_config,
        from_topo.get_dim("pipe"),
        flash_model_embed_param_count,
        flash_model_tblock_param_count,
        flash_model_head_param_count,
    )
    from_layer_mapping = {k: list(range(v[0], v[1])) for k, v in from_layer_mapping.items()}
    to_layer_mapping = partition_pipeline_layers(
        to_model_config,
        to_topo.get_dim("pipe"),
        flash_model_embed_param_count,
        flash_model_tblock_param_count,
        flash_model_head_param_count,
    )
    to_layer_mapping = {k: list(range(v[0], v[1])) for k, v in to_layer_mapping.items()}
    repart_strat = pipeline_repartition_strategy(from_layer_mapping, to_layer_mapping)

    if base.constants.has_model_name(from_model_name):
        with base.constants.model_scope(from_model_name):
            from_layer_indices = from_layer_mapping[base.constants.pipe_parallel_rank()]
            from_model_param_specs, _ = _build_param_spec(from_layer_indices, from_model_config,
                                                          from_topo.get_dim("model"))
    if base.constants.has_model_name(to_model_name):
        with base.constants.model_scope(to_model_name):
            to_layer_indices = to_layer_mapping[base.constants.pipe_parallel_rank()]
            to_model_param_specs, _ = _build_param_spec(to_layer_indices, to_model_config,
                                                        to_topo.get_dim("model"))

    comm_plan = []

    src_dp_size = from_topo.get_dim("data")

    # derive a global NCCL communication plan
    for (pp_i, pp_j), layer_indices in repart_strat.items():
        if len(layer_indices) == 0:
            continue

        for mp_i in range(src_mp_size):
            if dst_mp_size > src_mp_size:
                factor = dst_mp_size // src_mp_size
                mp_js = [i + factor * mp_i for i in range(factor)]
                receiver_mp_portion_id = 0
            else:
                factor = src_mp_size // dst_mp_size
                mp_js = [mp_i // factor]
                receiver_mp_portion_id = mp_i % factor
            for sender_mp_portion_id, mp_j in enumerate(mp_js):

                for dp_i in range(src_dp_size):
                    key = gpu_utils.ParamSyncPair(
                        src=from_model_name,
                        src_dp_rank=dp_i,
                        src_mp_rank=mp_i,
                        src_pp_rank=pp_i,
                        dst=to_model_name,
                        dst_mp_rank=mp_j,
                        dst_pp_rank=pp_j,
                    )
                    src = pg_info.param_sync_src_ranks[key]
                    group = pg_info.param_sync_groups[key]
                    dst_ranks = pg_info.param_sync_dst_ranks[key]

                    param_intervals = param_keys = receiver_param_intervals = None
                    max_param_interval_size = max_receiver_param_interval_size = -1
                    param_intervals_cpu = receiver_param_intervals_cpu = None
                    param_size = -1
                    if torch.distributed.get_rank() in dst_ranks or torch.distributed.get_rank() == src:
                        param_keys = _keys_from_layer_indices(from_model_config, layer_indices)

                        if torch.distributed.get_rank() == src:

                            param_intervals_cpu = _param_intervals_from_keys(
                                model_name=from_model_name,
                                config=from_model_config,
                                mp_size=src_mp_size,
                                param_spec=from_model_param_specs,
                                sd_keys=param_keys,
                                portion_size=max(dst_mp_size // src_mp_size, 1),
                                portion_rank=sender_mp_portion_id,
                            )
                            if len(param_intervals_cpu) > MAX_PYTORCH_N_INTERVALS:
                                param_intervals_cpu = _split_intervals(param_intervals_cpu,
                                                                       CUDA_INTERVAL_OP_CHUNK_SIZE)
                                max_param_interval_size = CUDA_INTERVAL_OP_CHUNK_SIZE
                            else:
                                max_param_interval_size = max(j - i for i, j in param_intervals_cpu)
                            param_intervals = torch.tensor(param_intervals_cpu,
                                                           dtype=torch.long,
                                                           device="cuda")
                        if torch.distributed.get_rank() in dst_ranks:
                            print(layer_indices)
                            receiver_param_intervals_cpu = _param_intervals_from_keys(
                                model_name=to_model_name,
                                config=to_model_config,
                                mp_size=dst_mp_size,
                                param_spec=to_model_param_specs,
                                sd_keys=param_keys,
                                portion_size=max(src_mp_size // dst_mp_size, 1),
                                portion_rank=receiver_mp_portion_id,
                            )
                            if len(receiver_param_intervals_cpu) > MAX_PYTORCH_N_INTERVALS:
                                receiver_param_intervals_cpu = _split_intervals(
                                    receiver_param_intervals_cpu, CUDA_INTERVAL_OP_CHUNK_SIZE)
                                max_receiver_param_interval_size = CUDA_INTERVAL_OP_CHUNK_SIZE
                            else:
                                max_receiver_param_interval_size = max(
                                    j - i for i, j in receiver_param_intervals_cpu)
                            receiver_param_intervals = torch.tensor(receiver_param_intervals_cpu,
                                                                    dtype=torch.long,
                                                                    device="cuda")
                        param_size = _param_size_from_keys(
                            config=from_model_config,
                            src_mp_size=src_mp_size,
                            sd_keys=param_keys,
                            src2dst_tp_size=max(dst_mp_size // src_mp_size, 1),
                            src2dst_tp_rank=sender_mp_portion_id,
                        )

                    for dst_rank in dst_ranks:
                        comm_plan.append(
                            ReparallelizeReceiverStep(
                                rank=dst_rank,
                                sender_mp_portion_id=sender_mp_portion_id,
                                receiver_mp_portion_id=receiver_mp_portion_id,
                                param_keys=param_keys,
                                sender_param_intervals_cpu=param_intervals_cpu,
                                sender_param_intervals=param_intervals,
                                sender_max_interval_size=max_param_interval_size,
                                receiver_param_intervals_cpu=receiver_param_intervals_cpu,
                                receiver_param_intervals=receiver_param_intervals,
                                receiver_max_interval_size=max_receiver_param_interval_size,
                                param_size=param_size,
                                param_dtype=dtype,
                                src=src,
                                group=group,
                            ))
                    comm_plan.append(
                        ReparallelizeSenderStep(
                            rank=src,
                            sender_mp_portion_id=sender_mp_portion_id,
                            receiver_mp_portion_id=receiver_mp_portion_id,
                            param_keys=param_keys,
                            param_intervals=param_intervals,
                            param_intervals_cpu=param_intervals_cpu,
                            max_interval_size=max_param_interval_size,
                            param_size=param_size,
                            group=group,
                            dst_ranks=dst_ranks,
                        ))
    for i, step in enumerate(comm_plan):
        if isinstance(step, ReparallelizeReceiverStep):
            continue
        step: ReparallelizeSenderStep
        required_by_nex_steps = False
        for nex_step in comm_plan[i + 1:]:
            if (isinstance(nex_step, ReparallelizeSenderStep) and nex_step.rank == step.rank
                    and nex_step.param_keys == step.param_keys):
                required_by_nex_steps = True
                break
        step.remove = not required_by_nex_steps

    return comm_plan


class FlashMQATModel(nn.Module):
    _parallelism_helpers: Dict[str, FlashMQATParallelismHelper] = {}
    _convert_helpers: Dict[str, FlashMQATConvertHelper] = {}

    def __init__(
        self,
        config: FlashMQATConfig,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        if dtype is None:
            dtype = torch.float16
        self.config = config
        self.dtype = dtype
        self.device = device

        self.layer_mapping = partition_pipeline_layers(
            config,
            base.constants.pipe_parallel_world_size(),
            flash_model_embed_param_count,
            flash_model_tblock_param_count,
            flash_model_head_param_count,
        )
        self.layer_idx_start = self.layer_mapping[base.constants.pipe_parallel_rank()][0]
        self.layer_idx_end = self.layer_mapping[base.constants.pipe_parallel_rank()][1]
        self.num_stages = base.constants.pipe_parallel_world_size()

        self.layers = nn.ModuleList()
        self.sequence_parallel = config.sequence_parallel

        self._instantiated = False
        self._instantiation_hooks = []

        self._reparallelize_targets: Dict[Tuple[ModelName, ModelName], ReparallelizeTraget] = {}

        # Flatten all parameters to a contiguous GPU buffer to reduce the time of CUDAFree and CUDAMalloc
        self._param_spec, self._param_size = _build_param_spec(
            list(range(self.layer_idx_start, self.layer_idx_end)),
            self.config,
            base.constants.model_parallel_world_size(),
        )
        self.contiguous_param = None

        self._offload_buffer = None
        self._offload_stream = torch.cuda.Stream()
        self._offload_event = torch.cuda.Event()
        self._offloaded = False

    def instantiate(self):
        assert not self._instantiated
        layers = []
        for idx in range(self.layer_idx_start, self.layer_idx_end):
            layers.append(self._build_layer(idx, self.config))

        self.layers = nn.ModuleList(layers)

        self.contiguous_param = torch.empty(self._param_size, dtype=self.dtype, device=self.device)
        map_param_to_contigous_memory(self.layers, self._param_spec, self.contiguous_param,
                                      self.layer_idx_start)

        for h in self._instantiation_hooks:
            h()

        self._instantiated = True
        self._instantiation_hooks = []

    def async_offload(self):
        assert not self._offloaded
        assert self._instantiated
        assert self.contiguous_param is not None
        if self._offload_buffer is None:
            self._offload_buffer = torch.empty_like(self.contiguous_param,
                                                    dtype=self.dtype,
                                                    device="cpu",
                                                    pin_memory=True)
        else:
            assert self._offload_buffer.shape == self.contiguous_param.shape
        dummy_tensor = torch.tensor((), device=self.device, dtype=self.dtype)
        self.contiguous_param = None
        for i, l in enumerate(self.layers):
            layer_idx = self.layer_idx_start + i
            with torch.cuda.stream(self._offload_stream):
                for k, p in l.named_parameters():
                    spec = self._param_spec[f"{layer_idx}.{k}"]
                    self._offload_buffer[spec.start_idx:spec.end_idx].copy_(p.data.view(-1),
                                                                            non_blocking=True)
                    p.data = dummy_tensor
        self._offload_event.record(self._offload_stream)
        self._offloaded = True

    def wait_for_offload(self):
        assert self._offloaded
        self._offload_event.synchronize()

    @property
    def num_layers(self):
        return self.layer_idx_end - self.layer_idx_start

    @property
    def is_critic(self):
        return self.config.is_critic

    def _build_layer(self, idx: int, config: FlashMQATConfig) -> nn.Module:
        dtype = self.dtype
        device = self.device
        if idx == 0:
            l = VocabPositionEmbedding(config, dtype=dtype, device=device)
        elif idx == config.n_layers + 1:
            l = self._build_output_head(config)
        else:
            l = FlashMQATBlock(
                config=config,
                layer_index=idx - 1,
                output_layernorm=(idx == config.n_layers),
                dtype=dtype,
                device=device,
            )
        return l

    def _build_output_head(self, config: FlashMQATConfig) -> nn.Module:
        dtype = self.dtype
        device = self.device
        if config.is_critic and config.sequence_parallel:
            l = SequenceParallelCriticHead(
                config.hidden_dim,
                1,
                bias=False,
                device=device,
                dtype=dtype,
            )
        elif not config.is_critic and base.constants.model_parallel_world_size() > 1:
            l = SequenceParallelActorHead(
                config.hidden_dim,
                config.vocab_size,
                bias=False,
                sequence_parallel=config.sequence_parallel,
                async_tensor_model_parallel_allreduce=not config.sequence_parallel,
                gradient_accumulation_fusion=config.gradient_accumulation_fusion,
                device=device,
                dtype=dtype,
            )
        else:
            l = OutputHead(
                config.hidden_dim,
                1 if config.is_critic else config.vocab_size,
                bias=False,
                device=device,
                dtype=dtype,
            )
        return l

    def gradient_checkpointing_enable(self, attn: Optional[bool] = False, mlp: Optional[bool] = False):
        for l in self.layers:
            if isinstance(l, FlashMQATBlock):
                l.gradient_checkpointing_enable(attn, mlp)

    @contextlib.contextmanager
    def gradient_checkpointing_disable(self):
        _states = []
        for l in self.layers:
            if isinstance(l, FlashMQATBlock):
                _states.append((l.ckpt_attn, l.ckpt_mlp, l.ckpt_full))
                l.gradient_checkpointing_disable()
        yield
        for l in self.layers:
            if isinstance(l, FlashMQATBlock):
                l.ckpt_attn, l.ckpt_mlp, l.ckpt_full = _states.pop(0)

    @contextlib.contextmanager
    def sequence_parallel_disable(self):
        x = self.sequence_parallel
        self.sequence_parallel = False
        yield
        self.sequence_parallel = x

    def __overlapped_load_forward(self, x: PipeTransferData,
                                  ys: List[PipeCacheData]) -> Tuple[PipeTransferData, List[PipeCacheData]]:
        assert len(ys) == self.num_layers
        raw_pp_input = x.pp_input
        self.contiguous_param = torch.empty(self._param_size, dtype=self.dtype, device=self.device)
        map_param_to_contigous_memory(self.layers, self._param_spec, self.contiguous_param,
                                      self.layer_idx_start)
        self.wait_for_offload()
        for layer_idx, y, l in zip(range(self.layer_idx_start, self.layer_idx_end), ys, self.layers):
            with torch.cuda.stream(self._offload_stream):
                # NOTE: although we can do more fine-grained overlapping, the overhead that can be
                # reduced is very small (~50ms), which is unnecessary for now.
                for k, v in l.named_parameters():
                    spec = self._param_spec[f"{layer_idx}.{k}"]
                    v.data.copy_(
                        self._offload_buffer[spec.start_idx:spec.end_idx].view(spec.shape),
                        non_blocking=True,
                    )
            torch.cuda.default_stream().wait_stream(self._offload_stream)
            if not self.sequence_parallel:
                with _disable_sequence_parallel_of_module(l):
                    x = l(x, y)
            else:
                x = l(x, y)
            x.pp_input = x.pp_output
        self._offloaded = False
        x.pp_input = raw_pp_input
        return x, ys

    def __forward(self, x: PipeTransferData,
                  ys: List[PipeCacheData]) -> Tuple[PipeTransferData, List[PipeCacheData]]:
        layers = self.layers
        assert len(ys) == len(layers)
        raw_pp_input = x.pp_input
        for i, (layer, y) in enumerate(zip(layers, ys)):
            if not self.sequence_parallel:
                with _disable_sequence_parallel_of_module(layer):
                    x = layer(x, y)  # This will set pp_output.
            else:
                x = layer(x, y)
            x.pp_input = x.pp_output
        # Finally, pp_input is the input of this pipeline stage (maybe across several layers),
        # pp_output is the output of this pipeline stage.
        # In the first stage, pp_input is None.
        x.pp_input = raw_pp_input
        return x, ys

    def forward(self, x: PipeTransferData,
                ys: List[PipeCacheData]) -> Tuple[PipeTransferData, List[PipeCacheData]]:
        if x.max_seqlen is not None and not isinstance(x.max_seqlen, int):
            x.max_seqlen = int(x.max_seqlen)
        if x.cu_seqlens is not None and not isinstance(x.cu_seqlens, torch.IntTensor):
            x.cu_seqlens = x.cu_seqlens.int()

        # Copy input tensor to a pinned buffer.
        mp_size = base.constants.model_parallel_world_size()
        batch_length = None
        if ys[0].input_ids is not None:
            batch_length = ys[0].input_ids.shape[0]
        if x.pp_input is not None:
            batch_length = x.pp_input.shape[0]
        assert batch_length is not None
        padded_batch_length = (batch_length + mp_size - 1) // mp_size * mp_size
        pad_size = padded_batch_length - batch_length

        if self.sequence_parallel and pad_size > 0 and ys[0].input_ids is not None:
            _cu_seqlens = x.cu_seqlens
            _max_seqlen = x.max_seqlen
            _input_ids = ys[0].input_ids
            _pp_input = x.pp_input

            x.cu_seqlens = torch.nn.functional.pad(x.cu_seqlens, (0, 1), value=padded_batch_length)
            x.max_seqlen = max(x.max_seqlen, padded_batch_length - batch_length)
            if ys[0].input_ids is not None:
                input_ids_buf = base.constants.get_global_memory_buffer().get_tensor(
                    (padded_batch_length,),
                    dtype=torch.long,
                    name="flash_model_input_ids",
                    force_zero=True,
                )
                input_ids_buf[:batch_length] = ys[0].input_ids
                ys[0].input_ids = input_ids_buf

            if x.pp_input is not None:
                pp_input_buf = base.constants.get_global_memory_buffer().get_tensor(
                    (padded_batch_length, *x.pp_input.shape[1:]),
                    dtype=x.pp_input.dtype,
                    name="flash_model_pp_input",
                    force_zero=True,
                )
                pp_input_buf[:batch_length] = x.pp_input
                x.pp_input = pp_input_buf

        # Main forward calls.
        if not self._offloaded:
            x, ys = self.__forward(x, ys)
        else:
            x, ys = self.__overlapped_load_forward(x, ys)

        # Resume from padding.
        if self.sequence_parallel and pad_size > 0 and ys[0].input_ids is not None:
            x.pp_output = x.pp_output[:-pad_size]

            x.pp_input = _pp_input
            ys[0].input_ids = _input_ids
            x.cu_seqlens = _cu_seqlens
            x.max_seqlen = _max_seqlen

            if x.store_kv_cache:
                for y in ys:
                    if y.k_cache is not None:
                        y.k_cache = y.k_cache[:-pad_size]
                    if y.v_cache is not None:
                        y.v_cache = y.v_cache[:-pad_size]
        return x, ys

    def _forward(
        self,
        input_ids: torch.LongTensor,
        cu_seqlens: torch.IntTensor,
        position_ids: torch.LongTensor,
        hidden_states: Optional[torch.Tensor],
        k_caches: Optional[List[torch.Tensor]],
        v_caches: Optional[List[torch.Tensor]],
        cache_seqlens: Optional[torch.IntTensor],
        max_seqlen: Optional[int],
    ):
        if k_caches is None:
            assert v_caches is None
            assert cache_seqlens is None
            k_caches = [None] * self.num_layers
            v_caches = [None] * self.num_layers

        h = hidden_states
        for idx, l in enumerate(self.layers):
            if isinstance(l, VocabPositionEmbedding):
                h = l._forward(input_ids, position_ids)
            elif isinstance(l, FlashMQATBlock):
                h, _, _ = l._forward(
                    h,
                    cu_seqlens=cu_seqlens,
                    k_cache=k_caches[idx],
                    v_cache=v_caches[idx],
                    cache_seqlens=cache_seqlens,
                    max_seqlen=max_seqlen,
                    attention_mask=None,
                )
            elif isinstance(l, (OutputHead, SequenceParallelCriticHead, SequenceParallelActorHead)):
                h = l._forward(h)
            else:
                raise NotImplementedError(f"Unsupported layer type {type(l)}")

        return h

    def state_dict(self):
        """Map layer indices to global layer indices."""
        state_dict = super().state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.lstrip("layers.")
            local_idx = int(k.split(".")[0])
            name = k.split(".", 1)[1]
            new_state_dict[f"{local_idx + self.layer_idx_start}.{name}"] = v
        return new_state_dict

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.split(".", 1)[1]
            global_idx = int(k.split(".")[0])
            new_state_dict[f"layers.{global_idx - self.layer_idx_start}.{name}"] = v
        return super().load_state_dict(
            new_state_dict,
            strict=strict,
            assign=assign,
        )

    # Template function used for converting HF model to FlashMQAT, similar to C++ template but is ugly in python.
    def _config_from_hf_template(
        config_converter: Callable[[transformers.PretrainedConfig], FlashMQATConfig],
        from_model: Optional[transformers.PreTrainedModel] = None,
        model_path: Optional[str] = None,
        is_critic: bool = False,
        sequence_parallel: bool = False,
        gradient_accumulation_fusion: bool = False,
    ) -> FlashMQATConfig:
        if model_path is not None:
            hf_config = transformers.AutoConfig.from_pretrained(os.path.join(model_path, "config.json"))
        else:
            assert from_model is not None
            hf_config = from_model.config
        config = config_converter(hf_config)
        config.is_critic = is_critic
        config.sequence_parallel = sequence_parallel
        config.gradient_accumulation_fusion = gradient_accumulation_fusion
        return config

    # Template function used for converting HF model to FlashMQAT, similar to C++ template but is ugly in python.
    def _from_hf_template(
        cls,
        config_converter: Callable[[transformers.PretrainedConfig], FlashMQATConfig],
        state_dict_converter: Callable[[Dict, FlashMQATConfig], Dict],
        from_model: Optional[transformers.PreTrainedModel] = None,
        model_path: Optional[str] = None,
        init_from_scratch: bool = False,
        is_critic: bool = False,
        sequence_parallel: bool = False,
        gradient_accumulation_fusion: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        config = FlashMQATModel._config_from_hf_template(
            config_converter=config_converter,
            model_path=model_path,
            from_model=from_model,
            is_critic=is_critic,
            sequence_parallel=sequence_parallel,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
        )
        model = cls(config=config, dtype=dtype, device=device)
        assert not model._instantiated
        if not init_from_scratch:
            if model_path is not None:
                model._instantiation_hooks.append(
                    lambda: model.load_from_hf(model_path, init_critic_from_actor=is_critic))
            else:
                model._instantiation_hooks.append(
                    lambda: model.load_state_dict(state_dict_converter(from_model.state_dict(), config)))
        return model

    # Template function used for FlashMQAT to HF models, similar to C++ template but is ugly in python.
    def _to_hf_template(config, state_dict, output_dir, hf_base_model_path, state_dict_converter_to_hf):
        save_to_disk(
            state_dict_converter_to_hf(FlashMQATModel.from_pipe_state_dict(config, state_dict), config),
            output_dir,
            with_hf_format=True,
            hf_base_model_path=hf_base_model_path,
        )

    @staticmethod
    def register_hf_model(
        model_name: str,
        config_converter: Callable[[transformers.PretrainedConfig], FlashMQATConfig],
        state_dict_converter: Callable[[Dict, FlashMQATConfig], Dict],
        embedding_param_names: Callable[[FlashMQATConfig], List[str]],
        tblock_param_names: Callable[[FlashMQATConfig, int], List[str]],
        head_param_names: Callable[[FlashMQATConfig], List[str]],
        state_dict_converter_to_hf: Optional[Callable[[Dict, FlashMQATConfig], Dict]] = None,
    ):
        """Register a HuggingFace model with `model_name`, such that models can be converted back-and-forth.
        # TODO: the documentation is OOD.

        Example usage:

        ```
        # 1. Register a model called `starcoder` with helper functions.
        # Check `impl/model/nn/flash_mqat/flash_from_hf_impl.py` for details.
        FlashMQATModel.register_hf_model("starcoder",
                                         convert_config_starcoder,
                                         state_dict_from_starcoder,
                                         state_dict_to_starcoder)

        # 2. Obtain the config
        config: FlashMQATConfig = FlashMQATModel.config_from_starcoder(model_path)

        # 3. Obtain config and state_dict (also support init_from_scratch=True)
        config, state_dict = FlashMQATModel.config_and_param_from_starcoder(model_path)

        # 4. Directly construct from HuggingFace model (also support init_from_scratch=True)
        model = FlashMQATModel.from_starcoder(model_path="/lustre/public/pretrained_model_weights/starcoder-16bit")

        # 5. Dump to HuggingFace model
        FlashMQATModel.dump_to_starcoder(model.config,
                                         model.state_dict(),
                                         save_path,
                                         "/lustre/public/pretrained_model_weights/starcoder-16bit")

        # 6. Use the dumped weights
        from impl.model.nn.utils.save_load import load_from_disk
        config = transformers.AutoConfig.from_pretrained(model_path)
        hf_model = transformers.AutoModelForCausalLM.from_config(config)
        hf_model.load_state_dict(load_from_disk(save_path))
        ```

        """
        setattr(
            FlashMQATModel,
            f"from_{model_name}",
            classmethod(
                functools.partial(
                    FlashMQATModel._from_hf_template,
                    config_converter=config_converter,
                    state_dict_converter=state_dict_converter,
                )),
        )
        setattr(
            FlashMQATModel,
            f"config_from_{model_name}",
            staticmethod(
                functools.partial(
                    FlashMQATModel._config_from_hf_template,
                    config_converter=config_converter,
                )),
        )
        if state_dict_converter_to_hf:
            setattr(
                FlashMQATModel,
                f"dump_to_{model_name}",
                staticmethod(
                    functools.partial(
                        FlashMQATModel._to_hf_template,
                        state_dict_converter_to_hf=state_dict_converter_to_hf,
                    )),
            )
        FlashMQATModel._parallelism_helpers[model_name] = FlashMQATParallelismHelper(
            embedding_param_names,
            tblock_param_names,
            head_param_names,
        )
        FlashMQATModel._convert_helpers[model_name] = FlashMQATConvertHelper(config_converter,
                                                                             state_dict_converter,
                                                                             state_dict_converter_to_hf)

    def load_from_hf(self, load_dir: str, init_critic_from_actor: bool = False):
        from impl.model.nn.flash_mqat.flash_from_hf_impl import HF_ARCH_TO_MODEL_TYPE

        tik = time.perf_counter()
        with open(os.path.join(load_dir, "config.json"), "r") as f:
            hf_config = json.load(f)
        model_type = HF_ARCH_TO_MODEL_TYPE[hf_config["architectures"][0]]
        ph = self._parallelism_helpers[model_type]
        layer_indices = range(self.layer_idx_start, self.layer_idx_end)

        required_hf_sd_names = []
        for lidx in layer_indices:
            if lidx == 0:
                required_hf_sd_names += ph.embedding_param_names(self.config)
            elif lidx == self.config.n_layers + 1:
                required_hf_sd_names += ph.head_param_names(self.config)
            else:
                required_hf_sd_names += ph.tblock_param_names(self.config, lidx - 1)

        if os.path.exists(os.path.join(load_dir, "pytorch_model.bin.index.json")):
            with open(os.path.join(load_dir, "pytorch_model.bin.index.json"), "r") as f:
                hf_sd_mapping = json.load(f)["weight_map"]
            files_to_load = set(hf_sd_mapping[name] for name in required_hf_sd_names)
        else:
            files_to_load = ["pytorch_model.bin"]
        setup_time = time.perf_counter() - tik

        load_times, partition_times = [], []
        state_dict = {}
        for fn in files_to_load:
            load_tik = time.perf_counter()
            # set map_location to be CPU is a little bit faster
            sd = torch.load(os.path.join(load_dir, fn), map_location="cpu")
            partition_tik = time.perf_counter()
            sd = {k: v for k, v in sd.items() if k in required_hf_sd_names}
            sd = self._convert_helpers[model_type].state_dict_converter(sd, self.config)
            state_dict.update(
                mp_partition_flash_mqat_state_dict(
                    sd,
                    self.config,
                    base.constants.model_parallel_world_size(),
                    base.constants.model_parallel_rank(),
                ))
            load_times.append(partition_tik - load_tik)
            partition_times.append(time.perf_counter() - partition_tik)

        copy_tik = time.perf_counter()
        if init_critic_from_actor and f"{self.config.n_layers + 1}.weight" in state_dict:
            state_dict.pop(f"{self.config.n_layers + 1}.weight")
            self.load_state_dict(state_dict, strict=False)
        else:
            self.load_state_dict(state_dict, strict=True)
        copy_time = time.perf_counter() - copy_tik
        load_times = "[" + ", ".join(f"{t:.2f}" for t in load_times) + "]"
        partition_times = "[" + ", ".join(f"{t:.2f}" for t in partition_times) + "]"
        if os.getenv("FLASH_MQAT_LOG_LOAD_TIME", None) == "1":
            logger.info(
                f"Loading from HuggingFace Model setup time cost={setup_time:.2f}s, load time cost={load_times}, "
                f"partition time cost={partition_times}, copy time cost={copy_time:.2f}s")

    def load_from_saved_flash_model(self, load_dir: str, init_critic_from_actor: bool = False):
        with open(os.path.join(load_dir, "flash_mqat_config.json"), "r") as f:
            ckpt_config = FlashMQATConfig(**json.load(f))
        for k, v in dataclasses.asdict(ckpt_config).items():
            if k not in [
                    "is_critic",
                    "sequence_parallel",
                    "gradient_accumulation_fusion",
                    "ckpt_attn",
                    "ckpt_mlp",
            ] and v != getattr(self.config, k):
                raise ValueError(
                    f"Can't load a checkpoint with different config (key `{k}`, "
                    f"value in checkpoint is `{v}`, current value is `{getattr(self.config, k)}`).")

        pp_rank = base.constants.pipe_parallel_rank()
        mp_rank = base.constants.model_parallel_rank()
        mp_size = base.constants.model_parallel_world_size()

        ckpt_spec = get_ckpt_spec(load_dir)
        if ckpt_spec.mp_size % mp_size != 0 and mp_size % ckpt_spec.mp_size != 0:
            raise ValueError(f"Trying to load checkpoint {load_dir} with mp_size={ckpt_spec.mp_size}, "
                             f"which is neither the multiple nor a factor of current mp_size={mp_size}.")

        if (self.config.n_kv_heads % mp_size == 0) != (self.config.n_kv_heads % ckpt_spec.mp_size == 0):
            raise RuntimeError(
                "The partition methods of KV heads of the checkpoint and the current model are not compatible. "
                "To load the checkpoint, KV heads must be able to evenly partitioned (i.e.,  #kv_heads % mp_size == 0) "
                "or unable to be partitioned (i.e., #kv_heads % mp_size != 0) for both models. "
                f"Number of kv heads={self.config.n_kv_heads}, mp_size={mp_size}, ckpt mp_size={ckpt_spec.mp_size}.",
            )

        ckpt_layer_partition = partition_pipeline_layers(
            self.config,
            ckpt_spec.pp_size,
            flash_model_embed_param_count,
            flash_model_tblock_param_count,
            flash_model_head_param_count,
        )

        ckpt_layer_mapping_list = {k: list(range(v[0], v[1])) for k, v in ckpt_layer_partition.items()}
        self_layer_mapping_list = {k: list(range(v[0], v[1])) for k, v in self.layer_mapping.items()}

        repartition_strategy = pipeline_repartition_strategy(self_layer_mapping_list, ckpt_layer_mapping_list)
        repartition_strategy = {k: v for k, v in repartition_strategy.items() if k[0] == pp_rank}

        if mp_size <= ckpt_spec.mp_size:
            factor = ckpt_spec.mp_size // mp_size
            interested_mp_ranks = list(range(factor * mp_rank, factor * (mp_rank + 1)))
            state_dict = {}
            for (_, target_pp_rank), global_layer_indices in repartition_strategy.items():
                mp_sds = []
                for i in interested_mp_ranks:
                    sd = load_from_disk(load_dir,
                                        fn_pattern=r".*" + f"-pp-{target_pp_rank:02d}-mp-{i:02d}-" +
                                        r"s-(\d{2}).*")
                    sd = {k: v for k, v in sd.items() if int(k.split(".")[0]) in global_layer_indices}
                    mp_sds.append(sd)
                state_dict.update(mp_merge_flash_mqat_state_dict(mp_sds, self.config))
        else:
            factor = mp_size // ckpt_spec.mp_size
            target_mp_rank = mp_rank // factor
            state_dict = {}
            for (_, target_pp_rank), global_layer_indices in repartition_strategy.items():
                sd = load_from_disk(
                    load_dir,
                    fn_pattern=r".*" + f"-pp-{target_pp_rank:02d}-mp-{target_mp_rank:02d}-" + r"s-(\d{2}).*",
                )
                sd = {k: v for k, v in sd.items() if int(k.split(".")[0]) in global_layer_indices}
                sd = mp_partition_flash_mqat_state_dict(sd,
                                                        self.config,
                                                        mp_size=factor,
                                                        mp_rank=mp_rank % factor)
                state_dict.update(sd)

        if init_critic_from_actor and f"{self.config.n_layers + 1}.weight" in state_dict:
            state_dict.pop(f"{self.config.n_layers + 1}.weight")
            self.load_state_dict(state_dict, strict=False)
        else:
            self.load_state_dict(state_dict, strict=True)

    def save(
        self,
        save_dir: str,
        epoch: Optional[int] = None,
        epoch_step: Optional[int] = None,
        global_step: Optional[int] = None,
    ):
        pp_rank = base.constants.pipe_parallel_rank()
        dp_rank = base.constants.data_parallel_rank()
        mp_rank = base.constants.model_parallel_rank()
        if dp_rank > 0:  # only save on dp_rank = 0
            return

        subfolder = ""
        if epoch is not None:
            subfolder += f"epoch{epoch}"
        if epoch_step is not None:
            subfolder += f"epochstep{epoch_step}"
        if global_step is not None:
            subfolder += f"globalstep{global_step}"
        save_dir = os.path.join(save_dir, subfolder)
        os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, "flash_mqat_config.json"), "w") as f:
            json.dump(dataclasses.asdict(self.config), f)

        save_to_disk(
            self.state_dict(),
            save_dir,
            output_fn=f"pytorch_model-pp-{pp_rank:02d}-mp-{mp_rank:02d}-s-" + "{shard:02d}.bin",
            save_type="pt",
            n_shards=int(os.getenv("FLASH_MQAT_N_SHARDS", "3")),
            with_hf_format=True,
        )

    def build_reparallelization_plan(
        self,
        from_model_name: ModelName,
        to_model_name: ModelName,
        from_topo: base.topology.PipeModelDataParallelTopology,
        to_topo: base.topology.PipeModelDataParallelTopology,
        to_model_config: FlashMQATConfig,
        pg_info: gpu_utils.NCCLProcessGroupInfo,
        from_model_config: None | FlashMQATConfig = None,
    ):
        if from_model_config is None:
            from_model_config = self.config
        to_layer_mapping = partition_pipeline_layers(
            to_model_config,
            to_topo.get_dim("pipe"),
            flash_model_embed_param_count,
            flash_model_tblock_param_count,
            flash_model_head_param_count,
        )
        to_layers_handle_dict = {}
        to_layer_indices = []
        if base.constants.has_model_name(to_model_name):
            with base.constants.model_scope(to_model_name):
                to_pp_rank = base.constants.pipe_parallel_rank()
                to_layer_indices = list(
                    range(to_layer_mapping[to_pp_rank][0], to_layer_mapping[to_pp_rank][1]))
                for _to_layer_idx in to_layer_indices:
                    l = self._build_layer(_to_layer_idx, to_model_config)
                    for v in l.parameters():
                        v.data = torch.tensor((), dtype=self.dtype, device=self.device)
                    to_layers_handle_dict[_to_layer_idx] = l
        to_param_spec, to_param_size = _build_param_spec(to_layer_indices, to_model_config,
                                                         to_topo.get_dim("model"))
        if len(to_layer_indices) > 0:
            to_layer_idx_start = min(to_layer_indices)
            to_layer_idx_end = max(to_layer_indices) + 1
        else:
            to_layer_idx_start = to_layer_idx_end = -1
        to_layers_handle = nn.ModuleList([to_layers_handle_dict[i] for i in to_layer_indices])

        comm_plan = _derive_reparallelize_comm_plan(
            from_model_name=from_model_name,
            to_model_name=to_model_name,
            from_topo=from_topo,
            to_topo=to_topo,
            from_model_config=from_model_config,
            to_model_config=to_model_config,
            pg_info=pg_info,
            dtype=self.dtype,
        )
        rtgt = ReparallelizeTraget(
            comm_plan=comm_plan,
            to_param_spec=to_param_spec,
            to_param_size=to_param_size,
            to_layers_handle=to_layers_handle,
            to_layer_start_idx=to_layer_idx_start,
            to_layer_end_idx=to_layer_idx_end,
        )
        self._reparallelize_targets[(from_model_name, to_model_name)] = rtgt

    def build_reparallelized_layers_async(
        self,
        from_model_name: ModelName,
        to_model_name: ModelName,
        from_topo: base.topology.PipeModelDataParallelTopology,
        to_topo: base.topology.PipeModelDataParallelTopology,
        to_model_config: FlashMQATConfig,
        pg_info: gpu_utils.NCCLProcessGroupInfo,
    ) -> Tuple[nn.ModuleList, torch.Tensor]:
        # FIXME: remote synchronization deletes the local model, but sometimes it is uncessary.

        if (from_model_name, to_model_name) not in self._reparallelize_targets:
            self.build_reparallelization_plan(
                from_model_name,
                to_model_name,
                from_topo,
                to_topo,
                to_model_config,
                pg_info,
            )
        rtgt = self._reparallelize_targets[(from_model_name, to_model_name)]

        # The following tensor holds the contiguous memory of incoming parameters
        to_contiguous_param = torch.empty(rtgt.to_param_size, dtype=self.dtype, device="cuda")
        map_param_to_contigous_memory(
            rtgt.to_layers_handle,
            rtgt.to_param_spec,
            to_contiguous_param,
            rtgt.to_layer_start_idx,
        )

        comm_volume = 0

        src_mp_size = from_topo.get_dim("model")
        dst_mp_size = to_topo.get_dim("model")
        n_receiver_portions = max(src_mp_size // dst_mp_size, 1)
        for step in rtgt.comm_plan:
            if isinstance(step, ReparallelizeReceiverStep) and step.rank == torch.distributed.get_rank():

                if step.rank == step.src:
                    buf = slice_intervals(
                        self.contiguous_param,
                        step.sender_param_intervals,
                        step.sender_param_intervals_cpu,
                        max_interval_size=step.sender_max_interval_size,
                        output_size=step.param_size,
                    )
                else:
                    buf = torch.zeros(step.param_size, dtype=step.param_dtype, device="cuda")
                    comm_volume += buf.numel()
                    torch.distributed.broadcast(buf, src=step.src, group=step.group)

                set_intervals(
                    src=buf,
                    dst=to_contiguous_param,
                    intervals=step.receiver_param_intervals,
                    intervals_cpu=step.receiver_param_intervals_cpu,
                    max_interval_size=step.receiver_max_interval_size,
                )

            if isinstance(step, ReparallelizeSenderStep) and step.rank == torch.distributed.get_rank():
                if step.group is not None:
                    buf = slice_intervals(
                        self.contiguous_param,
                        step.param_intervals,
                        step.param_intervals_cpu,
                        step.max_interval_size,
                        step.param_size,
                    )
                    torch.distributed.broadcast(buf, src=step.rank, group=step.group)
                if step.remove:
                    for param_key in step.param_keys:
                        layer_idx, k = param_key.split(".", 1)
                        layer_idx = int(layer_idx)
                        dummy_tensor = torch.tensor((), dtype=self.dtype, device=self.device)
                        recursive_getattr(self.layers[layer_idx - self.layer_idx_start],
                                          k).data = (dummy_tensor)

        # assert len(state_dict) == 0

        # release the local GPU memory
        self.contiguous_param = None
        self.layers = None
        return rtgt.to_layers_handle, to_contiguous_param

    def patch_reparallelization(self, x):
        self.layers, self.contiguous_param = x


# a helper function to make flash_mqat look like huggingface model
def generate_helper(
        self: FlashMQATModel,
        tokenizer: transformers.PreTrainedTokenizerFast,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        packed_input_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        k_caches: Optional[List[torch.Tensor]] = None,
        v_caches: Optional[List[torch.Tensor]] = None,
        cache_seqlens: Optional[torch.Tensor] = None,
        gconfig: GenerationConfig = dataclasses.field(default_factory=GenerationConfig),
) -> DuckGenerationOutput:
    current_forward = self.forward
    self.forward = functools.partial(FlashMQATModel.forward, self)
    seq, scores, mask, _, _ = generate(
        model=self,
        tokenizer=tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        packed_input_ids=packed_input_ids,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        k_caches=k_caches,
        v_caches=v_caches,
        cache_seqlens=cache_seqlens,
        gconfig=gconfig,
    )
    self.forward = current_forward
    return DuckGenerationOutput(seq, scores, mask)


# a helper function to make flash_mqat look like huggingface model
def forward_helper(
    self: FlashMQATModel,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    packed_input_ids: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
) -> DuckModelOutput:
    assert (packed_input_ids is None) == (cu_seqlens is None) == (max_seqlen is None)
    build_packed = False
    if attention_mask is None and input_ids is not None:
        attention_mask = torch.ones_like(input_ids)
    if packed_input_ids is None and attention_mask is not None:
        build_packed = True
        packed_input_ids, indices, cu_seqlens, max_seqlen = unpad_input(input_ids, attention_mask)
        batch_size, seqlen = input_ids.shape[:2]
    if packed_input_ids is not None:
        x = PipeTransferData(cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        ys = [PipeCacheData(input_ids=packed_input_ids)
              ] + [PipeCacheData() for _ in range(self.config.n_layers + 1)]
    else:
        x = PipeTransferData()
        ys = [PipeCacheData(input_ids=input_ids)] + [PipeCacheData() for _ in range(self.config.n_layers + 1)]
    scores = FlashMQATModel.forward(self, x, ys)[0].pp_output
    if build_packed:
        scores = pad_input(scores, indices, batch_size, seqlen)
    return scores


def add_helper_functions(m: FlashMQATModel):
    m.forward = functools.partial(forward_helper, m)
    m.generate = functools.partial(generate_helper, m)
    return m


def make_flash_model(
    name: ModelName,
    device: torch.device,
    model_path: str,
    from_type: str,
    dtype: Optional[str] = None,
    hf_model_type: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    sequence_parallel: bool = False,
    gradient_accumulation_fusion: bool = False,
) -> api.model.Model:
    if dtype == "fp16" or dtype == None:
        dtype = torch.float16
    elif dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp32":
        dtype == torch.float32
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")

    tokenizer = None
    if from_type == "hf_as_critic":
        # Convert a HuggingFace model into FlashMQAT.
        m = getattr(FlashMQATModel, f"from_{hf_model_type}")(
            model_path=model_path,
            dtype=dtype,
            device=device,
            is_critic=True,
            init_from_scratch=False,
            sequence_parallel=sequence_parallel,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
        )
        tokenizer = api.huggingface.load_hf_tokenizer(model_path)
    elif from_type == "hf_as_actor":
        # Convert a HuggingFace model into FlashMQAT.
        m = getattr(FlashMQATModel, f"from_{hf_model_type}")(
            model_path=model_path,
            dtype=dtype,
            device=device,
            is_critic=False,
            init_from_scratch=False,
            sequence_parallel=sequence_parallel,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
        )
        tokenizer = api.huggingface.load_hf_tokenizer(model_path)
    elif from_type == "actor_as_critic":
        # initialize a critic from actor
        with open(os.path.join(model_path, "flash_mqat_config.json"), "r") as f:
            config = FlashMQATConfig(**json.load(f))
        config.is_critic = True
        config.sequence_parallel = sequence_parallel
        config.gradient_accumulation_fusion = gradient_accumulation_fusion
        m = FlashMQATModel(config=config, dtype=dtype, device=device)
        m.load_from_saved_flash_model(model_path, init_critic_from_actor=True)
    elif from_type == "random_actor":
        # randomly initialize a actor
        m = getattr(FlashMQATModel, f"from_{hf_model_type}")(
            model_path=tokenizer_path,
            dtype=dtype,
            device=device,
            is_critic=False,
            init_from_scratch=True,
            sequence_parallel=sequence_parallel,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
        )
    elif from_type == "random_critic":
        # randomly initialize a critic
        m = getattr(FlashMQATModel, f"from_{hf_model_type}")(
            model_path=tokenizer_path,
            dtype=dtype,
            device=device,
            is_critic=True,
            init_from_scratch=True,
            sequence_parallel=sequence_parallel,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
        )
    else:
        # actor loads from saved actor or critic loads from saved critic
        assert from_type == "self"
        with open(os.path.join(model_path, "flash_mqat_config.json"), "r") as f:
            config = FlashMQATConfig(**json.load(f))
        config.sequence_parallel = sequence_parallel
        config.gradient_accumulation_fusion = gradient_accumulation_fusion
        m = FlashMQATModel(config=config, dtype=dtype, device=device)
        m.load_from_saved_flash_model(model_path, init_critic_from_actor=False)

    if tokenizer is None:
        tokenizer = api.huggingface.load_hf_tokenizer(tokenizer_path)

    if base.constants.pipe_parallel_world_size() == 1:
        m = add_helper_functions(m)
    return api.model.Model(name, m, tokenizer, device, dtype=dtype)


api.model.register_model("flash_mqat", make_flash_model)
