from collections import defaultdict
from typing import Any, List, Tuple
import logging

import torch

from impl.model.utils.data import PipeCacheData, PipeTransferData
import impl.model.backend.stream_pipe_engine.p2p as p2p

logger = logging.getLogger("tensor_utils")


def get_shape(tensor):
    return tensor.shape if torch.is_tensor(tensor) else None


def print_data_shapes(name, rank, mbid, x, ys):
    if rank == 0:
        logger.debug(f"{name}: rank {rank} mbid {mbid}")
        logger.debug(f"shapes: x.pp_input {get_shape(x.pp_input)}, x.pp_output {get_shape(x.pp_output)},"
                     f" x.cu_seqlens {get_shape(x.cu_seqlens)}")
        for i, y in enumerate(ys):
            logger.debug(f"shapes: ys[{i}].input_ids {get_shape(y.input_ids)}, "
                         f"ys[{i}].k_cache {get_shape(y.k_cache)}, ys[{i}].v_cache {get_shape(y.v_cache)}, "
                         f"ys[{i}].cache_seqlens {get_shape(y.cache_seqlens)}")


class TensorBuffer:

    def __init__(self):
        self.tensors = defaultdict(dict)
        self.others = defaultdict(dict)

    def put_non_tensor(self, name: str, mbid: int, tensor_tuple: Any):
        self.others[name][mbid] = tensor_tuple

    def get_non_tensor(self, name: str, mbid: int, remove: bool = False):
        if remove:
            return self.others[name].pop(mbid)
        else:
            return self.others[name][mbid]

    def put(self, name: str, mbid: int, x: torch.Tensor):
        self.tensors[name][mbid] = x

    def alloc(self,
              name: str,
              mbid: int,
              shape: Tuple[int],
              dtype: torch.dtype,
              device: torch.device,
              require_grads: bool = False):
        self.tensors[name][mbid] = torch.zeros(shape, dtype=dtype, device=device, requires_grad=require_grads)

    def get(self, name: str, mbid: int, remove: bool = False):
        if remove:
            return self.tensors[name].pop(mbid)
        else:
            return self.tensors[name][mbid]

    def clear(self):
        self.tensors = defaultdict(dict)
        self.tensor_tuples = defaultdict(dict)


def send_pipe_transfer_data(x: PipeTransferData, dst_stage: int):
    p2p.send(x.pp_input, dst_stage)
    # XXX: sending all tensors in pipe transfer data introduce big communication overheads
    # always only send the activation
    # tensor_tuple = (x.pp_output, x.cu_seqlens, x.max_seqlen,
    #                 x.store_kvcache, x.attention_mask)
    # p2p.send_tensor_tuple_meta(tensor_tuple, dst_stage)
    # for t in tensor_tuple:
    #     if t is not None:
    #         p2p.send(t, dst_stage)


def recv_pipe_transfer_data(buf: torch.Tensor, src_stage: int, others):
    # "others" is a dict contain other informations required for reconstructing PipeTransferData
    p2p.recv(buf, src_stage)
    # keys = ["pp_output", "cu_seqlens", "max_seqlen", "store_kvcache", "attention_mask"]
    # tensor_tuple = p2p.recv_tensor_tuple_meta(src_stage)
    # others = dict()
    # for k, t in zip(keys, tensor_tuple):
    #     if t is not None:
    #         p2p.recv(t, src_stage)
    #     others[k] = t
    return PipeTransferData(pp_input=buf, **others)


def send_grad(grad: torch.Tensor, dst_stage: int):
    p2p.send(grad, dst_stage)


def recv_grad(buf: torch.Tensor, src_stage: int):
    p2p.recv(buf, src_stage)
    return buf
