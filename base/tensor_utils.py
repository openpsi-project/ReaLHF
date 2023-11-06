from collections import defaultdict
from typing import List, Tuple
import logging

import torch

from impl.model.utils.data import PipeCacheData, PipeTransferData

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
    """ In DeepSpeedPipelineEngine, type of tensors include: 
            1. activations and grads (stored and passed between stages, shapes: (total_seq_lens, hidden_dim))
            2. fixed parameters (passed between stages along with activations)
            3. kv caches (only stored, shapes: (bs, kvcache_seqlen, n_kvheads, head_dim))
    TensorBuffer class allocate, stores and manage above tensors. In a pipeline stage, tensors of type 1. and 3. are 
    stored in a big block which should be only initialized once. Pipeline engine accesses tensors via tensor type, 
    stage_id and micro_batch_id. Once a tensor is accessed, TensorBuffer translate the info into slice index and 
    return the tensor via simple slicing to reduce memory operations.
    """

    def __init__(self):
        self.activation_cache = None  # initialize when pipeline engine methods are called
        # to obtain more information about data
        self.activation_index_mapping = {}
        self.activation_right = 0

        self.grad_cache = None
        self.grad_index_mapping = {}
        self.grad_right = 0

        self.k_cache = None
        self.v_cache = None
        self.kv_cache_index_mapping = {}
        self.kv_cache_right = 0

        self.others = defaultdict(dict)  # for small tensors,
        # store them as a dict of format
        # {name: {mbid: tensor_tuple}}

    def init_activation_store(self, total_batch_size: int, max_seq_len: int, hidden_dim: int,
                              init_grad_store: bool, device: torch.device):
        if self.activation_cache is not None:
            raise RuntimeError("activation store has already been initialized")
        self.activation_cache = torch.zeros((total_batch_size * max_seq_len, hidden_dim),
                                            dtype=torch.float16,
                                            device=device,
                                            requires_grad=True)
        if init_grad_store:
            self.grad_cache = torch.zeros((total_batch_size * max_seq_len, hidden_dim),
                                          dtype=torch.float16,
                                          device=device)

    def init_kv_cache_store(self, total_batch_size: int, max_seq_len: int, max_new_tokens: int, n_kv: int,
                            head_dim: int, num_layers: int, device: torch.device):
        if self.k_cache is not None:
            raise RuntimeError("kv cache store has already been initialized")
        max_cache_size = max(max_seq_len + max_new_tokens, 128)
        self.k_cache = [
            torch.zeros((total_batch_size, max_cache_size, n_kv, head_dim),
                        dtype=torch.float16,
                        device=device) for _ in range(num_layers)
        ]
        self.v_cache = [
            torch.zeros((total_batch_size, max_cache_size, n_kv, head_dim),
                        dtype=torch.float16,
                        device=device) for _ in range(num_layers)
        ]

    def store_activation(self, mbid: int, x: torch.Tensor):
        if x == None:
            self.activation_index_mapping[mbid] = (self.activation_right, 0)
        length = x.shape[0]
        self.activation_cache[self.activation_right:self.activation_right + length] = x
        self.activation_index_mapping[mbid] = (self.activation_right, length)
        self.activation_right += length
        return self.activation_cache[self.activation_right - length:self.activation_right]

    def store_grad(self, mbid: int, x: torch.Tensor):
        if x == None:
            raise ValueError("grad tensor x should never be None")
        length = x.shape[0]
        self.grad_cache[self.grad_right:self.grad_right + length] = x
        self.grad_index_mapping[mbid] = (self.grad_right, length)
        self.grad_right += length
        return self.grad_cache[self.grad_right - length:self.grad_right]

    def store_kvcache(self, mbid: int, k_cache_list: List[torch.Tensor], v_cache_list: List[torch.Tensor]):
        length = max([k_cache.shape[0] if k_cache is not None else 0 \
                      for k_cache in k_cache_list])
        if length == 0:
            self.kv_cache_index_mapping[mbid] = (self.kv_cache_right, length)
            return None, None
        assert len(k_cache_list) == len(v_cache_list) == len(self.k_cache)
        for layer_id, (k_cache, v_cache) in enumerate(zip(k_cache_list, v_cache_list)):
            if k_cache is None:
                continue
            self.k_cache[layer_id][self.kv_cache_right:self.kv_cache_right + length] = k_cache
            self.v_cache[layer_id][self.kv_cache_right:self.kv_cache_right + length] = v_cache
        self.kv_cache_index_mapping[mbid] = (self.kv_cache_right, length)
        self.kv_cache_right += length
        return self.k_cache[self.kv_cache_right-length:self.kv_cache_right], \
               self.v_cache[self.kv_cache_right-length:self.kv_cache_right]

    def store_others(self, name: str, mbid: int, tensor_tuple: Tuple[torch.Tensor]):
        self.others[name][mbid] = tensor_tuple
        return tensor_tuple

    def load_activation(self, mbid: int):
        start, length = self.activation_index_mapping[mbid]
        if length == 0:
            return None
        return self.activation_cache[start:start + length]

    def load_grad(self, mbid: int):
        start, length = self.grad_index_mapping[mbid]
        assert length > 0
        return self.grad_cache[start:start + length]

    def load_kvcache(self, mbid: int):
        start, length = self.kv_cache_index_mapping[mbid]
        if length == 0:
            n_layers = len(self.k_cache)
            return [None for _ in range(n_layers)], \
                   [None for _ in range(n_layers)]
        k_caches, v_caches = [], []
        for k_cache, v_cache in zip(self.k_cache, self.v_cache):
            k_caches.append(k_cache[start:start + length])
            v_caches.append(v_cache[start:start + length])
        return k_caches, v_caches

    def load_others(self, name: str, mbid: int):
        return self.others[name][mbid]

    def store_pipe_transfer_data(self, mbid: int, x: PipeTransferData):
        activation_handle = self.store_activation(mbid, x.pp_input)
        others = (x.pp_output, x.cu_seqlens, x.max_seqlen, x.store_kvcache, x.attention_mask)
        other_handle = self.store_others("pipe_transfer_data", mbid, others)

    def store_pipe_cache_datas(self, mbid: int, ys: List[PipeCacheData]):
        assert len(ys) == len(self.k_cache)
        k_caches, v_caches, others = [], [], []
        for i, y in enumerate(ys):
            k_caches.append(y.k_cache)
            v_caches.append(v_caches)
            others.append((y.input_ids, y.cache_seqlens))
        k_cache_handle, v_cache_handle = self.store_kvcache(mbid, k_caches, v_caches)
        others_handle = self.store_others("pipe_cache_datas", mbid, others)

    def load_pipe_transfer_data(self, mbid: int):
        activation = self.load_activation(mbid)
        others = self.load_others("pipe_transfer_data", mbid)
        return PipeTransferData(activation, *others)

    def load_pipe_cache_datas(self, mbid: int):
        k_caches, v_caches = self.load_kvcache(mbid)
        others = self.load_others("pipe_cache_datas", mbid)
        ys = []
        for i, (k_cache, v_cache) in enumerate(zip(k_caches, v_caches)):
            ys.append(PipeCacheData(k_cache, v_cache, *others[i]))
        return ys

    def clear_all_cache(self):
        self.activation_index_mapping = {}
        self.activation_right = 0
        self.grad_index_mapping = {}
        self.grad_right = 0
        self.kv_cache_index_mapping = {}
        self.kv_cache_right = 0
