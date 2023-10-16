from typing import Dict, List, Union
import abc
import collections

import numpy as np
import torch

import base.datapack as datapack
import base.namedarray as namedarray

InterfaceReturnDataType = Union[namedarray.NamedArray, Dict]


class ParallelDataRouter:

    @staticmethod
    def gather_from(src: List[InterfaceReturnDataType]) -> InterfaceReturnDataType:
        """Gather outputs of a data-parallel model RPC."""
        if isinstance(src[0], Dict):
            return {k: np.mean([r[k] for r in src]) for k in src[0].keys()}
        elif isinstance(src[0], namedarray.NamedArray):
            return namedarray.recursive_aggregate(src, lambda x: torch.cat(x, dim=0))
        else:
            raise NotImplementedError()

    @abc.abstractstaticmethod
    def scatter_to(src: namedarray.NamedArray, n_dp: int) -> List[namedarray.NamedArray]:
        """Scatter the input of a data-parallel model RPC."""
        pass


class PaddedBatchParallelDataRouter(ParallelDataRouter):

    @staticmethod
    def gather_from(*src: List[InterfaceReturnDataType]) -> InterfaceReturnDataType:
        return ParallelDataRouter.gather_from(src)

    @staticmethod
    def scatter_to(src: namedarray.NamedArray, n_dp: int) -> List[namedarray.NamedArray]:
        datas = namedarray.split(src, n_dp)
        for x in datas:
            x.register_metadata(**src.metadata)
        return datas


class PackedParallelDataRouter(ParallelDataRouter):

    @staticmethod
    def gather_from(src: List[InterfaceReturnDataType]) -> InterfaceReturnDataType:
        if not all(['input_lens' in x for x in src]):
            raise RuntimeError("input_lens must be in the return data when using packed data router. "
                               f"Current keys: {[list(x.keys()) for x in src]}.")
        return ParallelDataRouter.gather_from(src)

    @staticmethod
    def scatter_to(src: namedarray.NamedArray, n_dp: int) -> List[namedarray.NamedArray]:
        if 'input_lens' not in src and 'cu_seqlens' not in src:
            raise RuntimeError(
                "input_lens or cu_seqlens must be in the return data when using packed data router. "
                f"Current keys: {list(src.keys())}.")
        if 'input_lens' not in src:
            src['input_lens'] = torch.diff(src['cu_seqlens'])

        print(src['input_lens'])
        partitions = datapack.min_abs_diff_partition(src['input_lens'].cpu().numpy().astype(np.int64), n_dp)

        input_lens: List[torch.IntTensor] = [src['input_lens'][start:end] for start, end in partitions]
        cu_seqlens = [torch.cat([x.new_zeros(1), torch.cumsum(x, dim=0)]) for x in input_lens]
        offsets = torch.tensor([sum(x) for x in input_lens], dtype=torch.int32).cumsum(0)
        offsets = torch.cat([offsets.new_zeros(1), offsets[:-1]])

        # These are used by log probabilities, which are one-step shorter than packed inputed ids.
        short1input_lens = [x - 1 for x in input_lens]
        short1cu_seqlens = [torch.cat([x.new_zeros(1), torch.cumsum(x, dim=0)]) for x in short1input_lens]
        short1offsets = torch.tensor([sum(x) for x in short1input_lens], dtype=torch.int32).cumsum(0)
        short1offsets = torch.cat([short1offsets.new_zeros(1), short1offsets[:-1]])

        splitted_data = [collections.defaultdict() for _ in range(n_dp)]
        for k, v in src.items():
            for i, sp in enumerate(splitted_data):
                # NOTE: This is a terrible implementation, because we must know how to split each tensor according to its semantics.
                # Different tensor has different shape and semantics,
                # e.g., packed_seq has shape [tot_seqlen] and should be splited according to cumulative lengths,
                # packed_logprobs has shape [tot_seqlen - bs] (each sequence is one-step shorter) and should be splitted
                # according to short-1 cumulative lengths, seq_no_eos_mask has shape [bs] and performs similar as input_lens, ...
                # so we must enumerate each possible key and deal with them separately.,
                if k == 'input_lens':
                    sp[k] = input_lens[i]
                elif k == 'seq_no_eos_mask':
                    start, end = partitions[i]
                    sp[k] = v[start:end]
                elif k in ['packed_seq', 'packed_logits_mask', 'prompt_mask', 'packed_input_ids']:
                    sp[k] = v[offsets[i]:offsets[i] + cu_seqlens[i][-1]]
                elif k == 'packed_logprobs':
                    sp[k] = v[short1offsets[i]:short1offsets[i] + short1cu_seqlens[i][-1]]
                elif k == "cu_seqlens":
                    sp[k] = cu_seqlens[i] if not 'packed_logprobs' in src else short1cu_seqlens[i]
                else:
                    raise RuntimeError(f"Unknown key {k} in packed data. We don't know how to split it. "
                                       f"Implemented keys include ")
        splitted_data = [namedarray.from_dict(dict(x)) for x in splitted_data]
        for x in splitted_data:
            x.register_metadata(**src.metadata)
        return splitted_data
