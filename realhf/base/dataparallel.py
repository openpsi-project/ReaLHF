from typing import Dict, List, Optional, Tuple, Union
import abc
import collections

import numpy as np
import torch

import realhf.base.datapack as datapack
import realhf.base.logging as logging
import realhf.base.namedarray as namedarray

logger = logging.getLogger("dataparallel")

InterfaceReturnDataType = Union[namedarray.NamedArray, Dict]


class ParallelDataBroker:

    @staticmethod
    def gather_from(
        src: List[InterfaceReturnDataType],
    ) -> InterfaceReturnDataType:
        """Gather outputs of a data-parallel model RPC."""
        if isinstance(src[0], Dict):
            cnt, stats = {}, {}
            for reply in src:
                for k, v in reply.items():
                    cnt[k] = cnt.get(k, 0) + 1
                    stats[k] = stats.get(k, 0) + v
            return {
                k: v / cnt
                for k, v, cnt in zip(stats.keys(), stats.values(), cnt.values())
            }
        elif isinstance(src[0], namedarray.NamedArray):
            if "input_lens" in src[0]:
                input_lens = torch.cat([x["input_lens"] for x in src], dim=0)
            elif "cu_seqlens" in src[0]:
                input_lens = torch.cat(
                    [x["cu_seqlens"][1:] - x["cu_seqlens"][:-1] for x in src],
                    dim=0,
                )
            elif "prompt_lens" in src[0]:
                input_lens = torch.cat([x["prompt_lens"] for x in src], dim=0)
            elif "prompt_cu_seqlens" in src[0]:
                input_lens = torch.cat(
                    [
                        x["prompt_cu_seqlens"][1:] - x["prompt_cu_seqlens"][:-1]
                        for x in src
                    ],
                    dim=0,
                )
            res = namedarray.recursive_aggregate(
                src, lambda x: torch.cat(x, dim=0)
            )
            if "cu_seqlens" in src[0] and len(src[0]["cu_seqlens"].shape) == 1:
                res["cu_seqlens"] = torch.cat(
                    [input_lens.new_zeros(1), torch.cumsum(input_lens, dim=0)]
                )
            elif (
                "prompt_cu_seqlens" in src[0]
                and len(src[0]["prompt_cu_seqlens"].shape) == 1
            ):
                res["prompt_cu_seqlens"] = torch.cat(
                    [input_lens.new_zeros(1), torch.cumsum(input_lens, dim=0)]
                )
            return res
        else:
            raise NotImplementedError(
                f"Don't know how to gather data of type {[type(x) for x in src]}."
            )

    @abc.abstractstaticmethod
    def scatter_to(
        src: namedarray.NamedArray, n_dp: int, **kwargs
    ) -> List[namedarray.NamedArray]:
        """Scatter the input of a data-parallel model RPC."""
        pass


class PaddedBatchParallelDataBroker(ParallelDataBroker):

    @staticmethod
    def gather_from(
        src: List[InterfaceReturnDataType],
    ) -> InterfaceReturnDataType:
        return ParallelDataBroker.gather_from(src)

    @staticmethod
    def scatter_to(
        src: namedarray.NamedArray, n_dp: int, **kwargs
    ) -> List[namedarray.NamedArray]:
        datas = namedarray.split(src, n_dp)
        for x in datas:
            x.register_metadata(**src.metadata)
        return datas


class PackedParallelDataBroker(ParallelDataBroker):

    @staticmethod
    def gather_from(
        src: List[InterfaceReturnDataType],
    ) -> InterfaceReturnDataType:
        return ParallelDataBroker.gather_from(src)

    @staticmethod
    def scatter_to(
        src: namedarray.NamedArray,
        n_dp: int,
        return_sizes=False,
        partitions: Optional[List[Tuple[int, int]]] = None,
        min_size: int = 1,
        return_partitions=False,
    ) -> List[namedarray.NamedArray]:
        if "input_lens" not in src:
            if "cu_seqlens" in src:
                raw_input_lens = src["cu_seqlens"][1:] - src["cu_seqlens"][:-1]
            elif "prompt_lens" in src or "prompt_cu_seqlens" in src:
                if "prompt_lens" in src:
                    raw_input_lens = src["prompt_lens"]
                else:
                    raw_input_lens = (
                        src["prompt_cu_seqlens"][1:]
                        - src["prompt_cu_seqlens"][:-1]
                    )
            else:
                raise RuntimeError(
                    "input_lens/cu_seqlens/prompt_lens/prompt_cu_seqlens "
                    "must be in the return data when using packed data broker. "
                    f"Current keys: {list(src.keys())}."
                )
        else:
            raw_input_lens = src["input_lens"]
        assert src.metadata.get("seqlens", None) is not None
        if "seqlens" in src.metadata:
            seqlens_cpu = src.metadata["seqlens"]
            assert isinstance(seqlens_cpu, list)
            seqlens_cpu = np.array(seqlens_cpu, dtype=np.int64)
        else:
            logger.warning(
                "seqlens not found in metadata. Using input_lens to calculate partitions. "
                "This will create an unnecessary host-device synchronization point."
            )
            seqlens_cpu = raw_input_lens.cpu().numpy().astype(np.int64)

        if partitions is None:
            partitions = datapack.min_abs_diff_partition(
                seqlens_cpu, n_dp, min_size
            )

        input_lens: List[torch.IntTensor] = [
            raw_input_lens[start:end].int() for start, end in partitions
        ]
        cu_seqlens = [
            torch.cat([x.new_zeros(1), torch.cumsum(x, dim=0)]).int()
            for x in input_lens
        ]
        batch_sizes = [cu_seqlen.shape[0] - 1 for cu_seqlen in cu_seqlens]

        # These are used by log probabilities, which are one-step shorter than packed inputed ids.
        # We use numpy/list for indexing to avoid host-device synchronization
        batch_seqlens = [
            sum(seqlens_cpu[start:end]) for start, end in partitions
        ]
        offsets = [0] + np.cumsum(batch_seqlens).tolist()[:-1]
        short1batch_seqlens = [
            sum(seqlens_cpu[start:end]) - (end - start)
            for start, end in partitions
        ]
        short1offsets = [0] + np.cumsum(short1batch_seqlens).tolist()[:-1]

        splitted_data = [collections.defaultdict() for _ in range(n_dp)]
        for k, v in src.items():
            for i, sp in enumerate(splitted_data):
                # NOTE: This is a terrible implementation, because we must know how to split each tensor according to its semantics.
                # Different tensor has different shape and semantics,
                # e.g., packed_seq has shape [tot_seqlen] and should be splited according to cumulative lengths,
                # packed_logprobs has shape [tot_seqlen - bs] (each sequence is one-step shorter) and should be splitted
                # according to short-1 cumulative lengths, seq_no_eos_mask has shape [bs] and performs similar as input_lens, ...
                # so we must enumerate each possible key and deal with them separately, etc.
                if v is None:
                    sp[k] = None
                elif k in ["prompt_cu_seqlens", "cu_seqlens"]:
                    continue
                elif k in [
                    "prompt_lens",
                    "input_lens",
                    "seq_no_eos_mask",
                    "rewards",
                    "reward_score",
                    "group_factor",
                    "pos_input_lens",
                    "group_input_lens",
                    "seqlogp",
                ]:
                    start, end = partitions[i]
                    sp[k] = v[start:end]
                elif k in [
                    "packed_seq",
                    "packed_logits_mask",
                    "prompt_mask",
                    "packed_input_ids",
                    "values",
                    "logits_mask",
                    "packed_prompts",
                ]:
                    sp[k] = v[offsets[i] : offsets[i] + batch_seqlens[i]]
                elif k in [
                    "packed_logprobs",
                    "packed_ref_logprobs",
                    "old_logp",
                    "ref_logp",
                    "advantages",
                    "ppo_loss_mask",
                    "kl_rewards",
                    "returns",
                ]:
                    sp[k] = v[
                        short1offsets[i] : short1offsets[i]
                        + short1batch_seqlens[i]
                    ]
                elif not torch.is_tensor(src[k]):
                    # for constant, preserve value for each splitted instance
                    sp[k] = src[k]
                else:
                    raise RuntimeError(
                        f"Unknown key {k} in packed data. We don't know how to split it. "
                        f"Check base/dataparallel.py for implemented keys."
                    )
        if "cu_seqlens" in src.keys():
            for x, slens in zip(splitted_data, input_lens):
                x["cu_seqlens"] = torch.nn.functional.pad(
                    slens.cumsum(dim=0), (1, 0)
                ).int()
        if "prompt_cu_seqlens" in src.keys():
            raw_prompt_lens = (
                src["prompt_cu_seqlens"][1:] - src["prompt_cu_seqlens"][:-1]
            )
            all_prompt_lens: List[torch.IntTensor] = [
                raw_prompt_lens[start:end].int() for start, end in partitions
            ]
            for x, pslens in zip(splitted_data, all_prompt_lens):
                x["prompt_cu_seqlens"] = torch.nn.functional.pad(
                    pslens.cumsum(dim=0), (1, 0)
                ).int()
        splitted_data = [namedarray.from_dict(dict(x)) for x in splitted_data]
        for x in splitted_data:
            x.register_metadata(**src.metadata)

        res = [splitted_data]
        if return_sizes:
            res.append(batch_sizes)
        if return_partitions:
            res.append(partitions)
        if not return_sizes and not return_partitions:
            return res[0]
        else:
            return res


def get_broker(type_: str) -> ParallelDataBroker:
    if type_ == "padded_batch":
        return PaddedBatchParallelDataBroker
    elif type_ == "packed":
        return PackedParallelDataBroker
    else:
        raise RuntimeError(f"Unknown data broker type {type_}.")
