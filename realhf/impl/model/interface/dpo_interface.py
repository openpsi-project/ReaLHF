from typing import Dict
import dataclasses

import torch
import torch.distributed as dist
import torch.utils.data
import tqdm

from realhf.base import constants
from realhf.base.namedarray import from_dict, NamedArray, recursive_apply
from realhf.impl.model.nn.real_llm_api import ReaLModel
from realhf.impl.model.utils.functional import gather_packed_shifted_log_probs
import realhf.api.core.model_api as model_api
import realhf.base.logging as logging
import realhf.impl.model.utils.dpo_functional as dpo_functional

logger = logging.getLogger("Packed DPO Interface")


def _dpo_loss_from_model_outputs(
    logits: torch.FloatTensor,
    packed_input_ids: torch.LongTensor,
    cu_seqlens: torch.IntTensor,
    prompt_lens: torch.IntTensor,
    seqlogp: torch.FloatTensor,
    beta: float,
    **kwargs,
):
    input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    logprobs = gather_packed_shifted_log_probs(
        logits, cu_seqlens, packed_input_ids
    ).float()

    assert (prompt_lens > 0).all(), prompt_lens
    logprob_sum = []
    offset = 0
    for i in range(prompt_lens.shape[0]):
        logprob_sum.append(
            logprobs[
                offset + prompt_lens[i] - 1 : offset + input_lens[2 * i] - 1
            ].sum()
        )
        offset += input_lens[2 * i] - 1
        logprob_sum.append(
            logprobs[
                offset + prompt_lens[i] - 1 : offset + input_lens[2 * i + 1] - 1
            ].sum()
        )
        offset += input_lens[2 * i + 1] - 1
    assert offset == sum(input_lens) - input_lens.shape[0], (
        offset,
        sum(input_lens),
        input_lens.shape,
    )

    pi_seqlogp = torch.stack(logprob_sum)

    loss, pos_score, neg_score, kl = dpo_functional.dpo_loss(
        pi_logps=pi_seqlogp,
        ref_logps=seqlogp.view(-1),
        beta=beta,
    )

    # Logging.
    logging_loss = (loss * prompt_lens.shape[0]).detach()
    n_seqs = torch.tensor(
        [prompt_lens.shape[0]], dtype=torch.float32, device=loss.device
    )
    dist.all_reduce(
        n_seqs, op=dist.ReduceOp.SUM, group=constants.data_parallel_group()
    )
    dist.all_reduce(
        pos_score, op=dist.ReduceOp.SUM, group=constants.data_parallel_group()
    )
    dist.all_reduce(
        neg_score, op=dist.ReduceOp.SUM, group=constants.data_parallel_group()
    )
    dist.all_reduce(
        logging_loss,
        op=dist.ReduceOp.SUM,
        group=constants.data_parallel_group(),
    )
    dist.all_reduce(
        kl, op=dist.ReduceOp.SUM, group=constants.data_parallel_group()
    )

    return loss, dict(
        loss=logging_loss,
        pos_score=pos_score,
        neg_score=neg_score,
        n_seqs=n_seqs,
        kl=kl,
    )


@dataclasses.dataclass
class DPOInterface(model_api.ModelInterface):
    beta: float
    enable_save: bool = True

    @torch.no_grad()
    def inference(self, model: model_api.Model, data: NamedArray) -> NamedArray:
        module = model.module
        module.eval()
        data = recursive_apply(data, lambda x: x.to(model.device))

        n_pairs = data["pos_input_lens"].shape[0]
        pair_lens = torch.tensor(
            data.metadata["seqlens"], dtype=torch.int32, device=model.device
        )
        input_lens: torch.IntTensor = torch.stack(
            [data["pos_input_lens"], pair_lens - data["pos_input_lens"]], 1
        ).view(-1)
        prompt_lens: torch.IntTensor = data["prompt_lens"]
        cu_seqlens = torch.cat(
            [input_lens.new_zeros(1), input_lens.cumsum(0)]
        ).int()
        input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = int(max(input_lens))

        seqlens_cpu = input_lens.cpu().numpy().tolist()

        logits = module.forward(
            seqlens_cpu=seqlens_cpu,
            packed_input_ids=data["packed_input_ids"],
            cu_seqlens=cu_seqlens,
        )
        if logits is None:
            return None

        logprobs = gather_packed_shifted_log_probs(
            logits, cu_seqlens, data["packed_input_ids"]
        ).float()

        assert (prompt_lens > 0).all(), prompt_lens
        logprob_sum = []
        offset = 0
        for i in range(prompt_lens.shape[0]):
            logprob_sum.append(
                logprobs[
                    offset + prompt_lens[i] - 1 : offset + input_lens[2 * i] - 1
                ].sum()
            )
            offset += input_lens[2 * i] - 1
            logprob_sum.append(
                logprobs[
                    offset
                    + prompt_lens[i]
                    - 1 : offset
                    + input_lens[2 * i + 1]
                    - 1
                ].sum()
            )
            offset += input_lens[2 * i + 1] - 1
        assert offset == sum(input_lens) - input_lens.shape[0], (
            offset,
            sum(input_lens),
            input_lens.shape,
        )

        x = from_dict(dict(seqlogp=torch.stack(logprob_sum).view(n_pairs, -1)))
        x.register_metadata(**data.metadata)
        return x

    def train_step(self, model: model_api.Model, data: NamedArray) -> Dict:
        data = recursive_apply(data, lambda x: x.to(model.device))

        packed_input_ids: torch.Tensor = data["packed_input_ids"]
        pair_lens = torch.tensor(
            data.metadata["seqlens"], dtype=torch.int32, device=model.device
        )
        neg_input_lens = pair_lens - data["pos_input_lens"]
        input_lens: torch.Tensor = torch.stack(
            [data["pos_input_lens"], neg_input_lens], 1
        ).view(-1)
        prompt_lens: torch.IntTensor = data["prompt_lens"]
        cu_seqlens = torch.cat(
            [input_lens.new_zeros(1), input_lens.cumsum(0)], 0
        ).int()
        max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))

        module = model.module
        module.eval()

        loss_fn_kwargs = dict(
            input_lens=pair_lens,
            prompt_lens=prompt_lens,
            seqlogp=data["seqlogp"],
            beta=self.beta,
        )
        stats = module.train_batch(
            seqlens_cpu=data.metadata["seqlens"],
            packed_input_ids=packed_input_ids,
            cu_seqlens=cu_seqlens,
            input_lens_for_partition=pair_lens,
            version_steps=model.version.global_step,
            loss_fn=_dpo_loss_from_model_outputs,
            **loss_fn_kwargs,
        )

        cur_epoch = model.version.epoch
        model.inc_version()

        res = {}
        if stats:
            res = dict(
                loss=float(stats["loss"]) / int(stats["n_seqs"]),
                pos_score=float(stats["pos_score"]) / int(stats["n_seqs"]),
                neg_score=float(stats["neg_score"]) / int(stats["n_seqs"]),
                kl=float(stats["kl"]) / int(stats["n_seqs"]),
                n_seqs=int(stats["n_seqs"]),
            )
        return res

    def save(self, model: model_api.Model, save_dir: str):
        if not self.enable_save:
            return
        module = model.module
        if not isinstance(module, ReaLModel):
            module = module.module
        module.save_to_hf(
            tokenizer=model.tokenizer,
            save_dir=save_dir,
        )


model_api.register_interface("dpo", DPOInterface)
