import dataclasses
import functools
from typing import Dict

import torch
import torch.distributed as dist
import torch.utils.data

import realhf.api.core.model_api as model_api
import realhf.base.logging as logging
import realhf.impl.model.utils.dpo_functional as dpo_functional
from realhf.api.core.data_api import SequenceSample
from realhf.base import constants
from realhf.base.datapack import flat2d
from realhf.impl.model.nn.real_llm_api import ReaLModel
from realhf.impl.model.utils.functional import gather_packed_shifted_log_probs

logger = logging.getLogger("Packed DPO Interface")


def _dpo_loss_from_model_outputs(
    logits: torch.FloatTensor,
    input_: SequenceSample,
    beta: float,
):
    input_lens = torch.tensor(flat2d(input_.seqlens["packed_input_ids"]))
    cu_seqlens = torch.nn.functional.pad(input_lens.cumsum(0), (1, 0)).int()
    packed_input_ids = input_.data["packed_input_ids"]
    prompt_lens = input_.data["prompt_lens"]

    seqlogp = input_.data["seqlogp"]

    logprobs = gather_packed_shifted_log_probs(
        logits, cu_seqlens, packed_input_ids
    ).float()

    assert (prompt_lens > 0).all(), prompt_lens
    logprob_sum = []
    offset = 0
    for i in range(prompt_lens.shape[0]):
        pair_input_lens = input_.seqlens["packed_input_ids"][i]
        prompt_len = prompt_lens[i]
        assert len(pair_input_lens) % 2 == 0
        for j in range(len(pair_input_lens) // 2):
            pos_len = pair_input_lens[2 * j]
            neg_len = pair_input_lens[2 * j + 1]
            logprob_sum.append(
                logprobs[offset + prompt_len - 1 : offset + pos_len - 1].sum()
            )
            offset += pos_len - 1
            logprob_sum.append(
                logprobs[offset + prompt_len - 1 : offset + neg_len - 1].sum()
            )
            offset += neg_len - 1
    assert offset == sum(input_lens) - input_lens.shape[0], (
        offset,
        sum(input_lens),
        input_lens.shape,
        prompt_lens.shape,
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
    dist.all_reduce(n_seqs, op=dist.ReduceOp.SUM, group=constants.data_parallel_group())
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
    dist.all_reduce(kl, op=dist.ReduceOp.SUM, group=constants.data_parallel_group())

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
    def inference(
        self,
        model: model_api.Model,
        input_: SequenceSample,
        n_mbs=None,
    ) -> SequenceSample:
        module = model.module
        module.eval()

        input_lens = torch.tensor(flat2d(input_.seqlens["packed_input_ids"]))
        cu_seqlens = torch.nn.functional.pad(input_lens.cumsum(0), (1, 0)).int()

        # This post_hook will gather log probabilities in mini-batches,
        # reducing peak memory usage.
        def calc_logprobs(logits, input_):
            logprobs = gather_packed_shifted_log_probs(
                logits, cu_seqlens, input_.data["packed_input_ids"]
            )
            return logprobs

        logprobs = module.forward(
            input_=input_,
            num_micro_batches=n_mbs,
            post_hook=calc_logprobs,
        )

        if logprobs is None:
            return None

        prompt_lens = input_.data["prompt_lens"]
        logprobs = logprobs.float()

        assert (prompt_lens > 0).all(), prompt_lens
        logprob_sum = []
        offset = 0
        for i in range(prompt_lens.shape[0]):
            pair_input_lens = input_.seqlens["packed_input_ids"][i]
            prompt_len = prompt_lens[i]
            assert len(pair_input_lens) % 2 == 0
            for j in range(len(pair_input_lens) // 2):
                pos_len = pair_input_lens[2 * j]
                neg_len = pair_input_lens[2 * j + 1]
                logprob_sum.append(
                    logprobs[offset + prompt_len - 1 : offset + pos_len - 1].sum()
                )
                offset += pos_len - 1
                logprob_sum.append(
                    logprobs[offset + prompt_len - 1 : offset + neg_len - 1].sum()
                )
                offset += neg_len - 1
        assert offset == sum(input_lens) - input_lens.shape[0], (
            offset,
            sum(input_lens),
            input_lens.shape,
            prompt_lens.shape,
        )

        seqlogp = torch.stack(logprob_sum)
        res = SequenceSample(
            keys=["seqlogp"],
            trailing_shapes=dict(seqlogp=()),
            dtypes=dict(seqlogp=torch.float32),
            ids=input_.ids,
            data=dict(seqlogp=seqlogp),
            seqlens=dict(
                seqlogp=[[len(slens)] for slens in input_.seqlens["packed_input_ids"]]
            ),
        )
        return res

    def train_step(
        self, model: model_api.Model, input_: SequenceSample, n_mbs=None
    ) -> Dict:
        module = model.module

        # Determining whether to disable dropout is a bit tricky.
        # We disable it by default.
        module.eval()

        stats = module.train_batch(
            input_=input_,
            version_steps=model.version.global_step,
            loss_fn=functools.partial(_dpo_loss_from_model_outputs, beta=self.beta),
            num_micro_batches=n_mbs,
        )

        model.inc_version()

        res = {}
        global_stats = constants.log_global_stats_tracker(
            return_dict=True, clear_stats_after_logging=True
        )
        if stats:
            res = dict(
                loss=float(stats["loss"]) / int(stats["n_seqs"]),
                pos_score=float(stats["pos_score"]) / int(stats["n_seqs"]),
                neg_score=float(stats["neg_score"]) / int(stats["n_seqs"]),
                kl=float(stats["kl"]) / int(stats["n_seqs"]),
                n_seqs=int(stats["n_seqs"]),
                **global_stats,
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
