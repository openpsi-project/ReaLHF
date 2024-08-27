from typing import Dict

import torch
import torch.distributed as dist
import torch.utils.data
import tqdm

import realhf.api.core.model_api as model_api
import realhf.base.constants as constants
from realhf.api.core.data_api import SequenceSample
from realhf.base.datapack import flat2d
from realhf.impl.model.nn.real_llm_api import ReaLModel
from realhf.impl.model.utils.functional import (
    build_shift_one_indices,
    gather_packed_shifted_log_probs,
)


def compute_packed_sft_loss(
    logits: torch.Tensor,
    input_: SequenceSample,
) -> torch.Tensor:
    packed_input_ids: torch.Tensor = input_.data["packed_input_ids"]
    input_lens = torch.tensor(flat2d(input_.seqlens["packed_input_ids"]))
    cu_seqlens = torch.nn.functional.pad(input_lens.cumsum(0), (1, 0)).int()
    prompt_mask = input_.data["prompt_mask"]

    shift_one_indices = build_shift_one_indices(logits, cu_seqlens)
    logprobs = gather_packed_shifted_log_probs(
        logits, cu_seqlens, packed_input_ids
    ).float()
    prompt_mask = prompt_mask[shift_one_indices]
    logprobs = torch.where(prompt_mask, 0, logprobs)

    loss_sum = -logprobs.sum()

    with torch.no_grad():
        seqlogp = torch.zeros(
            cu_seqlens.shape[0] - 1, device=logits.device, dtype=torch.float64
        )
        for i in range(cu_seqlens.shape[0] - 1):
            m = prompt_mask[cu_seqlens[i] - i : cu_seqlens[i + 1] - i - 1]
            logp = logprobs[cu_seqlens[i] - i : cu_seqlens[i + 1] - i - 1]
            assert cu_seqlens[i + 1] - i - 1 <= logprobs.shape[0], (
                cu_seqlens,
                logprobs.shape,
            )
            seqlogp[i] = torch.where(m, 0.0, logp).sum() / (
                m.numel() - m.count_nonzero()
            )

    logging_ppl = (-seqlogp).exp().sum()
    token_denorm = prompt_mask.numel() - prompt_mask.count_nonzero()
    seq_denorm = torch.tensor(
        [cu_seqlens.shape[0] - 1], dtype=torch.float32, device=logits.device
    )

    # Logging loss and perplexity.
    logging_loss = loss_sum.detach().clone()
    logging_token_denorm = token_denorm.detach().clone().float()
    dist.all_reduce(
        logging_ppl, op=dist.ReduceOp.SUM, group=constants.data_parallel_group()
    )
    dist.all_reduce(
        logging_loss,
        op=dist.ReduceOp.SUM,
        group=constants.data_parallel_group(),
    )
    dist.all_reduce(
        seq_denorm, op=dist.ReduceOp.SUM, group=constants.data_parallel_group()
    )
    dist.all_reduce(
        logging_token_denorm,
        op=dist.ReduceOp.SUM,
        group=constants.data_parallel_group(),
    )

    loss = loss_sum / token_denorm
    return loss, {
        "loss": logging_loss,
        "ppl": logging_ppl,
        "n_tokens": logging_token_denorm,
        "n_seqs": seq_denorm,
    }


class SFTInterface(model_api.ModelInterface):

    def train_step(self, model: model_api.Model, x: SequenceSample, n_mbs=None) -> Dict:
        module = model.module

        module.train()

        stat = module.train_batch(
            input_=x,
            loss_fn=compute_packed_sft_loss,
            num_micro_batches=n_mbs,
            version_steps=model.version.global_step,
        )

        model.inc_version()

        res = dict()
        global_stats = constants.log_global_stats_tracker(
            return_dict=True, clear_stats_after_logging=True
        )
        if stat:
            res = dict(
                loss=float(stat["loss"]) / int(stat["n_tokens"]),
                ppl=float(stat["ppl"]) / int(stat["n_seqs"]),
                n_tokens=int(stat["n_tokens"]),
                n_seqs=int(stat["n_seqs"]),
                **global_stats,
            )
        return res

    def save(self, model: model_api.Model, save_dir: str):
        module = model.module
        if not isinstance(module, ReaLModel):
            module = module.module
        module.save_to_hf(
            tokenizer=model.tokenizer,
            save_dir=save_dir,
        )

    @torch.no_grad()
    def evaluate(
        self,
        model_: model_api.Model,
        eval_dataloader: torch.utils.data.DataLoader,
    ) -> Dict:
        device = model_.device
        module = model_.module

        module.eval()
        losses = n_seqs = ppl = n_tokens = 0

        for step, x in enumerate(tqdm.tqdm(eval_dataloader)):
            x: SequenceSample

            res = module.eval_batch(
                input_=x.cuda(),
                loss_fn=compute_packed_sft_loss,
                num_micro_batches=constants.pipe_parallel_world_size(),
            )

            if res is not None:
                _, stat = res
                losses += stat["loss"]
                n_tokens += stat["n_tokens"]
                n_seqs += stat["n_seqs"]
                ppl += stat["ppl"]

        global_stats = constants.log_global_stats_tracker(
            return_dict=True, clear_stats_after_logging=True
        )
        if res is not None:
            return dict(
                loss=float(losses / n_tokens),
                ppl=float(ppl / n_seqs),
                n_tokens=int(n_tokens),
                n_seqs=int(n_seqs),
                **global_stats,
            )
        return dict()


model_api.register_interface("sft", SFTInterface)
