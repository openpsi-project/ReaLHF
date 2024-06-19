from typing import Dict, List, Optional

import deepspeed
import torch
import torch.distributed as dist
import torch.utils.data
import tqdm

from realhf.base.namedarray import from_dict, NamedArray, recursive_apply
from realhf.impl.model.nn.real_llm_api import ReaLModel
from realhf.impl.model.nn.real_llm_generate import GenerationConfig
from realhf.impl.model.utils.functional import (
    build_shift_one_indices,
    gather_packed_shifted_log_probs,
)
import realhf.api.core.model_api as model_api
import realhf.base.constants as constants


def compute_packed_sft_loss(
    logits: torch.Tensor,
    packed_input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    prompt_mask: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    # **kwargs is used to ensure the correctness of invoking this function
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

    def train_step(self, model: model_api.Model, data: NamedArray) -> Dict:
        data = recursive_apply(data, lambda x: x.to(model.device))
        packed_input_ids: torch.Tensor = data[
            "packed_input_ids"
        ]  # shape [tot_seqlen]
        prompt_mask: torch.BoolTensor = data[
            "prompt_mask"
        ]  # shape [tot_seqlen]
        module: deepspeed.DeepSpeedEngine = model.module

        module.train()

        seqlens_cpu = data.metadata["seqlens"]
        max_seqlen = max(seqlens_cpu)
        cu_seqlens = torch.nn.functional.pad(
            torch.tensor(
                seqlens_cpu, dtype=torch.int32, device=model.device
            ).cumsum(0),
            (1, 0),
        )

        loss_fn_kwargs = dict(
            prompt_mask=prompt_mask,
            input_lens=cu_seqlens[1:]
            - cu_seqlens[
                :-1
            ],  # this is used to partition other loss_fn_kwargs into microbatches
        )
        stat = module.train_batch(
            seqlens_cpu=seqlens_cpu,
            packed_input_ids=packed_input_ids,
            cu_seqlens=cu_seqlens,
            loss_fn=compute_packed_sft_loss,
            num_micro_batches=constants.pipe_parallel_world_size() * 2,
            version_steps=model.version.global_step,
            **loss_fn_kwargs,
        )

        model.inc_version()

        res = dict()
        if stat:
            res = dict(
                loss=float(stat["loss"]) / int(stat["n_tokens"]),
                ppl=float(stat["ppl"]) / int(stat["n_seqs"]),
                n_tokens=int(stat["n_tokens"]),
                n_seqs=int(stat["n_seqs"]),
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

        for step, data in enumerate(tqdm.tqdm(eval_dataloader)):
            seqlens_cpu = data.metadata["seqlens"]

            data = recursive_apply(data, lambda x: x.to(device))
            packed_input_ids: torch.Tensor = data[
                "packed_input_ids"
            ]  # shape [tot_seqlen]
            prompt_mask: torch.BoolTensor = data[
                "prompt_mask"
            ]  # shape [tot_seqlen]

            max_seqlen = max(seqlens_cpu)
            cu_seqlens = torch.nn.functional.pad(
                torch.tensor(
                    seqlens_cpu, dtype=torch.int32, device=model_.device
                ).cumsum(0),
                (1, 0),
            )

            loss_fn_kwargs = dict(
                prompt_mask=prompt_mask,
                input_lens=cu_seqlens[1:] - cu_seqlens[:-1],
            )
            stat = module.eval_batch(
                seqlens_cpu=seqlens_cpu,
                packed_input_ids=packed_input_ids,
                cu_seqlens=cu_seqlens,
                loss_fn=compute_packed_sft_loss,
                num_micro_batches=constants.pipe_parallel_world_size(),
                **loss_fn_kwargs,
            )

            if stat:
                losses += stat["loss"]
                n_tokens += stat["n_tokens"]
                n_seqs += stat["n_seqs"]
                ppl += stat["ppl"]

        res = dict()
        if stat:
            return dict(
                loss=float(losses / n_tokens),
                ppl=float(ppl / n_seqs),
                n_tokens=int(n_tokens),
                n_seqs=int(n_seqs),
            )
        return res

    @torch.no_grad()
    def inference(self, model: model_api.Model, data: NamedArray) -> Dict:
        device = model.device
        module = model.module
        module.eval()

        data = recursive_apply(data, lambda x: x.to(device))
        packed_input_ids: torch.Tensor = data["packed_input_ids"]
        cu_seqlens: torch.Tensor = data["cu_seqlens"]
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())
        seqlens_cpu = data.metadata["seqlens"]

        logits = module.forward(
            seqlens_cpu=seqlens_cpu,
            packed_input_ids=packed_input_ids,
            cu_seqlens=cu_seqlens,
        )
        x = from_dict(dict(logits=logits))
        x.register_metadata(**data.medata)
        return x

    # for testing only
    @torch.no_grad()
    def generate(
        self,
        model: model_api.Model,
        data: NamedArray,
        gconfig: GenerationConfig,
    ) -> NamedArray:
        module = model.module

        module.eval()

        data = recursive_apply(data, lambda x: x.to(model.device))

        res = module.generate(
            seqlens_cpu=data.metadata["seqlens"],
            tokenizer=model.tokenizer,
            packed_input_ids=data["packed_input_ids"],
            cu_seqlens=data["cu_seqlens"],
            gconfig=gconfig,
        )
        if res is None:
            return dict()

        gen_tokens, logprobs, logits_mask, *_ = res

        return dict(
            gen_tokens=gen_tokens,
            log_probs=logprobs,
            logits_mask=logits_mask,
        )


model_api.register_interface("sft", SFTInterface)
