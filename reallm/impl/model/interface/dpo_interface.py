from typing import Dict
import dataclasses

import torch
import torch.utils.data
import tqdm

from reallm.base.namedarray import from_dict, NamedArray, recursive_apply
from reallm.impl.model.backend.pipe_engine.ds_pipe_engine import (PipelinableModelRunner,
                                                                  PipelinableModelRunnerWithZeRO)
from reallm.impl.model.nn.real_llm_api import ReaLModel
from reallm.impl.model.utils.functional import gather_packed_shifted_log_probs
import reallm.api.core.model_api as model_api
import reallm.base.logging as logging
import reallm.impl.model.utils.dpo_functional as dpo_functional

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
    logprobs = gather_packed_shifted_log_probs(logits, cu_seqlens, packed_input_ids).float()

    logprob_sum = []
    offset = 0
    for i in range(prompt_lens.shape[0]):
        logprob_sum.append(logprobs[offset + prompt_lens[i] - 1:offset + input_lens[2 * i] - 1].sum())
        offset += input_lens[2 * i] - 1
        logprob_sum.append(logprobs[offset + prompt_lens[i] - 1:offset + input_lens[2 * i + 1] - 1].sum())
        offset += input_lens[2 * i + 1] - 1
    assert offset == sum(input_lens) - input_lens.shape[0], (
        offset,
        sum(input_lens),
        input_lens.shape,
    )

    pi_seqlogp = torch.stack(logprob_sum)

    loss, kl_rewards = dpo_functional.dpo_loss(pi_logps=pi_seqlogp, ref_logps=seqlogp.view(-1), beta=beta)

    loss = loss.mean()

    return loss, dict(
        loss=loss.detach().cpu(),
        kl_rewards=kl_rewards.detach().mean().cpu(),
    )


@dataclasses.dataclass
class DPOInterface(model_api.ModelInterface):
    beta: float
    enable_save: bool = True

    @torch.inference_mode()
    def inference(self, model: model_api.Model, data: NamedArray) -> NamedArray:
        module = model.module
        module.eval()
        data = recursive_apply(data, lambda x: x.to(model.device))

        n_pairs = data["pos_input_lens"].shape[0]
        pair_lens = torch.tensor(data.metadata["seqlens"], dtype=torch.int32, device=model.device)
        input_lens: torch.IntTensor = torch.stack(
            [data["pos_input_lens"], pair_lens - data["pos_input_lens"]], 1).view(-1)
        prompt_lens: torch.IntTensor = data["prompt_lens"]
        cu_seqlens = torch.cat([input_lens.new_zeros(1), input_lens.cumsum(0)]).int()
        input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = int(max(input_lens))

        seqlens_cpu = input_lens.cpu().numpy().tolist()

        if isinstance(module, (PipelinableModelRunner, PipelinableModelRunnerWithZeRO)):
            res = module.forward(
                seqlens_cpu=seqlens_cpu,
                packed_input_ids=data["packed_input_ids"],
                cu_seqlens=cu_seqlens,
            )
            if res is None:
                return None
            logits = res
        else:
            if hasattr(module, "module"):
                module = module.module
            logits: torch.FloatTensor = module(
                packed_input_ids=data["packed_input_ids"],
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            ).logits

        logprobs = gather_packed_shifted_log_probs(logits, cu_seqlens, data["packed_input_ids"]).float()

        logprob_sum = []
        offset = 0
        for i in range(prompt_lens.shape[0]):
            logprob_sum.append(logprobs[offset + prompt_lens[i] - 1:offset + input_lens[2 * i] - 1].sum())
            offset += input_lens[2 * i] - 1
            logprob_sum.append(logprobs[offset + prompt_lens[i] - 1:offset + input_lens[2 * i + 1] - 1].sum())
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
        pair_lens = torch.tensor(data.metadata["seqlens"], dtype=torch.int32, device=model.device)
        neg_input_lens = pair_lens - data["pos_input_lens"]
        input_lens: torch.Tensor = torch.stack([data["pos_input_lens"], neg_input_lens], 1).view(-1)
        prompt_lens: torch.IntTensor = data["prompt_lens"]
        cu_seqlens = torch.cat([input_lens.new_zeros(1), input_lens.cumsum(0)], 0).int()
        max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))

        module = model.module
        module.eval()

        if isinstance(module, PipelinableModelRunnerWithZeRO):
            loss_fn_kwargs = dict(
                input_lens=pair_lens,
                prompt_lens=prompt_lens,
                seqlogp=data["seqlogp"],
                beta=self.beta,
            )
            loss, stats = module.train_batch(
                seqlens_cpu=data.metadata["seqlens"],
                packed_input_ids=packed_input_ids,
                cu_seqlens=cu_seqlens,
                input_lens_for_partition=pair_lens,
                loss_fn=_dpo_loss_from_model_outputs,
                **loss_fn_kwargs,
            )
        else:
            logits: torch.FloatTensor = module(
                packed_input_ids=packed_input_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            ).logits

            loss, stats = _dpo_loss_from_model_outputs(
                logits=logits,
                packed_input_ids=packed_input_ids,
                cu_seqlens=cu_seqlens,
                prompt_lens=prompt_lens,
                seqlogp=data["seqlogp"],
                beta=self.beta,
            )

            module.backward(loss)
            module.step()

        cur_epoch = model.version.epoch
        model.inc_version()

        if stats is None:
            stats = {}
        return {k: float(v) for k, v in stats.items()}

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
