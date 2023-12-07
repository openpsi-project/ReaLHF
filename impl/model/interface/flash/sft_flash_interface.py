from typing import Dict, List, Optional

import deepspeed
import torch
import torch.utils.data
import tqdm

from base.namedarray import from_dict, NamedArray, recursive_apply
from impl.model.backend.pipe_engine.ds_pipe_engine import DeepSpeedPipelineEngine
from impl.model.utils.functional import gather_packed_shifted_log_probs
from impl.model.utils.save_load import save_hf_or_lora_model, save_pipeline_model
import api.data
import api.model


def compute_packed_sft_loss(
    logits: torch.Tensor,
    packed_input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    prompt_mask: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    # **kwargs is used to ensure the correctness of invoking this function
    logprobs = gather_packed_shifted_log_probs(logits, cu_seqlens, packed_input_ids)
    shift_one_indices = torch.cat([
        torch.arange(cu_seqlens[i] + 1, cu_seqlens[i + 1], dtype=torch.long, device=cu_seqlens.device)
        for i in range(cu_seqlens.shape[0] - 1)
    ])
    loss_mask = 1 - prompt_mask[shift_one_indices].float()
    loss = -(logprobs * loss_mask).sum() / loss_mask.sum()
    return loss, {"loss": loss.detach().cpu()}


class PackedSupervisedFinetuningInterface(api.model.ModelInterface):

    def train_step(self, model: api.model.Model, data: NamedArray) -> Dict:
        data = recursive_apply(data, lambda x: x.to(model.device))
        packed_input_ids: torch.Tensor = data['packed_input_ids']  # shape [tot_seqlen]
        cu_seqlens: torch.Tensor = data['cu_seqlens']
        prompt_mask: torch.BoolTensor = data['prompt_mask']  # shape [tot_seqlen]
        module: deepspeed.DeepSpeedEngine = model.module
        max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))

        module.train()

        if isinstance(module, DeepSpeedPipelineEngine):
            loss_fn_kwargs = dict(
                prompt_mask=prompt_mask,
                input_lens=cu_seqlens[1:] -
                cu_seqlens[:-1],  # this is used to partition other loss_fn_kwargs into microbatches
            )
            loss, _ = module.train_batch(packed_input_ids=packed_input_ids,
                                         cu_seqlens=cu_seqlens,
                                         loss_fn=compute_packed_sft_loss,
                                         **loss_fn_kwargs)
        else:
            logits = module(packed_input_ids=packed_input_ids, cu_seqlens=cu_seqlens,
                            max_seqlen=max_seqlen).logits.float()
            loss, _ = compute_packed_sft_loss(logits, packed_input_ids, cu_seqlens, prompt_mask)
            module.backward(loss)
            module.step()

        cur_epoch = model.version.epoch
        model.inc_version()
        if model.version.epoch > cur_epoch:
            module.tput_timer.update_epoch_count()

        res = dict()
        if loss is not None:
            res['loss'] = float(loss)
        return res

    def save(self, model: api.model.Model, save_dir: str):
        if isinstance(model.module, DeepSpeedPipelineEngine):
            save_pipeline_model(model, save_dir)
        else:
            save_hf_or_lora_model(model, save_dir)

    @torch.inference_mode()
    def evaluate(self, model_: api.model.Model, eval_dataloader: torch.utils.data.DataLoader) -> Dict:
        device = model_.device
        module = model_.module

        module.eval()
        losses = 0
        n_seqs = 0

        for step, data in enumerate(tqdm.tqdm(eval_dataloader)):
            data = recursive_apply(from_dict(data), lambda x: x.to(device))
            packed_input_ids: torch.Tensor = data['packed_input_ids']  # shape [tot_seqlen]
            cu_seqlens: torch.Tensor = data['cu_seqlens']
            prompt_mask: torch.BoolTensor = data['prompt_mask']  # shape [tot_seqlen]
            max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))

            if isinstance(module, DeepSpeedPipelineEngine):
                loss_fn_kwargs = dict(
                    prompt_mask=prompt_mask,
                    input_lens=cu_seqlens[1:] - cu_seqlens[:-1],
                )
                loss, _ = module.eval_batch(packed_input_ids,
                                            cu_seqlens,
                                            loss_fn=compute_packed_sft_loss,
                                            **loss_fn_kwargs)
            else:
                logits = module(packed_input_ids=packed_input_ids,
                                cu_seqlens=cu_seqlens,
                                max_seqlen=max_seqlen).logits.float()
                loss, _ = compute_packed_sft_loss(logits, packed_input_ids, cu_seqlens, prompt_mask)

            if loss is not None:
                losses += (cu_seqlens.shape[0] - 1) * loss.float()
                n_seqs += cu_seqlens.shape[0] - 1

        res = dict()
        if n_seqs > 0:
            losses = losses / n_seqs
            try:
                perplexity = torch.exp(losses).item()
            except OverflowError:
                perplexity = float("inf")
            return dict(ppl=perplexity)
        return res


api.model.register_interface("flash_sft", PackedSupervisedFinetuningInterface)
