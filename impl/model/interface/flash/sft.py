from typing import Dict, List, Optional

import deepspeed
import torch
import torch.utils.data
import tqdm
import transformers

from base.namedarray import from_dict, NamedArray, recursive_apply
from impl.model.utils.data import gather_packed_shifted_log_probs
from impl.model.utils.save import save_hf_format
import api.data
import api.model


def compute_packed_sft_loss(logits: torch.Tensor, packed_input_ids: torch.Tensor, cu_seqlens: torch.Tensor,
                            loss_mask: torch.Tensor) -> torch.Tensor:
    logprobs = gather_packed_shifted_log_probs(logits, cu_seqlens, packed_input_ids)
    shift_one_indices = torch.cat([
        torch.arange(cu_seqlens[i] + 1, cu_seqlens[i + 1], dtype=torch.long, device=cu_seqlens.device)
        for i in range(cu_seqlens.shape[0] - 1)
    ])
    loss_mask = loss_mask[shift_one_indices]
    return -(logprobs * loss_mask).sum() / loss_mask.sum()


class PackedSupervisedFinetuningInterface(api.model.ModelInterface):

    def train_step(self, model: api.model.Model, data: NamedArray) -> Dict:
        data = recursive_apply(data, lambda x: x.to(model.device))
        packed_input_ids: torch.Tensor = data['packed_input_ids'].squeeze()  # shape [tot_seqlen]
        total_seq_len = packed_input_ids.shape[0]
        cu_seqlens: torch.Tensor = data['cu_seqlens'].squeeze()
        n_seqs = (cu_seqlens == total_seq_len).nonzero()[0][0]
        cu_seqlens = cu_seqlens[:n_seqs + 1]  # shape [bs + 1]
        prompt_mask: torch.BoolTensor = data['prompt_mask'].squeeze()  # shape [tot_seqlen]
        module: deepspeed.DeepSpeedEngine = model.module
        max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))

        module.train()

        logits = module(packed_input_ids=packed_input_ids, cu_seqlens=cu_seqlens,
                        max_seqlen=max_seqlen).logits.float()
        loss = compute_packed_sft_loss(logits, packed_input_ids, cu_seqlens, 1 - prompt_mask.float())
        module.backward(loss)
        module.step()
        return dict(loss=loss.item())

    def save(self, model: api.model.Model, save_dir: str):
        save_hf_format(model.module, model.tokenizer, save_dir)

    @torch.inference_mode()
    def evaluate(self, model_: api.model.Model, eval_dataloader: torch.utils.data.DataLoader) -> Dict:
        device = model_.device
        module = model_.module

        module.eval()
        losses = 0
        n_tokens = 0

        for step, data in enumerate(tqdm.tqdm(eval_dataloader)):
            data = recursive_apply(from_dict(data), lambda x: x.to(device))
            packed_input_ids: torch.Tensor = data['packed_input_ids'].squeeze()  # shape [tot_seqlen]
            total_seq_len = packed_input_ids.shape[0]
            cu_seqlens: torch.Tensor = data['cu_seqlens'].squeeze()
            n_seqs = (cu_seqlens == total_seq_len).nonzero()[0][0]
            cu_seqlens = cu_seqlens[:n_seqs + 1]  # shape [bs + 1]
            prompt_mask: torch.BoolTensor = data['prompt_mask'].squeeze()  # shape [tot_seqlen]
            max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))

            logits = module(packed_input_ids=packed_input_ids, cu_seqlens=cu_seqlens,
                            max_seqlen=max_seqlen).logits.float()
            loss = compute_packed_sft_loss(logits, packed_input_ids, cu_seqlens, 1 - prompt_mask)

            losses += (1 - prompt_mask).sum() * loss.float()
            n_tokens += (1 - prompt_mask).sum()
        losses = losses / n_tokens
        try:
            perplexity = torch.exp(losses).item()
        except OverflowError:
            perplexity = float("inf")
        return dict(ppl=perplexity)


api.model.register_interface("flash_sft", PackedSupervisedFinetuningInterface)