from statistics import mean
from typing import Dict, List, Optional
import logging

import deepspeed
import torch
import torch.utils.data
import tqdm

from base.namedarray import from_dict, NamedArray, recursive_apply
from impl.model.backend.pipe_engine.ds_pipe_engine import DeepSpeedPipelineEngine
from impl.model.nn.flash_mqat import GenerationConfig
from impl.model.utils.data import gather_packed_shifted_log_probs, PipeCacheData, PipeTransferData
from impl.model.utils.save import save_hf_or_lora_model
import api.data
import api.model

logger = logging.getLogger("pipe_flash_sft")


def compute_packed_sft_loss(logits: torch.Tensor, packed_input_ids: torch.Tensor, cu_seqlens: torch.Tensor,
                            prompt_mask: torch.Tensor, **kwargs) -> torch.Tensor:
    loss_mask = 1 - prompt_mask.float()
    logprobs = gather_packed_shifted_log_probs(logits, cu_seqlens, packed_input_ids)
    shift_one_indices = torch.cat([
        torch.arange(cu_seqlens[i] + 1, cu_seqlens[i + 1], dtype=torch.long, device=cu_seqlens.device)
        for i in range(cu_seqlens.shape[0] - 1)
    ])
    loss_mask = loss_mask[shift_one_indices]
    loss = -(logprobs * loss_mask).sum() / loss_mask.sum()
    return loss, {"loss": loss.detach().cpu()}


class PipePackedSupervisedFinetuningInterface(api.model.ModelInterface):

    def train_step(self, model: api.model.Model, data: NamedArray) -> Dict:
        data = recursive_apply(data, lambda x: x.to(model.device))
        packed_input_ids: torch.Tensor = data['packed_input_ids'].squeeze()  # shape [tot_seqlen]
        total_seq_len = packed_input_ids.shape[0]
        cu_seqlens: torch.Tensor = data['cu_seqlens'].squeeze()
        n_seqs = (cu_seqlens == total_seq_len).nonzero()[0][0]
        cu_seqlens = cu_seqlens[:n_seqs + 1]  # shape [bs + 1]
        prompt_mask: torch.BoolTensor = data['prompt_mask'].squeeze()  # shape [tot_seqlen]
        module: DeepSpeedPipelineEngine = model.module
        max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))

        module.train()

        loss_fn_kwargs = dict(
            prompt_mask=prompt_mask,
            input_lens=cu_seqlens[1:] - cu_seqlens[:-1],
        )

        stats = module.train_batch(packed_input_ids=packed_input_ids,
                                   cu_seqlens=cu_seqlens,
                                   loss_fn=compute_packed_sft_loss,
                                   **loss_fn_kwargs)

        cur_epoch = model.version.epoch
        model.inc_version()
        if model.version.epoch > cur_epoch:
            module.tput_timer.update_epoch_count()

        # agg_loss = average loss of data parallel batches
        if len(stats) > 0:
            agg_loss = mean([s["loss"].detach().item() for s in stats])
            return dict(loss=agg_loss)
        else:
            return dict()

    @torch.inference_mode()  # one time evaluate
    def inference(self, model_: api.model.Model, data: NamedArray) -> Dict:
        device = model_.device
        module = model_.module

        module.eval()
        losses = 0
        n_tokens = 0

        # one time inference
        data = recursive_apply(from_dict(data), lambda x: x.to(device))
        packed_input_ids: torch.Tensor = data['packed_input_ids'].squeeze()  # shape [tot_seqlen]
        total_seq_len = packed_input_ids.shape[0]
        cu_seqlens: torch.Tensor = data['cu_seqlens'].squeeze()
        n_seqs = (cu_seqlens == total_seq_len).nonzero()[0][0]
        cu_seqlens = cu_seqlens[:n_seqs + 1]  # shape [bs + 1]
        prompt_mask: torch.BoolTensor = data['prompt_mask'].squeeze()  # shape [tot_seqlen]
        max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))

        fwd_output = module(packed_input_ids=packed_input_ids, cu_seqlens=cu_seqlens, prompt_mask=prompt_mask)
        if fwd_output is not None:
            logits = fwd_output.logits.float()

            loss = compute_packed_sft_loss(logits, packed_input_ids, cu_seqlens, prompt_mask)

            losses += (1 - prompt_mask.float()).sum() * loss.float()
            n_tokens += (1 - prompt_mask.float()).sum()

            return dict(losses=losses.detach().item())
        else:
            return dict()

    def save(self, model: api.model.Model, save_dir: str):
        model.module.save(save_dir)

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

            fwd_output = module(packed_input_ids=packed_input_ids, cu_seqlens=cu_seqlens)

            if fwd_output is not None:
                logits = fwd_output.logits.float()
                loss, _ = compute_packed_sft_loss(logits, packed_input_ids, cu_seqlens,
                                                  1 - prompt_mask.float())

                losses += (1 - prompt_mask.float()).sum() * loss.float()
                n_tokens += (1 - prompt_mask.float()).sum()
                losses = losses / n_tokens
                try:
                    perplexity = torch.exp(losses).item()
                except OverflowError:
                    perplexity = float("inf")
                return dict(ppl=perplexity)
            else:
                return dict()

    @torch.inference_mode()
    def generate(self,
                 model_: api.model.Model,
                 data: NamedArray,
                 gconfig: Optional[GenerationConfig] = None) -> Dict:
        packed_input_ids = data['packed_input_ids'].squeeze()
        cu_seqlens = data['cu_seqlens'].squeeze()
        if gconfig is None:
            gconfig = GenerationConfig(
                min_new_tokens=10,
                max_new_tokens=20,
            )
        module = model_.module
        tokenizer = model_.tokenizer

        module.eval()

        logger.debug(f"gconfig: {gconfig}")
        res = module.generate(tokenizer=tokenizer,
                              packed_input_ids=packed_input_ids,
                              cu_seqlens=cu_seqlens,
                              gconfig=gconfig)

        if res is not None:
            gen_tokens, log_probs, logits_mask, _, prompt_logits = res
            return dict(gen_tokens=gen_tokens,
                        log_probs=log_probs,
                        logits_mask=logits_mask,
                        prompt_logits=prompt_logits)
        else:
            return dict()


api.model.register_interface("pipe_flash_sft", PipePackedSupervisedFinetuningInterface)
