from typing import Dict, List, Optional, Tuple
import os

import deepspeed
import torch
import torch.utils.data
import tqdm

from base.namedarray import from_dict, NamedArray, recursive_apply
from impl.model.backend.pipe_engine.ds_pipe_engine import DeepSpeedPipelineEngine
from impl.model.nn.flash_mqat.flash_generate import GenerationConfig
from impl.model.utils.functional import gather_packed_shifted_log_probs
from impl.model.utils.save_load import save_pipeline_model
import api.data
import api.model
import base.logging as logging

logger = logging.getLogger("pipe_flash_sft")


def compute_packed_sft_loss(logits: torch.Tensor, packed_input_ids: torch.Tensor, cu_seqlens: torch.Tensor,
                            prompt_mask: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict]:
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

        r = module.train_batch(packed_input_ids=packed_input_ids,
                               cu_seqlens=cu_seqlens,
                               loss_fn=compute_packed_sft_loss,
                               **loss_fn_kwargs)

        cur_epoch = model.version.epoch
        model.inc_version()
        if model.version.epoch > cur_epoch:
            module.tput_timer.update_epoch_count()

        # agg_loss = average loss of data parallel batches
        if r is not None:
            avg_loss, stats = r
            logger.info("loss: %s", avg_loss)
            return dict(losses=avg_loss)
        else:
            return dict()

    def save(self, model: api.model.Model, save_dir: str):
        # os.makedirs(save_dir, exist_ok=True)
        # model.module.save(save_dir)
        save_pipeline_model(model, save_dir)

    @torch.inference_mode()
    def evaluate(self, model_: api.model.Model, eval_dataloader: torch.utils.data.DataLoader) -> Dict:
        device = model_.device
        module = model_.module
        assert isinstance(module, DeepSpeedPipelineEngine)

        module.eval()
        losses = 0
        n_seqs = 0

        for step, data in enumerate(tqdm.tqdm(eval_dataloader)):
            data = recursive_apply(from_dict(data), lambda x: x.to(device))
            packed_input_ids: torch.Tensor = data['packed_input_ids'].squeeze()  # shape [tot_seqlen]
            total_seq_len = packed_input_ids.shape[0]
            cu_seqlens: torch.Tensor = data['cu_seqlens'].squeeze()
            n_seqs = (cu_seqlens == total_seq_len).nonzero()[0][0]
            cu_seqlens = cu_seqlens[:n_seqs + 1]  # shape [bs + 1]
            prompt_mask: torch.BoolTensor = data['prompt_mask'].squeeze()  # shape [tot_seqlen]
            max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))

            loss_fn_kwargs = dict(
                prompt_mask=prompt_mask,
                input_lens=cu_seqlens[1:] - cu_seqlens[:-1],
            )

            r = module.eval_batch(packed_input_ids,
                                  cu_seqlens,
                                  loss_fn=compute_packed_sft_loss,
                                  **loss_fn_kwargs)
            if r is not None:
                loss, _ = r
                losses += (cu_seqlens.shape[0] - 1) * loss
                n_seqs += cu_seqlens.shape[0] - 1

        if n_seqs > 0:
            losses = losses / n_seqs
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
                max_new_tokens=256,
            )
        module = model_.module
        tokenizer = model_.tokenizer

        module.eval()

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
