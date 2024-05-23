from typing import Dict, List, Optional

import deepspeed
import torch
import torch.utils.data
import tqdm

from reallm.base.namedarray import from_dict, NamedArray, recursive_apply
from reallm.impl.model.backend.pipe_engine.ds_pipe_engine import DeepSpeedPipelineEngine
from reallm.impl.model.nn.real_llm_api import ReaLModel
from reallm.impl.model.nn.real_llm_generate import generate, GenerationConfig
from reallm.impl.model.parallelism.model_parallel.modules import vocab_parallel_cross_entropy
from reallm.impl.model.utils.functional import (build_leave_one_indices, build_shift_one_indices,
                                                gather_packed_shifted_log_probs)
import reallm.api.core.data_api as data_api
import reallm.api.core.model_api as model_api
import reallm.base.constants as constants
import reallm.base.dataparallel as dataparallel

try:
    from flash_attn.bert_padding import unpad_input
except ModuleNotFoundError:
    pass


def compute_packed_sft_loss(
    logits: torch.Tensor,
    packed_input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    prompt_mask: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    # **kwargs is used to ensure the correctness of invoking this function
    shift_one_indices = build_shift_one_indices(logits, cu_seqlens)
    logprobs = gather_packed_shifted_log_probs(logits, cu_seqlens, packed_input_ids).float()
    prompt_mask = prompt_mask[shift_one_indices]
    # float16 will overflow here
    loss = -torch.where(prompt_mask, 0, logprobs).sum() / (prompt_mask.numel() - prompt_mask.count_nonzero())
    return loss, {"loss": loss.detach()}


class SFTInterface(model_api.ModelInterface):

    def train_step(self, model: model_api.Model, data: NamedArray) -> Dict:
        data = recursive_apply(data, lambda x: x.to(model.device))
        packed_input_ids: torch.Tensor = data['packed_input_ids']  # shape [tot_seqlen]
        cu_seqlens: torch.Tensor = data['cu_seqlens']
        prompt_mask: torch.BoolTensor = data['prompt_mask']  # shape [tot_seqlen]
        module: deepspeed.DeepSpeedEngine = model.module
        max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))

        module.train()

        seqlens_cpu = data.metadata["seqlens"]

        if isinstance(module, DeepSpeedPipelineEngine):
            loss_fn_kwargs = dict(
                prompt_mask=prompt_mask,
                input_lens=cu_seqlens[1:] -
                cu_seqlens[:-1],  # this is used to partition other loss_fn_kwargs into microbatches
            )
            loss, _ = module.train_batch(
                seqlens_cpu=seqlens_cpu,
                packed_input_ids=packed_input_ids,
                cu_seqlens=cu_seqlens,
                loss_fn=compute_packed_sft_loss,
                num_micro_batches=constants.pipe_parallel_world_size() * 2,
                **loss_fn_kwargs,
            )
        else:
            logits = module(packed_input_ids=packed_input_ids, cu_seqlens=cu_seqlens,
                            max_seqlen=max_seqlen).logits
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

    def save(self, model: model_api.Model, save_dir: str):
        module = model.module
        if not isinstance(module, ReaLModel):
            module = module.module
        module.save_to_hf(tokenizer=model.tokenizer,
                          save_dir=save_dir,
                          epoch=model.version.epoch,
                          epoch_step=model.version.epoch_step,
                          global_step=model.version.global_step)

    @torch.inference_mode()
    def evaluate(self, model_: model_api.Model, eval_dataloader: torch.utils.data.DataLoader) -> Dict:
        device = model_.device
        module = model_.module

        module.eval()
        losses = 0
        n_seqs = 0

        for step, data in enumerate(tqdm.tqdm(eval_dataloader)):
            data = recursive_apply(from_dict(data), lambda x: x.to(device))
            packed_input_ids: torch.Tensor = data["packed_input_ids"]  # shape [tot_seqlen]
            cu_seqlens: torch.Tensor = data["cu_seqlens"].int()
            prompt_mask: torch.BoolTensor = data["prompt_mask"]  # shape [tot_seqlen]
            max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))
            seqlens_cpu = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy().tolist()

            if isinstance(module, DeepSpeedPipelineEngine):
                loss_fn_kwargs = dict(
                    prompt_mask=prompt_mask,
                    input_lens=cu_seqlens[1:] - cu_seqlens[:-1],
                )
                loss, _ = module.eval_batch(seqlens_cpu=seqlens_cpu,
                                            packed_input_ids=packed_input_ids,
                                            cu_seqlens=cu_seqlens,
                                            loss_fn=compute_packed_sft_loss,
                                            num_micro_batches=constants.pipe_parallel_world_size(),
                                            **loss_fn_kwargs)
            else:
                logits = module(packed_input_ids=packed_input_ids,
                                cu_seqlens=cu_seqlens,
                                max_seqlen=max_seqlen).logits
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

        if isinstance(module, DeepSpeedPipelineEngine):
            logits = module.forward(
                seqlens_cpu=seqlens_cpu,
                packed_input_ids=packed_input_ids,
                cu_seqlens=cu_seqlens,
            )
        else:
            logits = model.module(packed_input_ids=packed_input_ids,
                                  cu_seqlens=cu_seqlens,
                                  max_seqlen=max_seqlen).logits
        return dict(logits=logits)

    # for testing only
    @torch.no_grad()
    def generate(self, model: model_api.Model, data: NamedArray, gconfig: GenerationConfig) -> NamedArray:
        module = model.module

        module.eval()

        data = recursive_apply(data, lambda x: x.to(model.device))
        # prompts: torch.LongTensor = data["prompts"]
        # prompt_att_mask: torch.BoolTensor = data["prompt_att_mask"]
        # bs, prompt_max_len = prompts.shape[:2]

        # assert isinstance(module, DeepSpeedPipelineEngine)
        # if isinstance(module, DeepSpeedPipelineEngine):
        # packed_input_ids, _, cu_seqlens, _ = unpad_input(prompts, prompt_att_mask)

        if isinstance(module, DeepSpeedPipelineEngine):
            res = module.generate(
                seqlens_cpu=data.metadata["seqlens"],
                tokenizer=model.tokenizer,
                packed_input_ids=data['packed_input_ids'],
                cu_seqlens=data['cu_seqlens'],
                gconfig=gconfig,
            )
            if res is None:
                return dict()

            gen_tokens, logprobs, logits_mask, *_ = res
        else:
            res = module.generate(
                tokenizer=model.tokenizer,
                packed_input_ids=data['packed_input_ids'],
                cu_seqlens=data['cu_seqlens'],
                max_seqlen=max(data['cu_seqlens'][1:] - data['cu_seqlens'][:-1]),
                gconfig=gconfig,
            )
            gen_tokens = res.sequences
            logprobs = res.scores
            logits_mask = res.logits_mask

        return dict(
            gen_tokens=gen_tokens,
            log_probs=logprobs,
            logits_mask=logits_mask,
        )


model_api.register_interface("sft", SFTInterface)
