from collections import defaultdict
from typing import Dict, List, Optional
import time

from deepspeed.runtime.engine import DeepSpeedEngine
import deepspeed
import torch
import torch.utils.data
import tqdm

from profiler.engine import ProfileEngine

from base.dataparallel import PackedParallelDataBroker
from base.namedarray import from_dict, NamedArray, recursive_apply
from impl.model.backend.pipe_engine.ds_pipe_engine import DeepSpeedPipelineEngine
from impl.model.backend.pipe_inf import InferencePipelineEngine
from impl.model.nn.flash_mqat.flash_generate import generate, GenerationConfig
from impl.model.parallelism.model_parallel.modules import vocab_parallel_cross_entropy
from impl.model.utils.data import PipeCacheData, PipeTransferData
from impl.model.utils.functional import (build_leave_one_indices, build_shift_one_indices,
                                         gather_packed_shifted_log_probs)
from impl.model.utils.save_load import save_hf_or_lora_model
import api.data
import api.model
import base.constants
import base.dataparallel

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


class ProfileInterface(api.model.ModelInterface):
    n_minibatches: int = 4

    def train_step(self, model: api.model.Model, data: NamedArray, gen_tokens: int = 128) -> Dict:
        module: deepspeed.DeepSpeedEngine = model.module
        data = recursive_apply(data, lambda x: x.to(model.device))
        datas = PackedParallelDataBroker.scatter_to(data,
                                                    self.n_minibatches,
                                                    min_size=2 * base.constants.pipe_parallel_world_size())
        offset = 0
        batch_seqlens = data.metadata['seqlens']

        for d in datas:
            packed_input_ids: torch.Tensor = d['packed_input_ids']  # shape [tot_seqlen]
            cu_seqlens: torch.Tensor = d['cu_seqlens']
            prompt_mask: torch.BoolTensor = d['prompt_mask']  # shape [tot_seqlen]
            input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))
            seqlens = batch_seqlens[offset:offset + input_lens.shape[0]]
            offset += input_lens.shape[0]

            module.train()

            loss_fn_kwargs = dict(
                prompt_mask=prompt_mask,
                input_lens=cu_seqlens[1:] -
                cu_seqlens[:-1],  # this is used to partition other loss_fn_kwargs into microbatches
            )
            if isinstance(module, DeepSpeedPipelineEngine):
                loss, _ = module.train_batch(
                    seqlens_cpu=seqlens,
                    packed_input_ids=packed_input_ids,
                    cu_seqlens=cu_seqlens,
                    loss_fn=compute_packed_sft_loss,
                    num_micro_batches=2 * base.constants.pipe_parallel_world_size(),
                    **loss_fn_kwargs,
                )
            else:
                logits = module(packed_input_ids=packed_input_ids,
                                cu_seqlens=cu_seqlens,
                                max_seqlen=max_seqlen)
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

    @torch.no_grad()
    def inference(self, model: api.model.Model, data: NamedArray, gen_tokens: int = 128) -> Dict:
        device = model.device
        module = model.module
        module.eval()
        # assert isinstance(module, ProfileEngine)

        data = recursive_apply(data, lambda x: x.to(device))
        packed_input_ids: torch.Tensor = data["packed_input_ids"]
        cu_seqlens: torch.Tensor = data["cu_seqlens"]
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())

        if isinstance(module, (DeepSpeedPipelineEngine, InferencePipelineEngine)):
            res = module.forward(seqlens_cpu=data.metadata['seqlens'],
                                 packed_input_ids=data["packed_input_ids"],
                                 cu_seqlens=cu_seqlens,
                                 num_micro_batches=base.constants.pipe_parallel_world_size())
            if res is None:
                return None
            logits = res
        else:
            if hasattr(module, "module"):
                module = module.module
            with module.sequence_parallel_disable():
                res = module(packed_input_ids=data["packed_input_ids"],
                             cu_seqlens=cu_seqlens,
                             max_seqlen=max_seqlen)
            logits = res

        return dict(logits=logits)

    @torch.no_grad()
    def generate(self, model: api.model.Model, data: NamedArray, gen_tokens: int = 256) -> NamedArray:
        module = model.module
        module.eval()
        # assert isinstance(module, ProfileEngine)

        data = recursive_apply(data, lambda x: x.to(model.device))
        gconfig = GenerationConfig(min_new_tokens=gen_tokens, max_new_tokens=gen_tokens)
        packed_prompts = data["packed_input_ids"]
        cu_seqlens = data["cu_seqlens"]
        self.pipe_gen_n_mbs = base.constants.pipe_parallel_world_size()
        prompt_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        bs = prompt_lengths.shape[0]

        if isinstance(module, (DeepSpeedPipelineEngine, InferencePipelineEngine)):
            res = module.generate(
                seqlens_cpu=data.metadata['seqlens'],
                tokenizer=model.tokenizer,
                packed_input_ids=packed_prompts,
                cu_seqlens=cu_seqlens,
                gconfig=gconfig,
                num_micro_batches=self.pipe_gen_n_mbs,
            )
            if res is None:
                return None

            gen_tokens, logprobs, logits_mask, *_ = res
            # logger.info(f"gen_tokens shape {gen_tokens.shape}")
        else:
            # unwrap deepspeed engine here
            if hasattr(module, "module"):
                module = module.module
            with module.sequence_parallel_disable():
                gen_res = module.generate(
                    tokenizer=model.tokenizer,
                    packed_input_ids=packed_prompts,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=int(max(prompt_lengths)),
                    gconfig=gconfig,
                )
            gen_tokens = gen_res.sequences
            logprobs = gen_res.scores
            logits_mask = gen_res.logits_mask

        return dict(
            gen_tokens=gen_tokens,
            log_probs=logprobs,
            logits_mask=logits_mask,
        )


api.model.register_interface("profile", ProfileInterface)
