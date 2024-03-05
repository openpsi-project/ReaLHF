from typing import Dict, List, Optional
import time

from deepspeed.runtime.engine import DeepSpeedEngine
import deepspeed
import torch
import torch.utils.data
import tqdm

from profiler.engine import ProfileEngine

from base.namedarray import from_dict, NamedArray, recursive_apply
from impl.model.backend.pipe_engine.ds_pipe_engine import DeepSpeedPipelineEngine
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

    def train_step(self, model: api.model.Model, data: NamedArray) -> Dict:
        data = recursive_apply(data, lambda x: x.to(model.device))
        packed_input_ids: torch.Tensor = data['packed_input_ids']  # shape [tot_seqlen]
        cu_seqlens: torch.Tensor = data['cu_seqlens']
        prompt_mask: torch.BoolTensor = data['prompt_mask']  # shape [tot_seqlen]
        module: deepspeed.DeepSpeedEngine = model.module
        max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))

        module.train()
        assert isinstance(module, ProfileEngine)

        loss_fn_kwargs = dict(
            prompt_mask=prompt_mask,
            input_lens=cu_seqlens[1:] -
            cu_seqlens[:-1],  # this is used to partition other loss_fn_kwargs into microbatches
        )
        loss, _ = module.train_batch(
            packed_input_ids=packed_input_ids,
            cu_seqlens=cu_seqlens,
            loss_fn=compute_packed_sft_loss,
            num_micro_batches=1,
            **loss_fn_kwargs,
        )

        cur_epoch = model.version.epoch
        model.inc_version()
        if model.version.epoch > cur_epoch:
            module.tput_timer.update_epoch_count()

        res = dict()
        if loss is not None:
            res['loss'] = float(loss)
        return res

    def save(self, model: api.model.Model, save_dir: str):
        model.module.save(save_dir,
                          epoch=model.version.epoch,
                          epoch_step=model.version.epoch_step,
                          global_step=model.version.global_step)

    @torch.no_grad()
    def inference(self, model: api.model.Model, data: NamedArray) -> Dict:
        device = model.device
        module = model.module
        module.eval()
        assert isinstance(module, ProfileEngine)

        data = recursive_apply(data, lambda x: x.to(device))
        packed_input_ids: torch.Tensor = data["packed_input_ids"]
        cu_seqlens: torch.Tensor = data["cu_seqlens"]
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max())

        logits = module.forward(packed_input_ids=packed_input_ids, cu_seqlens=cu_seqlens, num_micro_batches=1)

        return dict(logits=logits)

    @torch.no_grad()
    def generate(self, model: api.model.Model, data: NamedArray) -> NamedArray:
        module = model.module
        module.eval()
        assert isinstance(module, ProfileEngine)

        data = recursive_apply(data, lambda x: x.to(model.device))
        gconfig = GenerationConfig(min_new_tokens=3, max_new_tokens=3)

        res = module.generate(
            tokenizer=model.tokenizer,
            packed_input_ids=data['packed_input_ids'],
            cu_seqlens=data['cu_seqlens'],
            gconfig=gconfig,
            num_micro_batches=1,
        )
        if res is None:
            return dict()

        gen_tokens, logprobs, logits_mask, *_ = res

        return dict(
            gen_tokens=gen_tokens,
            log_probs=logprobs,
            logits_mask=logits_mask,
        )


class ProfileLayerInterface(api.model.ModelInterface):

    @torch.no_grad()
    def inference(self, model: api.model.Model, data: NamedArray) -> NamedArray:
        module = model.module
        assert isinstance(module, DeepSpeedEngine)
        module.eval()

        x = PipeTransferData(
            pp_input=data['pp_input'],
            cu_seqlens=data['cu_seqlens'],
            max_seqlen=data['max_seqlen'],
            attention_mask=data['attention_mask'],
        )
        y = PipeCacheData(
            k_cache=data['k_cache'],
            v_cache=data['v_cache'],
            cache_seqlens=data['cache_seqlens'],
        )

        x = recursive_apply(x, lambda x: x.to(model.device))
        y = recursive_apply(y, lambda y: y.to(model.device))

        st = time.monotonic_ns()
        x = module.forward(x, y)
        t = (time.monotonic_ns() - st) // int(1e3)  # micro seconds
        return dict(cost=t,)

    def train_forward(self, model: api.model.Model, data: NamedArray) -> float:
        module = model.module
        assert isinstance(module, DeepSpeedEngine)
        module.eval()

        x = PipeTransferData(
            pp_input=data['pp_input'],
            cu_seqlens=data['cu_seqlens'],
            max_seqlen=data['max_seqlen'],
            attention_mask=data['attention_mask'],
        )
        y = PipeCacheData()
        x = recursive_apply(x, lambda x: x.to(model.device))

        st = time.monotonic_ns()
        x = module.forward(x, y)
        t = (time.monotonic_ns() - st) // int(1e3)  # micro seconds

        # some random loss function
        r = torch.rand(*x.pp_output.shape, device=x.pp_output.device, dtype=x.pp_output.dtype)
        loss = torch.sum(x.pp_output * r)

        return dict(
            loss=loss,
            cost=t,
        )

    def train_backward(self, model: api.model.Model, data: NamedArray) -> NamedArray:
        module = model.module
        assert isinstance(module, DeepSpeedEngine)
        module.eval()

        loss = data['loss']

        st = time.monotonic_ns()
        r = module.backward(loss)
        t = (time.monotonic_ns() - st) // int(1e3)
        return dict(cost=t,)

    def optimizer_step(self, model: api.model.Model, data: NamedArray) -> NamedArray:
        module = model.module
        assert isinstance(module, DeepSpeedEngine)
        module.eval()

        st = time.monotonic_ns()
        module.step()
        t = (time.monotonic_ns() - st) // int(1e3)
        return dict(cost=t,)


api.model.register_interface("profile", ProfileInterface)
api.model.register_interface("profile_layer", ProfileLayerInterface)
