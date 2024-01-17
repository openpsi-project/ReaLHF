from typing import Dict, Optional, Tuple
import dataclasses

import torch

from base.namedarray import NamedArray, recursive_apply
from impl.model.backend.pipe_engine.stream_pipe_engine import EngineFuture, StreamPipeEngine
from impl.model.interface.flash.sft_flash_interface import compute_packed_sft_loss
from impl.model.nn.flash_mqat.flash_generate import GenerationConfig
import api.model
import base.logging as logging

try:
    from flash_attn.bert_padding import unpad_input
except ModuleNotFoundError:
    pass

logger = logging.getLogger("StreamPipeTestInterface")


@dataclasses.dataclass
class StreamPipeTestInterface(api.model.ModelInterface):

    def __post_init__(self):
        super().__post_init__()
        self._is_future_interface = True
        self.register_post_hook("train_step", self.__collect_train_step)
        self.register_post_hook("generate", self.__collect_generate)

    def train_step(self,
                   model: api.model.Model,
                   data: NamedArray,
                   num_micro_batches: Optional[int] = None) -> Tuple[EngineFuture, NamedArray]:
        module = model.module
        assert isinstance(module, StreamPipeEngine)
        data = recursive_apply(data, lambda x: x.to(model.device))
        packed_input_ids: torch.Tensor = data['packed_input_ids']  # shape [tot_seqlen]
        cu_seqlens: torch.Tensor = data['cu_seqlens']
        prompt_mask: torch.BoolTensor = data['prompt_mask']  # shape [tot_seqlen]
        max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))

        module.train()

        loss_fn_kwargs = dict(
            prompt_mask=prompt_mask,
            input_lens=cu_seqlens[1:] -
            cu_seqlens[:-1],  # this is used to partition other loss_fn_kwargs into microbatches
        )
        future = module.train_batch(packed_input_ids=packed_input_ids,
                                    cu_seqlens=cu_seqlens,
                                    loss_fn=compute_packed_sft_loss,
                                    num_micro_batches=num_micro_batches,
                                    **loss_fn_kwargs)
        return future, data

    def __collect_train_step(self, model: api.model.Model, data: NamedArray, future: EngineFuture) -> Dict:
        loss, _ = future.result()
        module = model.module

        cur_epoch = model.version.epoch
        model.inc_version()
        if model.version.epoch > cur_epoch:
            module.tput_timer.update_epoch_count()

        res = dict()
        if loss is not None:
            res['loss'] = float(loss)
        return res

    @torch.no_grad()
    def generate(self,
                 model: api.model.Model,
                 data: NamedArray,
                 gconfig: GenerationConfig,
                 num_micro_batches: Optional[int] = None) -> Tuple[EngineFuture, NamedArray]:
        module = model.module
        assert isinstance(module, StreamPipeEngine)
        data = recursive_apply(data, lambda x: x.to(model.device))
        packed_input_ids: torch.Tensor = data['packed_input_ids']

        data = recursive_apply(data, lambda x: x.to(model.device))
        prompts: torch.LongTensor = data["prompts"]
        prompt_att_mask: torch.BoolTensor = data["prompt_att_mask"]

        packed_input_ids, _, cu_seqlens, _ = unpad_input(prompts, prompt_att_mask)

        future = module.generate(
            tokenizer=model.tokenizer,
            packed_input_ids=packed_input_ids,
            cu_seqlens=cu_seqlens,
            gconfig=gconfig,
            num_micro_batches=num_micro_batches,
        )
        return future, data

    def __collect_generate(self, model: api.model.Model, data: NamedArray, future: EngineFuture) -> Dict:
        res = future.result()
        if res is not None:
            gen_tokens, logprobs, logits_mask, *_ = res
            return dict(
                gen_tokens=gen_tokens,
                log_probs=logprobs,
                logits_mask=logits_mask,
            )
        else:
            return dict()


api.model.register_interface("stream_pipe_test", StreamPipeTestInterface)
