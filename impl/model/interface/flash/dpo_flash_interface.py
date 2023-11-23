from typing import Dict
import dataclasses
import os

import deepspeed
import torch
import tqdm

from base.namedarray import from_dict, NamedArray, recursive_apply
from impl.model.utils.data import gather_packed_shifted_log_probs
from impl.model.utils.save import save_hf_or_lora_model
import api.model
import base.logging as logging
import impl.model.utils.dpo_functional as dpo_functional

logger = logging.getLogger("Packed DPO Interface")


@dataclasses.dataclass
class PackedDirectPerferenceOptimizationInterface(api.model.ModelInterface):
    beta: float
    enable_save: bool = True

    @torch.inference_mode()
    def inference(self, model: api.model.Model, data: NamedArray) -> NamedArray:
        module = model.module
        module.eval()
        data = recursive_apply(data, lambda x: x.to(model.device))

        input_lens: torch.IntTensor = data['pair_input_lens']
        prompt_lens: torch.IntTensor = data['prompt_lens']
        cu_seqlens = torch.cat([input_lens.new_zeros(1), input_lens.cumsum(0)])
        input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = int(max(input_lens))

        logits: torch.FloatTensor = module(packed_input_ids=data['packed_input_ids'],
                                           cu_seqlens=cu_seqlens,
                                           max_seqlen=max_seqlen).logits.float()
        logprobs = gather_packed_shifted_log_probs(logits, cu_seqlens, data['packed_input_ids'])

        logprob_sum = []
        offset = 0
        for i in range(prompt_lens.shape[0]):
            logprob_sum.append(logprobs[offset + prompt_lens[i] - 1:offset + input_lens[2 * i] - 1].sum())
            offset += input_lens[2 * i] - 1
            logprob_sum.append(logprobs[offset + prompt_lens[i] - 1:offset + input_lens[2 * i + 1] - 1].sum())
            offset += input_lens[2 * i + 1] - 1
        assert offset == sum(input_lens) - input_lens.shape[0], (offset, sum(input_lens), input_lens.shape)

        return from_dict(dict(seqlogp=torch.stack(logprob_sum).cpu()))

    def train_step(self, model: api.model.Model, data: NamedArray) -> NamedArray:
        data = recursive_apply(data, lambda x: x.to(model.device))

        packed_input_ids: torch.Tensor = data['packed_input_ids']
        input_lens: torch.Tensor = data['pair_input_lens']
        prompt_lens: torch.IntTensor = data['prompt_lens']
        cu_seqlens = torch.cat([input_lens.new_zeros(1), input_lens.cumsum(0)], 0)
        max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))

        module: deepspeed.DeepSpeedEngine = model.module
        module.eval()

        logits: torch.FloatTensor = module(packed_input_ids=packed_input_ids,
                                           cu_seqlens=cu_seqlens,
                                           max_seqlen=max_seqlen).logits.float()
        logprobs = gather_packed_shifted_log_probs(logits, cu_seqlens, data['packed_input_ids'])

        logprob_sum = []
        offset = 0
        for i in range(prompt_lens.shape[0]):
            logprob_sum.append(logprobs[offset + prompt_lens[i] - 1:offset + input_lens[2 * i] - 1].sum())
            offset += input_lens[2 * i] - 1
            logprob_sum.append(logprobs[offset + prompt_lens[i] - 1:offset + input_lens[2 * i + 1] - 1].sum())
            offset += input_lens[2 * i + 1] - 1
        assert offset == sum(input_lens) - input_lens.shape[0], (offset, sum(input_lens), input_lens.shape)

        pi_seqlogp = torch.stack(logprob_sum)
        ref_seqlogp = data['pair_ref_seqlogp']

        loss, kl_rewards = dpo_functional.dpo_loss(pi_logps=pi_seqlogp, ref_logps=ref_seqlogp, beta=self.beta)

        loss = loss.mean()
        module.backward(loss)
        module.step()

        cur_epoch = model.version.epoch
        model.inc_version()
        if model.version.epoch > cur_epoch:
            module.tput_timer.update_epoch_count()

        return dict(
            loss=loss.detach().item(),
            kl_rewards=kl_rewards.detach().mean().item(),
        )

    def save(self, model: api.model.Model, output_dir):
        if not self.enable_save:
            return
        from impl.model.nn.lora import is_lora_model
        save_hf_or_lora_model(model, output_dir)
        if is_lora_model(model.module):
            save_path = os.path.abspath(
                os.path.join(
                    output_dir,
                    f"epoch{model.version.epoch}step{model.version.epoch_step}",
                ))
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.module.module.head.state_dict(), os.path.join(save_path, "rw_v_head.bin"))


api.model.register_interface("flash_dpo", PackedDirectPerferenceOptimizationInterface)
