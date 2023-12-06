import dataclasses
import os

import deepspeed
import torch
import torch.utils.data

from base.namedarray import from_dict, NamedArray, recursive_apply
from impl.model.utils.functional import gather_shifted_log_probs
from impl.model.utils.save_load import save_hf_or_lora_model
import api.model
import impl.model.utils.dpo_functional as dpo_functional


@dataclasses.dataclass
class DirectPreferenceOptimizationInterface(api.model.ModelInterface):
    beta: float
    enable_save: bool = True

    @torch.inference_mode()
    def inference(self, model: api.model.Model, data: NamedArray) -> NamedArray:
        module = model.module
        module.eval()
        data = recursive_apply(data, lambda x: x.to(model.device))

        input_ids: torch.LongTensor = torch.cat([data['pos_input_ids'], data['neg_input_ids']], dim=0)
        attention_mask: torch.BoolTensor = torch.cat([data['pos_attention_mask'], data['neg_attention_mask']],
                                                     dim=0)

        logits: torch.FloatTensor = module(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        logp = gather_shifted_log_probs(logits, input_ids)

        logp_mask = attention_mask.clone()
        for i, prompt_len in enumerate(data['prompt_lens']):
            logp_mask[i, :prompt_len] = 0
        logp = logp * logp_mask[:, 1:]

        # returned shape: [n_pos_seqs, 2]
        return from_dict(dict(seqlogp=logp.sum(1).view(2, -1).transpose(0, 1).cpu()))

    def train_step(self, model: api.model.Model, data: NamedArray) -> NamedArray:
        data = recursive_apply(data, lambda x: x.to(model.device))

        module: deepspeed.DeepSpeedEngine = model.module
        module.eval()

        input_ids: torch.LongTensor = torch.cat([data['pos_input_ids'], data['neg_input_ids']], dim=0)
        attention_mask: torch.BoolTensor = torch.cat([data['pos_attention_mask'], data['neg_attention_mask']],
                                                     dim=0)
        logits: torch.FloatTensor = module(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        logp = gather_shifted_log_probs(logits, input_ids)

        logp_mask = attention_mask.clone()
        for i, prompt_len in enumerate(data['prompt_lens']):
            logp_mask[i, :prompt_len] = 0
        seqlogp = (logp * logp_mask[:, 1:]).sum(1).view(2, -1).transpose(0, 1).flatten()

        ref_seqlogp = data['ref_seqlogp'].flatten()

        loss, kl_rewards = dpo_functional.dpo_loss(pi_logps=seqlogp, ref_logps=ref_seqlogp, beta=self.beta)

        loss = loss.mean()
        module.backward(loss)
        module.step()

        cur_epoch = model.version.epoch
        model.inc_version()
        if model.version.epoch > cur_epoch:
            module.tput_timer.update_epoch_count()

        return dict(loss=loss.detach().item(), kl_rewards=kl_rewards.mean().item())

    def save(self, model: api.model.Model, output_dir):
        if not self.enable_save:
            return
        save_hf_or_lora_model(model, output_dir)


api.model.register_interface("dpo", DirectPreferenceOptimizationInterface)
