from typing import Callable, Dict, List, Literal, Optional, Union
import collections
import dataclasses
import functools
import logging
import math
import os
import re

import deepspeed
import torch
import torch.nn as nn
import tqdm
import transformers

from base.namedarray import from_dict, NamedArray, recursive_aggregate, recursive_apply
from impl.model.utils import masked_normalization, save_hf_model
import api.model
import api.utils

logger = logging.getLogger("WPS Actor Critic")


def remove_code_comments(code: str) -> str:
    prompt_line, *code_lines = code.split('\n')
    lines_to_pop = []
    for i in range(len(code_lines)):
        if "//" not in code_lines[i]:
            if i == 0:
                code_lines[i] = code_lines[i].lstrip()
            continue
        code_lines[i] = code_lines[i][:code_lines[i].index("//")].rstrip()
        if not re.match(r".*[0-9a-zA-Z@].*", code_lines[i]):
            lines_to_pop.append(i)
    for j in reversed(lines_to_pop):
        code_lines.pop(j)
    code = '\n'.join([prompt_line] + code_lines)
    assert "//" not in code
    return code


def get_eos_indices(
    input_ids: torch.LongTensor,
    tokenizer: transformers.PreTrainedTokenizerFast,
) -> torch.LongTensor:
    assert not torch.any(input_ids[:, 0] == tokenizer.eos_token_id)
    seq_len = input_ids.shape[1]
    eos_mask = (input_ids == tokenizer.eos_token_id).float()
    seq_no_eos_mask = (eos_mask.sum(1) == 0).float()
    eos_indices = eos_mask.argmax(1)
    eos_indices = (eos_indices * (1 - seq_no_eos_mask) + seq_no_eos_mask * (seq_len - 1)).long()
    return eos_indices


@dataclasses.dataclass
class WPSRewardUnpairedInterface(api.model.ModelInterface):
    remove_code_comments: bool = False
    pos_weight: float = 1.0

    @torch.no_grad()
    def inference(self, model: api.model.Model, data: NamedArray) -> NamedArray:
        module = model.module
        module.eval()
        data = recursive_apply(data, lambda x: x.to(model.device))
        scores: torch.FloatTensor = module(input_ids=data['input_ids'], attention_mask=data['attention_mask'])
        prompt_len = data['prompts'].size()[-1]
        eos_indices = get_eos_indices(data['input_ids'][:, prompt_len:], model.tokenizer) + prompt_len
        chosen_end_scores = scores.gather(-1, eos_indices.unsqueeze(-1)).squeeze(-1)

        ###################### logging ######################
        seq_strs = model.tokenizer.batch_decode(data['input_ids'],
                                                clean_up_tokenization_spaces=False,
                                                skip_special_tokens=True)
        for seq_str, score in zip(seq_strs, chosen_end_scores):
            logger.info(f"reward is {score.item()}, sequence is: {seq_str}")
        #####################################################

        return from_dict(dict(scores=chosen_end_scores))

    def train_step(self, model: api.model.Model, batch: NamedArray) -> NamedArray:
        device = model.device
        rm_model = model.module
        rm_model.train()

        if self.remove_code_comments:
            max_token_len = batch['input_ids'].shape[1]
            seq_strs = model.tokenizer.batch_decode(batch['input_ids'],
                                                    clean_up_tokenization_spaces=False,
                                                    skip_special_tokens=True)
            for j, seq_str in enumerate(seq_strs):
                seq_str = remove_code_comments(seq_str)
                seq_strs[j] = seq_str + model.tokenizer.eos_token

            tokens = model.tokenizer(seq_strs,
                                     max_length=max_token_len,
                                     padding="max_length",
                                     truncation=True,
                                     return_tensors="pt")
            batch['input_ids'] = tokens['input_ids']
            batch['attention_mask'] = tokens['attention_mask']

        batch = recursive_apply(batch, lambda x: x.to(device))
        labels = batch['correctness_labels']
        eos_indices = get_eos_indices(batch['input_ids'], model.tokenizer)

        scores = rm_model(input_ids=batch['input_ids'],
                          attention_mask=batch['attention_mask'],
                          use_cache=False)
        scores = torch.gather(scores, -1, eos_indices.unsqueeze(-1)).squeeze(-1)

        bs = batch['input_ids'].shape[0]
        pos_weight = self.pos_weight
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            scores,
            labels.half(),
            pos_weight=torch.full((bs,), fill_value=pos_weight, dtype=torch.half, device=scores.device),
        )
        rm_model.backward(loss)
        rm_model.step()

        cur_epoch = model.version.epoch
        model.inc_version()
        if model.version.epoch > cur_epoch:
            rm_model.tput_timer.update_epoch_count()

        return dict(loss=loss.detach().item())

    def save(self, model: api.model.Model, output_dir):
        save_hf_model(model, output_dir)

    @torch.no_grad()
    def evaluate(self, model_: api.model.Model, eval_dataloader: torch.utils.data.DataLoader) -> Dict:
        device = model_.device
        model = model_.module

        model.eval()
        correct_predictions = 0
        total_predictions = 0

        for step, batch in enumerate(tqdm.tqdm(eval_dataloader)):
            batch = recursive_apply(from_dict(batch), lambda x: x.to(device))
            labels = batch['correctness_labels']
            eos_indices = get_eos_indices(batch['input_ids'], model_.tokenizer)
            scores = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )
            scores = torch.gather(scores, -1, eos_indices.unsqueeze(-1)).squeeze(-1)
            bs = scores.shape[0]
            correct_predictions += ((scores > 0.0) == labels).sum()
            total_predictions += bs

        return dict(acc=float(correct_predictions / total_predictions))


api.model.register_interface("wps_reward_unpaired", WPSRewardUnpairedInterface)


def gather_shifted_log_probs(logits: torch.FloatTensor, labels: torch.LongTensor):
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    log_probs = torch.nn.functional.log_softmax(logits - logits.max(-1, keepdim=True).values, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def generate_logits_ignoring_mask(logits: torch.FloatTensor,
                                  top_p: Optional[float] = 1.0,
                                  top_k: Optional[int] = -1):
    if top_p is None:
        top_p = 1.0
    if top_k is None:
        top_k = -1
    assert 0 < top_p <= 1.0
    if top_k < 0 or top_k > logits.size(-1):
        top_k = logits.size(-1)
    if top_p == 1.0 and top_k == logits.size(-1):
        return torch.zeros_like(logits, dtype=torch.bool)

    sorted_logits, sorted_indices = torch.sort(logits, descending=False, dim=-1)
    sorted_logits: torch.FloatTensor
    sorted_indices: torch.LongTensor
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
    # scatter sorted tensors to original indexing
    top_p_indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)

    # Remove all tokens with a probability less than the last token of the top-k
    top_k_indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]

    return top_p_indices_to_remove.logical_or(top_k_indices_to_remove).bool()


def actor_loss_fn(logprobs: torch.Tensor, old_logprobs: torch.Tensor, advantages: torch.Tensor,
                  loss_mask: torch.FloatTensor, eps_clip: float):
    ## policy gradient loss
    ratio = torch.exp(logprobs - old_logprobs)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip)
    pg_loss = (torch.max(pg_loss1, pg_loss2) * loss_mask).sum() / loss_mask.sum()
    clip_ratio = ((pg_loss2 > pg_loss1).float().detach() * loss_mask).sum() / loss_mask.sum()
    return pg_loss, clip_ratio.detach(), (ratio.detach() * loss_mask).sum() / loss_mask.sum()


def critic_loss_fn(value: torch.Tensor, old_value: torch.Tensor, target_value: torch.Tensor,
                   loss_mask: torch.FloatTensor, value_eps_clip: float):
    value_loss_original = (value - target_value).pow(2)
    value_clipped = old_value + (value - old_value).clamp(-value_eps_clip, value_eps_clip)
    value_loss_clipped = (value_clipped - target_value).pow(2)
    value_loss = torch.max(value_loss_original, value_loss_clipped)
    return 0.5 * (value_loss * loss_mask).sum() / loss_mask.sum()


@torch.no_grad()
def compute_rewards(kl_ctl: float, clip_reward_value: float, log_probs: torch.FloatTensor,
                    ref_log_probs: torch.FloatTensor, reward_score: torch.FloatTensor,
                    seq_eos_indices: torch.LongTensor):
    eos_indices = seq_eos_indices - 1  # -1 because log_probs is one token shorter than input_ids
    kl_rewards = -kl_ctl * (log_probs - ref_log_probs)
    for i in range(kl_rewards.shape[0]):
        # We also mask the loss at the EOS token because the probability outputed
        # at this position does not matter for the generated sequence.
        kl_rewards[i, eos_indices[i]:] = 0.0
    score_rewards = torch.zeros_like(kl_rewards)
    reward_clip = torch.clamp(reward_score, -clip_reward_value, clip_reward_value)
    # This is assigned to the token before EOS, which rewards the output of the EOS token.
    score_rewards.scatter_(-1, (eos_indices - 1).unsqueeze(-1), reward_clip.unsqueeze(-1))
    return kl_rewards, kl_rewards + score_rewards


@torch.no_grad()
def get_advantages_and_returns(gamma: float, lam: float, values: torch.FloatTensor,
                               rewards: torch.FloatTensor):
    # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
    lastgaelam = 0
    advantages_reversed = []
    length = rewards.size()[-1]
    for t in reversed(range(length)):
        nextvalues = values[:, t + 1] if t < length - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    return advantages.detach(), returns


@dataclasses.dataclass
class WPSActorInterface(api.model.ModelInterface):
    mini_batch_size: int = 8
    ppo_epochs: int = 1
    kl_ctl: float = 0.1
    discount: float = 1.0
    gae_lambda: float = 0.95
    eps_clip: float = 0.2
    value_eps_clip: float = 0.2
    max_reward_clip: float = 5.0

    @torch.no_grad()
    def generate(self, model: api.model.Model, data: NamedArray) -> NamedArray:
        # FIXME: the "answer:" prefix
        module = model.module
        tokenizer = model.tokenizer
        module.eval()
        module = module.module if isinstance(module, deepspeed.DeepSpeedEngine) else module

        assert module.generation_config.pad_token_id == model.tokenizer.pad_token_id
        assert module.generation_config.eos_token_id == model.tokenizer.eos_token_id
        if module.generation_config.max_new_tokens is not None:
            max_token_len = module.generation_config.max_new_tokens + data.prompts.shape[1]
        else:
            max_token_len = module.generation_config.max_length

        data = recursive_apply(data, lambda x: x.to(model.device))
        seq = module.generate(data.prompts,
                              attention_mask=data.prompt_att_mask,
                              generation_config=module.generation_config)

        pad_token_id = model.tokenizer.pad_token_id
        eos_token_id = model.tokenizer.eos_token_id
        pad_length = max_token_len - seq.shape[1]
        if pad_length > 0:
            seq = torch.nn.functional.pad(seq, pad=(0, pad_length), mode='constant', value=pad_token_id)
        attention_mask = torch.logical_and(seq.not_equal(pad_token_id), (seq.not_equal(eos_token_id))).long()

        logits: torch.FloatTensor = module(input_ids=seq, attention_mask=attention_mask).logits
        logits_ignoring_mask = generate_logits_ignoring_mask(logits, module.generation_config.top_p,
                                                             module.generation_config.top_k)
        # FIXME: add logits mask
        # logits.masked_fill_(logits_ignoring_mask.bool(), torch.finfo(logits.dtype).min)
        logp = gather_shifted_log_probs(logits, seq)

        res = from_dict(
            dict(
                debug_id=debug_id,
                seq=seq,
                attention_mask=attention_mask,
                logp=logp,
                logits_ignoring_mask=logits_ignoring_mask,
            ),)
        return recursive_apply(res, lambda x: x.cpu())

    @torch.no_grad()
    def inference(self, model: api.model.Model, data: NamedArray) -> NamedArray:
        module = model.module
        module.eval()
        data = recursive_apply(data, lambda x: x.to(model.device))
        logits = module(input_ids=data['input_ids'], attention_mask=data['attention_mask']).logits
        # logits.masked_fill_(data['logits_ignoring_mask'].bool(), torch.finfo(logits.dtype).min)
        logp = gather_shifted_log_probs(logits, data['input_ids'])
        return from_dict(dict(logp=logp.cpu()))

    def _ppo_actor_step(self, ppo_epoch: int, module: api.model.NeuralNetwork,
                        tokenizer: transformers.PreTrainedTokenizerFast, sample: NamedArray) -> Dict:
        # FIXME:
        module.eval()
        logits_ignoring_mask = sample['logits_ignoring_mask']
        new_logits: torch.FloatTensor = module(input_ids=sample['input_ids'],
                                               attention_mask=sample['attention_mask']).logits
        # FIXME:
        # new_logits.masked_fill_(logits_ignoring_mask.bool(), torch.finfo(new_logits.dtype).min)
        new_logp = gather_shifted_log_probs(new_logits, sample['input_ids'])

        old_logp: torch.Tensor = sample['logp']
        ref_logp: torch.Tensor = sample['ref_logp']

        prompt_len = sample['prompts'].size()[-1]
        shifted_start = prompt_len - 1
        loss_mask = sample['attention_mask'][:, 1:].clone()
        # Mask the probability of prompts. All tokens after EOS (including EOS) are masked, too.
        loss_mask[:, :shifted_start] = 0

        eos_indices = get_eos_indices(sample['input_ids'][:, prompt_len:], tokenizer)
        eos_indices = eos_indices + prompt_len

        kl_rewards, rewards = compute_rewards(self.kl_ctl, self.max_reward_clip, old_logp, ref_logp,
                                              sample['rewards'], eos_indices)
        advantages, _ = get_advantages_and_returns(self.discount, self.gae_lambda,
                                                   sample['values'][:, shifted_start:],
                                                   rewards[:, shifted_start:])
        # adv_norm = masked_normalization(advantages)
        adv_norm = advantages
        adv_norm = torch.cat([torch.zeros_like(sample['values'][:, :shifted_start]), adv_norm], dim=1)

        logits_ignoring_mask = sample['logits_ignoring_mask']
        new_logits: torch.FloatTensor = module(input_ids=sample['input_ids'],
                                               attention_mask=sample['attention_mask']).logits
        # FIXME:
        # new_logits.masked_fill_(logits_ignoring_mask.bool(), torch.finfo(new_logits.dtype).min)
        new_logp = gather_shifted_log_probs(new_logits, sample['input_ids'])

        loss, clip_ratio, importance_weight = actor_loss_fn(new_logp, old_logp, adv_norm, loss_mask,
                                                            self.eps_clip)

        module.backward(loss)
        module.step()

        return dict(
            reward=rewards.mean().detach(),
            kl_reward=kl_rewards.mean().detach(),
            advantage=advantages.mean().detach(),
            clip_ratio=clip_ratio.detach(),
            importance_weight=importance_weight.detach(),
            actor_loss=loss.detach(),
        )

    def train_step(self, model_: api.model.Model, sample: NamedArray) -> Dict:
        # TODO: add imitation learning auxilary loss
        # TODO: add EMA
        model = model_.module
        tokenizer = model_.tokenizer
        model.train()
        assert sample['input_ids'].shape[0] % self.mini_batch_size == 0
        n_minibatch = sample['input_ids'].shape[0] // self.mini_batch_size

        sample = recursive_apply(sample, lambda x: x.to(model_.device))

        train_stats = collections.defaultdict(lambda: 0)
        for ppo_i in range(self.ppo_epochs):
            shuffle_indices = torch.randperm(sample['input_ids'].shape[0])
            for mini_bs_i in range(0, sample['input_ids'].shape[0], self.mini_batch_size):
                indices = shuffle_indices[mini_bs_i:mini_bs_i + self.mini_batch_size]
                stats = self._ppo_actor_step(ppo_i, model, tokenizer, sample[indices])
                for k, v in stats.items():
                    train_stats[k] += v

        cur_epoch = model_.version.epoch
        model_.inc_version()
        if model_.version.epoch > cur_epoch:
            model.tput_timer.update_epoch_count()

        train_stats = dict(train_stats)
        for k, v in train_stats.items():
            v = v.detach() / self.ppo_epochs / n_minibatch
            train_stats[k] = api.utils.get_all_reduce_mean(v).item()

        return train_stats

    def save(self, model: api.model.Model, output_dir):
        save_hf_model(model, output_dir)


api.model.register_interface("wps_actor", WPSActorInterface)


@dataclasses.dataclass
class WPSCriticInterface(api.model.ModelInterface):
    mini_batch_size: int = 8
    ppo_epochs: int = 1
    kl_ctl: float = 0.1
    discount: float = 1.0
    gae_lambda: float = 0.95
    eps_clip: float = 0.2
    value_eps_clip: float = 0.2
    max_reward_clip: float = 5.0

    @torch.no_grad()
    def inference(self, model: api.model.Model, data: NamedArray) -> NamedArray:
        module = model.module
        module.eval()
        data = recursive_apply(data, lambda x: x.to(model.device))
        scores = module(input_ids=data['input_ids'], attention_mask=data['attention_mask'])
        prompt_len = data['prompts'].shape[1]
        eos_indices = get_eos_indices(data['input_ids'][:, prompt_len:], model.tokenizer) + prompt_len
        for i in range(scores.shape[0]):
            scores[i, eos_indices[i]:] = 0
        return from_dict(dict(scores=scores[:, :-1]))

    def _ppo_critic_step(self, ppo_epoch: int, module: api.model.NeuralNetwork,
                         tokenizer: transformers.PreTrainedTokenizerFast, sample: NamedArray) -> Dict:
        new_values = module(input_ids=sample['input_ids'], attention_mask=sample['attention_mask'])[:, :-1]

        old_logp: torch.Tensor = sample['logp']
        ref_logp: torch.Tensor = sample['ref_logp']

        prompt_len = sample['prompts'].size()[-1]
        shifted_start = prompt_len - 1
        loss_mask = sample['attention_mask'][:, 1:].clone()
        # Mask the probability of prompts. All tokens after EOS (including EOS) are masked, too.
        loss_mask[:, :shifted_start] = 0

        eos_indices = get_eos_indices(sample['input_ids'][:, prompt_len:], tokenizer)
        eos_indices = eos_indices + prompt_len

        _, rewards = compute_rewards(self.kl_ctl, self.max_reward_clip, old_logp, ref_logp, sample['rewards'],
                                     eos_indices)
        _, returns = get_advantages_and_returns(self.discount, self.gae_lambda,
                                                sample['values'][:, shifted_start:], rewards[:,
                                                                                             shifted_start:])
        returns = torch.cat([torch.zeros_like(sample['values'][:, :shifted_start]), returns], dim=1)

        new_values = module(input_ids=sample['input_ids'], attention_mask=sample['attention_mask'])[:, :-1]
        loss = critic_loss_fn(new_values, sample['values'], returns, loss_mask, self.value_eps_clip)

        module.backward(loss)
        module.step()

        return dict(loss=loss.detach())

    def train_step(self, model_: api.model.Model, sample: NamedArray) -> Dict:
        model = model_.module
        tokenizer = model_.tokenizer
        model.train()
        assert sample['input_ids'].shape[0] % self.mini_batch_size == 0
        n_minibatch = sample['input_ids'].shape[0] // self.mini_batch_size

        sample = recursive_apply(sample, lambda x: x.to(model_.device))

        train_stats = collections.defaultdict(lambda: 0)
        for ppo_i in range(self.ppo_epochs):
            shuffle_indices = torch.randperm(sample['input_ids'].shape[0])
            for mini_bs_i in range(0, sample['input_ids'].shape[0], self.mini_batch_size):
                indices = shuffle_indices[mini_bs_i:mini_bs_i + self.mini_batch_size]
                stats = self._ppo_critic_step(ppo_i, model, tokenizer, sample[indices])
                for k, v in stats.items():
                    train_stats[k] += v

        cur_epoch = model_.version.epoch
        model_.inc_version()
        if model_.version.epoch > cur_epoch:
            model.tput_timer.update_epoch_count()

        train_stats = dict(train_stats)
        for k, v in train_stats.items():
            v = v.detach() / self.ppo_epochs / n_minibatch
            train_stats[k] = api.utils.get_all_reduce_mean(v).item()

        return train_stats

    def save(self, model: api.model.Model, output_dir):
        save_hf_model(model, output_dir)


api.model.register_interface('wps_critic', WPSCriticInterface)
