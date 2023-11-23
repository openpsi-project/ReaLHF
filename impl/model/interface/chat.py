from typing import Callable, Dict, List, Literal, Optional, Union
import collections
import dataclasses
import functools
import math
import os
import re
import time

import deepspeed
import torch
import torch.nn as nn
import torch.utils.data
import tqdm
import transformers

from base.namedarray import from_dict, NamedArray, recursive_aggregate, recursive_apply
from impl.model.utils.data import get_eos_indices, masked_normalization
from impl.model.utils.save import save_hf_or_lora_model
import api.huggingface
import api.model
import base.logging as logging

logger = logging.getLogger("Chat Interface")


def gather_shifted_log_probs(logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
    logits = logits[:, :-1]
    labels = labels[:, 1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


def generate_logits_ignoring_mask(logits: torch.FloatTensor,
                                  top_p: Optional[float] = 1.0,
                                  top_k: Optional[int] = -1) -> torch.BoolTensor:
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


def actor_loss_fn(logprobs: torch.FloatTensor, old_logprobs: torch.FloatTensor, advantages: torch.FloatTensor,
                  loss_mask: torch.FloatTensor, eps_clip: float):
    # clone inference tensors
    old_logprobs = old_logprobs.clone()
    advantages = advantages.clone()

    ratio = torch.exp((logprobs - old_logprobs) * loss_mask)
    clipped_ratio = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    pg_loss = (torch.max(pg_loss1, pg_loss2) * loss_mask).sum() / loss_mask.sum()
    proportion_clipped = (pg_loss1 < pg_loss2)
    proportion_clipped = (proportion_clipped.float() * loss_mask).sum() / loss_mask.sum()
    return pg_loss, proportion_clipped, (ratio.detach() * loss_mask).sum() / loss_mask.sum()


def critic_loss_fn(value: torch.FloatTensor,
                   old_value: torch.FloatTensor,
                   target_value: torch.FloatTensor,
                   loss_mask: torch.FloatTensor,
                   value_eps_clip: float,
                   loss_fn_type: str = 'huber') -> torch.FloatTensor:

    if loss_fn_type == 'huber':
        loss_fn = functools.partial(torch.nn.functional.huber_loss, reduction='none', delta=10.0)
    elif loss_fn_type == 'mse':
        loss_fn = functools.partial(torch.nn.functional.mse, reduction='none')
    else:
        raise NotImplementedError(f"Unknown loss fn type: {loss_fn_type}")

    target_value = target_value.clone()  # clone a inference tensor
    value_loss_original = loss_fn(value, target_value)
    value_clipped = old_value + (value - old_value).clamp(-value_eps_clip, value_eps_clip)
    value_loss_clipped = loss_fn(value_clipped, target_value)
    value_loss = torch.max(value_loss_original, value_loss_clipped)
    proportion_clipped = (value_loss_clipped > value_loss_original)
    proportion_clipped = (proportion_clipped.float() * loss_mask).sum() / loss_mask.sum()
    return (value_loss * loss_mask).sum() / loss_mask.sum(), proportion_clipped


@torch.no_grad()
def compute_rewards(kl_ctl: float, clip_reward_value: float, log_probs: torch.FloatTensor,
                    ref_log_probs: torch.FloatTensor, reward_score: torch.FloatTensor,
                    eos_indices: torch.LongTensor, seq_no_eos_mask: torch.FloatTensor):
    kl_rewards = -kl_ctl * (log_probs - ref_log_probs)
    for i in range(kl_rewards.shape[0]):
        kl_rewards[i, eos_indices[i]:] = 0.0
    score_rewards = torch.zeros_like(kl_rewards)
    reward_clip = torch.clamp(reward_score, -clip_reward_value, clip_reward_value)
    # This is assigned to the token before EOS, which rewards the output of the EOS token.
    score_rewards.scatter_(-1, (eos_indices - 1).unsqueeze(-1), reward_clip.unsqueeze(-1))
    score_rewards = score_rewards * (1 - seq_no_eos_mask.unsqueeze(1))  # only compute final rewards with EOS
    return kl_rewards, kl_rewards + score_rewards


@torch.no_grad()
def get_advantages_and_returns(gamma: float, lam: float, values: torch.FloatTensor,
                               rewards: torch.FloatTensor):
    # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
    assert values.shape[1] == rewards.shape[1] + 1
    lastgaelam = 0
    advantages_reversed = []
    length = rewards.size()[-1]
    for t in reversed(range(length)):
        nextvalues = values[:, t + 1]
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values[:, :-1]
    return advantages.detach(), returns


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = torch.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value = self.value * mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass


@dataclasses.dataclass
class ChatActorInterface(api.model.ModelInterface):
    mini_batch_size: int = 8
    ppo_epochs: int = 1
    kl_ctl: float = 0.1
    discount: float = 1.0
    gae_lambda: float = 0.95
    eps_clip: float = 0.2
    value_eps_clip: float = 0.2
    max_reward_clip: float = 5.0
    early_stop_kl: Optional[float] = None  # by default, 0.1
    early_stop_imp_ratio: Optional[float] = None  # by default, 10.0
    adaptive_kl_ctl: bool = False
    adaptive_kl_target: Optional[float] = 6
    adaptive_kl_horizon: Optional[float] = 10000

    def __post_init__(self):
        if self.adaptive_kl_ctl:
            assert self.adaptive_kl_target is not None
            assert self.adaptive_kl_horizon is not None
            self.kl_adapter = AdaptiveKLController(self.kl_ctl, self.adaptive_kl_target,
                                                   self.adaptive_kl_horizon)
        else:
            self.kl_adapter = FixedKLController(self.kl_ctl)
        self.kl_ctl = None

    @torch.no_grad()
    def generate(self, model: api.model.Model, data: NamedArray) -> NamedArray:
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

        # TODO: fix generation config
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

        # module.eval()
        # TODO: optimize this
        logits: torch.FloatTensor = module(input_ids=seq, attention_mask=attention_mask).logits.float()
        # logits_ignoring_mask = generate_logits_ignoring_mask(logits, module.generation_config.top_p,
        #                                                      module.generation_config.top_k)
        # logits.masked_fill_(logits_ignoring_mask.bool(), torch.finfo(logits.dtype).min)
        logp = gather_shifted_log_probs(logits, seq)

        res = from_dict(
            dict(
                seq=seq,
                attention_mask=attention_mask,
                logp=logp,
                # logits_ignoring_mask=logits_ignoring_mask,
            ),)
        res = recursive_apply(res, lambda x: x.cpu())
        return res

    @torch.no_grad()
    def inference(self, model: api.model.Model, data: NamedArray) -> NamedArray:
        module = model.module
        module.eval()
        data = recursive_apply(data, lambda x: x.to(model.device))
        logits: torch.FloatTensor = module(input_ids=data['input_ids'],
                                           attention_mask=data['attention_mask']).logits.float()
        # logits.masked_fill_(data['logits_ignoring_mask'].bool(), torch.finfo(logits.dtype).min)
        logp = gather_shifted_log_probs(logits, data['input_ids'])
        return from_dict(dict(logp=logp.cpu()))

    def _ppo_actor_step(self, ppo_epoch: int, module: api.model.NeuralNetwork,
                        tokenizer: transformers.PreTrainedTokenizerFast, sample: NamedArray,
                        version_steps: int) -> Dict:
        mini_bs = sample.length(0)

        # st = time.perf_counter()
        # module.train()
        # module.eval()
        t0 = time.perf_counter()
        # logits_ignoring_mask = sample['logits_ignoring_mask']
        new_logits: torch.FloatTensor = module(input_ids=sample['input_ids'],
                                               attention_mask=sample['attention_mask'],
                                               use_cache=False).logits  # .float()
        t1 = time.perf_counter()
        # new_logits.masked_fill_(logits_ignoring_mask.bool(), torch.finfo(new_logits.dtype).min)
        new_logp = gather_shifted_log_probs(new_logits, sample['input_ids'])

        old_logp: torch.Tensor = sample['logp']
        ref_logp: torch.Tensor = sample['ref_logp']

        prompt_len = sample['prompts'].size()[-1]
        shifted_start = prompt_len - 1
        loss_mask = sample['attention_mask'][:, 1:].clone()
        loss_mask[:, :shifted_start] = 0

        eos_indices, seq_no_eos_mask = get_eos_indices(sample['input_ids'][:, prompt_len:], tokenizer)
        eos_indices = eos_indices + prompt_len
        for i in range(eos_indices.shape[0]):
            if not seq_no_eos_mask[i]:
                loss_mask[i, eos_indices[i] - 1] = 1

        kl_rewards, rewards = compute_rewards(self.kl_adapter.value, self.max_reward_clip, old_logp, ref_logp,
                                              sample['rewards'], eos_indices, seq_no_eos_mask)
        advantages, returns = get_advantages_and_returns(self.discount, self.gae_lambda,
                                                         sample['values'][:, shifted_start:],
                                                         rewards[:, shifted_start:])

        mean_ref_kl = (kl_rewards.detach() * loss_mask).sum() / loss_mask.sum()
        self.kl_adapter.update(mean_ref_kl, n_steps=torch.distributed.get_world_size() * mini_bs)

        # adv_norm = masked_normalization(advantages)
        adv_norm = advantages

        loss, clip_ratio, importance_weight = actor_loss_fn(new_logp[:, shifted_start:],
                                                            old_logp[:, shifted_start:], adv_norm,
                                                            loss_mask[:, shifted_start:], self.eps_clip)

        # if self.early_stop_imp_ratio is not None and importance_weight > self.early_stop_imp_ratio:
        #     logger.warning(f"Current importance ratio {importance_weight.item():.4f} is larger "
        #                    f"than early stop threshold {self.early_stop_imp_ratio}. Abandon this minibatch.")
        #     loss = loss * 0.0

        # prompts: torch.LongTensor = sample['prompts']
        # ans: torch.LongTensor = sample['input_ids'][:, prompt_len:]
        # prompt_non_pad_ratio = (prompts != tokenizer.pad_token_id).float().mean()
        # prompt_truncate_ratio = (prompts[:, 0] != tokenizer.pad_token_id).float().mean()
        # generated_non_pad_ratio = (ans != tokenizer.pad_token_id).float().mean()
        # generated_truncate_ratio = (ans[:, -1] != tokenizer.pad_token_id).float().mean()

        # ignoring_logits_ratio = logits_ignoring_mask.float().mean()

        # approx_kl = ((old_logp[:, shifted_start:] - new_logp[:, shifted_start:].detach()) *
        #              loss_mask[:, shifted_start:]).sum() / loss_mask[:, shifted_start:].sum()

        # stats = dict(
        #     task_reward=sample['rewards'].mean().detach(),
        #     kl_reward=(kl_rewards.detach() * loss_mask).sum(1).mean(),
        #     ppo_approx_kl=approx_kl,
        #     cur_kl_ctl=torch.tensor(self.kl_adapter.value).to(approx_kl),
        #     advantage=advantages.mean().detach(),
        #     actor_loss=loss.detach(),
        #     actor_clip_ratio=clip_ratio.detach(),
        #     importance_weight=importance_weight.detach(),
        #     prompt_non_pad_ratio=prompt_non_pad_ratio,
        #     prompt_truncate_ratio=prompt_truncate_ratio,
        #     generated_non_pad_ratio=generated_non_pad_ratio,
        #     generated_truncate_ratio=generated_truncate_ratio,
        #     # ignoring_logits_ratio=ignoring_logits_ratio,
        # )

        # if self.early_stop_kl is not None and api.huggingface.get_all_reduce_mean(approx_kl) > self.early_stop_kl:
        #     logger.warning(f"Current approximate KL divergence {approx_kl.item():.4f} is larger "
        #                    f"than early stop threshold {self.early_stop_kl}. Abort actor update.")
        #     return {}

        torch.cuda.synchronize()
        t2 = time.perf_counter()
        module.backward(loss)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        module.step(lr_kwargs={'epoch': version_steps})
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        logger.info(
            f"_ppo_actor_step {t4 - t0:.3f}, forward {t1 - t0:.3f}, loss {t2 - t1:.3f}, backward {t3 - t2:.3f}, opt step {t4 - t3:.3f}"
        )
        return {}

    def train_step(self, model_: api.model.Model, sample: NamedArray) -> Dict:
        # TODO: add imitation learning auxilary loss
        # TODO: add EMA
        model = model_.module
        tokenizer = model_.tokenizer
        model.eval()
        assert sample['input_ids'].shape[0] % self.mini_batch_size == 0
        n_minibatch = sample['input_ids'].shape[0] // self.mini_batch_size

        t0 = time.perf_counter()
        sample = recursive_apply(sample, lambda x: x.to(model_.device))
        t1 = time.perf_counter()
        train_stats = collections.defaultdict(lambda: 0)
        for ppo_i in range(self.ppo_epochs):
            shuffle_indices = torch.randperm(sample['input_ids'].shape[0])
            for mini_bs_i in range(0, sample['input_ids'].shape[0], self.mini_batch_size):
                indices = shuffle_indices[mini_bs_i:mini_bs_i + self.mini_batch_size]
                stats = self._ppo_actor_step(ppo_i, model, tokenizer, sample[indices],
                                             model_.version.global_step)
                # for k, v in stats.items():
                #     train_stats[k] += v
        t2 = time.perf_counter()
        cur_epoch = model_.version.epoch
        model_.inc_version()
        if model_.version.epoch > cur_epoch:
            model.tput_timer.update_epoch_count()

        # train_stats = dict(train_stats)
        # for k, v in train_stats.items():
        #     v = v.detach() / self.ppo_epochs / n_minibatch
        # train_stats[k] = api.huggingface.get_all_reduce_mean(v).item()
        # train_stats[k] = v.item()
        t3 = time.perf_counter()
        # logger.info(f"actor train_step {t3 - t0:.3f}, recursive_apply {t1 - t0:.3f}, ppo_actor_step {t2 - t1:.3f}, other {t3 - t2:.3f}")
        return {}

    def save(self, model: api.model.Model, output_dir):
        # TODO: for benchmark experiment, do not save models
        # save_hf_or_lora_model(model, output_dir)
        pass


api.model.register_interface("chat_actor", ChatActorInterface)


@dataclasses.dataclass
class ChatCriticInterface(api.model.ModelInterface):
    mini_batch_size: int = 8
    ppo_epochs: int = 1
    kl_ctl: float = 0.1
    discount: float = 1.0
    gae_lambda: float = 0.95
    eps_clip: float = 0.2
    value_eps_clip: float = 0.2
    max_reward_clip: float = 5.0
    adaptive_kl_ctl: bool = False
    adaptive_kl_target: Optional[float] = 6
    adaptive_kl_horizon: Optional[float] = 10000

    def __post_init__(self):
        if self.adaptive_kl_ctl:
            assert self.adaptive_kl_target is not None
            assert self.adaptive_kl_horizon is not None
            self.kl_adapter = AdaptiveKLController(self.kl_ctl, self.adaptive_kl_target,
                                                   self.adaptive_kl_horizon)
        else:
            self.kl_adapter = FixedKLController(self.kl_ctl)
        self.kl_ctl = None

    @torch.no_grad()
    def inference(self, model: api.model.Model, data: NamedArray) -> NamedArray:
        module = model.module
        module.eval()
        data = recursive_apply(data, lambda x: x.to(model.device))
        scores = module(input_ids=data['input_ids'], attention_mask=data['attention_mask']).float()

        prompt_len = data['prompts'].shape[1]
        eos_indices, seq_no_eos_mask = get_eos_indices(data['input_ids'][:, prompt_len:], model.tokenizer)
        eos_indices = eos_indices + prompt_len
        for i in range(scores.shape[0]):
            if not seq_no_eos_mask[i]:
                scores[i, eos_indices[i]:] = 0

        return from_dict(dict(scores=scores))

    def _ppo_critic_step(self, ppo_epoch: int, module: api.model.NeuralNetwork,
                         tokenizer: transformers.PreTrainedTokenizerFast, sample: NamedArray,
                         version_steps: int) -> Dict:
        mini_bs = sample.length(0)

        module.eval()
        new_values = module(input_ids=sample['input_ids'],
                            attention_mask=sample['attention_mask'],
                            use_cache=False).float()

        old_logp: torch.Tensor = sample['logp']
        ref_logp: torch.Tensor = sample['ref_logp']

        prompt_len = sample['prompts'].size()[-1]
        shifted_start = prompt_len - 1
        loss_mask = sample['attention_mask'][:, 1:].clone()
        loss_mask[:, :shifted_start] = 0

        eos_indices, seq_no_eos_mask = get_eos_indices(sample['input_ids'][:, prompt_len:], tokenizer)
        eos_indices = eos_indices + prompt_len
        for i in range(eos_indices.shape[0]):
            if not seq_no_eos_mask[i]:
                loss_mask[i, eos_indices[i] - 1] = 1

        old_values = sample['values']
        kl_rewards, rewards = compute_rewards(self.kl_adapter.value, self.max_reward_clip, old_logp, ref_logp,
                                              sample['rewards'], eos_indices, seq_no_eos_mask)
        _, returns = get_advantages_and_returns(self.discount, self.gae_lambda, old_values[:, shifted_start:],
                                                rewards[:, shifted_start:])

        mean_ref_kl = (kl_rewards.detach() * loss_mask).sum() / loss_mask.sum()
        self.kl_adapter.update(mean_ref_kl, n_steps=torch.distributed.get_world_size() * mini_bs)

        loss, clip_ratio = critic_loss_fn(new_values[:, shifted_start:-1], old_values[:, shifted_start:-1],
                                          returns, loss_mask[:, shifted_start:], self.value_eps_clip)

        module.backward(loss)
        module.step(lr_kwargs={'epoch': version_steps})

        return dict(
            value_loss=loss.detach(),
            value_clip_ratio=clip_ratio.detach(),
            values=new_values.mean().detach(),
            returns=returns.mean().detach(),
        )

    def train_step(self, model_: api.model.Model, sample: NamedArray) -> Dict:
        model = model_.module
        tokenizer = model_.tokenizer
        assert sample['input_ids'].shape[0] % self.mini_batch_size == 0
        n_minibatch = sample['input_ids'].shape[0] // self.mini_batch_size

        sample = recursive_apply(sample, lambda x: x.to(model_.device))

        train_stats = collections.defaultdict(lambda: 0)
        for ppo_i in range(self.ppo_epochs):
            shuffle_indices = torch.randperm(sample['input_ids'].shape[0])
            for mini_bs_i in range(0, sample['input_ids'].shape[0], self.mini_batch_size):
                indices = shuffle_indices[mini_bs_i:mini_bs_i + self.mini_batch_size]
                stats = self._ppo_critic_step(ppo_i, model, tokenizer, sample[indices],
                                              model_.version.global_step)
                for k, v in stats.items():
                    train_stats[k] += v

        cur_epoch = model_.version.epoch
        model_.inc_version()
        if model_.version.epoch > cur_epoch:
            model.tput_timer.update_epoch_count()

        train_stats = dict(train_stats)
        for k, v in train_stats.items():
            v = v.detach() / self.ppo_epochs / n_minibatch
            train_stats[k] = api.huggingface.get_all_reduce_mean(v).item()

        return train_stats

    def save(self, model: api.model.Model, output_dir):
        # TODO: for benchmark experiment, do not save models
        # save_hf_or_lora_model(model, output_dir)
        pass


api.model.register_interface('chat_critic', ChatCriticInterface)


@dataclasses.dataclass
class ChatRewardInterface(api.model.ModelInterface):

    def __post_init__(self):
        self.train_total_correct_predictions = 0
        self.train_total_predictions = 0

    @torch.no_grad()
    def inference(self, model: api.model.Model, data: NamedArray) -> NamedArray:
        module = model.module
        module.eval()
        data = recursive_apply(data, lambda x: x.to(model.device))
        scores: torch.FloatTensor = module(input_ids=data['input_ids'],
                                           attention_mask=data['attention_mask']).float()
        prompt_len = data['prompts'].size()[-1]
        eos_indices, _ = get_eos_indices(data['input_ids'][:, prompt_len:], model.tokenizer)
        eos_indices = eos_indices + prompt_len
        chosen_end_scores = scores.gather(-1, eos_indices.unsqueeze(-1)).squeeze(-1)

        ###################### logging ######################
        seq_strs = model.tokenizer.batch_decode(data['input_ids'],
                                                clean_up_tokenization_spaces=False,
                                                skip_special_tokens=True)
        # for seq_str, score in zip(seq_strs, chosen_end_scores):
        #     logger.info(f"reward is {score.item()}, sequence is: {seq_str}")
        #####################################################

        return from_dict(dict(scores=chosen_end_scores.cpu()))

    def train_step(self, model: api.model.Model, batch: NamedArray) -> NamedArray:
        pass

    def save(self, model: api.model.Model, output_dir):
        pass

    @torch.no_grad()
    def evaluate(self, model_: api.model.Model, eval_dataloader: torch.utils.data.DataLoader) -> Dict:
        pass


api.model.register_interface("chat_reward", ChatRewardInterface)
