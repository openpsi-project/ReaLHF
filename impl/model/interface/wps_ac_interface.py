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
import torch.utils.data
import tqdm
import transformers

from base.namedarray import from_dict, NamedArray, recursive_aggregate, recursive_apply
from impl.model.utils.data import gather_shifted_log_probs, get_eos_indices, masked_normalization
from impl.model.utils.logits_warper import top_k_top_p_logits
from impl.model.utils.save import save_hf_or_lora_model
import api.huggingface
import api.model
import impl.model.utils.ppo_functional as ppo_functional

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


@dataclasses.dataclass
class WPSRewardUnpairedInterface(api.model.ModelInterface):
    remove_code_comments: bool = False
    pos_weight: float = 1.0

    @torch.inference_mode()
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
        for seq_str, score in zip(seq_strs, chosen_end_scores):
            logger.debug(f"reward is {score.item()}, sequence is: {seq_str}")
        #####################################################

        return from_dict(dict(scores=chosen_end_scores.cpu()))

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
        eos_indices, _ = get_eos_indices(batch['input_ids'], model.tokenizer)

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
        from impl.model.nn.lora import is_lora_model
        save_hf_or_lora_model(model, output_dir)
        if is_lora_model(model.module):
            save_path = os.path.abspath(
                os.path.join(
                    output_dir,
                    f"epoch{model.version.epoch}step{model.version.epoch_step}",
                ))
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.module.v_head.state_dict(), os.path.join(save_path, "rw_v_head.bin"))

    @torch.inference_mode()
    def evaluate(self, model_: api.model.Model, eval_dataloader: torch.utils.data.DataLoader) -> Dict:
        device = model_.device
        model = model_.module

        model.eval()
        correct_predictions = 0
        total_predictions = 0

        for step, batch in enumerate(tqdm.tqdm(eval_dataloader)):
            batch = recursive_apply(from_dict(batch), lambda x: x.to(device))
            labels = batch['correctness_labels']
            eos_indices, _ = get_eos_indices(batch['input_ids'], model_.tokenizer)
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
    early_stop_kl: Optional[float] = None
    early_stop_imp_ratio: Optional[float] = None
    adaptive_kl_ctl: bool = False
    adaptive_kl_target: Optional[float] = 6
    adaptive_kl_horizon: Optional[float] = 10000

    def __post_init__(self):
        if self.adaptive_kl_ctl:
            assert self.adaptive_kl_target is not None
            assert self.adaptive_kl_horizon is not None
            self.kl_adapter = ppo_functional.AdaptiveKLController(self.kl_ctl, self.adaptive_kl_target,
                                                                  self.adaptive_kl_horizon)
        else:
            self.kl_adapter = ppo_functional.FixedKLController(self.kl_ctl)
        self.kl_ctl = None

    @torch.inference_mode()
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

        prompt_lens = data.prompt_att_mask.sum(1)
        data = recursive_apply(data, lambda x: x.to(model.device))
        seq = module.generate(
            data.prompts,  # NOTE: left-pad
            attention_mask=data.prompt_att_mask,  # NOTE: left-pad
            generation_config=module.generation_config,
            use_cache=True,
        )

        pad_token_id = model.tokenizer.pad_token_id
        eos_token_id = model.tokenizer.eos_token_id
        pad_length = max_token_len - seq.shape[1]
        if pad_length > 0:
            seq = torch.nn.functional.pad(seq, pad=(0, pad_length), mode='constant', value=pad_token_id)
        # NOTE
        attention_mask = torch.logical_and(seq.not_equal(pad_token_id), (seq.not_equal(eos_token_id))).long()
        eos_indices = attention_mask.shape[1] - torch.argmax((attention_mask.flip(1) != 0).float(), dim=1)
        eos_indices.clip_(max=attention_mask.shape[1] - 1)
        batch_indices = torch.arange(attention_mask.shape[0], device=model.device, dtype=torch.long)
        attention_mask[batch_indices, eos_indices] = 1

        module.eval()
        logits: torch.FloatTensor = module(input_ids=seq, attention_mask=attention_mask).logits.float()
        top_k_top_p_logits(logits,
                           top_k=module.generation_config.top_k,
                           top_p=module.generation_config.top_p,
                           inplace=True,
                           ordered=False)
        bs = prompt_lens.shape[0]
        # NOTE
        if module.generation_config.min_new_tokens is not None and module.generation_config.min_new_tokens > 0:
            prompt_padded_len = data.prompts.shape[1]
            logits[:, prompt_padded_len - 1:prompt_padded_len - 1 + module.generation_config.min_new_tokens,
                   eos_token_id] = torch.finfo(torch.float32).min
        else:  # use min_length instead
            for i in range(bs):
                prompt_len = prompt_lens[i]
                if module.generation_config.min_length > prompt_len:
                    logits[i, prompt_len - 1:module.generation_config.min_length - 1,
                           eos_token_id] = torch.finfo(torch.float32).min
        logits_ignoring_mask = logits == torch.finfo(logits.dtype).min
        logp = gather_shifted_log_probs(logits, seq)

        res = from_dict(
            dict(
                seq=seq,
                attention_mask=attention_mask,
                logp=logp,
                logits_ignoring_mask=logits_ignoring_mask,
            ),)
        return recursive_apply(res, lambda x: x.cpu())

    @torch.inference_mode()
    def inference(self, model: api.model.Model, data: NamedArray) -> NamedArray:
        module = model.module
        module.eval()
        data = recursive_apply(data, lambda x: x.to(model.device))
        logits: torch.FloatTensor = module(input_ids=data['input_ids'],
                                           attention_mask=data['attention_mask']).logits.float()
        if data['logits_ignoring_mask'] is not None:
            logits.masked_fill_(data['logits_ignoring_mask'].bool(), torch.finfo(logits.dtype).min)
        logp = gather_shifted_log_probs(logits, data['input_ids'])
        return from_dict(dict(logp=logp.cpu()))

    def _ppo_actor_step(self, ppo_epoch: int, module: api.model.NeuralNetwork,
                        tokenizer: transformers.PreTrainedTokenizerFast, sample: NamedArray,
                        version_steps: int) -> Dict:
        mini_bs = sample.length(0)

        module.eval()
        logits_ignoring_mask = sample['logits_ignoring_mask']
        new_logits: torch.FloatTensor = module(input_ids=sample['input_ids'],
                                               attention_mask=sample['attention_mask'],
                                               use_cache=False).logits.float()
        if logits_ignoring_mask is not None:
            new_logits.masked_fill_(logits_ignoring_mask.bool(), torch.finfo(new_logits.dtype).min)
        new_logp = gather_shifted_log_probs(new_logits, sample['input_ids'])

        old_logp: torch.Tensor = sample['logp']
        ref_logp: torch.Tensor = sample['ref_logp']

        prompt_len = sample['prompts'].size()[-1]
        loss_mask = sample['attention_mask'].clone()
        loss_mask[:, :prompt_len] = 0
        loss_mask = loss_mask[:, 1:]
        shifted_start = prompt_len - 1

        eos_indices, seq_no_eos_mask = get_eos_indices(sample['input_ids'][:, prompt_len:], tokenizer)
        eos_indices = eos_indices + prompt_len

        kl_rewards, rewards = ppo_functional.compute_rewards(self.kl_adapter.value, self.max_reward_clip,
                                                             old_logp, ref_logp, sample['rewards'],
                                                             eos_indices, seq_no_eos_mask)
        advantages, returns = ppo_functional.get_advantages_and_returns(
            gamma=self.discount,
            lam=self.gae_lambda,
            values=sample['values'][:, shifted_start:],
            rewards=rewards[:, shifted_start:],
            seq_no_eos_mask=seq_no_eos_mask,
        )

        mean_ref_kl = (kl_rewards.detach() * loss_mask).sum() / loss_mask.sum()
        self.kl_adapter.update(mean_ref_kl, n_steps=torch.distributed.get_world_size() * mini_bs)

        # adv_norm = masked_normalization(advantages)
        adv_norm = advantages

        loss, loss_stat = ppo_functional.actor_loss_fn(
            logprobs=new_logp[:, shifted_start:],
            old_logprobs=old_logp[:, shifted_start:],
            advantages=adv_norm,
            loss_mask=loss_mask[:, shifted_start:],
            eps_clip=self.eps_clip,
        )

        importance_weight = loss_stat['importance_weight']
        clip_ratio = loss_stat['clip_ratio']
        if self.early_stop_imp_ratio is not None and importance_weight > self.early_stop_imp_ratio:
            logger.warning(f"Current importance ratio {importance_weight.item():.4f} is larger "
                           f"than early stop threshold {self.early_stop_imp_ratio}. Abandon this minibatch.")
            loss = loss * 0.0

        prompts: torch.LongTensor = sample['prompts']
        ans: torch.LongTensor = sample['input_ids'][:, prompt_len:]
        prompt_non_pad_ratio = (prompts != tokenizer.pad_token_id).float().mean()
        prompt_truncate_ratio = (prompts[:, 0] != tokenizer.pad_token_id).float().mean()
        generated_non_pad_ratio = (ans != tokenizer.pad_token_id).float().mean()
        generated_truncate_ratio = (ans[:, -1] != tokenizer.pad_token_id).float().mean()

        # ignoring_logits_ratio = logits_ignoring_mask.float().mean()

        approx_kl = ((old_logp[:, shifted_start:] - new_logp[:, shifted_start:].detach()) *
                     loss_mask[:, shifted_start:]).sum() / loss_mask[:, shifted_start:].sum()

        stats = dict(
            task_reward=sample['rewards'].mean().detach(),
            kl_reward=(kl_rewards.detach() * loss_mask).sum(1).mean(),
            ppo_approx_kl=approx_kl,
            cur_kl_ctl=torch.tensor(self.kl_adapter.value).to(approx_kl),
            advantage=advantages.mean().detach(),
            actor_loss=loss.detach(),
            actor_clip_ratio=clip_ratio,
            importance_weight=importance_weight,
            prompt_non_pad_ratio=prompt_non_pad_ratio,
            prompt_truncate_ratio=prompt_truncate_ratio,
            generated_non_pad_ratio=generated_non_pad_ratio,
            generated_truncate_ratio=generated_truncate_ratio,
        )

        if logits_ignoring_mask is not None:
            ignoring_logits_ratio = logits_ignoring_mask.float().mean()
            stats['ignoring_logits_ratio'] = ignoring_logits_ratio

        if self.early_stop_kl is not None and api.huggingface.get_all_reduce_mean(
                approx_kl) > self.early_stop_kl:
            logger.warning(f"Current approximate KL divergence {approx_kl.item():.4f} is larger "
                           f"than early stop threshold {self.early_stop_kl}. Abort actor update.")
            return stats

        module.backward(loss)
        module.step(lr_kwargs={'epoch': version_steps})
        return stats

    def train_step(self, model_: api.model.Model, sample: NamedArray) -> Dict:
        # TODO: add imitation learning auxilary loss
        # TODO: add EMA
        model = model_.module
        tokenizer = model_.tokenizer
        model.eval()
        assert sample['input_ids'].shape[0] % self.mini_batch_size == 0
        n_minibatch = sample['input_ids'].shape[0] // self.mini_batch_size

        sample = recursive_apply(sample, lambda x: x.to(model_.device))

        train_stats = collections.defaultdict(lambda: 0)
        for ppo_i in range(self.ppo_epochs):
            shuffle_indices = torch.randperm(sample['input_ids'].shape[0])
            for mini_bs_i in range(0, sample['input_ids'].shape[0], self.mini_batch_size):
                indices = shuffle_indices[mini_bs_i:mini_bs_i + self.mini_batch_size]
                stats = self._ppo_actor_step(ppo_i, model, tokenizer, sample[indices],
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
        save_hf_or_lora_model(model, output_dir)


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
    adaptive_kl_ctl: bool = False
    adaptive_kl_target: Optional[float] = 6
    adaptive_kl_horizon: Optional[float] = 10000

    def __post_init__(self):
        if self.adaptive_kl_ctl:
            assert self.adaptive_kl_target is not None
            assert self.adaptive_kl_horizon is not None
            self.kl_adapter = ppo_functional.AdaptiveKLController(self.kl_ctl, self.adaptive_kl_target,
                                                                  self.adaptive_kl_horizon)
        else:
            self.kl_adapter = ppo_functional.FixedKLController(self.kl_ctl)
        self.kl_ctl = None

    @torch.inference_mode()
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
        new_values: torch.FloatTensor = module(input_ids=sample['input_ids'],
                                               attention_mask=sample['attention_mask'],
                                               use_cache=False).float()

        old_logp: torch.Tensor = sample['logp']
        ref_logp: torch.Tensor = sample['ref_logp']

        prompt_len = sample['prompts'].size()[-1]
        loss_mask = sample['attention_mask'].clone()
        loss_mask[:, :prompt_len] = 0
        loss_mask = loss_mask[:, 1:]
        shifted_start = prompt_len - 1

        eos_indices, seq_no_eos_mask = get_eos_indices(sample['input_ids'][:, prompt_len:], tokenizer)
        eos_indices = eos_indices + prompt_len

        old_values = sample['values']
        kl_rewards, rewards = ppo_functional.compute_rewards(self.kl_adapter.value, self.max_reward_clip,
                                                             old_logp, ref_logp, sample['rewards'],
                                                             eos_indices, seq_no_eos_mask)
        _, returns = ppo_functional.get_advantages_and_returns(gamma=self.discount,
                                                               lam=self.gae_lambda,
                                                               values=old_values[:, shifted_start:],
                                                               rewards=rewards[:, shifted_start:],
                                                               seq_no_eos_mask=seq_no_eos_mask)

        mean_ref_kl = (kl_rewards.detach() * loss_mask).sum() / loss_mask.sum()
        self.kl_adapter.update(mean_ref_kl, n_steps=torch.distributed.get_world_size() * mini_bs)

        loss, loss_stat = ppo_functional.critic_loss_fn(
            value=new_values[:, shifted_start:-1],
            old_value=old_values[:, shifted_start:-1],
            target_value=returns,
            loss_mask=loss_mask[:, shifted_start:],
            value_eps_clip=self.value_eps_clip,
        )
        clip_ratio = loss_stat['clip_ratio']

        module.backward(loss)
        module.step(lr_kwargs={'epoch': version_steps})

        return dict(
            value_loss=loss.detach(),
            value_clip_ratio=clip_ratio,
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
        save_hf_or_lora_model(model, output_dir)


api.model.register_interface('wps_critic', WPSCriticInterface)


@dataclasses.dataclass
class WPSPlackettLuceRewardInterface(api.model.ModelInterface):

    def __post_init__(self):
        self.train_total_correct_predictions = 0
        self.train_total_predictions = 0

    @torch.inference_mode()
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
        for seq_str, score in zip(seq_strs, chosen_end_scores):
            logger.debug(f"reward is {score.item()}, sequence is: {seq_str}")
        #####################################################

        return from_dict(dict(scores=chosen_end_scores.cpu()))

    def train_step(self, model: api.model.Model, batch: NamedArray) -> NamedArray:
        device = model.device
        rm_model = model.module
        rm_model.train()

        batch = recursive_apply(batch, lambda x: x.to(device))
        labels = batch['labels']

        bs, c_dim = batch['input_ids'].shape[:2]
        eos_indices, _ = get_eos_indices(batch['input_ids'].flatten(end_dim=1), model.tokenizer)
        eos_indices = eos_indices.view(bs, c_dim)

        scores: torch.FloatTensor = rm_model(
            input_ids=batch['input_ids'].flatten(end_dim=1),
            attention_mask=batch['attention_mask'].flatten(end_dim=1),
            use_cache=False,
        ).float()  # [bs * c_dim, seq_len]

        scores = scores.view(bs, c_dim, -1)
        scores = scores.gather(-1, eos_indices.unsqueeze(-1)).squeeze(-1)  # [bs, c_dim]

        scores = torch.cat([torch.zeros((bs, 1), dtype=scores.dtype, device=scores.device), scores], dim=1)
        loss = torch.nn.functional.cross_entropy(scores, labels, reduction='mean')
        logger.info(f"scores: {scores}, loss: {loss}.")

        rm_model.backward(loss)
        rm_model.step()

        correct_predictions = (scores.argmax(-1) == labels).float().sum().detach().item()
        self.train_total_correct_predictions += correct_predictions
        self.train_total_predictions += bs
        acc = self.train_total_correct_predictions / self.train_total_predictions

        cur_epoch = model.version.epoch
        model.inc_version()
        if model.version.epoch > cur_epoch:
            rm_model.tput_timer.update_epoch_count()
            self.train_total_predictions = self.train_total_correct_predictions = 0

        return dict(loss=loss.detach().item(), acc=acc)

    def save(self, model: api.model.Model, output_dir):
        from impl.model.nn.lora import is_lora_model
        save_hf_or_lora_model(model, output_dir)
        if is_lora_model(model.module):
            save_path = os.path.abspath(
                os.path.join(
                    output_dir,
                    f"epoch{model.version.epoch}step{model.version.epoch_step}",
                ))
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.module.v_head.state_dict(), os.path.join(save_path, "rw_v_head.bin"))

    @torch.inference_mode()
    def evaluate(self, model_: api.model.Model, eval_dataloader: torch.utils.data.DataLoader) -> Dict:
        device = model_.device
        model = model_.module
        tokenizer = model_.tokenizer

        model.eval()
        correct_predictions = 0
        total_predictions = 0
        loss = 0

        for step, batch in enumerate(tqdm.tqdm(eval_dataloader)):
            batch = recursive_apply(from_dict(batch), lambda x: x.to(device))
            labels = batch['labels']
            bs, c_dim = batch['input_ids'].shape[:2]

            eos_indices, _ = get_eos_indices(batch['input_ids'].flatten(end_dim=1), tokenizer)
            eos_indices = eos_indices.view(bs, c_dim)

            scores: torch.FloatTensor = model(
                input_ids=batch['input_ids'].flatten(end_dim=1),
                attention_mask=batch['attention_mask'].flatten(end_dim=1),
                use_cache=False,
            ).float().view(bs, c_dim, -1)
            scores = scores.gather(-1, eos_indices.unsqueeze(-1)).squeeze(-1)
            scores = torch.cat([torch.zeros((bs, 1), dtype=scores.dtype, device=scores.device), scores],
                               dim=1)
            loss += torch.nn.functional.cross_entropy(scores, labels, reduction='sum')
            correct_predictions += (scores.argmax(-1) == labels).sum()
            total_predictions += bs

        return dict(acc=float(correct_predictions / total_predictions), loss=float(loss / total_predictions))


api.model.register_interface("wps_plackett_luce_reward", WPSPlackettLuceRewardInterface)
