from typing import Dict, Optional
import dataclasses
import itertools

try:
    from flash_attn.bert_padding import unpad_input
except ModuleNotFoundError:
    pass

import os

import torch

from base.namedarray import from_dict, NamedArray, recursive_apply
from impl.model.backend.pipe_engine import DeepSpeedPipelineEngine, StreamPipeEngine
from impl.model.utils.data import gather_packed_shifted_log_probs
from impl.model.utils.save import save_pipeline_model
import api.huggingface
import api.model
import base.logging as logging
import impl.model.nn.flash_mqat as flash_mqat
import impl.model.utils.ppo_functional as ppo_functional

logger = logging.getLogger("PipePackedPPOInterface")


def _ppo_actor_step(
        logits: torch.FloatTensor,  # [tot_seqlen, vocab_size]
        packed_input_ids: torch.LongTensor,  # [tot_seqlen]
        cu_seqlens: torch.LongTensor,  # [bs+1] 
        old_logp: torch.FloatTensor,  # [tot_seqlen-bs]
        ref_logp: torch.FloatTensor,  # [tot_seqlen-bs]
        reward_score: torch.FloatTensor,  # [bs]
        values: torch.FloatTensor,  # [tot_seqlen]
        prompt_mask: torch.FloatTensor,  # [tot_seqlen]
        seq_no_eos_mask: torch.FloatTensor,  # [bs]
        pad_token_id: int,  # const
        eos_token_id: int,  # const
        kl_adapter_value: int,  # const
        max_reward_clip: int,  # const
        discount: int,  # const
        gae_lambda: int,  # const
        eps_clip: int,  # const
        logits_mask: Optional[torch.BoolTensor] = None,  # [tot_seqlen, vocab_size]
        **kwargs):
    """ Loss function for ppo actor step, all inputs should be splitted into pipeline micro batches,
    returns loss and logging stats.
    """
    # input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    short1cu_seqlens = cu_seqlens.clone()
    short1cu_seqlens[1:] -= torch.ones_like(cu_seqlens[1:]).cumsum(0)
    if logits_mask is not None:
        logits.masked_fill_(logits_mask.logical_not(), torch.finfo(logits.dtype).min)

    new_logp = gather_packed_shifted_log_probs(logits, cu_seqlens, packed_input_ids)
    kl_rewards, rewards = ppo_functional.get_packed_rewards(
        kl_ctl=kl_adapter_value,
        clip_reward_value=max_reward_clip,
        log_probs=old_logp,
        ref_log_probs=ref_logp,
        reward_score=reward_score,
        short1cu_seqlens=short1cu_seqlens,
        seq_no_eos_mask=seq_no_eos_mask,
    )
    advantages, _ = ppo_functional.get_packed_advantages_and_returns(
        gamma=discount,
        lam=gae_lambda,
        values=values,
        rewards=rewards,
        short1cu_seqlens=short1cu_seqlens,
        seq_no_eos_mask=seq_no_eos_mask,
    )
    loss_mask = (1 - prompt_mask.float()) * (packed_input_ids != pad_token_id).logical_and(
        packed_input_ids != eos_token_id).float()
    shift_one_indices = torch.cat([
        torch.arange(cu_seqlens[i] + 1, cu_seqlens[i + 1], dtype=torch.long, device=cu_seqlens.device)
        for i in range(cu_seqlens.shape[0] - 1)
    ])
    loss_mask = loss_mask[shift_one_indices]

    loss, loss_stat = ppo_functional.actor_loss_fn(logprobs=new_logp,
                                                   old_logprobs=old_logp,
                                                   advantages=advantages,
                                                   eps_clip=eps_clip,
                                                   loss_mask=loss_mask)

    # mean_ref_kl = (kl_rewards.detach() * loss_mask).sum() / loss_mask.sum()
    importance_weight = loss_stat['importance_weight']
    clip_ratio = loss_stat['clip_ratio']
    approx_kl = ((old_logp - new_logp).detach() * loss_mask).sum() / loss_mask.sum()
    stats = dict(
        task_reward=reward_score.mean().detach(),
        kl_reward=(kl_rewards.detach() * loss_mask).sum().mean(),
        ppo_approx_kl=approx_kl,
        cur_kl_ctl=torch.tensor(kl_adapter_value).to(approx_kl),
        advantage=advantages.mean().detach(),
        actor_loss=loss.detach(),
        actor_clip_ratio=clip_ratio,
        importance_weight=importance_weight,
    )
    if logits_mask is not None:
        stats['ignoring_logits_ratio'] = (1 - logits_mask).float().mean()
    return loss, stats


@dataclasses.dataclass
class PipePackedActorInterface(api.model.ModelInterface):

    generation_config: Optional[Dict] = None

    mini_batch_size: int = 8

    kl_ctl: float = 0.1

    discount: float = 1.0
    gae_lambda: float = 1.0

    eps_clip: float = 0.2
    value_eps_clip: float = 0.2
    max_reward_clip: float = 5.0

    early_stop_kl: Optional[float] = None  # e.g. 0.1
    early_stop_imp_ratio: Optional[float] = None  # e.g., 10.0

    adaptive_kl_ctl: bool = False
    adaptive_kl_target: Optional[float] = 6
    adaptive_kl_horizon: Optional[float] = 10000

    force_no_logits_mask: bool = False

    def __post_init__(self):
        if self.adaptive_kl_ctl:
            assert self.adaptive_kl_target is not None
            assert self.adaptive_kl_horizon is not None
            self.kl_adapter = ppo_functional.AdaptiveKLController(self.kl_ctl, self.adaptive_kl_target,
                                                                  self.adaptive_kl_horizon)
        else:
            self.kl_adapter = ppo_functional.FixedKLController(self.kl_ctl)
        self.kl_ctl = None

    def save(self, model: api.model.Model, save_dir: str):
        # os.makedirs(save_dir, exist_ok=True)
        # model.module.save(save_dir)
        save_pipeline_model(model, save_dir)

    @torch.no_grad()
    def generate(self, model: api.model.Model, data: NamedArray) -> NamedArray:
        module = model.module
        assert isinstance(module, DeepSpeedPipelineEngine) or isinstance(module, StreamPipeEngine)

        module.eval()

        data = recursive_apply(data, lambda x: x.to(model.device))
        prompts: torch.LongTensor = data['prompts']
        prompt_att_mask: torch.BoolTensor = data['prompt_att_mask']
        bs, prompt_max_len = prompts.shape[:2]

        packed_input_ids, _, cu_seqlens, _ = unpad_input(prompts, prompt_att_mask)

        res = module.generate(
            tokenizer=model.tokenizer,
            packed_input_ids=packed_input_ids,
            cu_seqlens=cu_seqlens,
            gconfig=flash_mqat.GenerationConfig(**self.generation_config),
        )

        if res is None:  # if not pipeline last stage, module.generate return nothing.
            return None

        # pipeline last stage, post process generate results
        gen_tokens, logprobs, logits_mask, *_ = res

        pad_token_id = model.tokenizer.pad_token_id
        eos_token_id = model.tokenizer.eos_token_id
        seq_no_eos_mask = (gen_tokens[:, -1] != eos_token_id).logical_and(gen_tokens[:, -1] != pad_token_id)
        # We also want gen_lengths to include the eos token, where the reward model outputs a score for this sequence.
        gen_lengths = (gen_tokens != pad_token_id).logical_and(gen_tokens != eos_token_id).sum(dim=-1) + 1
        gen_lengths = gen_lengths.clip(max=gen_tokens.shape[-1])

        prompt_lengths = prompt_att_mask.sum(1)

        # Pack generated sequences and logprobs.
        prompts_list, prompt_log_probs_list, prompt_logits_mask_list = [], [], []
        gen_tokens_list, gen_log_probs_list, gen_logits_mask_list = [], [], []
        for i in range(bs):
            prompt_len, gen_len = prompt_lengths[i].item(), gen_lengths[i].item()

            # Prompts are left-padded. Besides, prompt_log_probs is one-step shorter than prompts.
            prompts_list.append(prompts[i, prompt_max_len - prompt_len:])
            prompt_log_probs_list.append(logprobs.new_zeros(prompt_len - 1))
            if logits_mask is not None and not self.force_no_logits_mask:
                # logits mask is NOT one-step shorter because it directly operates on logits outputed by the model.
                prompt_logits_mask_list.append(logits_mask.new_ones((prompt_len, logits_mask.shape[-1])))

            # Generated tokens are right-padded.
            gen_tokens_list.append(gen_tokens[i, :gen_len])
            gen_log_probs_list.append(logprobs[i, :gen_len])
            if logits_mask is not None and not self.force_no_logits_mask:
                gen_logits_mask_list.append(logits_mask[i, :gen_len])

        # For complete sequences, EOS token is included. Otherwise the sequence may end with arbitrary token.
        # cu_seqlens marks the boundary of these sequences, no matter whether they are complete or not.
        packed_seq = torch.cat(list(itertools.chain.from_iterable(zip(prompts_list, gen_tokens_list))))
        seq_lengths = prompt_lengths + gen_lengths
        cu_seqlens = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=seq_lengths.device),
             seq_lengths.cumsum(0)])
        packed_logprobs = torch.cat(
            list(itertools.chain.from_iterable(zip(prompt_log_probs_list, gen_log_probs_list))))
        assert packed_seq.shape[0] == packed_logprobs.shape[0] + bs, (packed_seq.shape, packed_logprobs.shape,
                                                                      bs)
        packed_logits_mask = None
        if not self.force_no_logits_mask and gen_logits_mask_list:
            packed_logits_mask = torch.cat(
                list(itertools.chain.from_iterable(zip(prompt_logits_mask_list, gen_logits_mask_list))))

        prompt_mask = zip(
            [torch.ones(plen, dtype=torch.bool, device=model.device) for plen in prompt_lengths],
            [torch.zeros(glen, dtype=torch.bool, device=model.device) for glen in gen_lengths],
        )
        prompt_mask = torch.cat(list(itertools.chain.from_iterable(prompt_mask)))

        res = dict(
            seq_no_eos_mask=seq_no_eos_mask,
            packed_seq=packed_seq,
            cu_seqlens=cu_seqlens,
            packed_logprobs=packed_logprobs.float(),
            packed_logits_mask=packed_logits_mask if not self.force_no_logits_mask else None,
            prompt_mask=prompt_mask,
        )
        return recursive_apply(from_dict(res), lambda x: x.cpu())

    @torch.no_grad()
    def inference(self, model: api.model.Model, data: NamedArray) -> NamedArray:
        module = model.module
        module.eval()
        data = recursive_apply(data, lambda x: x.to(model.device))

        cu_seqlens = data['cu_seqlens']
        input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = int(max(input_lens))

        res = module(packed_input_ids=data['packed_seq'], cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        if res is None:  # if not pipeline last stage, module.generate return nothing.
            return None

        logits: torch.FloatTensor = res.logits.float()

        if data['packed_logits_mask'] is not None:
            logits.masked_fill_(data['packed_logits_mask'].logical_not(), torch.finfo(logits.dtype).min)
        logprobs = gather_packed_shifted_log_probs(logits, cu_seqlens, data['packed_seq'])
        return from_dict(dict(logprobs=logprobs.cpu()))

    def train_step(self, model: api.model.Model, data: NamedArray) -> Dict:
        module = model.module
        tokenizer = model.tokenizer
        assert isinstance(module, DeepSpeedPipelineEngine) or isinstance(module, StreamPipeEngine)
        # We call module.eval() because dropout causes the computation of incorrect of log probs.
        module.eval()
        data = recursive_apply(data, lambda x: x.to(model.device))

        module.set_version_steps(model.version.global_step)

        cu_seqlens = data['cu_seqlens']
        input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        loss_fn_kwargs = dict(
            input_lens=input_lens,
            old_logp=data['packed_logprobs'],
            ref_logp=data['packed_ref_logprobs'],
            reward_score=data['rewards'],
            values=data['values'],
            prompt_mask=data['prompt_mask'],
            seq_no_eos_mask=data['seq_no_eos_mask'],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            kl_adapter_value=self.kl_adapter.value,
            max_reward_clip=self.max_reward_clip,
            discount=self.discount,
            gae_lambda=self.gae_lambda,
            eps_clip=self.eps_clip,
            logits_mask=data['packed_logits_mask'],
        )

        r = module.train_batch(
            packed_input_ids=data['packed_seq'],
            cu_seqlens=data['cu_seqlens'],
            loss_fn=_ppo_actor_step,
            **loss_fn_kwargs,
        )

        cur_epoch = model.version.epoch
        model.inc_version()
        if model.version.epoch > cur_epoch:
            module.tput_timer.update_epoch_count()

        if r is None:
            return dict()
        else:
            avg_loss, train_stats = r
            return {"loss": avg_loss}


api.model.register_interface('pipe_flash_actor', PipePackedActorInterface)
