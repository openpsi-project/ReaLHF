from typing import Dict, Optional, Tuple
import functools

import torch


class KLController:

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self):
        raise NotImplementedError()


class AdaptiveKLController(KLController):
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


class FixedKLController(KLController):
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass


def actor_loss_fn(
    logprobs: torch.FloatTensor,
    old_logprobs: torch.FloatTensor,
    advantages: torch.FloatTensor,
    eps_clip: float,
    loss_mask: Optional[torch.BoolTensor] = None,
) -> Tuple[torch.Tensor, Dict]:
    """Compute PPO actor loss function.
    
    There is no shape requirements for the inputs, but they must have the same shape.
    Either [bs, max_seqlen] for batch padded inputs or [tot_seqlen] for padded inputs.

    Args:
        logprobs (torch.FloatTensor): Log probabilities of actions.
        old_logprobs (torch.FloatTensor): Old log probabilities of actions.
        advantages (torch.FloatTensor): GAE (normalized) advantages.
        eps_clip (float): Clip ratio of PPO.
        loss_mask (Optional[torch.BoolTensor], optional): Mask for loss computation.
            1 if valid else 0. Defaults to None.

    Returns:
        Tuple[torch.Tensor, Dict]: Scalar loss and statistics.
    """
    # clone inference tensors
    if old_logprobs.is_inference():
        old_logprobs = old_logprobs.clone()
    if advantages.is_inference():
        advantages = advantages.clone()

    if loss_mask is not None:
        loss_mask_count = loss_mask.count_nonzero()
        # For numerical stability.
        ratio = torch.where(loss_mask, torch.exp(logprobs - old_logprobs), 0)
    else:
        ratio = torch.exp(logprobs - old_logprobs)

    clipped_ratio = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio

    if loss_mask is not None:
        pg_loss = torch.where(loss_mask, torch.max(pg_loss1, pg_loss2), 0).sum() / loss_mask_count
    else:
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    clip_mask = (pg_loss1.detach() < pg_loss2.detach())
    if loss_mask is not None:
        proportion_clipped = clip_mask.logical_and_(loss_mask).count_nonzero() / loss_mask_count
        importance_weight = torch.where(loss_mask, ratio.detach(), 0).sum() / loss_mask_count
    else:
        proportion_clipped = clip_mask.count_nonzero()
        importance_weight = ratio.detach().mean()
    # Remain torch.CudaTensor here for all-reduce after train step.
    stat = dict(clip_ratio=proportion_clipped, importance_weight=importance_weight)

    return pg_loss, stat


def critic_loss_fn(value: torch.FloatTensor,
                   old_value: torch.FloatTensor,
                   target_value: torch.FloatTensor,
                   value_eps_clip: float,
                   loss_mask: Optional[torch.FloatTensor] = None,
                   loss_fn_type: str = 'huber') -> Tuple[torch.Tensor, Dict]:
    """Compute PPO critic loss function given padded batch inputs.

    There is no shape requirements for the inputs, but they must have the same shape.
    Either [bs, max_seqlen] for batch padded inputs or [tot_seqlen] for padded inputs.
    
    Args:
        value (torch.FloatTensor): Values. The position of the final token is not included.
            (The whole generated sequence is not a state.)
        old_value (torch.FloatTensor): Old values.
        target_value (torch.FloatTensor): Returns computed by GAE.
        value_eps_clip (float): Clip ratio.
        loss_mask (Optional[torch.FloatTensor], optional): Mask for loss computation.
            1 if valid else 0. Defaults to None.
        loss_fn_type (str, optional): Type of loss function. Defaults to 'huber'.

    Returns:
        Tuple[torch.Tensor, Dict]: Scalar loss and statistics.
    """
    if loss_fn_type == 'huber':
        loss_fn = functools.partial(torch.nn.functional.huber_loss, reduction='none', delta=10.0)
    elif loss_fn_type == 'mse':
        loss_fn = functools.partial(torch.nn.functional.mse_loss, reduction='none')
    else:
        raise NotImplementedError(f"Unknown loss fn type: {loss_fn_type}")

    if target_value.is_inference():
        target_value = target_value.clone()  # clone a inference tensor

    # TODO: bf16 support for autocast
    with torch.autocast("cuda"):
        value_loss_original = loss_fn(value, target_value)

    value_clipped = old_value + (value - old_value).clamp(-value_eps_clip, value_eps_clip)

    with torch.autocast("cuda"):
        value_loss_clipped = loss_fn(value_clipped, target_value)

    value_loss = torch.max(value_loss_original, value_loss_clipped)

    clip_mask = (value_loss_clipped.detach() > value_loss_original.detach())
    if loss_mask is not None:
        mask_count = loss_mask.count_nonzero()
        proportion_clipped = clip_mask.logical_and_(loss_mask).count_nonzero() / mask_count
    else:
        proportion_clipped = clip_mask.count_nonzero()

    stat = dict(clip_ratio=proportion_clipped)

    if loss_mask is not None:
        value_loss = torch.where(loss_mask, value_loss, 0).sum() / mask_count
    else:
        value_loss = value_loss.mean()

    return value_loss, stat


@torch.no_grad()
def compute_rewards(
    kl_ctl: float,
    clip_reward_value: float,
    log_probs: torch.FloatTensor,
    ref_log_probs: torch.FloatTensor,
    reward_score: torch.FloatTensor,
    eos_indices: torch.LongTensor,
    seq_no_eos_mask: torch.FloatTensor,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """Compute rewards for padded batch inputs.

    Args:
        kl_ctl (float): Coefficient of KL rewards.
        clip_reward_value (float): Clip value of rewards. Preventing extreme reward numbers.
        log_probs (torch.FloatTensor): Log probabilities of actions. Shape [bs, max_seqlen - 1].
        ref_log_probs (torch.FloatTensor): Log probabilities outputed by reference models. Shape [bs, max_seqlen - 1].
        reward_score (torch.FloatTensor): Scores outputed by the reward model. Shape [bs].
        eos_indices (torch.LongTensor): Indices of EOS tokens. Shape [bs].
            Used for adding score rewards onto KL rewards.
        seq_no_eos_mask (torch.FloatTensor): Indicator of no EOS tokens in a sequence. Shape [bs].
            1 if no EOS else 0.

    Returns:
        Tuple[torch.FloatTensor, torch.FloatTensor]: Pure KL rewards and total rewards.
            Both of shape [bs, max_seqlen - 1].
    """
    kl_rewards = -kl_ctl * (log_probs - ref_log_probs)
    for i in range(kl_rewards.shape[0]):
        # Set KL rewards *at EOS* and after EOS to 0.
        # The final *state* is the sequence without EOS, so the final KL reward is assigned to this state.
        # The next "environment step" indicates a "done" by outputting an EOS token, therefore no rewards afterwards.
        kl_rewards[i, eos_indices[i]:] = 0.0

    reward_clip = torch.clamp(reward_score, -clip_reward_value, clip_reward_value)
    score_rewards = torch.zeros_like(kl_rewards)
    # This is assigned to the token before EOS, which rewards the output of the EOS token.
    score_rewards.scatter_(-1, (eos_indices - 1).unsqueeze(-1), reward_clip.unsqueeze(-1))
    score_rewards = score_rewards * (1 - seq_no_eos_mask.unsqueeze(1))  # only compute final rewards with EOS
    return kl_rewards, kl_rewards + score_rewards


@torch.no_grad()
def get_advantages_and_returns(
    gamma: float,
    lam: float,
    values: torch.FloatTensor,
    rewards: torch.FloatTensor,
    seq_no_eos_mask: torch.FloatTensor,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """Compute GAE and returns given padded batch inputs.

    Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
    
    Args:
        gamma (float): Discount factor.
        lam (float): GAE discount factor.
        values (torch.FloatTensor): Values of shape [bs, max_seqlen] (with bootstrapped values).
        rewards (torch.FloatTensor): Rewards of shape [bs, max_seqlen - 1].
        seq_no_eos_mask (torch.FloatTensor): Indicator of no EOS tokens in a sequence. Shape [bs].
            1 if no EOS else 0. Used for masking out bootstrap values for terminated sequences.

    Returns:
        Tuple[torch.FloatTensor, torch.FloatTensor]: GAE and returns of shape [bs, max_seqlen - 1].
    """
    assert values.shape[1] == rewards.shape[1] + 1
    lastgaelam = 0
    advantages_reversed = []
    length = rewards.size()[-1]
    for t in reversed(range(length)):
        nextvalues = values[:, t + 1]
        if t == length - 1:
            nextvalues *= seq_no_eos_mask
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values[:, :-1]
    return advantages, returns


@torch.no_grad()
def get_packed_rewards(
    kl_ctl: float,
    clip_reward_value: float,
    log_probs: torch.FloatTensor,
    ref_log_probs: torch.FloatTensor,
    reward_score: torch.FloatTensor,
    short1cu_seqlens: torch.IntTensor,
    seq_no_eos_mask: torch.BoolTensor,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    # Here log_probs/ref_log_probs is one-step shorter than packed_input_ids (the last step is removed),
    # so the log_probs at the EOS token is not included in this tensor.
    # We directly add reward scores of each sequence onto the final token of each sequence.
    tot_rewards = -kl_ctl * (log_probs - ref_log_probs)
    kl_rewards = tot_rewards.clone()
    reward_score = reward_score.clip(-clip_reward_value, clip_reward_value)
    tot_rewards[short1cu_seqlens[1:] - 1] += torch.where(seq_no_eos_mask, 0,0, reward_score)
    return kl_rewards, tot_rewards


def _pygae1d_nolp_misalign(
    rewards: torch.FloatTensor,
    values: torch.FloatTensor,
    cu_seqlens_: torch.IntTensor,
    bootstrap: torch.FloatTensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    cu_seqlens = cu_seqlens_.clone()
    cu_seqlens[1:] += torch.ones_like(cu_seqlens_[1:]).cumsum(0)

    bs = cu_seqlens_.shape[0] - 1
    assert values.shape[0] == rewards.shape[0] + bs
    advantages_reversed = []
    returns_reversed = []
    for i in reversed(range(bs)):
        v_offset = cu_seqlens[i]
        r_offset, r_end = cu_seqlens_[i], cu_seqlens_[i + 1]
        assert cu_seqlens[i + 1] - v_offset - 1 == r_end - r_offset
        lastgaelam = 0
        for t in reversed(range(r_end - r_offset)):
            nextvalues = values[v_offset + t + 1]
            if t == r_end - r_offset - 1:
                nextvalues *= bootstrap[i]
            delta = rewards[r_offset + t] + gamma * nextvalues - values[v_offset + t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
            returns_reversed.append(lastgaelam + values[v_offset + t])

    advantages = torch.stack(advantages_reversed[::-1])
    returns = torch.stack(returns_reversed[::-1])
    return advantages, returns


@torch.no_grad()
def get_packed_advantages_and_returns(
    gamma: float,
    lam: float,
    values: torch.FloatTensor,
    rewards: torch.FloatTensor,
    short1cu_seqlens: torch.IntTensor,
    seq_no_eos_mask: torch.FloatTensor,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    try:
        import cugae
        return cugae.cugae1d_nolp_misalign_func(rewards, values, short1cu_seqlens.int(),
                                                seq_no_eos_mask.bool(), gamma, lam)
    except ModuleNotFoundError:
        return _pygae1d_nolp_misalign(rewards, values, short1cu_seqlens, seq_no_eos_mask, gamma, lam)
