from typing import Tuple, Dict
import torch
import functools


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


def actor_loss_fn(
    logprobs: torch.FloatTensor,
    old_logprobs: torch.FloatTensor,
    advantages: torch.FloatTensor,
    loss_mask: torch.FloatTensor,
    eps_clip: float,
) -> Tuple[torch.Tensor, Dict]:
    """Compute PPO actor loss function given padded batch inputs.

    Args:
        logprobs (torch.FloatTensor): Log probabilities of actions. Shape [bs, max_seqlen - 1].
        old_logprobs (torch.FloatTensor): Old log probabilities of actions. Shape [bs, max_seqlen - 1].
        advantages (torch.FloatTensor): GAE (normalized) advantages. Shape [bs, max_seqlen - 1].
        loss_mask (torch.FloatTensor): Mask for loss computation. Shape [bs, max_seqlen - 1].
            1 if valid else 0.
        eps_clip (float): Clip ratio of PPO.

    Returns:
        Tuple[torch.Tensor, Dict]: Loss scalar and statistics.
    """
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
    importance_weight = (ratio.detach() * loss_mask).sum() / loss_mask.sum()
    # Remain torch.CudaTensor here for all-reduce after train step.
    stat = dict(clip_ratio=proportion_clipped.detach(), importance_weight=importance_weight.detach())

    return pg_loss, stat


def critic_loss_fn(value: torch.FloatTensor,
                   old_value: torch.FloatTensor,
                   target_value: torch.FloatTensor,
                   loss_mask: torch.FloatTensor,
                   value_eps_clip: float,
                   loss_fn_type: str = 'huber') -> Tuple[torch.Tensor, Dict]:
    """Compute PPO critic loss function given padded batch inputs.

    Args:
        value (torch.FloatTensor): Values of shape [bs, max_seqlen - 1].
            The position of the final token is not included. (The whole generated sequence is not a state.)
        old_value (torch.FloatTensor): Old values of shape [bs, max_seqlen - 1].
        target_value (torch.FloatTensor): Target value computed by GAE of shape [bs, max_seqlen - 1].
        loss_mask (torch.FloatTensor): Mask for loss computation. Shape [bs, max_seqlen - 1].
            1 if valid else 0.
        value_eps_clip (float): Clip ratio.
        loss_fn_type (str, optional): Type of loss function. Defaults to 'huber'.

    Returns:
        Tuple[torch.Tensor, Dict]: Loss scalar and statistics.
    """

    if loss_fn_type == 'huber':
        loss_fn = functools.partial(torch.nn.functional.huber_loss, reduction='none', delta=10.0)
    elif loss_fn_type == 'mse':
        loss_fn = functools.partial(torch.nn.functional.mse_loss, reduction='none')
    else:
        raise NotImplementedError(f"Unknown loss fn type: {loss_fn_type}")

    target_value = target_value.clone()  # clone a inference tensor

    value_loss_original = loss_fn(value, target_value)
    value_clipped = old_value + (value - old_value).clamp(-value_eps_clip, value_eps_clip)
    value_loss_clipped = loss_fn(value_clipped, target_value)

    value_loss = torch.max(value_loss_original, value_loss_clipped)

    proportion_clipped = (value_loss_clipped > value_loss_original)
    proportion_clipped = (proportion_clipped.float() * loss_mask).sum() / loss_mask.sum()
    stat = dict(clip_ratio=proportion_clipped.detach())

    return (value_loss * loss_mask).sum() / loss_mask.sum(), stat


@torch.inference_mode()
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


@torch.inference_mode()
def get_advantages_and_returns(
    gamma: float,
    lam: float,
    values: torch.FloatTensor,
    rewards: torch.FloatTensor,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """Compute GAE and returns given padded batch inputs.

    Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
    
    Args:
        gamma (float): Discount factor.
        lam (float): GAE discount factor.
        values (torch.FloatTensor): Values of shape [bs, max_seqlen] (with bootstrapped values).
        rewards (torch.FloatTensor): Rewards of shape [bs, max_seqlen - 1].

    Returns:
        Tuple[torch.FloatTensor, torch.FloatTensor]: GAE and returns of shape [bs, max_seqlen - 1].
    """
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
    return advantages.detach(), returns.detach()