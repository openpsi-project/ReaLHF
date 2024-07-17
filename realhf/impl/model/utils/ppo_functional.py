import functools
from typing import Dict, Optional, Tuple

import torch
import torch.distributed
from torch.cuda.amp import custom_bwd, custom_fwd

import realhf.base.constants as constants
from realhf.impl.model.parallelism.model_parallel.utils import VocabUtility
from realhf.impl.model.utils.functional import build_leave_one_indices


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
    assert logprobs.dtype == torch.float32
    assert old_logprobs.dtype == torch.float32
    assert advantages.dtype == torch.float32

    # clone inference tensors
    if old_logprobs.is_inference():
        old_logprobs = old_logprobs.clone()
    if advantages.is_inference():
        advantages = advantages.clone()

    if loss_mask is not None:
        loss_mask_count = loss_mask.count_nonzero()
        # For numerical stability.
        ratio = torch.where(loss_mask, torch.exp(logprobs - old_logprobs), 0)
        approx_kl = torch.where(loss_mask, (logprobs - old_logprobs).detach(), 0.0)
    else:
        ratio = torch.exp(logprobs - old_logprobs)
        approx_kl = (logprobs - old_logprobs).detach()

    clipped_ratio = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio

    if loss_mask is not None:
        pg_loss = (
            torch.where(loss_mask, torch.max(pg_loss1, pg_loss2), 0).sum()
            / loss_mask_count
        )
    else:
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    clip_mask = pg_loss1.detach() < pg_loss2.detach()
    if loss_mask is not None:
        proportion_clipped = (
            clip_mask.logical_and_(loss_mask).count_nonzero() / loss_mask_count
        )
        importance_weight = (
            torch.where(loss_mask, ratio.detach(), 0).sum() / loss_mask_count
        )
        approx_kl = approx_kl.sum() / loss_mask_count
    else:
        proportion_clipped = clip_mask.count_nonzero()
        importance_weight = ratio.detach().mean()
        approx_kl = approx_kl.mean()
    # Remain torch.CudaTensor here for all-reduce after train step.
    stat = dict(
        clip_ratio=proportion_clipped,
        importance_weight=importance_weight,
        approx_kl=approx_kl,
    )

    return pg_loss, stat


class _VocabParallelMemoryEfficientPPOLoss(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        vocab_parallel_logits,
        cu_seqlens,
        packed_input_ids,
        ppo_loss_mask,
        old_logprobs,
        advantages,
        eps_clip,
    ):

        target = labels = torch.nn.functional.pad(
            packed_input_ids[1:], (0, 1), value=0.0
        )
        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        torch.distributed.all_reduce(
            logits_max,
            op=torch.distributed.ReduceOp.MAX,
            group=constants.model_parallel_group(),
        )
        # Subtract the maximum value.
        vocab_parallel_logits = vocab_parallel_logits - logits_max.unsqueeze(dim=-1)

        # Get the partition's vocab indecies
        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = constants.model_parallel_rank()
        world_size = constants.model_parallel_world_size()
        vocab_start_index, vocab_end_index = get_vocab_range(
            partition_vocab_size, rank, world_size
        )

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(
            start=0, end=logits_2d.size()[0], device=logits_2d.device
        )
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        # All reduce is needed to get the chunks from other GPUs.
        torch.distributed.all_reduce(
            predicted_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=constants.model_parallel_group(),
        )

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = vocab_parallel_logits
        torch.exp(vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        torch.distributed.all_reduce(
            sum_exp_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=constants.model_parallel_group(),
        )

        # Loss = log(sum(exp(logits))) - predicted-logit.
        new_logprobs = predicted_logits - torch.log(sum_exp_logits)

        # Normalize and optionally smooth logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        leave_one_indices = build_leave_one_indices(new_logprobs, cu_seqlens)
        new_logprobs = new_logprobs[leave_one_indices] * ppo_loss_mask
        new_logprobs = new_logprobs.float()

        # For numerical stability.
        ratio = torch.where(ppo_loss_mask, torch.exp(new_logprobs - old_logprobs), 0)

        loss_mask_count = ppo_loss_mask.count_nonzero()
        approx_kl = (
            torch.where(
                ppo_loss_mask, (old_logprobs - new_logprobs).detach(), 0.0
            ).sum()
            / loss_mask_count
        )
        importance_weight = (
            torch.where(ppo_loss_mask, ratio.detach(), 0).sum() / loss_mask_count
        )

        clipped_ratio = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * clipped_ratio

        clip_mask = (pg_loss1 < pg_loss2).detach()
        proportion_clipped = (
            clip_mask.logical_and_(ppo_loss_mask).count_nonzero() / loss_mask_count
        )

        # Store softmax, target-mask and masked-target for backward pass.
        ctx.save_for_backward(
            exp_logits,
            target_mask,
            masked_target_1d,
            leave_one_indices,
            ppo_loss_mask,
            ratio,
            advantages,
        )
        ctx.eps_clip = eps_clip

        return (
            torch.max(pg_loss1, pg_loss2),
            importance_weight,
            proportion_clipped,
            approx_kl,
        )

    @staticmethod
    def backward(ctx, grad_output, g1, g2, g3):
        # Retreive tensors from the forward path.
        (
            softmax,
            target_mask,
            masked_target_1d,
            leave_one_indices,
            ppo_loss_mask,
            ratio,
            advantages,
        ) = ctx.saved_tensors
        eps_clip = ctx.eps_clip

        # ppo backward
        clipped_ratio = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * clipped_ratio
        pg_loss1_larger_mask = pg_loss1 > pg_loss2
        unclip_mask = (ratio >= 1.0 - eps_clip) & (ratio <= 1.0 + eps_clip)

        g_ = -grad_output * advantages
        grad_ratio = torch.where(pg_loss1_larger_mask & unclip_mask, g_, 0.0)
        grad_new_logp = torch.where(ppo_loss_mask, grad_ratio * ratio, 0.0)
        _grad_new_logp = grad_new_logp.new_zeros(softmax.shape[0], dtype=torch.float16)
        _grad_new_logp[leave_one_indices] = grad_new_logp.half()

        # All the inputs have softmax as thier gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)

        softmax_update = 1.0 - target_mask.view(-1).float()
        grad_2d[arange_1d, masked_target_1d] -= softmax_update

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(_grad_new_logp.unsqueeze(dim=-1))

        return grad_input, None, None, None, None, None, None


class _MemoryEfficientPPOActorLossFn(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        logits,
        cu_seqlens,
        packed_input_ids,
        ppo_loss_mask,
        old_logprobs,
        advantages,
        eps_clip,
    ):
        labels = torch.nn.functional.pad(packed_input_ids[1:], (0, 1), value=0.0)
        _new_logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        new_logprobs_labels = _new_logprobs.gather(
            dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)
        leave_one_indices = build_leave_one_indices(logits, cu_seqlens)
        new_logprobs = torch.where(
            ppo_loss_mask, new_logprobs_labels[leave_one_indices], 0.0
        )
        new_logprobs = new_logprobs.float()

        # For numerical stability.
        ratio = torch.where(ppo_loss_mask, torch.exp(new_logprobs - old_logprobs), 0)

        loss_mask_count = ppo_loss_mask.count_nonzero()
        approx_kl = (
            torch.where(
                ppo_loss_mask, (old_logprobs - new_logprobs).detach(), 0.0
            ).sum()
            / loss_mask_count
        )
        importance_weight = (
            torch.where(ppo_loss_mask, ratio.detach(), 0).sum() / loss_mask_count
        )

        clipped_ratio = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * clipped_ratio

        clip_mask = (pg_loss1 < pg_loss2).detach()
        proportion_clipped = (
            clip_mask.logical_and_(ppo_loss_mask).count_nonzero() / loss_mask_count
        )

        ctx.save_for_backward(
            logits, leave_one_indices, labels, ppo_loss_mask, ratio, advantages
        )
        ctx.eps_clip = eps_clip

        return (
            torch.max(pg_loss1, pg_loss2),
            importance_weight,
            proportion_clipped,
            approx_kl,
        )

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output, g1, g2, g3):
        logits, leave_one_indices, labels, ppo_loss_mask, ratio, advantages = (
            ctx.saved_tensors
        )
        eps_clip = ctx.eps_clip

        clipped_ratio = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * clipped_ratio
        pg_loss1_larger_mask = pg_loss1 > pg_loss2
        unclip_mask = (ratio >= 1.0 - eps_clip) & (ratio <= 1.0 + eps_clip)

        g_ = -grad_output * advantages
        grad_ratio = torch.where(pg_loss1_larger_mask & unclip_mask, g_, 0.0)
        grad_new_logp = torch.where(ppo_loss_mask, grad_ratio * ratio, 0.0)
        _grad_new_logp = grad_new_logp.new_zeros(logits.shape[0], dtype=torch.float16)
        _grad_new_logp[leave_one_indices] = grad_new_logp.half()

        grad_logits = torch.nn.functional.softmax(logits, dim=-1)
        grad_logits.scatter_add_(
            1, labels.view(-1, 1), torch.full_like(grad_logits, fill_value=-1)
        )
        grad_logits.mul_(_grad_new_logp.unsqueeze(1))

        return grad_logits, None, None, None, None, None, None, None


def memory_efficient_ppo_loss_fn(
    logits,
    cu_seqlens,
    packed_input_ids,
    ppo_loss_mask,
    old_logprobs,
    advantages,
    eps_clip,
):
    if constants.model_parallel_world_size() == 1:
        return _MemoryEfficientPPOActorLossFn.apply(
            logits,
            cu_seqlens,
            packed_input_ids,
            ppo_loss_mask,
            old_logprobs,
            advantages,
            eps_clip,
        )
    else:
        return _VocabParallelMemoryEfficientPPOLoss.apply(
            logits,
            cu_seqlens,
            packed_input_ids,
            ppo_loss_mask,
            old_logprobs,
            advantages,
            eps_clip,
        )


def _huber_loss(x: torch.Tensor, y: torch.Tensor, delta: float):
    diff = torch.abs(x - y)
    return torch.where(diff < delta, 0.5 * diff**2, delta * (diff - 0.5 * delta))


def _mse_loss(x: torch.Tensor, y: torch.Tensor):
    return 0.5 * (x - y) ** 2


def critic_loss_fn(
    value: torch.FloatTensor,
    old_value: torch.FloatTensor,
    target_value: torch.FloatTensor,
    value_eps_clip: float,
    loss_mask: Optional[torch.FloatTensor] = None,
    loss_fn_type: str = "mse",
) -> Tuple[torch.Tensor, Dict]:
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
    assert value.dtype == torch.float32
    assert old_value.dtype == torch.float32
    assert target_value.dtype == torch.float32

    if loss_fn_type == "huber":
        loss_fn = functools.partial(_huber_loss, delta=10.0)
    elif loss_fn_type == "mse":
        loss_fn = _mse_loss
    else:
        raise NotImplementedError(f"Unknown loss fn type: {loss_fn_type}")

    if target_value.is_inference():
        target_value = target_value.clone()  # clone a inference tensor

    value_loss_original = loss_fn(value, target_value)

    value_clipped = old_value + (value - old_value).clamp(
        -value_eps_clip, value_eps_clip
    )

    value_loss_clipped = loss_fn(value_clipped, target_value)

    value_loss = torch.max(value_loss_original, value_loss_clipped)

    with torch.no_grad():
        clip_mask = value_loss_clipped.detach() > value_loss_original.detach()
        if loss_mask is not None:
            mask_count = loss_mask.count_nonzero()
            proportion_clipped = (
                clip_mask.logical_and_(loss_mask).count_nonzero() / mask_count
            )
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
        kl_rewards[i, eos_indices[i] :] = 0.0

    reward_clip = torch.clamp(reward_score, -clip_reward_value, clip_reward_value)
    score_rewards = torch.zeros_like(kl_rewards)
    # This is assigned to the token before EOS, which rewards the output of the EOS token.
    score_rewards.scatter_(
        -1, (eos_indices - 1).unsqueeze(-1), reward_clip.unsqueeze(-1)
    )
    score_rewards = score_rewards * (
        1 - seq_no_eos_mask.unsqueeze(1)
    )  # only compute final rewards with EOS
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
    tot_rewards[short1cu_seqlens[1:] - 1] += torch.where(
        seq_no_eos_mask, 0, reward_score
    )
    return kl_rewards, tot_rewards


def pygae2d_olp(
    rewards: torch.FloatTensor,
    values: torch.FloatTensor,
    dones: torch.BoolTensor,
    truncates: torch.BoolTensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    episode_length = int(rewards.shape[1])
    masks = 1 - dones.float()
    truncate_mask = 1 - truncates.float()
    delta = rewards + gamma * values[:, 1:] * masks[:, 1:] - values[:, :-1]
    adv = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[:, 0])
    m = gamma * lam * masks[:, 1:]
    step = episode_length - 1
    while step >= 0:
        # if env is terminated compulsively, then abandon the finnal step
        # i.e. advantage of final step is 0, values target of final step is predicted values
        gae = (delta[:, step] + m[:, step] * gae) * truncate_mask[:, step + 1]
        adv[:, step] = gae
        step -= 1
    return adv, adv + values[:, :-1]


def pygae1d_nolp_misalign(
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


def pygae2d_nolp(
    rewards: torch.FloatTensor,
    values: torch.FloatTensor,
    on_reset: torch.BoolTensor,
    truncates: torch.BoolTensor,
    gamma: float,
    lam: float,
) -> torch.FloatTensor:
    on_reset = on_reset.float()
    truncates = truncates.float()
    episode_length = int(rewards.shape[1])
    delta = rewards + gamma * values[:, 1:] * (1 - on_reset[:, 1:]) - values[:, :-1]

    gae = torch.zeros_like(rewards[:, 0])
    adv = torch.zeros_like(rewards)

    # 1. If the next step is a new episode, GAE doesn't propagate back
    # 2. If the next step is a truncated final step, the backpropagated GAE is -V(t),
    #    which is not correct. We ignore it such that the current GAE is r(t-1)+ɣV(t)-V(t-1)
    # 3. If the next step is a done final step, the backpropagated GAE is zero.
    m = gamma * lam * (1 - on_reset[:, 1:]) * (1 - truncates[:, 1:])

    step = episode_length - 1
    while step >= 0:
        gae = delta[:, step] + m[:, step] * gae
        adv[:, step] = gae
        step -= 1

    return adv, adv + values[:, :-1]


def cugae1d_nolp_misalign_func(
    rewards: torch.FloatTensor,
    values: torch.FloatTensor,
    cu_seqlens: torch.IntTensor,
    truncate: torch.BoolTensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """Compute GAE over a batch of packed sequences with different lengths.

    This function assumes that rewards and values are packed into an 1D tensor.
    Values are longer than rewards by the number of sequences in rewards because of bootstrapping.
    cu_seqlens marks the bounary of sequences in rewards.

    The final step of each sequence is *NOT* overlapped with the first step of the next sequence,
    and rewards/values do not have the same length, so this function is suffixed with
    "nolp" (non-overlap) and "misalign".

    Args:
        rewards (torch.FloatTensor): Shape [total_seqlen], rewards across sequences.
        values (torch.FloatTensor): Shape [total_seqlen + batch_size], values across sequences.
            Values are bootstrapped, so it's longer than rewards.
        cu_seqlens (torch.IntTensor): Marker of sequence boundaries in rewards,
            e.g., [0, s1, s1+s2, ..., total_seqlen]. It should starts with 0 and ends with total_seqlen.
        truncate (torch.BoolTensor): Whether each sequence is truncated because of exceeding max length.
            If truncate, the next value of the last step will be bootstraped, otherwise 0.
        gamma (float): Discount factor.
        lam (float): GAE discount factor.

    Returns:
        Tuple[torch.FloatTensor, torch.FloatTensor]: Advantages and returns (value targets).
            Both have the same shape as rewards.
    """
    import realhf._C.cugae as gae_cuda

    assert len(rewards.shape) == len(values.shape) == len(cu_seqlens.shape) == 1
    assert cu_seqlens[0] == 0 and cu_seqlens[-1] == rewards.shape[0]
    return gae_cuda.gae_1d_nolp_misalign(
        rewards, values, cu_seqlens, truncate, gamma, lam
    )


def cugae2d_olp_func(
    rewards: torch.FloatTensor,
    values: torch.FloatTensor,
    dones: torch.BoolTensor,
    truncates: torch.BoolTensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """Compute GAE over batched sequences with variable lengths, assuming
    overlapped sequences.

    This function assumes that rewards and values are batched as 2D tensors.
    The first dimension is batch_size and the second dimension is the number of collected timesteps.
    Each batch slot may contain multiple sequences, and sequences may have different lengths.
    The length of each sequence is marked by dones.

    `dones` marks the termination of each sequence, no matter it's truncated or not.
    `truncates` marks truncation and its nonzero indices must be the subset of `dones`.
    If truncate, abandon GAE computation of the last step (because we don't have the bootstrapped
    value in this case) and start from the second last step.

    The final step of each sequence *is overlapped* by the first step of the next sequence,
    i.e., auto-reset, which has widely used in libraries such as gym. In other words, the
    steps where `dones` is True are actually the first steps of sequences. Therefore,
    this function is suffixed with "olp" (overlap).

    Args:
        rewards (torch.FloatTensor): Shape [batch_size, seqlen].
        values (torch.FloatTensor): Shape [batch_size, seqlen + 1], with one more bootstrap step.
        dones (torch.BoolTensor): Shape [batch_size, seqlen + 1].
        truncates (torch.BoolTensor): Shape [batch_size, seqlen + 1].
        gamma (float): Discount factor.
        lam (float): GAE discount factor.

    Returns:
        Tuple[torch.FloatTensor, torch.FloatTensor]: Advantages and returns (value targets).
            Both have the same shape as rewards.
    """
    import realhf._C.cugae as gae_cuda

    truncates_indices = truncates.nonzero()
    assert torch.all(dones[truncates_indices[:, 0], truncates_indices[:, 1]])
    done_indices = dones.nonzero()
    num_dones = dones.float().sum(1)
    max_num_dones = int(num_dones.max())
    cu_num_dones = torch.nn.functional.pad(num_dones.cumsum(0), (1, 0), value=0).int()
    is_truncate = truncates[done_indices[:, 0], done_indices[:, 1]]
    return gae_cuda.gae_2d_olp(
        rewards,
        values,
        dones,
        done_indices[:, 1].int(),
        cu_num_dones,
        max_num_dones,
        is_truncate,
        gamma,
        lam,
    )


def cugae2d_nolp_func(
    rewards: torch.FloatTensor,
    values: torch.FloatTensor,
    on_reset: torch.BoolTensor,
    truncates: torch.BoolTensor,
    gamma: float,
    lam: float,
) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
    """Compute GAE over batched sequences with variable lengths, assuming non-
    overlapped sequences.

    This function assumes that rewards and values are batched as 2D tensors.
    The first dimension is batch_size and the second dimension is the number of collected timesteps.
    Each batch slot may contain multiple sequences, and sequences may have different lengths.
    The length of each sequence is marked by `on_reset`.

    `on_reset` marks the beginning of each sequence. `truncates` marks truncation.
    If truncate, values will be bootstrapped from the `done` step.

    The final step of each sequence is *NOT* overlapped by the first step of the next sequence.
    Each sequence will be complete. The last step should only have observations but no rewards.
    This is used in SRL. Therefore, this function is suffixed with "nolp" (non-overlap).

    Args:
        rewards (torch.FloatTensor): Shape [batch_size, seqlen].
        values (torch.FloatTensor): Shape [batch_size, seqlen + 1], with one more bootstrap step.
        dones (torch.BoolTensor): Shape [batch_size, seqlen + 1].
        truncates (torch.BoolTensor): Shape [batch_size, seqlen + 1].
        gamma (float): Discount factor.
        lam (float): GAE discount factor.

    Returns:
        Tuple[torch.FloatTensor, torch.FloatTensor]: Advantages and returns (value targets).
            Both have the same shape as rewards.
    """
    import realhf._C.cugae as gae_cuda

    dones = on_reset[:, 1:]
    truncates_indices = truncates[:, :-1].nonzero()
    assert torch.all(dones[truncates_indices[:, 0], truncates_indices[:, 1]])
    on_reset_indices = on_reset.nonzero()
    num_resets = on_reset.float().sum(1)
    max_num_resets = int(num_resets.max())
    cu_num_resets = torch.nn.functional.pad(num_resets.cumsum(0), (1, 0), value=0).int()
    truncates = torch.cat(
        [torch.zeros_like(truncates[:, 0:1]), truncates[:, :-1]], dim=1
    )
    bootstrap = truncates[on_reset_indices[:, 0], on_reset_indices[:, 1]]
    return gae_cuda.gae_2d_nolp(
        rewards,
        values,
        on_reset,
        on_reset_indices[:, 1].int(),
        cu_num_resets,
        max_num_resets,
        bootstrap,
        gamma,
        lam,
    )


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
        return cugae1d_nolp_misalign_func(
            rewards,
            values,
            short1cu_seqlens.int(),
            seq_no_eos_mask.bool(),
            gamma,
            lam,
        )
    except ModuleNotFoundError:
        return pygae1d_nolp_misalign(
            rewards, values, short1cu_seqlens, seq_no_eos_mask, gamma, lam
        )
