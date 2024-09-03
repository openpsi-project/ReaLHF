import collections
import dataclasses
import functools
from typing import *

import torch
import torch.distributed as dist

import realhf.api.core.model_api as model_api
import realhf.base.constants as constants
import realhf.base.logging as logging
from realhf.api.core.data_api import SequenceSample
from realhf.base.datapack import flat2d

logger = logging.getLogger("GRPO Interface")


def _grpo_loss(
    logits: torch.FloatTensor,  # [tot_seqlen, vocab_size]
    input_: SequenceSample,
    kl_adapter: Any,  # const
    eps_clip: float,  # const
    early_stop_imp_ratio: Optional[float],  # const
    early_stop_kl: Optional[float],  # const
) -> Tuple[torch.FloatTensor, Dict]:
    # NOTE: import here to avoid cuda initialization
    import realhf.impl.model.utils.ppo_functional as ppo_functional
    from realhf.impl.model.utils.functional import (
        apply_logits_mask,
        gather_packed_shifted_log_probs,
    )

    packed_input_ids = input_.data["packed_input_ids"]
    seqlens = torch.tensor(flat2d(input_.seqlens["packed_input_ids"]), device="cuda")
    cu_seqlens = torch.nn.functional.pad(seqlens.cumsum(0), (1, 0)).int()
    ppo_loss_mask = input_.data["ppo_loss_mask"]
    advantages = input_.data["advantages"].float()
    old_logp = input_.data["old_logp"].float()
    ref_logp = input_.data["ref_logp"].float()

    logits_mask = input_.data["packed_logits_mask"]
    if logits_mask is not None:
        apply_logits_mask(logits, logits_mask)

    logprobs = gather_packed_shifted_log_probs(
        logits, cu_seqlens, packed_input_ids
    ).float()
    loss, ppo_stat = ppo_functional.actor_loss_fn(
        logprobs=logprobs,
        old_logprobs=old_logp,
        advantages=advantages,
        eps_clip=eps_clip,
        loss_mask=ppo_loss_mask,
    )

    ref_kl = ref_logp - logprobs

    token_denorm = ppo_loss_mask.count_nonzero().float()

    actor_loss = torch.where(ppo_loss_mask, loss.detach(), 0.0).sum()
    kl_loss = torch.where(ppo_loss_mask, ref_kl.exp() - ref_kl - 1, 0.0).sum()

    loss += kl_adapter.value * kl_loss / ppo_loss_mask.count_nonzero()

    kl_loss = kl_loss.detach()
    importance_weight = ppo_stat["importance_weight"].float() * token_denorm
    clip_ratio = ppo_stat["clip_ratio"].float() * token_denorm
    approx_kl = ppo_stat["approx_kl"].float() * token_denorm
    advantages = torch.where(ppo_loss_mask, advantages, 0.0).sum()
    dist.all_reduce_coalesced(
        [
            token_denorm,
            importance_weight,
            clip_ratio,
            approx_kl,
            actor_loss,
            kl_loss,
            advantages,
        ],
        group=constants.data_parallel_group(),
    )

    # Early stopping.
    kl_adapter.update(kl_loss / token_denorm, n_steps=cu_seqlens.shape[0] - 1)
    _imp = importance_weight / token_denorm
    _kl = approx_kl / token_denorm
    if early_stop_imp_ratio is not None and _imp > early_stop_imp_ratio:
        logger.warning(
            f"Current importance ratio {_imp.item():.4f} is larger "
            f"than early stop threshold {early_stop_imp_ratio}. Abandon this minibatch."
        )
        loss = loss * 0.0
    if early_stop_kl is not None and _kl > early_stop_kl:
        logger.warning(
            f"Current approximate KL divergence {_kl.item():.4f} is larger "
            f"than early stop threshold {early_stop_kl}. Abort actor update."
        )
        loss = loss * 0.0

    stats = dict(
        ppo_approx_kl=approx_kl,
        actor_loss=actor_loss,
        kl_loss=kl_loss,
        actor_clip_ratio=clip_ratio,
        token_denorm=token_denorm,
        advantages=advantages,
        importance_weight=importance_weight,
    )

    return loss, stats


@dataclasses.dataclass
class GRPOInterface(model_api.ModelInterface):
    group_size: int
    n_minibatches: int = 4

    generation_config: model_api.GenerationHyperparameters = dataclasses.field(
        default_factory=model_api.GenerationHyperparameters
    )

    kl_ctl: float = 0.1

    adv_norm: bool = True
    discount: float = 0.99

    eps_clip: float = 0.2
    max_reward_clip: float = 5.0

    early_stop_kl: Optional[float] = None  # e.g. 0.1
    early_stop_imp_ratio: Optional[float] = None  # e.g., 10.0

    adaptive_kl_ctl: bool = False
    adaptive_kl_target: Optional[float] = 6
    adaptive_kl_horizon: Optional[float] = 10000

    enable_save: bool = True

    def __post_init__(self):
        from realhf.impl.model.utils import ppo_functional

        if self.adaptive_kl_ctl:
            assert self.adaptive_kl_target is not None
            assert self.adaptive_kl_horizon is not None
            self.kl_adapter = ppo_functional.AdaptiveKLController(
                self.kl_ctl, self.adaptive_kl_target, self.adaptive_kl_horizon
            )
        else:
            self.kl_adapter = ppo_functional.FixedKLController(self.kl_ctl)
        self.kl_ctl = None

    def save(self, model: model_api.Model, save_dir: str):
        # NOTE: import here to avoid cuda initialization
        from realhf.impl.model.nn.real_llm_api import ReaLModel

        if not self.enable_save:
            return
        module = model.module
        if not isinstance(module, ReaLModel):
            module = module.module
        module.save_to_hf(
            tokenizer=model.tokenizer,
            save_dir=save_dir,
        )

    @torch.no_grad()
    def generate(
        self, model: model_api.Model, input_: SequenceSample, n_mbs=None
    ) -> SequenceSample:
        # NOTE: import here to avoid cuda initialization
        from realhf.impl.model.nn.real_llm_generate import (
            concat_prompt_to_generation_output,
        )

        module = model.module

        module.eval()

        # Repeat the prompt for `self.group_size` times.
        packed_input_ids = input_.data["packed_input_ids"]
        new_input_ids = []
        offset = 0
        for x in input_.seqlens["packed_input_ids"]:
            new_input_ids += [
                packed_input_ids[offset : offset + x[0]]
            ] * self.group_size
            offset += x[0]
        assert offset == sum([x[0] for x in input_.seqlens["packed_input_ids"]])

        grouped_input = SequenceSample.from_default(
            ids=list(range(input_.bs * self.group_size)),
            seqlens=[
                int(x[0])
                for _ in range(self.group_size)
                for x in input_.seqlens["packed_input_ids"]
            ],
            data=dict(packed_input_ids=torch.cat(new_input_ids)),
        )

        res = module.generate(
            input_=grouped_input,
            tokenizer=model.tokenizer,
            gconfig=self.generation_config,
            num_micro_batches=n_mbs,
        )
        if res is None:
            return None

        gen_tokens, logprobs, logits_mask, *_ = res

        pad_token_id = model.tokenizer.pad_token_id
        eos_token_id = model.tokenizer.eos_token_id
        # We also want gen_lengths to include the eos token, where the reward model outputs a score for this sequence.
        gen_lengths = (gen_tokens != pad_token_id).logical_and(
            gen_tokens != eos_token_id
        ).sum(dim=-1) + 1
        gen_lengths = gen_lengths.clip(max=gen_tokens.shape[-1])

        (
            packed_input_ids,
            packed_logprobs,
            packed_logits_mask,
            seq_lengths,
            prompt_mask,
        ) = concat_prompt_to_generation_output(
            packed_prompts=grouped_input.data["packed_input_ids"],
            prompt_lengths=torch.tensor(
                flat2d(grouped_input.seqlens["packed_input_ids"]), device=model.device
            ),
            gen_tokens=gen_tokens,
            logprobs=logprobs,
            logits_mask=logits_mask,
            gen_lengths=gen_lengths,
        )

        # Partition generated data into groups.
        seqlens = [
            seq_lengths[i * self.group_size : (i + 1) * self.group_size].cpu().int()
            for i in range(input_.bs)
        ]
        data = dict(
            packed_input_ids=packed_input_ids,
            prompt_mask=prompt_mask,
            packed_logprobs=packed_logprobs,
            packed_logits_mask=(
                packed_logits_mask.bool()
                if not self.generation_config.force_no_logits_mask
                and packed_logits_mask is not None
                else None
            ),
        )

        res = SequenceSample(
            keys=[
                "packed_input_ids",
                "prompt_mask",
                "packed_logprobs",
                "packed_logits_mask",
            ],
            trailing_shapes=dict(
                packed_input_ids=(),
                prompt_mask=(),
                packed_logprobs=(),
                packed_logits_mask=(packed_logits_mask.shape[-1],),
            ),
            dtypes=dict(
                packed_input_ids=torch.long,
                prompt_mask=torch.bool,
                packed_logprobs=torch.float,
                packed_logits_mask=torch.bool,
            ),
            seqlens=dict(
                packed_input_ids=seqlens,
                packed_logits_mask=seqlens,
                packed_logprobs=[x - 1 for x in seqlens],
                prompt_mask=seqlens,
            ),
            data=data,
            ids=input_.ids,
        )
        return res

    @torch.no_grad()
    def inference(
        self, model: model_api.Model, input_: SequenceSample, n_mbs=None
    ) -> SequenceSample:
        from realhf.impl.model.utils.functional import (
            apply_logits_mask,
            gather_packed_shifted_log_probs,
        )

        module = model.module
        module.eval()

        # This post_hook will gather log probabilities in mini-batches,
        # reducing peak memory usage.
        def calc_logprobs(logits, input_):
            input_lens = torch.tensor(flat2d(input_.seqlens["packed_input_ids"]))
            cu_seqlens = torch.nn.functional.pad(input_lens.cumsum(0), (1, 0)).int()
            logits /= self.generation_config.temperature
            if (
                "packed_logits_mask" in input_.data
                and input_.data["packed_logits_mask"] is not None
            ):
                apply_logits_mask(logits, input_.data["packed_logits_mask"])

            logprobs = gather_packed_shifted_log_probs(
                logits, cu_seqlens, input_.data["packed_input_ids"]
            )
            return logprobs

        logprobs = module.forward(
            input_=input_,
            num_micro_batches=n_mbs,
            post_hook=calc_logprobs,
        )
        if logprobs is None:
            return None

        res = SequenceSample(
            keys=["packed_ref_logprobs"],
            ids=input_.ids,
            dtypes=dict(packed_ref_logprobs=logprobs.dtype),
            trailing_shapes=dict(packed_ref_logprobs=()),
            data=dict(packed_ref_logprobs=logprobs),
            seqlens=dict(
                packed_ref_logprobs=[
                    [xx - 1 for xx in x] for x in input_.seqlens["packed_input_ids"]
                ]
            ),
        )
        return res

    def train_step(
        self, model: model_api.Model, input_: SequenceSample, n_mbs=None
    ) -> Dict:
        # NOTE: import here to avoid cuda initialization
        from realhf.impl.model.utils.functional import masked_normalization
        from realhf.impl.model.utils.ppo_functional import (
            get_packed_advantages_and_returns,
        )

        module = model.module
        module.eval()

        # Get the useful sequence length indices.
        seqlens = torch.tensor(
            flat2d(input_.seqlens["packed_input_ids"]), device=model.device
        )
        cu_seqlens = torch.nn.functional.pad(seqlens.cumsum(0), (1, 0)).int()
        short1seqlens = seqlens - 1
        short1cu_seqlens = torch.nn.functional.pad(
            short1seqlens.cumsum(0), (1, 0)
        ).int()
        shift_one_indices = torch.cat(
            [
                torch.arange(
                    cu_seqlens[i] + 1,
                    cu_seqlens[i + 1],
                    dtype=torch.long,
                    device=cu_seqlens.device,
                )
                for i in range(cu_seqlens.shape[0] - 1)
            ]
        )

        # Get loss mask that filters prompts out.
        loss_mask = input_.data[f"prompt_mask"][shift_one_indices].logical_not()

        # Apply the mask to log probabilities.
        input_.data["packed_ref_logprobs"] *= loss_mask
        input_.data["packed_logprobs"] *= loss_mask

        # Gather rewards for all groups and normalize them.
        group_rewards = input_.data["rewards"].view(-1, self.group_size)
        rewards_mean = group_rewards.mean(1, keepdim=True)
        rewards_std = group_rewards.std(1, keepdim=True)
        all_rewards = (
            ((group_rewards - rewards_mean) / (rewards_std + 1e-5))
            .clip(-self.max_reward_clip, self.max_reward_clip)
            .view(-1)
        )
        assert all_rewards.shape[0] == input_.bs * self.group_size, (
            all_rewards.shape,
            input_.bs,
            self.group_size,
        )

        # Compute episode-level rewards.
        episode_rewards = torch.zeros(
            int(short1seqlens.sum()), dtype=torch.float32, device=model.device
        )
        episode_rewards.scatter_(
            0,
            short1seqlens.cumsum(0) - 1,
            all_rewards,
        )

        # Get discounted reward.
        adv, _ = get_packed_advantages_and_returns(
            gamma=1.0,
            lam=self.discount,
            rewards=episode_rewards,
            values=torch.zeros(
                int(seqlens.sum()), dtype=torch.float32, device=model.device
            ),
            short1cu_seqlens=short1cu_seqlens,
            seq_no_eos_mask=torch.zeros(
                input_.bs * self.group_size, dtype=torch.bool, device=model.device
            ),
        )
        # Optionally normalize computed advantages.
        if self.adv_norm:
            adv = masked_normalization(adv, mask=loss_mask)

        # Unpack grouped inputs to individual sequences for training.
        data_ = SequenceSample.from_default(
            seqlens=[int(x) for x in seqlens.cpu().numpy().tolist()],
            data=dict(
                packed_input_ids=input_.data["packed_input_ids"],
                ppo_loss_mask=loss_mask,
                advantages=adv,
                old_logp=input_.data["packed_logprobs"],
                ref_logp=input_.data["packed_ref_logprobs"],
                packed_logits_mask=(
                    None
                    if self.generation_config.force_no_logits_mask
                    else input_.data.get("packed_logits_mask")
                ),
            ),
            ids=list(range(input_.bs * self.group_size)),
        )

        # Split mini-batches and run PPO training. Mini-batches have balanced sizes
        datas = data_.split(self.n_minibatches, min_size=data_.bs // self.n_minibatches)
        train_stats = collections.defaultdict(float)
        for data in datas:
            stats = module.train_batch(
                input_=data,
                version_steps=model.version.global_step,
                loss_fn=functools.partial(
                    _grpo_loss,
                    kl_adapter=self.kl_adapter,
                    eps_clip=self.eps_clip,
                    early_stop_imp_ratio=self.early_stop_imp_ratio,
                    early_stop_kl=self.early_stop_kl,
                ),
                num_micro_batches=n_mbs,
            )
            if stats:
                for k, v in stats.items():
                    train_stats[k] += v

        model.inc_version()

        # Logging.
        rewards_group_mean_sq = group_rewards.square().mean(1).sum()
        rewards_group_mean = rewards_mean.sum()
        bs = torch.tensor([input_.bs], device=model.device, dtype=torch.float32)
        dist.all_reduce_coalesced(
            [rewards_group_mean, rewards_group_mean_sq, bs],
            group=constants.data_parallel_group(),
        )
        rewards_mean = float(rewards_group_mean / bs)
        rewards_std = float(torch.sqrt(rewards_group_mean_sq / bs - rewards_mean**2))
        if stats:
            token_denorm = int(stats["token_denorm"])
            stats = dict(
                ppo_approx_kl=float(stats["ppo_approx_kl"]) / token_denorm,
                actor_loss=float(stats["actor_loss"]) / token_denorm,
                kl_loss=float(stats["kl_loss"]) / token_denorm,
                kl_ctl=self.kl_adapter.value,
                actor_clip_ratio=float(stats["actor_clip_ratio"]) / token_denorm,
                importance_weight=float(stats["importance_weight"]) / token_denorm,
                advantages=float(stats["advantages"]) / token_denorm,
                rewards=rewards_mean,
                rewards_std=rewards_std,
                # FIXME: It only logs the MoE aux loss of the final PPO mini-batch.
                **constants.log_global_stats_tracker(
                    return_dict=True, clear_stats_after_logging=True
                ),
            )

        return dict(stats) if stats else {}
