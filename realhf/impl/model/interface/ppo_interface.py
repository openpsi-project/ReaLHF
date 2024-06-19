from typing import Dict, Optional, Tuple
import collections
import dataclasses
import itertools
import time

from deepspeed import DeepSpeedEngine
import torch
import torch.distributed as dist

from realhf.api.core import data_api
from realhf.base.namedarray import from_dict, NamedArray, recursive_apply
from realhf.impl.model.nn.real_llm_api import ReaLModel
from realhf.impl.model.nn.real_llm_generate import (
    concat_prompt_to_generation_output,
    GenerationConfig,
)
from realhf.impl.model.utils.functional import (
    apply_logits_mask,
    gather_packed_shifted_log_probs,
    masked_normalization,
)
from realhf.impl.model.utils.padding import unpad_input
import realhf.api.core.model_api as model_api
import realhf.base.constants as constants
import realhf.base.logging as logging
import realhf.impl.model.utils.ppo_functional as ppo_functional

logger = logging.getLogger("PackedPPOInterface")


def _ppo_actor_loss_from_model_outputs(
    logits: torch.FloatTensor,  # [tot_seqlen, vocab_size]
    packed_input_ids: torch.LongTensor,  # [tot_seqlen]
    cu_seqlens: torch.LongTensor,  # [bs+1]
    old_logp: torch.FloatTensor,  # [tot_seqlen-bs]
    ppo_loss_mask: torch.FloatTensor,  # [tot_seqlen-bs]
    advantages: torch.FloatTensor,  # [tot_seqlen-bs]
    kl_rewards: torch.FloatTensor,  # [tot_seqlen-bs]
    kl_adapter: ppo_functional.KLController,  # const
    eps_clip: int,  # const
    early_stop_imp_ratio: Optional[float],  # const
    early_stop_kl: Optional[float],  # const
    logits_mask: Optional[torch.BoolTensor] = None,  # [tot_seqlen, vocab_size]
    **kwargs,
) -> Tuple[torch.FloatTensor, Dict]:
    """Loss function for ppo actor step, all inputs should be splitted into pipeline micro batches,
    returns loss and logging stats.
    """
    if logits_mask is not None:
        apply_logits_mask(logits, logits_mask)

    n_tokens = ppo_loss_mask.count_nonzero()
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

    # FIXME: The memory efficient loss function is buggy. It does not produce gradients correctly.
    # assert ppo_loss_mask is not None
    # (loss, importance_weight, clip_ratio, approx_kl) = (ppo_functional.memory_efficient_ppo_loss_fn(
    #     logits=logits,
    #     cu_seqlens=cu_seqlens,
    #     packed_input_ids=packed_input_ids,
    #     ppo_loss_mask=ppo_loss_mask,
    #     old_logprobs=old_logp,
    #     advantages=advantages,
    #     eps_clip=eps_clip,
    # ))
    # loss = torch.where(ppo_loss_mask, loss, 0.0).sum() / ppo_loss_mask.count_nonzero()
    importance_weight = ppo_stat["importance_weight"].float() * n_tokens
    clip_ratio = ppo_stat["clip_ratio"].float() * n_tokens
    approx_kl = ppo_stat["approx_kl"].float() * n_tokens

    # Logging and early stopping according to KL (logp vs ref) or importance ratio (new logp vs old logp).
    mean_ref_kl = (kl_rewards.detach().float() * ppo_loss_mask).sum()
    logging_loss = torch.where(ppo_loss_mask, loss.detach().float(), 0.0).sum()
    dist.all_reduce(n_tokens, group=constants.data_parallel_group())
    dist.all_reduce(mean_ref_kl, group=constants.data_parallel_group())
    dist.all_reduce(importance_weight, group=constants.data_parallel_group())
    dist.all_reduce(clip_ratio, group=constants.data_parallel_group())
    dist.all_reduce(approx_kl, group=constants.data_parallel_group())
    dist.all_reduce(logging_loss, group=constants.data_parallel_group())

    # Early stopping.
    kl_adapter.update(mean_ref_kl / n_tokens, n_steps=cu_seqlens.shape[0] - 1)
    _imp = importance_weight / n_tokens
    _kl = approx_kl / n_tokens
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
        actor_loss=logging_loss,
        actor_clip_ratio=clip_ratio,
        importance_weight=importance_weight,
    )

    return loss, stats


@dataclasses.dataclass
class PPOActorInterface(model_api.ModelInterface):
    n_minibatches: int = 4

    generation_config: Optional[Dict] = None

    kl_ctl: float = 0.1

    adv_norm: bool = True
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

    enable_save: bool = True
    force_no_logits_mask: bool = False

    value_norm: bool = False
    value_norm_type: str = dataclasses.field(
        metadata={"choices": ["exp", "ma"]}, default="exp"
    )
    value_norm_beta: float = 0.99995
    value_norm_eps: float = 1e-5

    def __post_init__(self):
        super().__post_init__()
        if self.adaptive_kl_ctl:
            assert self.adaptive_kl_target is not None
            assert self.adaptive_kl_horizon is not None
            self.kl_adapter = ppo_functional.AdaptiveKLController(
                self.kl_ctl, self.adaptive_kl_target, self.adaptive_kl_horizon
            )
        else:
            self.kl_adapter = ppo_functional.FixedKLController(self.kl_ctl)
        if self.value_norm:
            from realhf.impl.model.modules import (
                ExponentialRunningMeanStd,
                MovingAverageRunningMeanStd,
            )

            if self.value_norm_type == "exp":
                self.rms = ExponentialRunningMeanStd(
                    beta=self.value_norm_beta, epsilon=self.value_norm_eps
                )
            elif self.value_norm_type == "ma":
                self.rms = MovingAverageRunningMeanStd()
            else:
                raise ValueError(
                    f"Unknown value_norm_type {self.value_norm_type}"
                )
        self.kl_ctl = None

    def save(self, model: model_api.Model, save_dir: str):
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
    def generate(self, model: model_api.Model, data: NamedArray) -> NamedArray:
        module = model.module

        module.eval()

        data = recursive_apply(data, lambda x: x.to(model.device))
        packed_prompts = data["packed_prompts"]
        prompt_lengths = torch.tensor(
            data.metadata["seqlens"],
            dtype=torch.int32,
            device=packed_prompts.device,
        )
        prompt_cu_seqlens = torch.nn.functional.pad(
            prompt_lengths.cumsum(0), (1, 0)
        )

        res = module.generate(
            seqlens_cpu=data.metadata["seqlens"],
            tokenizer=model.tokenizer,
            packed_input_ids=packed_prompts,
            cu_seqlens=prompt_cu_seqlens,
            gconfig=GenerationConfig(**self.generation_config),
        )
        if res is None:
            return None

        gen_tokens, logprobs, logits_mask, *_ = res

        pad_token_id = model.tokenizer.pad_token_id
        eos_token_id = model.tokenizer.eos_token_id
        seq_no_eos_mask = (gen_tokens[:, -1] != eos_token_id).logical_and(
            gen_tokens[:, -1] != pad_token_id
        )
        # We also want gen_lengths to include the eos token, where the reward model outputs a score for this sequence.
        gen_lengths = (gen_tokens != pad_token_id).logical_and(
            gen_tokens != eos_token_id
        ).sum(dim=-1) + 1
        gen_lengths = gen_lengths.clip(max=gen_tokens.shape[-1])

        (
            packed_seq,
            packed_logprobs,
            packed_logits_mask,
            seq_lengths,
            prompt_mask,
        ) = concat_prompt_to_generation_output(
            packed_prompts=packed_prompts,
            prompt_lengths=prompt_lengths,
            gen_tokens=gen_tokens,
            logprobs=logprobs,
            logits_mask=logits_mask,
            gen_lengths=gen_lengths,
        )

        cu_seqlens = torch.nn.functional.pad(gen_lengths.cumsum(0), (1, 0))

        res = dict(
            seq_no_eos_mask=seq_no_eos_mask,
            packed_seq=packed_seq,
            cu_seqlens=cu_seqlens,
            packed_logprobs=packed_logprobs,
            packed_logits_mask=(
                packed_logits_mask.bool()
                if not self.force_no_logits_mask
                and packed_logits_mask is not None
                else None
            ),
            prompt_mask=prompt_mask,
        )
        res = from_dict(res)
        seqlens = seq_lengths.cpu().numpy().tolist()
        res.register_metadata(seqlens=seqlens)
        return res

    @torch.no_grad()
    def inference(self, model: model_api.Model, data: NamedArray) -> NamedArray:
        module = model.module
        module.eval()
        data = recursive_apply(data, lambda x: x.to(model.device))

        cu_seqlens = data["cu_seqlens"].int()
        input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = int(max(input_lens))

        logits = module.forward(
            seqlens_cpu=data.metadata["seqlens"],
            packed_input_ids=data["packed_seq"],
            cu_seqlens=cu_seqlens,
        )
        if logits is None:
            return None

        logits /= GenerationConfig(**self.generation_config).temperature
        if (
            "packed_logits_mask" in data
            and data["packed_logits_mask"] is not None
        ):
            apply_logits_mask(logits, data["packed_logits_mask"])
        logprobs = gather_packed_shifted_log_probs(
            logits, cu_seqlens, data["packed_seq"]
        )
        res = from_dict(dict(logprobs=logprobs))
        res.register_metadata(seqlens=data.metadata["seqlens"])
        return res

    def train_step(self, model: model_api.Model, data_: NamedArray) -> Dict:
        module = model.module
        tokenizer = model.tokenizer
        # We call module.eval() because dropout causes the computation of incorrect of log probs.
        module.eval()

        old_logp: torch.FloatTensor = data_["packed_logprobs"].float()
        ref_logp: torch.FloatTensor = data_["packed_ref_logprobs"].float()
        prompt_mask = data_["prompt_mask"]
        cu_seqlens = data_["cu_seqlens"].int()
        reward_score = data_["rewards"].float()
        values = data_["values"].float()
        seq_no_eos_mask = data_["seq_no_eos_mask"]

        if self.value_norm:
            denormalized_values = self.rms.denormalize(values)
        else:
            denormalized_values = values

        for i in range(seq_no_eos_mask.shape[0]):
            if not seq_no_eos_mask[i]:
                # Set value at the EOS token to be zero.
                denormalized_values[cu_seqlens[i + 1] - 1] = 0.0
                values[cu_seqlens[i + 1] - 1] = 0.0

        # Shift the loss mask by one token for each packed sequences.
        input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        short1cu_seqlens = cu_seqlens.clone()
        short1cu_seqlens[1:] -= torch.ones_like(cu_seqlens[1:]).cumsum(0)
        loss_mask = prompt_mask.logical_not()
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
        loss_mask = loss_mask[shift_one_indices]

        # Apply the mask to log probabilities.
        ref_logp *= loss_mask
        old_logp *= loss_mask

        # Compute rewards and GAEs.
        kl_rewards, rewards = ppo_functional.get_packed_rewards(
            kl_ctl=self.kl_adapter.value,
            clip_reward_value=self.max_reward_clip,
            log_probs=old_logp,
            ref_log_probs=ref_logp,
            reward_score=reward_score,
            short1cu_seqlens=short1cu_seqlens,
            seq_no_eos_mask=seq_no_eos_mask,
        )
        advantages, returns = ppo_functional.get_packed_advantages_and_returns(
            gamma=self.discount,
            lam=self.gae_lambda,
            values=denormalized_values,
            rewards=rewards,
            short1cu_seqlens=short1cu_seqlens,
            seq_no_eos_mask=seq_no_eos_mask,
        )

        # Optionally perform normalization.
        if self.value_norm:
            self.rms.update(returns, mask=loss_mask)
        if self.adv_norm:
            advantages = masked_normalization(advantages, loss_mask)

        # Prepare data to be splitted into mini-batches.
        batch_seqlens = data_.metadata["seqlens"]
        data_ = from_dict(
            dict(
                advantages=advantages,
                old_logp=old_logp,
                ppo_loss_mask=loss_mask,
                packed_seq=data_["packed_seq"],
                cu_seqlens=data_["cu_seqlens"].int(),
                kl_rewards=kl_rewards,
                packed_logits_mask=(
                    data_["packed_logits_mask"]
                    if "packed_logits_mask" in data_
                    else None
                ),
            )
        )
        data_.register_metadata(seqlens=batch_seqlens)
        datas = data_api.split_sequences(
            data_,
            self.n_minibatches,
            min_size=constants.pipe_parallel_world_size() * 2,
        )

        ### Logging code starts. ###
        _n_seqs = torch.tensor(
            [reward_score.shape[0]], dtype=torch.float32, device=model.device
        )
        _n_tokens = loss_mask.count_nonzero()
        task_reward = reward_score.sum()
        _advantages = advantages.sum()
        _kl_rewards = (kl_rewards * loss_mask).sum()
        prompt_len = prompt_mask.count_nonzero().float()
        seq_len = (cu_seqlens[1:] - cu_seqlens[:-1]).float().sum()
        dist.all_reduce(_n_seqs, group=constants.data_parallel_group())
        dist.all_reduce(task_reward, group=constants.data_parallel_group())
        dist.all_reduce(_advantages, group=constants.data_parallel_group())
        dist.all_reduce(prompt_len, group=constants.data_parallel_group())
        dist.all_reduce(seq_len, group=constants.data_parallel_group())
        dist.all_reduce(_n_tokens, group=constants.data_parallel_group())
        dist.all_reduce(_kl_rewards, group=constants.data_parallel_group())
        global_stats = dict(
            task_reward=float(task_reward / _n_seqs),
            kl_reward=float(_kl_rewards / _n_tokens),
            advantage=float(_advantages / _n_tokens),
            avg_seq_len=float(seq_len / _n_seqs),
            avg_prompt_len=float(prompt_len / _n_seqs),
            n_tokens=int(_n_tokens),
            n_seqs=int(_n_seqs),
        )

        if data_["packed_logits_mask"] is not None:
            n_masked_vocabs = data_["packed_logits_mask"].count_nonzero()
            total_vocabs = torch.tensor(
                data_["packed_logits_mask"].numel(),
                dtype=torch.long,
                device=model.device,
            )
            dist.all_reduce(
                n_masked_vocabs, group=constants.data_parallel_group()
            )
            dist.all_reduce(total_vocabs, group=constants.data_parallel_group())
            global_stats["valid_vocab_ratio"] = float(
                (total_vocabs - n_masked_vocabs) / total_vocabs
            )
        ### Logging code ends. ###

        # NOTE: We cannot randomly shuffle data here because
        # data must have the same shape across different pipeline stages.
        train_stats = collections.defaultdict(lambda: 0)
        offset = 0
        for data in datas:
            cu_seqlens = data["cu_seqlens"]
            input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
            logits_mask = (
                data["packed_logits_mask"]
                if "packed_logits_mask" in data
                else None
            )
            seqlens = batch_seqlens[offset : offset + input_lens.shape[0]]
            offset += input_lens.shape[0]
            loss_fn_kwargs = dict(
                input_lens=input_lens,  # used for partition
                old_logp=data["old_logp"],
                ppo_loss_mask=data["ppo_loss_mask"],
                advantages=data["advantages"],
                kl_rewards=data["kl_rewards"],
                kl_adapter=self.kl_adapter,
                eps_clip=self.eps_clip,
                early_stop_imp_ratio=self.early_stop_imp_ratio,
                early_stop_kl=self.early_stop_kl,
                logits_mask=logits_mask,
            )

            stats = module.train_batch(
                seqlens_cpu=seqlens,
                packed_input_ids=data["packed_seq"],
                cu_seqlens=data["cu_seqlens"],
                version_steps=model.version.global_step,
                loss_fn=_ppo_actor_loss_from_model_outputs,
                **loss_fn_kwargs,
            )

            if stats:
                for k, v in stats.items():
                    train_stats[k] += v
        cur_epoch = model.version.epoch
        model.inc_version()

        if train_stats:
            train_stats = dict(
                ppo_approx_kl=float(train_stats["ppo_approx_kl"] / _n_tokens),
                actor_loss=float(train_stats["actor_loss"] / _n_tokens),
                actor_clip_ratio=float(
                    train_stats["actor_clip_ratio"] / _n_tokens
                ),
                importance_weight=float(
                    train_stats["importance_weight"] / _n_tokens
                ),
            )
            train_stats = dict(**train_stats, **global_stats)

        return dict(train_stats)


def _ppo_critic_loss_from_model_outputs(
    new_values: torch.FloatTensor,
    packed_input_ids: torch.LongTensor,
    cu_seqlens: torch.LongTensor,
    values: torch.FloatTensor,
    ppo_loss_mask: torch.FloatTensor,
    returns: torch.FloatTensor,
    kl_rewards: torch.FloatTensor,
    value_eps_clip: float,
    kl_adapter: ppo_functional.KLController,
    rms=None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Dict]:
    leave_one_indices = torch.cat(
        [
            torch.arange(
                cu_seqlens[i],
                cu_seqlens[i + 1] - 1,
                dtype=torch.long,
                device=cu_seqlens.device,
            )
            for i in range(cu_seqlens.shape[0] - 1)
        ]
    )
    new_values = new_values[leave_one_indices].squeeze(-1).float()
    values = values[leave_one_indices].squeeze(-1).float()

    loss, loss_stat = ppo_functional.critic_loss_fn(
        value=new_values,
        old_value=values,
        target_value=returns,
        value_eps_clip=value_eps_clip,
        loss_mask=ppo_loss_mask,
    )

    if rms is not None:
        denormalized_values = rms.denormalize(new_values)
    else:
        denormalized_values = new_values

    # Logging.
    n_tokens = ppo_loss_mask.count_nonzero()
    mean_ref_kl = (kl_rewards.detach().float() * ppo_loss_mask).sum()
    logging_loss = loss.detach().float() * n_tokens
    clip_ratio = loss_stat["clip_ratio"].float() * n_tokens
    denormalized_values = (
        torch.where(ppo_loss_mask, denormalized_values, 0.0)
        .sum()
        .detach()
        .float()
    )
    dist.all_reduce(n_tokens, group=constants.data_parallel_group())
    dist.all_reduce(mean_ref_kl, group=constants.data_parallel_group())
    dist.all_reduce(logging_loss, group=constants.data_parallel_group())
    dist.all_reduce(clip_ratio, group=constants.data_parallel_group())
    dist.all_reduce(denormalized_values, group=constants.data_parallel_group())

    # Update KL coefficient to be consistent with actor.
    kl_adapter.update(mean_ref_kl, n_steps=cu_seqlens.shape[0] - 1)

    return loss, dict(
        value_loss=logging_loss,
        value_clip_ratio=clip_ratio,
        denormalized_values=denormalized_values,
    )


@dataclasses.dataclass
class PPOCriticInterface(model_api.ModelInterface):
    n_minibatches: int = 4
    enable_save: bool = True
    kl_ctl: float = 0.1
    discount: float = 1.0
    gae_lambda: float = 0.95
    eps_clip: float = 0.2
    value_eps_clip: float = 0.2
    max_reward_clip: float = 5.0
    adaptive_kl_ctl: bool = False
    adaptive_kl_target: Optional[float] = 6
    adaptive_kl_horizon: Optional[float] = 10000
    value_norm: bool = False
    value_norm_type: str = dataclasses.field(
        metadata={"choices": ["exp", "ma"]}, default="exp"
    )
    value_norm_beta: float = 0.99995
    value_norm_eps: float = 1e-5

    def __post_init__(self):
        super().__post_init__()
        if self.adaptive_kl_ctl:
            assert self.adaptive_kl_target is not None
            assert self.adaptive_kl_horizon is not None
            self.kl_adapter = ppo_functional.AdaptiveKLController(
                self.kl_ctl, self.adaptive_kl_target, self.adaptive_kl_horizon
            )
        else:
            self.kl_adapter = ppo_functional.FixedKLController(self.kl_ctl)
        if self.value_norm:
            from realhf.impl.model.modules import (
                ExponentialRunningMeanStd,
                MovingAverageRunningMeanStd,
            )

            if self.value_norm_type == "exp":
                self.rms = ExponentialRunningMeanStd(
                    beta=self.value_norm_beta, epsilon=self.value_norm_eps
                )
            elif self.value_norm_type == "ma":
                self.rms = MovingAverageRunningMeanStd()
            else:
                raise ValueError(
                    f"Unknown value_norm_type {self.value_norm_type}"
                )
        self.kl_ctl = None

    def save(self, model: model_api.Model, save_dir: str):
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
    def inference(self, model: model_api.Model, data: NamedArray) -> NamedArray:
        module = model.module
        module.eval()
        data = recursive_apply(data, lambda x: x.to(model.device))

        cu_seqlens = data["cu_seqlens"].int()
        input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = int(max(input_lens))

        scores = module.forward(
            seqlens_cpu=data.metadata["seqlens"],
            packed_input_ids=data["packed_seq"],
            cu_seqlens=cu_seqlens,
        )
        if scores is None:
            return None
        scores = scores.squeeze(-1)
        res = from_dict(dict(scores=scores))
        res.register_metadata(seqlens=data.metadata["seqlens"])
        return res

    def train_step(self, model: model_api.Model, data_: NamedArray) -> Dict:
        module = model.module
        tokenizer = model.tokenizer
        # We call module.eval() because dropout causes the computation of incorrect of log probs.
        module.eval()
        data_ = recursive_apply(data_, lambda x: x.to(model.device))

        old_logp: torch.FloatTensor = data_["packed_logprobs"].float()
        ref_logp: torch.FloatTensor = data_["packed_ref_logprobs"].float()
        prompt_mask = data_["prompt_mask"]
        cu_seqlens = data_["cu_seqlens"].int()
        reward_score = data_["rewards"].float()
        values = data_["values"].float()
        seq_no_eos_mask = data_["seq_no_eos_mask"]

        if self.value_norm:
            denormalized_values = self.rms.denormalize(values)
        else:
            denormalized_values = values

        for i in range(seq_no_eos_mask.shape[0]):
            if not seq_no_eos_mask[i]:
                # Set value at the EOS token to be zero.
                denormalized_values[cu_seqlens[i + 1] - 1] = 0.0
                values[cu_seqlens[i + 1] - 1] = 0.0

        # Shift the loss mask by one token for each packed sequences.
        input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        short1cu_seqlens = cu_seqlens.clone()
        short1cu_seqlens[1:] -= torch.ones_like(cu_seqlens[1:]).cumsum(0)
        loss_mask = prompt_mask.logical_not()
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
        loss_mask = loss_mask[shift_one_indices]

        # Apply the mask to log probabilities.
        ref_logp *= loss_mask
        old_logp *= loss_mask

        # Compute rewards and GAEs.
        kl_rewards, rewards = ppo_functional.get_packed_rewards(
            kl_ctl=self.kl_adapter.value,
            clip_reward_value=self.max_reward_clip,
            log_probs=old_logp,
            ref_log_probs=ref_logp,
            reward_score=reward_score,
            short1cu_seqlens=short1cu_seqlens,
            seq_no_eos_mask=seq_no_eos_mask,
        )
        _, returns = ppo_functional.get_packed_advantages_and_returns(
            gamma=self.discount,
            lam=self.gae_lambda,
            values=denormalized_values,
            rewards=rewards,
            short1cu_seqlens=short1cu_seqlens,
            seq_no_eos_mask=seq_no_eos_mask,
        )

        # Optionally perform normalization.
        if self.value_norm:
            self.rms.update(returns, mask=loss_mask)
            normalized_returns = self.rms.normalize(returns)
        else:
            normalized_returns = returns

        # Prepare data to be splitted into mini-batches.
        batch_seqlens = data_.metadata["seqlens"]
        data_ = from_dict(
            dict(
                returns=normalized_returns,
                values=values,
                ppo_loss_mask=loss_mask,
                kl_rewards=kl_rewards,
                packed_seq=data_["packed_seq"],
                cu_seqlens=data_["cu_seqlens"],
            )
        )
        data_.register_metadata(seqlens=batch_seqlens)
        datas = data_api.split_sequences(
            data_,
            self.n_minibatches,
            min_size=constants.pipe_parallel_world_size() * 2,
        )

        # Logging.
        returns = torch.where(loss_mask, returns, 0.0).sum()
        n_tokens = loss_mask.count_nonzero()
        dist.all_reduce(returns, group=constants.data_parallel_group())
        dist.all_reduce(n_tokens, group=constants.data_parallel_group())
        global_stats = dict(returns=float(returns), n_tokens=int(n_tokens))

        # NOTE: We cannot randomly shuffle data here because data must the same shape across different pipeline stages.
        train_stats = collections.defaultdict(lambda: 0)
        offset = 0
        for data in datas:
            input_lens = data["cu_seqlens"][1:] - data["cu_seqlens"][:-1]
            seqlens_cpu = batch_seqlens[offset : offset + input_lens.shape[0]]
            offset += input_lens.shape[0]

            loss_kwargs = dict(
                input_lens=input_lens,
                values=data["values"],
                ppo_loss_mask=data["ppo_loss_mask"],
                returns=data["returns"],
                kl_rewards=data["kl_rewards"],
                value_eps_clip=self.value_eps_clip,
                kl_adapter=self.kl_adapter,
                rms=self.rms if self.value_norm else None,
            )

            stats = module.train_batch(
                seqlens_cpu=seqlens_cpu,
                packed_input_ids=data["packed_seq"],
                cu_seqlens=data["cu_seqlens"],
                version_steps=model.version.global_step,
                loss_fn=_ppo_critic_loss_from_model_outputs,
                **loss_kwargs,
            )

            if stats:
                for k, v in stats.items():
                    train_stats[k] += v

        cur_epoch = model.version.epoch
        model.inc_version()

        if train_stats:
            train_stats = dict(
                value_loss=float(train_stats["value_loss"] / n_tokens),
                value_clip_ratio=float(
                    train_stats["value_clip_ratio"] / n_tokens
                ),
                denormalized_values=float(
                    train_stats["denormalized_values"] / n_tokens
                ),
                returns=global_stats["returns"] / int(n_tokens),
                n_tokens=int(n_tokens),
            )

        return dict(train_stats)


model_api.register_interface("ppo_actor", PPOActorInterface)
model_api.register_interface("ppo_critic", PPOCriticInterface)
