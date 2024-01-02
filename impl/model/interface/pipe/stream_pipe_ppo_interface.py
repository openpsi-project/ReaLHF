from typing import Dict, Optional, Tuple
import dataclasses
import itertools

import torch

from base.namedarray import from_dict, NamedArray, recursive_apply
from impl.model.backend.pipe_engine.stream_pipe_engine import EngineFuture, StreamPipeEngine
from impl.model.interface.flash.ppo_flash_interface import _ppo_actor_loss_from_model_outputs
from impl.model.nn.flash_mqat.flash_generate import GenerationConfig
from impl.model.utils.functional import gather_packed_shifted_log_probs, masked_normalization
import api.model
import base.logging as logging
import impl.model.utils.ppo_functional as ppo_functional

logger = logging.getLogger("stream_pipe_test")


@dataclasses.dataclass
class StreamPipePPOActorInterface(api.model.ModelInterface):
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
    value_norm_type: str = dataclasses.field(metadata={"choices": ["exp", "ma"]}, default="exp")
    value_norm_beta: float = 0.99995
    value_norm_eps: float = 1e-5

    def __post_init__(self):
        if self.adaptive_kl_ctl:
            assert self.adaptive_kl_target is not None
            assert self.adaptive_kl_horizon is not None
            self.kl_adapter = ppo_functional.AdaptiveKLController(self.kl_ctl, self.adaptive_kl_target,
                                                                  self.adaptive_kl_horizon)
        else:
            self.kl_adapter = ppo_functional.FixedKLController(self.kl_ctl)
        if self.value_norm:
            from impl.model.utils.modules import ExponentialRunningMeanStd, MovingAverageRunningMeanStd

            if self.value_norm_type == "exp":
                self.rms = ExponentialRunningMeanStd(beta=self.value_norm_beta, epsilon=self.value_norm_eps)
            elif self.value_norm_type == "ma":
                self.rms = MovingAverageRunningMeanStd()
            else:
                raise ValueError(f"Unknown value_norm_type {self.value_norm_type}")
        self.kl_ctl = None

    @torch.no_grad()
    def generate(self, model: api.model.Model, data: NamedArray) -> Tuple[EngineFuture, NamedArray]:
        """ Returns future and data for post process
        """
        module = model.module
        assert isinstance(module, StreamPipeEngine)

        module.eval()

        data = recursive_apply(data, lambda x: x.to(model.device))
        packed_prompts = data["packed_prompts"]
        cu_seqlens = data["prompt_cu_seqlens"]
        prompt_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        bs = prompt_lengths.shape[0]

        future = module.generate(
            tokenizer=model.tokenizer,
            packed_input_ids=packed_prompts,
            cu_seqlens=cu_seqlens,
            gconfig=GenerationConfig(**self.generation_config),
        )

        data = from_dict(dict(
            packed_prompts=packed_prompts,
            cu_seqlens=cu_seqlens,
        ))
        return future, data

    @torch.no_grad()
    def postprocess_generate(self, model: api.model.Model, data: NamedArray, future: EngineFuture):
        assert future.done()
        gen_tokens, logprobs, logits_mask, *_ = future.result()

        # data = recursive_apply(data, lambda x: x.to(model.device))
        packed_prompts = data["packed_prompts"]
        cu_seqlens = data["prompt_cu_seqlens"]
        prompt_lengths = cu_seqlens[1:] - cu_seqlens[:-1]

        bs = prompt_lengths.shape[0]

        pad_token_id = model.tokenizer.pad_token_id
        eos_token_id = model.tokenizer.eos_token_id
        seq_no_eos_mask = (gen_tokens[:, -1] != eos_token_id).logical_and(gen_tokens[:, -1] != pad_token_id)
        # We also want gen_lengths to include the eos token, where the reward model outputs a score for this sequence.
        gen_lengths = (gen_tokens != pad_token_id).logical_and(gen_tokens != eos_token_id).sum(dim=-1) + 1
        gen_lengths = gen_lengths.clip(max=gen_tokens.shape[-1])

        # TODO: refactor the following whole bunch of sh*t.
        # Pack generated sequences and logprobs.
        prompts_list, prompt_log_probs_list, prompt_logits_mask_list = [], [], []
        gen_tokens_list, gen_log_probs_list, gen_logits_mask_list = [], [], []
        for i in range(bs):
            prompt_len, gen_len = prompt_lengths[i].item(), gen_lengths[i].item()

            # Prompts are left-padded. Besides, prompt_log_probs is one-step shorter than prompts.
            prompts_list.append(packed_prompts[cu_seqlens[i]:cu_seqlens[i + 1]])
            prompt_log_probs_list.append(logprobs.new_zeros(prompt_len - 1))
            if logits_mask is not None:
                prompt_logits_mask_list.append(logits_mask.new_ones((prompt_len - 1, logits_mask.shape[-1])))

            # Generated tokens are right-padded.
            gen_tokens_list.append(gen_tokens[i, :gen_len])
            gen_log_probs_list.append(logprobs[i, :gen_len])
            if logits_mask is not None:
                gen_logits_mask_list.append(
                    torch.cat([logits_mask[i, :gen_len],
                               logits_mask.new_ones(1, logits_mask.shape[-1])]))

        # For complete sequences, EOS token is included. Otherwise the sequence may end with arbitrary token.
        # cu_seqlens marks the boundary of these sequences, no matter whether they are complete or not.
        packed_seq = torch.cat(list(itertools.chain.from_iterable(zip(prompts_list, gen_tokens_list))))
        seq_lengths = prompt_lengths + gen_lengths
        cu_seqlens = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=seq_lengths.device),
             seq_lengths.cumsum(0)])
        packed_logprobs = torch.cat(
            list(itertools.chain.from_iterable(zip(prompt_log_probs_list, gen_log_probs_list))))
        assert packed_seq.shape[0] == packed_logprobs.shape[0] + bs, (
            packed_seq.shape,
            packed_logprobs.shape,
            bs,
        )
        packed_logits_mask = None
        if gen_logits_mask_list and not self.force_no_logits_mask:
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
            packed_logprobs=packed_logprobs,
            packed_logits_mask=packed_logits_mask.bool() if packed_logits_mask is not None else None,
            prompt_mask=prompt_mask,
        )
        return from_dict(res)

    def train_step(self, model: api.model.Model, data: NamedArray) -> EngineFuture:
        module = model.module
        assert isinstance(module, StreamPipeEngine)
        # We call module.eval() because dropout causes the computation of incorrect of log probs.
        module.eval()
        data = recursive_apply(data, lambda x: x.to(model.device))

        old_logp: torch.FloatTensor = data["packed_logprobs"].float()
        ref_logp: torch.FloatTensor = data["packed_ref_logprobs"].float()
        prompt_mask = data["prompt_mask"]
        cu_seqlens = data["cu_seqlens"]
        reward_score = data["rewards"].float()
        values = data["values"].float()
        seq_no_eos_mask = data["seq_no_eos_mask"]

        if self.value_norm:
            denormalized_values = self.rms.denormalize(values)
        else:
            denormalized_values = values

        for i in range(seq_no_eos_mask.shape[0]):
            if not seq_no_eos_mask[i]:
                # Set value at the EOS token to be zero.
                denormalized_values[cu_seqlens[i + 1] - 1] = 0.0
                values[cu_seqlens[i + 1] - 1] = 0.0

        input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        short1cu_seqlens = cu_seqlens.clone()
        short1cu_seqlens[1:] -= torch.ones_like(cu_seqlens[1:]).cumsum(0)
        loss_mask = prompt_mask.logical_not()
        shift_one_indices = torch.cat([
            torch.arange(cu_seqlens[i] + 1, cu_seqlens[i + 1], dtype=torch.long, device=cu_seqlens.device)
            for i in range(cu_seqlens.shape[0] - 1)
        ])
        loss_mask = loss_mask[shift_one_indices]

        ref_logp *= loss_mask
        old_logp *= loss_mask

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

        if self.value_norm:
            self.rms.update(returns, mask=loss_mask)

        if self.adv_norm:
            advantages = masked_normalization(advantages, loss_mask)

        cu_seqlens = data["cu_seqlens"]
        input_lens = cu_seqlens[1:] - cu_seqlens[:-1]
        logits_mask = data["packed_logits_mask"] if "packed_logits_mask" in data else None

        module.set_version_steps(model.version.global_step)
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

        future = module.train_batch(
            packed_input_ids=data["packed_seq"],
            cu_seqlens=data["cu_seqlens"],
            loss_fn=_ppo_actor_loss_from_model_outputs,
            **loss_fn_kwargs,
        )

        global_stats = from_dict(
            dict(
                task_reward=reward_score.mean().detach(),
                kl_reward=(kl_rewards.detach() * loss_mask).sum().mean(),
                advantage=advantages.mean().detach(),
            ))

        return future, global_stats

    def postprocess_train_step(self, model: api.model.Model, data: NamedArray, future: EngineFuture):
        assert future.done()
        global_stats = data.to_dict()
        module = model.module
        loss, stats = future.result()

        if stats:
            for k, v in stats.items():
                train_stats[k] += v

        cur_epoch = model.version.epoch
        model.inc_version()
        if model.version.epoch > cur_epoch:
            module.tput_timer.update_epoch_count()

        if train_stats:
            train_stats: Dict[str, torch.Tensor] = dict(train_stats, **global_stats)
            for k, v in train_stats.items():
                v = v.detach() / self.n_minibatches
                train_stats[k] = v.item()

        return loss


api.model.register_interface("stream_pipe_ppo_actor", StreamPipePPOActorInterface)
