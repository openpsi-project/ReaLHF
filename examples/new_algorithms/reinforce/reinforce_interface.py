import dataclasses
from typing import Dict, Tuple

import torch
import torch.distributed as dist

import realhf.api.core.model_api as model_api
import realhf.base.constants as constants
import realhf.base.logging as logging
from realhf.api.core.data_api import SequenceSample

logger = logging.getLogger("Reinforce Interface")


def _reinforce_loss_from_model_outputs(
    logits: torch.FloatTensor,  # [tot_seqlen, vocab_size]
    input_: SequenceSample,
) -> Tuple[torch.FloatTensor, Dict]:
    # NOTE: import here to avoid cuda initialization
    from realhf.impl.model.utils.functional import gather_packed_shifted_log_probs

    packed_input_ids = input_.data["packed_input_ids"]
    seqlens = torch.cat(input_.seqlens["packed_input_ids"]).cuda()
    cu_seqlens = torch.nn.functional.pad(seqlens.cumsum(0), (1, 0)).int()
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
    loss_mask = input_.data["prompt_mask"][shift_one_indices].logical_not()
    adv = input_.data["adv"]

    logprobs = gather_packed_shifted_log_probs(
        logits, cu_seqlens, packed_input_ids
    ).float()

    loss = -(logprobs * adv * loss_mask).sum() / loss_mask.count_nonzero()

    logging_loss = (logprobs * adv * loss_mask).sum()
    token_denorm = loss_mask.count_nonzero().float()
    bs = torch.tensor([input_.bs], dtype=torch.float32, device=logits.device)
    baseline_rewards = input_.data["greedy_rewards"].sum()
    rewards = input_.data["rewards"].sum()
    dp_group = constants.data_parallel_group()
    dist.all_reduce_coalesced(
        [logging_loss, bs, baseline_rewards, rewards, token_denorm], group=dp_group
    )

    return loss, dict(
        loss=logging_loss,
        baseline_rewards=baseline_rewards,
        token_denorm=token_denorm,
        rewards=rewards,
        bs=bs,
    )


@dataclasses.dataclass
class ReinforceInterface(model_api.ModelInterface):

    force_greedy: bool = False
    generation_config: model_api.GenerationHyperparameters = dataclasses.field(
        default_factory=model_api.GenerationHyperparameters
    )
    enable_save: bool = True
    discount: float = 0.99
    adv_norm: bool = True

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
        self, model: model_api.Model, input_: SequenceSample
    ) -> SequenceSample:
        # NOTE: import here to avoid cuda initialization
        from realhf.impl.model.nn.real_llm_generate import (
            concat_prompt_to_generation_output,
        )

        module = model.module

        module.eval()

        self.generation_config.greedy = self.force_greedy
        res = module.generate(
            input_=input_,
            tokenizer=model.tokenizer,
            gconfig=self.generation_config,
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
            _,
            _,
            seq_lengths,
            prompt_mask,
        ) = concat_prompt_to_generation_output(
            packed_prompts=input_.data["packed_input_ids"],
            prompt_lengths=torch.cat(input_.seqlens["packed_input_ids"]).to(
                model.device
            ),
            gen_tokens=gen_tokens,
            logprobs=logprobs,
            logits_mask=logits_mask,
            gen_lengths=gen_lengths,
        )

        seqlens = [
            torch.tensor([s], dtype=torch.int32)
            for s in seq_lengths.cpu().numpy().tolist()
        ]
        data = dict(
            packed_input_ids=packed_input_ids,
            prompt_mask=prompt_mask,
        )

        res = SequenceSample.from_default(
            ids=input_.ids,
            seqlens=seqlens,
            data=data,
        )
        return res

    def train_step(self, model: model_api.Model, input_: SequenceSample) -> Dict:
        # NOTE: import here to avoid cuda initialization
        from realhf.impl.model.utils.functional import masked_normalization
        from realhf.impl.model.utils.ppo_functional import (
            get_packed_advantages_and_returns,
        )

        module = model.module
        module.eval()

        seqlens = torch.cat(input_.seqlens["packed_input_ids"]).cuda()
        short1seqlens = seqlens - 1
        rewards = torch.zeros(
            int(short1seqlens.sum()), dtype=torch.float32, device=model.device
        )
        rewards.scatter_(
            0,
            short1seqlens.cumsum(0) - 1,
            input_.data["rewards"] - input_.data["greedy_rewards"],
        )
        short1cu_seqlens = torch.nn.functional.pad(
            short1seqlens.cumsum(0), (1, 0)
        ).int()

        adv, _ = get_packed_advantages_and_returns(
            gamma=1.0,
            lam=self.discount,
            rewards=rewards,
            values=torch.zeros(
                int(seqlens.sum()), dtype=torch.float32, device=model.device
            ),
            short1cu_seqlens=short1cu_seqlens,
            seq_no_eos_mask=torch.zeros(
                input_.bs, dtype=torch.bool, device=model.device
            ),
        )
        cu_seqlens = torch.nn.functional.pad(seqlens.cumsum(0), (1, 0)).int()

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
        loss_mask = input_.data["prompt_mask"][shift_one_indices].logical_not()

        if self.adv_norm:
            adv = masked_normalization(adv, mask=loss_mask)

        input_.update_(
            SequenceSample(
                keys=["adv"],
                trailing_shapes=dict(adv=()),
                data=dict(adv=adv),
                seqlens=dict(
                    adv=[short1seqlens[i : i + 1].cpu().int() for i in range(input_.bs)]
                ),
                ids=input_.ids,
                dtypes=dict(adv=torch.float32),
            )
        )

        stats = module.train_batch(
            input_=input_,
            version_steps=model.version.global_step,
            loss_fn=_reinforce_loss_from_model_outputs,
        )

        model.inc_version()

        if stats:
            bs = int(stats["bs"])
            stats = dict(
                loss=float(stats["loss"]) / int(stats["token_denorm"]),
                baseline_rewards=float(stats["baseline_rewards"] / bs),
                rewards=float(stats["rewards"] / bs),
            )
            stats["advantage"] = stats["rewards"] - stats["baseline_rewards"]

        return dict(stats) if stats else {}
