import dataclasses
from typing import Dict, Tuple

import torch
import torch.distributed as dist

import realhf.api.core.model_api as model_api
import realhf.base.constants as constants
import realhf.base.logging as logging
from realhf.api.core.data_api import SequenceSample
from realhf.impl.model.nn.real_llm_api import ReaLModel
from realhf.impl.model.nn.real_llm_generate import concat_prompt_to_generation_output
from realhf.impl.model.utils.functional import (
    gather_packed_shifted_log_probs,
)

logger = logging.getLogger("Reinforce Interface")


def _reinforce_loss_from_model_outputs(
    logits: torch.FloatTensor,  # [tot_seqlen, vocab_size]
    input_: SequenceSample,
) -> Tuple[torch.FloatTensor, Dict]:
    packed_input_ids = input_.data["packed_input_ids"]
    seqlens = torch.cat(input_.seqlens["packed_input_ids"]).cuda()
    cu_seqlens = torch.nn.functional.pad(seqlens.cumsum(0), (1, 0)).int()
    prompt_mask = input_.data["prompt_mask"]
    adv = input_.data["rewards"] - input_.data["greedy_rewards"]

    logprobs = gather_packed_shifted_log_probs(logits, cu_seqlens, packed_input_ids).float()

    seqlogp = []
    offset = 0
    short1offset = 0
    for i in range(input_.bs):
        logp = logprobs[short1offset : short1offset + seqlens[i] - 1]
        mask = prompt_mask[offset + 1 : offset + seqlens[i]].logical_not()
        seqlogp.append((logp * mask).sum())
        short1offset += seqlens[i] - 1
        offset += seqlens[i]
    assert offset == packed_input_ids.shape[0]
    assert short1offset == logprobs.shape[0]

    loss = -(torch.stack(seqlogp) * adv).mean()

    logging_loss = -(torch.stack(seqlogp) * adv).detach().sum()
    bs = torch.tensor([input_.bs], dtype=torch.float32, device=logits.device)
    baseline_rewards = input_.data["greedy_rewards"].sum()
    rewards = input_.data["rewards"].sum()
    dp_group = constants.data_parallel_group()
    dist.all_reduce_coalesced([logging_loss, bs, baseline_rewards, rewards], group=dp_group)

    return loss, dict(
        loss=logging_loss,
        baseline_rewards=baseline_rewards,
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
    def generate(self, model: model_api.Model, input_: SequenceSample) -> SequenceSample:
        module = model.module

        module.eval()

        # Remap the key `packed_prompts` to `packed_input_ids`,
        # because the pipe runner only recognizes `packed_input_ids`.
        x = SequenceSample.from_default(
            ids=input_.ids,
            seqlens=input_.seqlens["packed_prompts"],
            data=dict(packed_input_ids=input_.data["packed_prompts"]),
        )

        self.generation_config.greedy = self.force_greedy
        res = module.generate(
            input_=x,
            tokenizer=model.tokenizer,
            gconfig=self.generation_config,
        )
        if res is None:
            return None

        gen_tokens, logprobs, logits_mask, *_ = res

        pad_token_id = model.tokenizer.pad_token_id
        eos_token_id = model.tokenizer.eos_token_id
        seq_no_eos_mask = (gen_tokens[:, -1] != eos_token_id).logical_and(gen_tokens[:, -1] != pad_token_id)
        # We also want gen_lengths to include the eos token, where the reward model outputs a score for this sequence.
        gen_lengths = (gen_tokens != pad_token_id).logical_and(gen_tokens != eos_token_id).sum(dim=-1) + 1
        gen_lengths = gen_lengths.clip(max=gen_tokens.shape[-1])

        (
            packed_input_ids,
            _,
            _,
            seq_lengths,
            prompt_mask,
        ) = concat_prompt_to_generation_output(
            packed_prompts=input_.data["packed_prompts"],
            prompt_lengths=torch.cat(input_.seqlens["packed_prompts"]).to(model.device),
            gen_tokens=gen_tokens,
            logprobs=logprobs,
            logits_mask=logits_mask,
            gen_lengths=gen_lengths,
        )

        seqlens = [torch.tensor([s], dtype=torch.int32) for s in seq_lengths.cpu().numpy().tolist()]
        # NOTE: seq_no_eos_mask marks whether generation terminates with an EOS token.
        # We may want to mask out the reward if the generation terminates when
        # reaching the maximum sequence length.
        data = dict(
            seq_no_eos_mask=seq_no_eos_mask,
            packed_input_ids=packed_input_ids,
            prompt_mask=prompt_mask,
        )
        if self.force_greedy:
            data = {f"greedy_{k}": v for k, v in data.items()}

        res = SequenceSample.from_default(
            ids=input_.ids,
            seqlens=seqlens,
            data=data,
        )
        return res

    def train_step(self, model: model_api.Model, input_: SequenceSample) -> Dict:
        module = model.module
        module.eval()

        stats = module.train_batch(
            input_=input_,
            version_steps=model.version.global_step,
            loss_fn=_reinforce_loss_from_model_outputs,
        )

        model.inc_version()

        if stats:
            bs = int(stats["bs"])
            stats = dict(
                loss=float(stats["loss"] / bs),
                baseline_rewards=float(stats["baseline_rewards"] / bs),
                rewards=float(stats["rewards"] / bs),
            )
            stats["advantage"] = stats["rewards"] - stats["baseline_rewards"]

        return dict(stats) if stats else {}


model_api.register_interface("reinforce", ReinforceInterface)


@dataclasses.dataclass
class ReinforceRewardInterface(model_api.ModelInterface):
    output_scaling: float = 1.0
    output_bias: float = 0.0
    force_greedy: bool = False

    @torch.no_grad()
    def inference(self, model: model_api.Model, input_: SequenceSample) -> SequenceSample:
        module = model.module
        module.eval()

        if self.force_greedy:
            input_ = SequenceSample.from_default(
                ids=input_.ids,
                data={k.replace("greedy_", ""): v for k, v in input_.data.items()},
                seqlens=input_.seqlens["greedy_packed_input_ids"],
            )

        r = module.forward(input_=input_)
        if r is None:
            return
        scores = r.float()

        input_lens = torch.cat(input_.seqlens["packed_input_ids"])
        scores = scores.squeeze(-1)[input_lens.cumsum(0) - 1].float()  # [bs]
        scores = (scores - self.output_bias) * self.output_scaling

        ###################### logging ######################
        # input_ids = [packed_input_ids[start:end] for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:])]
        # seq_strs = model.tokenizer.batch_decode(input_ids,
        #                                         clean_up_tokenization_spaces=False,
        #                                         skip_special_tokens=True)
        # for seq_str, score in zip(seq_strs, scores):
        #     logger.info(
        #         f"reward is {colorama.Fore.RED}{score.item()}{colorama.Style.RESET_ALL}, "
        #         f"sequence is: {colorama.Fore.YELLOW + colorama.Style.DIM}{seq_str}{colorama.Style.RESET_ALL}"
        #     )
        #####################################################
        ret_data = dict(rewards=scores)
        if self.force_greedy:
            ret_data = dict(greedy_rewards=scores)
        res = SequenceSample.from_default(
            ids=input_.ids,
            seqlens=input_.seqlens["packed_input_ids"],
            data=ret_data,
        )
        return res

model_api.register_interface("reinforce_reward", ReinforceRewardInterface)