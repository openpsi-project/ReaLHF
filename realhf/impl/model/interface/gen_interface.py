import dataclasses
import fcntl
import json
import os

import colorama
import torch

import realhf.api.core.model_api as model_api
from realhf.api.core.data_api import SequenceSample
from realhf.base import constants, logging
from realhf.base.datapack import flat2d

logger = logging.getLogger("Generation Interface", "benchmark")


def acquire_lock(lock_file):
    fd = open(lock_file, "w")
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def release_lock(lock_fd):
    fcntl.flock(lock_fd, fcntl.LOCK_UN)
    lock_fd.close()


def write_dict_to_jsonl(dict_data, file_path, lock_file):
    lock_fd = acquire_lock(lock_file)
    try:
        with open(file_path, "a") as file:
            json.dump(dict_data, file)
            file.write("\n")
    finally:
        release_lock(lock_fd)


@dataclasses.dataclass
class GenerationInterface(model_api.ModelInterface):
    output_file: str | None = None
    generation_config: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self.gconfig = model_api.GenerationHyperparameters(**self.generation_config)

    @torch.no_grad()
    def generate(
        self,
        model: model_api.Model,
        input_: SequenceSample,
        n_mbs=None,
    ) -> SequenceSample:
        module = model.module

        module.eval()

        # Remap the key `packed_prompts` to `packed_input_ids`,
        # because the pipe runner only recognizes `packed_input_ids`.
        x = SequenceSample.from_default(
            ids=input_.ids,
            seqlens=input_.seqlens["packed_prompts"],
            data=dict(packed_input_ids=input_.data["packed_prompts"]),
        )

        res = module.generate(
            input_=x,
            tokenizer=model.tokenizer,
            gconfig=self.gconfig,
            num_micro_batches=n_mbs,
        )
        if res is None:
            return None

        gen_tokens, *_ = res

        res = {
            "generated_length": gen_tokens.shape[1],
            "batch_size": gen_tokens.shape[0],
        }
        if not (
            constants.model_parallel_rank() == 0 and constants.is_last_pipe_stage()
        ):
            # Not DP head, return stats.
            return res

        if self.output_file is not None:

            # Concatenate prompts with gen_tokens, decode, and output to file.
            prompt_lens = flat2d(input_.seqlens["packed_prompts"])
            gen_lengths = (gen_tokens != model.tokenizer.pad_token_id).logical_and(
                gen_tokens != model.tokenizer.eos_token_id
            ).sum(dim=-1) + 1
            gen_lengths = gen_lengths.clip(max=gen_tokens.shape[-1])
            assert len(gen_lengths) == len(prompt_lens) == input_.bs, (
                input_.bs,
                len(prompt_lens),
                len(gen_lengths),
            )

            prompt_tokens_lis = []
            ans_tokens_lis = []
            prompt_offset = 0
            for i, (prompt_len, gen_len) in enumerate(zip(prompt_lens, gen_lengths)):
                prompt_tokens_lis.append(
                    input_.data["packed_prompts"][
                        prompt_offset : prompt_offset + prompt_len
                    ]
                )
                ans_tokens_lis.append(gen_tokens[i, :gen_len])
                prompt_offset += prompt_len
            assert prompt_offset == sum(prompt_lens)
            seq_tokens_lis = [
                torch.cat([a, b]) for a, b in zip(prompt_tokens_lis, ans_tokens_lis)
            ]

            prompt_str = model.tokenizer.batch_decode(
                prompt_tokens_lis,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            ans_str = model.tokenizer.batch_decode(
                ans_tokens_lis,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            seq_str = model.tokenizer.batch_decode(
                seq_tokens_lis,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            lock_file = os.path.join(
                constants.LOG_ROOT,
                constants.experiment_name(),
                constants.trial_name(),
                "_gen.lock",
            )
            output_file = os.path.join(
                constants.LOG_ROOT,
                constants.experiment_name(),
                constants.trial_name(),
                self.output_file,
            )
            if constants.data_parallel_rank() == 0:
                logger.info(f"Dumping output to: {output_file}...")
            for p, a, s, _id in zip(prompt_str, ans_str, seq_str, input_.ids):
                d = dict(
                    prompt=p,
                    answer=a,
                    seq=s,
                    id=_id,
                )
                write_dict_to_jsonl(d, output_file, lock_file)
        else:
            # Decode and log the first generated sentence.
            l = input_.seqlens["packed_prompts"][0][0]
            tokens = torch.cat(
                [input_.data["packed_prompts"][:l], gen_tokens[0]]
            ).unsqueeze(0)
            out = model.tokenizer.batch_decode(
                tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            dp_rank = constants.data_parallel_rank()
            logger.info(
                f"DP rank {dp_rank}, the first generated sequence "
                f"is: {colorama.Fore.YELLOW + colorama.Style.DIM}{out[0]}{colorama.Style.RESET_ALL}"
            )

        return res


model_api.register_interface("generation", GenerationInterface)
