import dataclasses

import colorama
import torch

import realhf.api.core.model_api as model_api
from realhf.api.core.data_api import SequenceSample
from realhf.base import constants, logging

logger = logging.getLogger("Generation Interface", "benchmark")


@dataclasses.dataclass
class GenerationInterface(model_api.ModelInterface):
    generation_config: model_api.GenerationHyperparameters = dataclasses.field(
        default_factory=model_api.GenerationHyperparameters
    )

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
            gconfig=self.generation_config,
            num_micro_batches=n_mbs,
        )
        if res is None:
            return None

        gen_tokens, logprobs, *_ = res

        # Decode and log the first generated sentence.
        l = input_.seqlens["packed_prompts"][0][0]
        tokens = torch.cat(
            [input_.data["packed_prompts"][:l], gen_tokens[0]]
        ).unsqueeze(0)
        out = model.tokenizer.batch_decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        if constants.model_parallel_rank() == 0 and constants.is_last_pipe_stage():
            dp_rank = constants.data_parallel_rank()
            logger.info(
                f"DP rank {dp_rank}, the first generated sequence "
                f"is: {colorama.Fore.YELLOW + colorama.Style.DIM}{out[0]}{colorama.Style.RESET_ALL}"
            )

        res = {
            "generated_length": gen_tokens.shape[1],
            "batch_size": gen_tokens.shape[0],
        }
        return res


model_api.register_interface("generation", GenerationInterface)
