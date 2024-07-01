import dataclasses
from typing import Dict, Optional, Tuple

import torch

import realhf.api.core.model_api as model_api
from realhf.base.namedarray import NamedArray, recursive_apply


@dataclasses.dataclass
class GenerationInterface(model_api.ModelInterface):
    generation_config: model_api.GenerationHyperparameters = dataclasses.field(
        default_factory=model_api.GenerationHyperparameters
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
        prompt_cu_seqlens = torch.nn.functional.pad(prompt_lengths.cumsum(0), (1, 0))

        res = module.generate(
            seqlens_cpu=data.metadata["seqlens"],
            tokenizer=model.tokenizer,
            packed_input_ids=packed_prompts,
            cu_seqlens=prompt_cu_seqlens,
            gconfig=self.generation_config,
        )
        if res is None:
            return None

        gen_tokens, logprobs, *_ = res
        res = {
            "generated_length": gen_tokens.shape[1],
            "batch_size": gen_tokens.shape[0],
        }
        return res


model_api.register_interface("generation", GenerationInterface)
