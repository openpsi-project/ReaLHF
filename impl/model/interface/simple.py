# simple interface for testing
import dataclasses

import torch
import torch.nn as nn

from base.namedarray import from_dict, NamedArray, recursive_aggregate, recursive_apply
import api.model
import api.utils


@dataclasses.dataclass
class SimpleInterface(api.model.ModelInterface):
    # a simple interface used to test pipeline models

    @torch.no_grad()
    def inference(self, model: api.model.Model, data: NamedArray) -> NamedArray:
        module = model.module
        module.eval()

        logits = module(data.input_ids, attention_mask=data.attention_mask)
        return from_dict(dict(logits=logits))

    @torch.no_grad()
    def generate(self, model: api.model.Model, data: NamedArray) -> NamedArray:
        module = model.module
        tokenizer = model.tokenizer

        module.eval()

        pad_token_id = model.tokenizer.pad_token_id
        eos_token_id = model.tokenizer.eos_token_id

        # huggingface generate arguments for huggingface models
        # namedarray -> tensor -> namedarray
        seq = module.generate(data.prompts,
                              attention_mask=data.prompt_att_mask,
                              max_new_tokens=50,
                              pad_token_id=pad_token_id,
                              eos_token_id=eos_token_id)

        return from_dict(dict(seq=seq))

    def train_step(self, model: api.model.Model, data: NamedArray) -> NamedArray:
        pass


api.model.register_interface("simple", SimpleInterface)
