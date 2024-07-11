import dataclasses
from typing import *

import torch
import transformers

import realhf.api.core.model_api as model_api
import realhf.base.constants as constants
from realhf.api.core.data_api import SequenceSample
from realhf.impl.model.backend.pipe_runner import PipelineRunner
from realhf.impl.model.nn.real_llm_api import ReaLModel


class PipelinableInferenceEngine:

    def __init__(self, module: ReaLModel):
        self.module = module

        self.device = module.device
        self.dtype = module.dtype

        if constants.pipe_parallel_world_size() > 1:
            self.pipe_runner = PipelineRunner(module)

    def train(self, mode: bool = True):
        self.module.train(mode)
        return self

    def eval(self):
        self.module.eval()
        return self

    @torch.no_grad()
    def eval_batch(
        self,
        input_: SequenceSample,
        loss_fn: Callable,
        num_micro_batches: Optional[int] = None,
    ):
        if constants.pipe_parallel_world_size() > 1:
            return self.pipe_runner.eval_batch(
                input_=input_,
                loss_fn=loss_fn,
                num_micro_batches=num_micro_batches,
            )
        else:
            input_lens = torch.cat(input_.seqlens["packed_input_ids"], dim=0).cuda()
            max_seqlen = int(max(input_lens))
            cu_seqlens = torch.nn.functional.pad(input_lens.cumsum(0), (1, 0)).int()
            model_output = self.module(
                packed_input_ids=input_.data["packed_input_ids"],
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            ).logits
            _, stat = loss_fn(model_output, input_)
            return stat

    def forward(
        self,
        input_: SequenceSample,
        num_micro_batches: Optional[int] = None,
    ):
        if constants.pipe_parallel_world_size() > 1:
            return self.pipe_runner.forward(
                input_=input_,
                num_micro_batches=num_micro_batches,
            )
        else:
            input_lens = torch.cat(input_.seqlens["packed_input_ids"], dim=0).cuda()
            max_seqlen = int(max(input_lens))
            cu_seqlens = torch.nn.functional.pad(input_lens.cumsum(0), (1, 0)).int()
            model_output = self.module(
                packed_input_ids=input_.data["packed_input_ids"],
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            ).logits
            return model_output

    @torch.no_grad()
    def generate(
        self,
        input_: SequenceSample,
        tokenizer: transformers.PreTrainedTokenizerFast,
        gconfig: model_api.GenerationHyperparameters = dataclasses.field(
            default_factory=model_api.GenerationHyperparameters
        ),
        num_micro_batches: Optional[int] = None,
    ):
        if constants.pipe_parallel_world_size() > 1:
            return self.pipe_runner.generate(
                input_=input_,
                num_micro_batches=num_micro_batches,
                tokenizer=tokenizer,
                gconfig=gconfig,
            )
        else:
            input_lens = torch.cat(input_.seqlens["packed_input_ids"], dim=0).cuda()
            max_seqlen = int(max(input_lens))
            cu_seqlens = torch.nn.functional.pad(input_lens.cumsum(0), (1, 0)).int()
            res = self.module.generate(
                tokenizer=tokenizer,
                packed_input_ids=input_.data["packed_input_ids"],
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                gconfig=gconfig,
            )
            return res.sequences, res.scores, res.logits_mask


@dataclasses.dataclass
class PipelineInferenceBackend(model_api.ModelBackend):

    def _initialize(self, model: model_api.Model, spec: model_api.FinetuneSpec):
        model.module = PipelinableInferenceEngine(model.module)
        return model


model_api.register_backend("inference", PipelineInferenceBackend)
