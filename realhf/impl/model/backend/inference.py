from typing import *
import dataclasses

import torch
import transformers

from realhf.impl.model.backend.pipe_runner import PipelineRunner
from realhf.impl.model.nn.real_llm_api import ReaLModel
from realhf.impl.model.nn.real_llm_generate import GenerationConfig
import realhf.api.core.model_api as model_api
import realhf.base.constants as constants


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
        seqlens_cpu: List[int],
        packed_input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        loss_fn: Callable,
        input_lens_for_partition: Optional[torch.Tensor] = None,
        num_micro_batches: Optional[int] = None,
        **loss_fn_kwargs,
    ):
        if constants.pipe_parallel_world_size() > 1:
            return self.pipe_runner.eval_batch(
                seqlens_cpu=seqlens_cpu,
                packed_input_ids=packed_input_ids,
                cu_seqlens=cu_seqlens,
                loss_fn=loss_fn,
                input_lens_for_partition=input_lens_for_partition,
                num_micro_batches=num_micro_batches,
                **loss_fn_kwargs,
            )
        else:
            max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))
            model_output = self.module(
                packed_input_ids=packed_input_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            ).logits
            _, stat = loss_fn(
                model_output, packed_input_ids, cu_seqlens, **loss_fn_kwargs
            )
            return stat

    def forward(
        self,
        seqlens_cpu: List[int],
        packed_input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        num_micro_batches: Optional[int] = None,
    ):
        if constants.pipe_parallel_world_size() > 1:
            return self.pipe_runner.forward(
                packed_input_ids=packed_input_ids,
                cu_seqlens=cu_seqlens,
                seqlens_cpu=seqlens_cpu,
                num_micro_batches=num_micro_batches,
            )
        else:
            max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))
            return self.module(
                packed_input_ids=packed_input_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            ).logits

    @torch.no_grad()
    def generate(
        self,
        seqlens_cpu: List[int],
        packed_input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        tokenizer: transformers.PreTrainedTokenizerFast,
        gconfig: GenerationConfig = dataclasses.field(
            default_factory=GenerationConfig
        ),
        num_micro_batches: Optional[int] = None,
    ):
        if constants.pipe_parallel_world_size() > 1:
            return self.pipe_runner.generate(
                seqlens_cpu=seqlens_cpu,
                num_micro_batches=num_micro_batches,
                tokenizer=tokenizer,
                packed_input_ids=packed_input_ids,
                cu_seqlens=cu_seqlens,
                gconfig=gconfig,
            )
        else:
            max_seqlen = int(max(cu_seqlens[1:] - cu_seqlens[:-1]))
            res = self.module.generate(
                tokenizer=tokenizer,
                packed_input_ids=packed_input_ids,
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
