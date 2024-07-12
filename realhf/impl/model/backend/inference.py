import dataclasses
from typing import *

import torch
import torch.distributed as dist
import transformers

import realhf.api.core.model_api as model_api
import realhf.base.constants as constants
import realhf.base.logging as logging
from realhf.api.core.data_api import SequenceSample
from realhf.impl.model.backend.pipe_runner import PipelineRunner
from realhf.impl.model.nn.real_llm_api import ReaLModel

logger = logging.getLogger("PipelinableInferenceEngine")


class PipelinableInferenceEngine:

    def __init__(self, module: ReaLModel):
        self.module = module

        self.device = module.device
        self.dtype = module.dtype

        if constants.pipe_parallel_world_size() > 1:
            self.pipe_runner = PipelineRunner(module)
            self._log_trainable_params()

    def _log_trainable_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.module.parameters())
        num_params = sum([p.numel() for p in model_parameters])
        shared_params = 0
        if self.module.shared_embedding_or_output_weight() is not None:
            shared_params = self.module.shared_embedding_or_output_weight().numel()
        unique_params = num_params - shared_params

        params_tensor = torch.LongTensor(data=[num_params, unique_params]).to(
            self.device
        )
        dist.all_reduce(
            params_tensor, group=constants.grid().get_model_parallel_group()
        )
        params_tensor = params_tensor.tolist()
        total_params = params_tensor[0]
        unique_params = params_tensor[1]

        if constants.parallelism_rank() == 0:
            logger.info(
                f"CONFIG: default_train_mbs={self.pipe_runner.default_train_mbs} "
                f"default_inf_mbs={self.pipe_runner.default_inf_mbs} "
                f"num_layers(this stage)={self.module.num_layers} "
                f"pp_size={constants.pipe_parallel_world_size()} "
                f"dp_size={constants.data_parallel_world_size()} "
                f"mp_size={constants.model_parallel_world_size()} "
            )
        if constants.data_parallel_rank() == 0:
            logger.info(
                f"rank={constants.parallelism_rank()} "
                f"stage={constants.pipe_parallel_rank()} "
                f"layers={self.module.num_layers} "
                f"[{self.module.layer_idx_start}, {self.module.layer_idx_end}) "
                f"stage_params={num_params} ({num_params/1e6:0.3f}M) "
                f"total_params={total_params} ({total_params/1e6:0.3f}M) "
                f"unique_params={unique_params} ({unique_params/1e6:0.3f}M)"
            )

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
