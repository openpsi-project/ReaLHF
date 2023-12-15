from typing import Callable, List, Optional, Tuple
import dataclasses

import torch
import transformers

from impl.model.backend.pipe_engine.ds_pipe_engine import DeepSpeedPipelineEngine
from impl.model.nn.flash_mqat.flash_generate import GenerationConfig
from impl.model.utils.data import PipeCacheData, PipeTransferData


class StreamPipeEngine(DeepSpeedPipelineEngine):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self,
                packed_input_ids: torch.Tensor,
                cu_seqlens: torch.Tensor,
                input_lens_for_partition: Optional[torch.Tensor] = None):
        pass

    def train_batch(self,
                    packed_input_ids: torch.Tensor,
                    cu_seqlens: torch.Tensor,
                    loss_fn: Callable,
                    input_lens_for_partition: Optional[torch.Tensor] = None,
                    **loss_fn_kwargs):
        pass

    @torch.no_grad()
    def generate(
        self,
        packed_input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        tokenizer: transformers.PreTrainedTokenizerFast,
        gconfig: GenerationConfig = dataclasses.field(default_factory=GenerationConfig),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[PipeCacheData]]:
        pass

    def eval_batch(self,
                   packed_input_ids: torch.Tensor,
                   cu_seqlens: torch.Tensor,
                   loss_fn: Callable,
                   input_lens_for_partition: Optional[torch.Tensor] = None,
                   **loss_fn_kwargs):
        pass

    def _exec_schedule(self, *args, **kwargs):
        raise RuntimeError("StreamPipeEngine does not static schedule execution.")

    def run(self):
        pass
