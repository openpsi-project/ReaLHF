# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import logging

from deepspeed import comm as dist
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.engine import DeepSpeedEngine, MEMORY_OPT_ALLREDUCE_SIZE
import zmq

from . import p2p, schedule
from .pipe_module import PipelineError, PipelineModule


class PipelineEngine(DeepSpeedEngine):

    def __init__(self, *super_args, **super_kwargs):
        super().__init__(*super_args, **super_kwargs)
        assert isinstance(self.module, PipelineModule), "model must base PipelineModule"

        assert self.zero_optimization_stage(
        ) < 2, "ZeRO-2 and ZeRO-3 are incompatible with pipeline parallelism"

        if dist.get_rank() == 0:
            pass

    def generate(self, *kwargs):
        pass

    def stream_generate(self, *kwargs):
        pass

    def inference(self, *kwargs):
        pass

    def forward(self, *kwargs):
        pass

    def backward(self, *kwargs):
        pass

    def step(self, *kwargs):
        pass
