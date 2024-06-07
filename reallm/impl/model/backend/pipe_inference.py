from typing import *
import dataclasses

from reallm.impl.model.parallelism.pipeline_parallel.pipe_runner import PipelineRunner
import reallm.api.core.model_api as model_api


@dataclasses.dataclass
class PipelineInferenceBackend(model_api.ModelBackend):

    def _initialize(self, model: model_api.Model, spec: model_api.FinetuneSpec):
        model.module = PipelineRunner(model.module)
        return model


model_api.register_backend("pipe_inference", PipelineInferenceBackend)
