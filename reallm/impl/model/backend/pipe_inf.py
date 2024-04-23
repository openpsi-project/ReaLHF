import dataclasses

from reallm.impl.model.backend.pipe_engine.inf_pipe_engine import InferencePipelineEngine
import reallm.api.core.model as model_api


@dataclasses.dataclass
class PipelineInferenceBackend(model_api.ModelBackend):

    def _initialize(self, model: model_api.Model, spec: model_api.FinetuneSpec):
        model.module = InferencePipelineEngine(model.module)
        return model


model_api.register_backend("pipe_inference", PipelineInferenceBackend)
