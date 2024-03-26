import dataclasses
import api.model
from impl.model.backend.pipe_engine.inf_pipe_engine import InferencePipelineEngine


@dataclasses.dataclass
class PipelineInferenceBackend(api.model.ModelBackend):
    def _initialize(self, model: api.model.Model, spec: api.model.FinetuneSpec):
        model.module = InferencePipelineEngine(model.module)
        return model


api.model.register_backend("pipe_inference", PipelineInferenceBackend)
