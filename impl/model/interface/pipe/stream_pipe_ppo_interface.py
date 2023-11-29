import asyncio
import dataclasses

import torch

from base.namedarray import NamedArray
from impl.model.backend.stream_pipe_engine import StreamPipeEngine
import api.model
import base.logging as logging

logger = logging.getLogger("stream_pipe_ppo")


def actor_loss_fn(logits: torch.Tensor, packed_input_ids: torch.Tensor, cu_seqlens: torch.Tensor,
                  **kwargs) -> torch.Tensor:
    return torch.tensor(1)


@dataclasses.dataclass
class StreamPipeActorInterface(api.model.ModelInterface):

    def train_step(self, model: api.model.Model, data: NamedArray) -> asyncio.Future:
        engine = model.module
        assert isinstance(engine, StreamPipeEngine)

        future = engine.train_batch(data, actor_loss_fn)
        return future

    def init_stream_generate(self, model: api.model.Model, data: NamedArray) -> asyncio.Future:
        """ put all batched data into engine for stream generate, 
            return futures of generate results. 
        """
        engine = model.module
        assert isinstance(engine, StreamPipeEngine)

        futures = engine.init_stream_generate(data)
        return futures
