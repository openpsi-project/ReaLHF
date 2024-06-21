from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import dataclasses
import gc
import os
import time

import torch

import realhf.base.constants as constants
import realhf.impl.model.parallelism.model_parallel.custom_all_reduce as custom_all_reduce

CUDA_GRAPH_STORAGE: Dict[str, torch.cuda.CUDAGraph] = dict()
CUDA_GRAPH_INPUT_BUFFER: Dict[str, Dict[str, torch.Tensor]] = dict()
CUDA_GRAPH_OUTPUT_BUFFER: Dict[str, Dict[str, torch.Tensor]] = dict()


# @dataclasses.dataclass
# class TensorMetadata:
#     """Metadata for a pytorch tensor in GPU."""

#     shape: Tuple[int]
#     dtype: torch.dtype

#     @staticmethod
#     def from_tensor(t: torch.Tensor):
#         return TensorMetadata(t.shape, t.dtype)

#     def to_tensor(self):
#         return torch.zeros(self.shape, dtype=self.dtype, device="cuda")

# def metadata_to_buffer(metadata: Dict[str, Union[List[Any], Any]]):
#     buf = {}
#     for k, v in metadata.items():
#         if isinstance(v, list):
#             buf[k] = [vv.to_tensor() if isinstance(vv, TensorMetadata) else vv
#                       for vv in v]
#         elif isinstance(v, TensorMetadata):
#             buf[k] = v.to_tensor()
#         else:
#             buf[k] = v
#     return buf

# def buffer_to_metadata(buffer: Dict[str, Union[List[Any], Any]]):
#     metadata = {}
#     for k, v in buffer.items():
#         if isinstance(v, list):
#             metadata[k] = [TensorMetadata.from_tensor() if torch.is_tensor(v) else vv
#                            for vv in v]
#         elif torch.is_tensor(v):
#             metadata[k] = TensorMetadata.from_tensor()
#         else:
#             metadata[k] = v
#     return metadata


@torch.no_grad()
def capture_func(
    name: str,
    func: Callable,
    input_buffer: Dict[str, Any],
    force_recapture: bool = False,
    no_grad: bool = False,
) -> Tuple[
    torch.cuda.CUDAGraph, Dict[str, torch.Tensor], Dict[str, torch.Tensor]
]:
    """Capture a function with cuda graph, store the graph and input/output buffers by name.
    The input/output metadata should match the inputs and outputs of function.

    Args:
        name: The name of the function.
        func: The function to be captured.
        input_metadata: The metadata of input tensors.
        output_metadata: The metadata of output tensors.
        force_recapture: Whether to force recapture the function.
    """
    use_cuda_graph = os.environ.get("USE_CUDA_GRAPH", "0") == "1"
    if not use_cuda_graph:
        return None, None, None

    global CUDA_GRAPH_STORAGE
    global CUDA_GRAPH_INPUT_BUFFER
    global CUDA_GRAPH_OUTPUT_BUFFER
    if not force_recapture:
        if name in CUDA_GRAPH_STORAGE:
            assert name in CUDA_GRAPH_INPUT_BUFFER
            assert name in CUDA_GRAPH_OUTPUT_BUFFER
            return (
                CUDA_GRAPH_STORAGE[name],
                CUDA_GRAPH_INPUT_BUFFER[name],
                CUDA_GRAPH_OUTPUT_BUFFER[name],
            )

    print(f"initializing ca")
    if not custom_all_reduce.is_initialized():
        custom_all_reduce.init_custom_ar()
    print(f"initialized ca")
    assert (
        custom_all_reduce.is_initialized()
        or constants.model_parallel_world_size() == 1
    )
    assert not constants.sequence_parallel()

    maybe_no_grad = nullcontext() if not no_grad else torch.no_grad()
    st = time.monotonic()
    with custom_all_reduce.graph_capture(), maybe_no_grad:
        print(f"rank {torch.distributed.get_rank()}: Warmup for capture {name}")
        func(**input_buffer)
        torch.cuda.synchronize()

        print(
            f"rank {torch.distributed.get_rank()}: Capturing CUDA graph for {name}"
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = func(**input_buffer)

        torch.cuda.synchronize()

        print(
            f"rank {torch.distributed.get_rank()}: Capturing CUDA graph {name} "
            f"for decoding takes {time.monotonic() - st:.2f} seconds."
        )

    assert torch.is_tensor(output)
    output_buffer = dict(output=output)

    gc.collect()
    torch.cuda.empty_cache()

    CUDA_GRAPH_STORAGE[name] = graph
    CUDA_GRAPH_INPUT_BUFFER[name] = input_buffer
    CUDA_GRAPH_OUTPUT_BUFFER[name] = output_buffer
    return graph, input_buffer, output_buffer


def input_buffer_handle(graph_name: str, tensor_name: str):
    if graph_name not in CUDA_GRAPH_INPUT_BUFFER:
        return None
    if tensor_name not in CUDA_GRAPH_INPUT_BUFFER[graph_name]:
        raise ValueError(
            f"Tensor {tensor_name} not found in input buffer of graph {graph_name}, "
            f"Existing keys = {CUDA_GRAPH_INPUT_BUFFER[graph_name].keys()}"
        )
    return CUDA_GRAPH_INPUT_BUFFER[graph_name][tensor_name]


def output_buffer_handle(graph_name: str, tensor_name: str):
    if graph_name not in CUDA_GRAPH_OUTPUT_BUFFER:
        return None
    if tensor_name not in CUDA_GRAPH_OUTPUT_BUFFER[graph_name]:
        raise ValueError(
            f"Tensor {tensor_name} not found in output buffer of graph {graph_name}, "
            f"existing keys = {CUDA_GRAPH_OUTPUT_BUFFER[graph_name].keys()}"
        )
    return CUDA_GRAPH_OUTPUT_BUFFER[graph_name][tensor_name]


def get_graph(name: str):
    return CUDA_GRAPH_STORAGE.get(name, None)
