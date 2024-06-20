from contextlib import nullcontext
from typing import Callable, Dict, List, Optional, Tuple, Union
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


@dataclasses.dataclass
class TensorMetadata:
    """Metadata for a pytorch tensor in GPU."""

    shape: Optional[Tuple[int]] = None
    dtype: Optional[torch.dtype] = None

    @staticmethod
    def from_tensor(t: Optional[torch.Tensor]):
        if t is None:
            return TensorMetadata()
        return TensorMetadata(t.shape, t.dtype)

    def to_tensor(self):
        if self.shape is None:
            assert self.dtype is None
            return None
        return torch.zeros(self.shape, dtype=self.dtype, device="cuda")


@torch.no_grad()
def capture_func(
    name: str,
    func: Callable,
    input_metadata: Dict[str, Union[List[TensorMetadata], TensorMetadata]],
    output_metadata: Dict[
        str, Union[List[TensorMetadata], TensorMetadata]
    ] = None,
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

    if not custom_all_reduce.is_initialized():
        custom_all_reduce.init_custom_ar()
    assert (
        custom_all_reduce.is_initialized()
        or constants.model_parallel_world_size() == 1
    )
    assert not constants.sequence_parallel()

    input_buffer = {
        k: (
            v.to_tensor()
            if not isinstance(v, list)
            else [vv.to_tensor() for vv in v]
        )
        for k, v in input_metadata.items()
    }

    maybe_no_grad = nullcontext() if not no_grad else torch.no_grad()
    st = time.monotonic()
    with custom_all_reduce.graph_capture(), maybe_no_grad:
        func(**input_buffer)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = func(**input_buffer)

        torch.cuda.synchronize()

    print(
        f"Capturing CUDA graph {name} for decoding takes {time.monotonic() - st:.2f} seconds."
    )

    if output_metadata is not None:
        output_buffer = {
            k: (
                v.to_tensor()
                if not isinstance(v, list)
                else [vv.to_tensor() for vv in v]
            )
            for k, v in output_metadata.items()
        }
    else:
        if torch.is_tensor(output):
            output_buffer = dict(output=output)
        elif isinstance(output, (list, tuple)):
            output_buffer = dict(zip(range(len(output)), output))
        elif dataclasses.is_dataclass(output):
            output_buffer = dataclasses.asdict(output)
        elif isinstance(output, dict):
            output_buffer = output

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
            f"Tensor {tensor_name} not found in input buffer of graph {graph_name}"
        )
    return CUDA_GRAPH_INPUT_BUFFER[graph_name][tensor_name]


def output_buffer_handle(graph_name: str, tensor_name: str):
    if graph_name not in CUDA_GRAPH_OUTPUT_BUFFER:
        return None
    if tensor_name not in CUDA_GRAPH_INPUT_BUFFER[graph_name]:
        raise ValueError(
            f"Tensor {tensor_name} not found in output buffer of graph {graph_name}"
        )
    return CUDA_GRAPH_OUTPUT_BUFFER[graph_name][tensor_name]
