import dataclasses
import gc
import os
import time
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

import realhf.base.constants as constants
import realhf.base.logging as logging
import realhf.impl.model.parallelism.model_parallel.custom_all_reduce as custom_all_reduce

logger = logging.getLogger("CUDAGraph")

CUDA_GRAPH_STORAGE: Dict[str, torch.cuda.CUDAGraph] = dict()
CUDA_GRAPH_INPUT_BUFFER: Dict[str, Dict[str, torch.Tensor]] = dict()
CUDA_GRAPH_OUTPUT_BUFFER: Dict[str, Dict[str, torch.Tensor]] = dict()


def reinitialize_input_buffer(cuda_graph_name, new_buf):
    global CUDA_GRAPH_INPUT_BUFFER
    assert (
        cuda_graph_name in CUDA_GRAPH_INPUT_BUFFER
    ), f"CUDAGraph {cuda_graph_name} does not exist."

    buf = CUDA_GRAPH_INPUT_BUFFER[cuda_graph_name]
    for k, v in buf.items():
        if torch.is_tensor(v):
            v.copy_(new_buf[k])
        elif isinstance(v, list):
            for i, vv in enumerate(v):
                if torch.is_tensor(vv):
                    vv.copy_(new_buf[k][i])
        else:
            buf[k] = new_buf[k]


@torch.no_grad()
def capture_func(
    name: str,
    func: Callable,
    input_buffer: Dict[str, Any],
    force_recapture: bool = False,
    no_grad: bool = False,
) -> Tuple[torch.cuda.CUDAGraph, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Capture a function with cuda graph, store the graph and input/output buffers by name.
    The input/output metadata should match the inputs and outputs of function.

    :param name: The identifier of the CUDAGraph to be captured/reused.
    :type name: str
    :param func: The function to be captured.
    :type func: Callable
    :param input_buffer: The input buffer of the function.
    :type input_buffer: Dict[str, Any]
    :param force_recapture: Whether to force recapture the function.
    :type force_recapture: bool
    :param no_grad: Whether to run the function in no_grad context.
    :type no_grad: bool
    """
    global CUDA_GRAPH_STORAGE
    global CUDA_GRAPH_INPUT_BUFFER
    global CUDA_GRAPH_OUTPUT_BUFFER
    if not force_recapture:
        if name in CUDA_GRAPH_STORAGE:
            reinitialize_input_buffer(name, input_buffer)
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
        custom_all_reduce.is_initialized() or constants.model_parallel_world_size() == 1
    )
    assert not constants.sequence_parallel()

    maybe_no_grad = nullcontext() if not no_grad else torch.no_grad()
    st = time.monotonic()

    with custom_all_reduce.graph_capture(), maybe_no_grad:
        logger.info(f"Rank {dist.get_rank()}: Capturing CUDA graph for {name}")
        func(**input_buffer)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = func(**input_buffer)

        torch.cuda.synchronize()

        logger.info(
            f"Rank {dist.get_rank()}: Capturing CUDA graph {name} "
            f"for decoding takes {time.monotonic() - st:.2f} seconds."
        )

    assert torch.is_tensor(output)
    output_buffer = dict(output=output)

    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

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
