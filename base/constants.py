# log format constants
from base.topology import PipelineParallelGrid

# constants in experiment instance scope

_experiment_name = None
_trial_name = None

# constants in worker/process scope

_grid: PipelineParallelGrid = None


def set_grid(grid):
    global _grid
    _grid = grid


def grid() -> PipelineParallelGrid:
    if _grid == None:
        raise RuntimeError("Global constant `grid` is accessed before set.")
    return _grid


def pipe_parallel_rank() -> int:
    return grid().get_pipe_parallel_rank()


def pipe_parallel_world_size() -> int:
    return grid().get_pipe_parallel_world_size()


def pipe_parallel_group():
    return grid().get_pipe_parallel_group()


def model_parallel_rank() -> int:
    return grid().get_model_parallel_rank()


def model_parallel_world_size() -> int:
    return grid().get_model_parallel_world_size()


def model_parallel_group():
    return grid().get_model_parallel_group()


def data_parallel_rank() -> int:
    return grid().get_data_parallel_rank()


def data_parallel_world_size() -> int:
    return grid().get_data_parallel_world_size()


def data_parallel_group():
    return grid().get_data_parallel_group()
