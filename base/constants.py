# log format constants
LOG_FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
DATE_FORMAT = "%Y%m%d-%H:%M:%S"

# constants in experiment instance scope

_experiment_name = None
_trial_name = None

# constants in worker/process scope

_grid = None


def set_grid(grid):
    global _grid
    _grid = grid


def grid():
    if _grid == None:
        raise RuntimeError("Global constant `grid` is accessed before set.")
    return _grid
