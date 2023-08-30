from typing import Type
import importlib
import logging
import os
import traceback

from base.constants import DATE_FORMAT, LOG_FORMAT
import api.config

logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT, level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger("system base")

# NOTE: Workers are configured in the following order.
# Take special care when adding a new worker type.
WORKER_TYPES = ["data_worker", "model_worker", "master_worker"]


class Controller:

    def __init__(self, experiment_name, trial_name):
        assert "_" not in experiment_name, f"_ not allowed in experiment_name (args: -e) " \
                                           f"{experiment_name}, use '-' instead."
        assert "_" not in trial_name, f"_ not allowed in trial_name (args: -f) {trial_name}, use '-' instead."
        self.experiment_name = experiment_name
        self.trial_name = trial_name

        logger.info("Experiment: %s %s", self.experiment_name, self.trial_name)

    def start(self, experiment: api.config.Experiment, ignore_worker_error=False):
        """Start an experiment.
        Args:
            experiment: An experiment class, with `initial_setup` method returning workers configurations.
            ignore_worker_error: If True, do not stop experiment when part of worker(s) fail.
        """
        raise NotImplementedError()


def load_worker(worker_type: str) -> Type:
    assert worker_type in WORKER_TYPES, f"Invalid worker type {worker_type}"
    module = importlib.import_module(worker_type_to_module(worker_type))
    class_name = worker_type_to_class_name(worker_type)
    return getattr(module, class_name)


def worker_type_to_module(worker_type: str):
    return "system." + worker_type


def worker_type_to_class_name(worker_type: str):
    return "".join([w.capitalize() for w in worker_type.split("_")])


def run_worker(worker_type, experiment_name, trial_name, worker_name, worker_server_type):
    """Run one worker
    Args:
        worker_type: string, one of the worker types listed above,
        experiment_name: string, the experiment this worker belongs to,
        trial_name: string, the specific trial this worker belongs to,
        worker_name: name given to the worker, typically "<worker_type>/<worker_index>"
        worker_server_type: string, either 'zmq' or 'ray'.
    """
    worker_class = load_worker(worker_type)
    make_server_fn = getattr(importlib.import_module("system.worker_control"), "make_server")
    server = make_server_fn(type_=worker_server_type,
                            experiment_name=experiment_name,
                            trial_name=trial_name,
                            worker_name=worker_name)
    worker = worker_class(server=server)
    try:
        worker.run()
    except Exception as e:
        logging.error("Worker %s failed with exception: %s", worker_name, e)
        logging.error(traceback.format_exc())
        raise e


def make_controller(type_, *args, **kwargs) -> Controller:
    module = importlib.import_module("system.controller")
    if type_ == 'zmq':
        return getattr(module, "ZmqController")(*args, **kwargs)
    elif type_ == 'ray':
        return getattr(module, "RayController")(*args, **kwargs)
    else:
        raise NotImplementedError(f"Unknown controller type {type_}.")
