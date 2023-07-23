import collections
import dataclasses
import importlib
import logging
import os
import traceback

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
DATE_FORMAT = "%Y%m%d-%H:%M:%S"
logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT, level=os.environ.get("LOGLEVEL", "INFO"))

# NOTE: Workers are configured in the following order.
# Take special care when adding a new worker type.
WORKER_TYPES = ["data_worker", "model_worker", "master_worker"]


def load_worker(worker_type: str):
    assert worker_type in WORKER_TYPES, f"Invalid worker type {worker_type}"
    module = importlib.import_module(worker_type_to_module(worker_type))
    class_name = worker_type_to_class_name(worker_type)
    return getattr(module, class_name)


def worker_type_to_module(worker_type: str):
    return "system." + worker_type


def worker_type_to_class_name(worker_type: str):
    return "".join([w.capitalize() for w in worker_type.split("_")])


def run_worker(worker_type, experiment_name, trial_name, worker_name):
    """Run one worker
    Args:
        worker_type: string, one of the worker types listed above,
        experiment_name: string, the experiment this worker belongs to,
        trial_name: string, the specific trial this worker belongs to,
        worker_name: name given to the worker, typically "<worker_type>/<worker_index>"
    """
    worker_class = load_worker(worker_type)
    server = make_worker_server(experiment_name=experiment_name,
                                trial_name=trial_name,
                                worker_name=worker_name)
    worker = worker_class(server=server)
    try:
        worker.run()
    except Exception as e:
        logging.error("Worker %s failed with exception: %s", worker_name, e)
        logging.error(traceback.format_exc())
        raise e


def make_controller(*args, **kwargs):
    """Make a distributed reinforcement learning controller.
    Returns:
        a controller.
    """
    module = importlib.import_module("system.controller")
    return getattr(module, "Controller")(*args, **kwargs)


def make_worker_server(*args, **kwargs):
    """Make a worker server, so we can establish remote control to the worker.
    Returns:
        a worker server.
    """
    module = importlib.import_module("system.worker_control")
    return getattr(module, "make_server")(*args, **kwargs)
