import argparse
import json
import logging
import multiprocessing
import os

multiprocessing.set_start_method("spawn", force=True)

import base.gpu_utils
import base.name_resolve

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
DATE_FORMAT = "%Y%m%d-%H:%M:%S"
logger = logging.getLogger("Main-Workers")


def main_reset_name_resolve(args):
    base.name_resolve.clear_subtree(
        base.names.trial_root(experiment_name=args.experiment_name, trial_name=args.trial_name))


def main_worker(args):
    base.gpu_utils.isolate_cuda_device(args.worker_type, args.group_id, args.group_size, args.experiment_name,
                                       args.trial_name)
    import experiments
    import impl.data
    import impl.model
    import system

    logger.info(f"Run {args.worker_type} worker with args: %s", args)
    assert not args.experiment_name.startswith(
        "/"), f"Invalid experiment_name \"{args.experiment_name}\" starts with \"/\""
    system.run_worker(worker_type=args.worker_type,
                      experiment_name=args.experiment_name,
                      trial_name=args.trial_name,
                      worker_name=f"{args.worker_type}/{args.group_id}")


def main_controller(args):
    """
    Args:
        args: argparse result including:
            experiment_name:
            trial_name:
            config_index: the index of experiment configuration (experiment may return multiple configurations)
            ignore_worker_error: bool, if False, stop the experiment when any worker(s) fail.
    """
    import api.config
    import experiments
    import system
    logger.info("Running controller with args: %s", args)
    assert not args.experiment_name.startswith("/"), args.experiment_name
    controller = system.make_controller(experiment_name=args.experiment_name, trial_name=args.trial_name)
    experiment = api.config.make_experiment(args.experiment_name)
    controller.start(
        experiment=experiment,
        ignore_worker_error=args.ignore_worker_error,
    )


def main():
    logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT, level=os.environ.get("LOGLEVEL", "INFO"))

    parser = argparse.ArgumentParser(prog="marl")
    subparsers = parser.add_subparsers(dest="cmd", help="sub-command help")
    subparsers.required = True

    subparser = subparsers.add_parser("controller", help="run a controller of experiment")
    subparser.add_argument("--experiment_name", "-e", type=str, required=True)
    subparser.add_argument("--trial_name", "-f", type=str, required=True)
    subparser.add_argument("--ignore_worker_error", action="store_true")
    subparser.add_argument('--raise_worker_error', dest='ignore_worker_error', action='store_false')
    subparser.set_defaults(feature=False)
    subparser.set_defaults(func=main_controller)

    subparser = subparsers.add_parser("worker", help="run a standalone worker")
    subparser.add_argument("--worker_type", '-w', type=str, required=True)
    subparser.add_argument("--experiment_name", "-e", type=str, required=True)
    subparser.add_argument("--trial_name", "-f", type=str, required=True)
    subparser.add_argument("--group_id", "-i", type=int, required=True)
    subparser.add_argument("--group_size", "-g", type=int, required=False, default=1)
    subparser.set_defaults(func=main_worker)

    subparser = subparsers.add_parser("reset_name_resolve", help="reset name resolve repo for a trial")
    subparser.add_argument("--experiment_name", "-e", type=str, required=True)
    subparser.add_argument("--trial_name", "-f", type=str, required=True)
    subparser.set_defaults(func=main_reset_name_resolve)

    args = parser.parse_args()

    args.func(args)


if __name__ == "__main__":
    main()
