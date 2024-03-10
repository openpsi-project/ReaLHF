from typing import Dict, List, Optional
import argparse
import getpass
import os
import re

import profiler.experiments

from apps.main import _submit_workers
import api.config as config_package
import base.logging as logging
import base.name_resolve
import base.names
import scheduler.client
import system

logger = logging.getLogger("main", "system")

CONTROLLER_TIME_LIMIT = None
TRACE_TIMEOUT = (
    300  # Should be larger than TRACER_SAVE_INTERVAL_SECONDS defined in system/worker_base.py
)


def main_start(args):
    trial_name = args.trial_name or f"test-{getpass.getuser()}"
    expr_name = args.experiment_name
    experiment = config_package.make_experiment(args.experiment_name)
    sched = scheduler.client.make(mode="slurm", expr_name=expr_name, trial_name=trial_name)

    setup = experiment.scheduling_setup()

    base_environs = {
        "PYTHONPATH": os.path.dirname(os.path.dirname(__file__)),
        "WANDB_MODE": args.wandb_mode,
        "DLLM_MODE": args.mode.upper(),
        "DLLM_TRACE": "1" if args.trace else "0",
    }

    logger.info(f"Resetting name resolving repo...")
    try:
        base.name_resolve.clear_subtree(
            base.names.trial_root(experiment_name=args.experiment_name, trial_name=args.trial_name))
    except Exception as e:
        logger.warning(f"Resetting name resolving repo failed.")
        raise e
    logger.info(f"Resetting name resolving repo... Done.")

    logger.info(f"Running configuration: {experiment.__class__.__name__}")

    # Schedule controller
    controller_type = "zmq"
    # For local_ray mode, the controller will start all remote workers.
    sched.submit_array(
        worker_type="ctl",
        cmd=scheduler.client.control_cmd(
            expr_name,
            trial_name,
            args.debug,
            args.ignore_worker_error,
            controller_type,
        ),
        count=1,
        cpu=1,
        gpu=0,
        mem=1024,
        env_vars=base_environs,
        container_image=args.image_name or setup.controller_image,
        time_limit=CONTROLLER_TIME_LIMIT,
    )

    workers_configs = ((k, getattr(setup, k)) for k in system.WORKER_TYPES)

    for name, scheduling_setup in workers_configs:
        if not isinstance(scheduling_setup, list):
            scheduling_setup = [scheduling_setup]
        # For local or slurm mode, launch all workers.
        # For ray mode, launch the ray cluster for all workers via slurm.
        _submit_workers(
            sched,
            expr_name,
            trial_name,
            args.debug,
            name,
            scheduling_setup,
            base_environs,
            args.image_name,
            use_ray_cluster=(args.mode == "ray"),
        )

    timeout = None if not args.trace else TRACE_TIMEOUT  # run 5 mins to collect trace
    try:
        sched.wait(timeout=timeout)
    except (KeyboardInterrupt, scheduler.client.JobException, TimeoutError) as e:
        if args.trace and isinstance(e, TimeoutError):
            s = "#" * 30 + "  Trace complete. Killing all processes...  " + "#" * 30
            logger.info("\n" + "#" * len(s) + "\n" + s + "\n" + "#" * len(s))
        sched.stop_all()
        raise e


def main():
    parser = argparse.ArgumentParser(prog="profile")
    subparsers = parser.add_subparsers(dest="cmd", help="sub-command help")
    subparsers.required = True

    subparser = subparsers.add_parser("start", help="starts a profile experiment")
    subparser.add_argument("--experiment_name", "-e", type=str, required=True, help="name of the experiment")
    subparser.add_argument(
        "--trial_name",
        "-f",
        type=str,
        default=None,
        help="trial name; by default uses '<USER>-test'",
    )
    subparser.add_argument("--mode", default="slurm", choices=["slurm"])
    subparser.add_argument("--partition", default="dev", help="slurm partition to schedule the trial")
    subparser.add_argument("--wandb_mode",
                           type=str,
                           default="disabled",
                           choices=["online", "offline", "disabled"])
    subparser.add_argument(
        "--image_name",
        type=str,
        required=False,
        default=None,
        help="if specified, all workers will use this image. Useful in CI/CD pipeline.",
    )
    subparser.add_argument("--ignore_worker_error", action="store_true")
    subparser.add_argument("--debug",
                           action="store_true",
                           help="If True, activate all assertions in the code.")
    subparser.add_argument(
        "--remote_reset",
        action="store_true",
        help="If True, reset name resolve repo remotely in computation nodes. Otherwise reset locally.",
    )
    subparser.add_argument(
        "--trace",
        action="store_true",
        help="Whether to use VizTracer to trace the execution time of each line of python code.",
    )
    subparser.set_defaults(ignore_worker_error=False)
    subparser.set_defaults(func=main_start)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
