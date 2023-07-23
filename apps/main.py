from typing import List
import argparse
import getpass
import logging
import os
import re

import api.config as config_package
import experiments
import scheduler.client
import system

LOG_FORMAT = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s: %(message)s"
DATE_FORMAT = "%Y%m%d-%H:%M:%S"

logger = logging.getLogger("main")


def main_start(args):

    def submit_workers(expr_name, trial_name, debug, worker_type, scheduling_configs) -> List[str]:
        if len(scheduling_configs) == 0:
            return []

        scheduled_jobs = []
        for sch_cfg in scheduling_configs:
            job_environs = {
                "PYTHONPATH": os.path.dirname(os.path.dirname(__file__)),
                "WANDB_MODE": args.wandb_mode,
                "LOGLEVEL": args.LOGLEVEL,
                **sch_cfg.scheduling.env_vars,
            }
            cmd = scheduler.client.remote_worker_cmd(expr_name, trial_name, debug, worker_type)
            logger.debug(f"Scheduling worker {worker_type}, {scheduling_configs}")

            nodelist = sch_cfg.scheduling.nodelist
            exclude = sch_cfg.scheduling.exclude

            scheduled_jobs.append(
                sched.submit_array(worker_type,
                                   cmd,
                                   count=sch_cfg.count,
                                   cpu=sch_cfg.scheduling.cpu,
                                   gpu=sch_cfg.scheduling.gpu,
                                   gpu_type=sch_cfg.scheduling.gpu_type,
                                   mem=sch_cfg.scheduling.mem,
                                   container_image=args.image_name or sch_cfg.scheduling.container_image,
                                   nodelist=nodelist,
                                   exclude=exclude,
                                   env_vars=job_environs,
                                   hostfile=True))
        return scheduled_jobs

    trial_name = args.trial_name or f"test-{getpass.getuser()}"
    expr_name = args.experiment_name
    experiment = config_package.make_experiment(args.experiment_name)
    sched = scheduler.client.make(mode=args.mode, job_name=f"{args.experiment_name}_{trial_name}")

    setup = experiment.scheduling_setup()

    simple_env_vars = {"PYTHONPATH": os.path.dirname(os.path.dirname(__file__)), "LOGLEVEL": args.LOGLEVEL}

    logger.info(f"Resetting name resolving repo...")
    sched.submit("setup",
                 scheduler.client.setup_cmd(expr_name, trial_name, args.debug),
                 env_vars=simple_env_vars,
                 exclude='frl2g[084-086],frl2g008,frl2g093,frl2g094,frl8g[136-137]')

    sched.wait(timeout=120, update=True)
    logger.info(f"Resetting name resolving repo... Done.")

    logger.info(f"Running configuration: {experiment.__class__.__name__}")
    # Schedule controller
    sched.submit_array(task_name="ctl",
                       cmd=scheduler.client.control_cmd(expr_name, trial_name, args.debug,
                                                        args.ignore_worker_error),
                       count=1,
                       cpu=1,
                       gpu=0,
                       mem=1024,
                       env_vars=simple_env_vars,
                       exclude='frl2g[084-086],frl2g008,frl2g093,frl2g094,frl8g[136-137]')

    workers_configs = ((k, getattr(setup, k)) for k in system.WORKER_TYPES)

    for name, scheduling_setup in workers_configs:
        if not isinstance(scheduling_setup, list):
            scheduling_setup = [scheduling_setup]
        submit_workers(expr_name, trial_name, args.debug, name, scheduling_setup)

    try:
        sched.wait()
    except (KeyboardInterrupt, scheduler.client.TaskException):
        sched.stop_all()
        raise


def main_stop(args):
    mode = args.mode or "slurm"
    assert mode == "slurm", "Only slurm experiment is supported."
    sched = scheduler.client.make(mode=args.mode, job_name=f"{args.experiment_name}_{args.trial_name}")
    sched.find_all()
    sched.stop_all()


def main_find_config(args):
    exp_names = [x for x in config_package.ALL_EXPERIMENT_CLASSES if re.match(args.regex, x)]
    if len(exp_names) == 0:
        print("No matched experiment names.")
    if len(exp_names) > 20:
        response = input(f"Found {len(exp_names)} experiments, list all?(y/n)")
        if response != "y":
            return
    for exp_name in exp_names:
        print(exp_name)


def main():
    parser = argparse.ArgumentParser(prog="rlhf")
    subparsers = parser.add_subparsers(dest="cmd", help="sub-command help")
    subparsers.required = True

    subparser = subparsers.add_parser("start", help="starts an experiment")
    subparser.add_argument("--experiment_name", "-e", type=str, required=True, help="name of the experiment")
    subparser.add_argument("--trial_name",
                           "-f",
                           type=str,
                           default=None,
                           help="trial name; by default uses '<USER>-test'")
    subparser.add_argument("--mode", default="slurm", choices=["slurm"])
    subparser.add_argument("--partition", default="dev", help="slurm partition to schedule the trial")
    subparser.add_argument("--wandb_mode",
                           type=str,
                           default="disabled",
                           choices=["online", "offline", "disabled"])
    subparser.add_argument("--image_name",
                           type=str,
                           required=False,
                           default=None,
                           help="if specified, all workers will use this image. Useful in CI/CD pipeline.")
    subparser.add_argument("--LOGLEVEL", type=str, default="INFO")
    subparser.add_argument("--ignore_worker_error", action="store_true")
    subparser.add_argument("--debug",
                           action="store_true",
                           help="If True, activate all assertions in the code.")
    subparser.set_defaults(ignore_worker_error=False)
    subparser.set_defaults(func=main_start)

    subparser = subparsers.add_parser("stop", help="stops an experiment. only slurm experiment is supported.")
    subparser.add_argument("--experiment_name", "-e", type=str, required=True, help="name of the experiment")
    subparser.add_argument("--trial_name", "-f", type=str, required=True, help="name of the trial")
    subparser.add_argument("--mode", default="slurm", choices=["local", "slurm"])
    subparser.set_defaults(func=main_stop)

    subparser = subparsers.add_parser("find_config",
                                      help="find configuration by matching regular expression.")
    subparser.add_argument("--regex", "-r", type=str, required=True)
    subparser.set_defaults(func=main_find_config)

    args = parser.parse_args()
    logging.basicConfig(format=LOG_FORMAT, datefmt=DATE_FORMAT, level=getattr(args, "LOGLEVEL", "INFO"))
    args.func(args)


if __name__ == "__main__":
    main()
