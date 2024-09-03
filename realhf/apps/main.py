import argparse
import getpass
import os
import re
from typing import Dict, List, Optional

import realhf.api.core.system_api as config_package
import realhf.base.constants as constants
import realhf.base.logging as logging
import realhf.base.name_resolve as name_resolve
import realhf.base.names as names
import realhf.scheduler.client as sched_client
import realhf.system as system
from realhf.scheduler.client import JobException, JobState

logger = logging.getLogger("main", "system")

CONTROLLER_TIME_LIMIT = None


def scheduler_mode(mode: str) -> str:
    return mode if mode == "slurm" else "local"


def _submit_workers(
    sched: sched_client.SchedulerClient,
    expr_name: str,
    trial_name: str,
    debug: bool,
    worker_type: str,
    scheduling_configs: List[config_package.TasksGroup],
    environs: Dict[str, str],
    image_name: Optional[str] = None,
) -> List[str]:
    if len(scheduling_configs) == 0:
        return []

    scheduled_jobs = []
    for sch_cfg in scheduling_configs:
        job_environs = {**environs, **sch_cfg.scheduling.env_vars}
        cmd = sched_client.remote_worker_cmd(expr_name, trial_name, debug, worker_type)

        logger.debug(f"Scheduling worker {worker_type}, {scheduling_configs}")

        nodelist = sch_cfg.scheduling.nodelist
        exclude = sch_cfg.scheduling.exclude
        node_type = sch_cfg.scheduling.node_type
        container_image = image_name or sch_cfg.scheduling.container_image

        scheduled_jobs.append(
            sched.submit_array(
                worker_type=worker_type,
                cmd=cmd,
                count=sch_cfg.count,
                cpu=sch_cfg.scheduling.cpu,
                gpu=sch_cfg.scheduling.gpu,
                gpu_type=sch_cfg.scheduling.gpu_type,
                mem=sch_cfg.scheduling.mem,
                container_image=container_image,
                node_type=node_type,
                nodelist=nodelist,
                exclude=exclude,
                env_vars=job_environs,
                hostfile=True,
                multiprog=True,
                begin=sch_cfg.scheduling.begin,
                deadline=sch_cfg.scheduling.deadline,
                time_limit=sch_cfg.scheduling.time_limit,
            ),
        )
    return scheduled_jobs


def main_start(args, recover_count: int = 0):
    if recover_count == 0:
        constants.set_experiment_trial_names(args.experiment_name, args.trial_name)
    experiment = config_package.make_experiment(args.experiment_name)

    if args.mode == "local":
        assert (
            args.recover_mode == "disabled"
        ), "Recover mode is not supported for local runs!"
    # Use search cache for recover runs
    force_allocation_use_cache = (
        recover_count > 1 or args.recover_mode == "resume"
    ) and args.allocation_mode == "search"
    # handle args
    args.ignore_worker_error = (
        args.ignore_worker_error and args.recover_mode == "disabled"
    )
    trial_name = args.trial_name or f"test-{getpass.getuser()}"
    expr_name = args.experiment_name
    is_recover_run = (
        args.recover_mode == "auto" and recover_count > 0
    ) or args.recover_mode == "resume"
    save_recover_states = args.recover_mode != "disabled"

    cluster_spec_path = os.environ.get("CLUSTER_SPEC_PATH", "")
    if not cluster_spec_path:
        if args.mode == "slurm":
            raise ValueError(
                "Environment variable CLUSTER_SPEC_PATH must be set for slurm mode! "
                "See example/cluster_config.json for a template."
            )
        logger.warning(
            "Environment variable CLUSTER_SPEC_PATH is not set. "
            "Files of the experiment (logs, checkpoints, cache ...) "
            "will be saved to temporary directory of the system. "
            "To change the fileroot, set the fileroot option of your choice in your CLUSTER_SPEC_PATH."
        )

    # set env vars
    BASE_ENVIRONS = constants.get_env_vars(
        WANDB_MODE=args.wandb_mode,
        REAL_MODE=args.mode.upper(),
        CLUSTER_SPEC_PATH=cluster_spec_path,
        REAL_RECOVER_RUN="1" if is_recover_run else "0",
        REAL_SAVE_RECOVER_STATES="1" if save_recover_states else "0",
    )
    for k, v in BASE_ENVIRONS.items():
        os.environ[k] = v
    os.environ["REAL_IS_REMOTE"] = "0" if not force_allocation_use_cache else "1"

    # setup experiments
    if args.allocation_mode == "search":
        experiment._search()

    sched = sched_client.make(
        mode=scheduler_mode(args.mode),
        expr_name=expr_name,
        trial_name=trial_name,
    )

    setup = experiment.scheduling_setup()

    logger.info(f"Resetting name resolving repo...")

    try:
        name_resolve.clear_subtree(
            names.trial_root(
                experiment_name=args.experiment_name, trial_name=args.trial_name
            )
        )
    except Exception as e:
        logger.warning(f"Resetting name resolving repo failed.")
        raise e
    logger.info(f"Resetting name resolving repo... Done.")

    logger.info(f"Running configuration: {experiment.__class__.__name__}")

    # Schedule controller
    if args.mode == "ray":
        controller_type = "ray"
    else:
        controller_type = "zmq"
    # For ray mode, the controller will start all remote workers.
    sched.submit_array(
        worker_type="ctl",
        cmd=sched_client.control_cmd(
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
        env_vars=BASE_ENVIRONS,
        container_image=args.image_name or setup.controller_image,
        time_limit=CONTROLLER_TIME_LIMIT,
    )

    if args.mode != "ray":
        workers_configs = ((k, getattr(setup, k)) for k in system.WORKER_TYPES)

        for name, scheduling_setup in workers_configs:
            if not isinstance(scheduling_setup, list):
                scheduling_setup = [scheduling_setup]
            # For local or slurm mode, launch all workers.
            # For ray mode, nothing to do because workers will be
            # started by the controller, rather than the scheduler.
            _submit_workers(
                sched,
                expr_name,
                trial_name,
                args.debug,
                name,
                scheduling_setup,
                BASE_ENVIRONS,
                args.image_name,
            )

    try:
        sched.wait(
            check_status=(
                JobState.CANCELLED,
                JobState.FAILED,
                JobState.NOT_FOUND,
                JobState.COMPLETED,
            ),
            remove_status=(),
        )
    except (KeyboardInterrupt, JobException, TimeoutError) as e:
        recover_states = [
            JobState.CANCELLED,
            JobState.FAILED,
            JobState.NOT_FOUND,
        ]
        reason = e.reason if isinstance(e, JobException) else None
        recover_this = (
            args.recover_mode == "auto" and recover_count < args.recover_retries
        )
        recover_this = recover_this and reason in recover_states

        kill_signal = (
            "SIGKILL" if args.mode == "slurm" else "SIGTERM"
        )  # use sigkill to terminate slurm jobs
        sched.stop_all(
            "SIGINT" if (recover_this or args.recover_mode == "save") else kill_signal
        )
        if recover_this:
            logger.warning(
                f"Recovering from error {e}. Recover count: {recover_count+1}, "
                f"total recover count {args.recover_retries}"
            )
            main_start(args, recover_count=recover_count + 1)
        else:
            raise e


def main_stop(args):
    sched = sched_client.make(
        mode=scheduler_mode(args.mode),
        expr_name=args.experiment_name,
        trial_name=args.trial_name,
    )
    sched.find_all()
    sched.stop_all()


def main_find_config(args):
    exp_names = [
        x for x in config_package.ALL_EXPERIMENT_CLASSES if re.match(args.regex, x)
    ]
    if len(exp_names) == 0:
        print("No matched experiment names.")
    if len(exp_names) > 20:
        response = input(f"Found {len(exp_names)} experiments, list all?(y/n)")
        if response != "y":
            return
    for exp_name in exp_names:
        print(exp_name)


def main_profile_layers(args):
    from realhf.api.core.model_api import ModelFamily

    _main_profile_layers(
        ModelFamily(args.model_class, args.model_size, args.is_critic),
        args.model_path,
    )


def _main_profile_layers(model_family, model_path):
    from realhf.api.core.model_api import ModelFamily
    from realhf.base.slurm_utils import check_slurm_availability
    from realhf.base.testing import clear_name_resolve

    expr_name = trial_name = "profile"
    cmd = (
        f"python3 -m realhf.apps.profile_layers --expr_name {expr_name} --trial_name {trial_name} "
        f"--model_path {model_path} --model_name {model_family} "
    )

    if check_slurm_availability():
        if not os.environ.get("CLUSTER_SPEC_PATH", ""):
            raise ValueError(
                "Environment variable CLUSTER_SPEC_PATH must be set for slurm mode! "
                "See example/cluster_config.json for a template."
            )
        BASE_ENVIRONS = constants.get_env_vars(
            WANDB_MODE="disabled",
            REAL_MODE="slurm",
            CLUSTER_SPEC_PATH=os.environ.get("CLUSTER_SPEC_PATH", ""),
        )
        clear_name_resolve(expr_name, trial_name)
        sched = sched_client.make(
            mode="slurm", expr_name=expr_name, trial_name=trial_name
        )
        print(
            f"Profiling {model_family} layers, model path {model_path}, " f"cmd {cmd}"
        )
        sched.submit_array(
            worker_type="profile_layer",
            cmd=cmd,
            count=1,
            cpu=64,
            gpu=8,
            gpu_type="tesla",
            mem=500000,
            env_vars=BASE_ENVIRONS,
            container_image=config_package._LLM_GPU_IMAGE,
        )

        try:
            sched.wait(timeout=None)
        except (
            KeyboardInterrupt,
            sched_client.JobException,
            TimeoutError,
        ) as e:
            sched.stop_all()
            raise e
    else:
        try:
            print(
                f"Profiling {model_family} layers, model path {model_path}, "
                f"cmd {cmd}"
            )
            clear_name_resolve(expr_name, trial_name)
            os.system(cmd)
        except (
            KeyboardInterrupt,
            sched_client.JobException,
            TimeoutError,
        ) as e:
            raise e


def main():
    parser = argparse.ArgumentParser(prog="ReaLHF")
    subparsers = parser.add_subparsers(dest="cmd", help="sub-command help")
    subparsers.required = True

    subparser = subparsers.add_parser("start", help="starts an experiment")
    subparser.add_argument(
        "--experiment_name",
        "-e",
        type=str,
        required=True,
        help="name of the experiment",
    )
    subparser.add_argument(
        "--trial_name",
        "-f",
        type=str,
        default=None,
        help="trial name; by default uses '<USER>-test'",
    )
    subparser.add_argument(
        "--mode",
        default="slurm",
        choices=["local", "slurm", "ray"],
    )
    subparser.add_argument(
        "--partition",
        default="dev",
        help="slurm partition to schedule the trial",
    )
    subparser.add_argument(
        "--wandb_mode",
        type=str,
        default="disabled",
        choices=["online", "offline", "disabled"],
    )
    subparser.add_argument(
        "--image_name",
        type=str,
        required=False,
        default=None,
        help="if specified, all workers will use this image. Useful in CI/CD pipeline.",
    )
    subparser.add_argument("--ignore_worker_error", action="store_true")
    subparser.add_argument(
        "--debug",
        action="store_true",
        help="If True, activate all assertions in the code.",
    )
    subparser.add_argument(
        "--recover_mode",
        required=False,
        default="disabled",
        choices=["disabled", "auto", "save", "resume"],
        help="Recover mode, 'auto': automatically recover the last failed run; "
        "'save': save recover states if any error occurs; "
        "'resume': resume from saved recover states and save states if fail again; "
        "'disabled': do nothing when error occurs. ",
    )
    subparser.add_argument(
        "--recover_retries",
        type=int,
        required=False,
        default=1,
        help="Total number of trials for the system to recover automatically when a worker fails. "
        "Only effective when recover_mode is 'auto'.",
    )
    subparser.add_argument(
        "--allocation_mode",
        type=str,
        required=False,
        default="pipe_model",
        choices=["manual", "search", "heuristic", "pipe_model", "pipe_data"],
        help="Mode of GPU resource/model parallel strategy allocation.",
    )
    subparser.set_defaults(ignore_worker_error=False)
    subparser.set_defaults(func=main_start)

    subparser = subparsers.add_parser(
        "stop", help="stops an experiment. only slurm experiment is supported."
    )
    subparser.add_argument(
        "--experiment_name",
        "-e",
        type=str,
        required=True,
        help="name of the experiment",
    )
    subparser.add_argument(
        "--trial_name", "-f", type=str, required=True, help="name of the trial"
    )
    subparser.add_argument(
        "--mode",
        default="slurm",
        choices=["local", "slurm", "ray"],
    )
    subparser.set_defaults(func=main_stop)

    subparser = subparsers.add_parser(
        "find_config", help="find configuration by matching regular expression."
    )
    subparser.add_argument("--regex", "-r", type=str, required=True)
    subparser.set_defaults(func=main_find_config)

    subparser = subparsers.add_parser(
        "profile_layers", help="profile layers of a model."
    )
    subparser.add_argument("--model_class", type=str, required=True)
    subparser.add_argument("--model_size", type=int, required=True)
    subparser.add_argument("--is_critic", action="store_true")
    subparser.add_argument("--model_path", type=str, required=True)
    subparser.set_defaults(func=main_profile_layers)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
