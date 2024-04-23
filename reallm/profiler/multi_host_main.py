from typing import Dict, List, Optional
import argparse
import getpass
import os
import re

import profiler.experiments

import api.config.config_system as config_package
import reallm.base.logging as logging
import reallm.base.name_resolve
import reallm.base.names
import scheduler.client
import system

logger = logging.getLogger("main", "system")

CONTROLLER_TIME_LIMIT = None
TRACE_TIMEOUT = (
    500  # Should be larger than TRACER_SAVE_INTERVAL_SECONDS defined in system/worker_base.py
)


def _submit_workers(
    sched: scheduler.client.SchedulerClient,
    expr_name: str,
    trial_name: str,
    debug: bool,
    worker_type: str,
    scheduling_configs: List[config_package.TasksGroup],
    environs: Dict[str, str],
    image_name: Optional[str] = None,
    use_ray_cluster: bool = False,
) -> List[str]:
    if len(scheduling_configs) == 0:
        return []

    scheduled_jobs = []
    for sch_cfg in scheduling_configs:
        job_environs = {**environs, **sch_cfg.scheduling.env_vars}
        if use_ray_cluster:
            cmd = scheduler.client.ray_cluster_cmd(
                expr_name,
                trial_name,
                worker_type=worker_type,
            )
        else:
            cmd = scheduler.client.remote_worker_cmd(expr_name, trial_name, debug, worker_type)

        logger.debug(f"Scheduling worker {worker_type}, {scheduling_configs}")

        nodelist = sch_cfg.scheduling.nodelist
        exclude = sch_cfg.scheduling.exclude
        node_type = sch_cfg.scheduling.node_type
        container_image = image_name or sch_cfg.scheduling.container_image
        if use_ray_cluster:
            worker_type = f"rc_{worker_type}"

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
            ),)
    return scheduled_jobs


def main(args, if_raise=True):
    trial_name = args.trial_name
    expr_name = args.expr_name
    experiment = config_package.make_experiment(expr_name)
    sched = scheduler.client.make(mode="slurm", expr_name=expr_name, trial_name=trial_name)

    setup = experiment.scheduling_setup()

    base_environs = {
        "PYTHONPATH": os.path.dirname(os.path.dirname(__file__)),
        "WANDB_MODE": "disabled",
        "DLLM_MODE": "SLURM",
        "DLLM_TRACE": "0" if not args.trace else "1",
    }

    logger.info(f"Resetting name resolving repo...")
    try:
        base.name_resolve.clear_subtree(
            base.names.trial_root(experiment_name=expr_name, trial_name=trial_name))
    except Exception as e:
        logger.warning(f"Resetting name resolving repo failed.")
        if if_raise:
            raise e
        else:
            return False
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
            False,
            False,
            controller_type,
        ),
        count=1,
        cpu=1,
        gpu=0,
        mem=1024,
        env_vars=base_environs,
        container_image=setup.controller_image,
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
            False,
            name,
            scheduling_setup,
            base_environs,
            None,
            use_ray_cluster=False,
        )

    timeout = None if not args.trace else TRACE_TIMEOUT  # run 5 mins to collect trace
    try:
        sched.wait(timeout=timeout)
    except (KeyboardInterrupt, scheduler.client.JobException, TimeoutError) as e:
        sched.stop_all()
        if if_raise:
            raise e
        else:
            logger.warning(f"Experiment {expr_name} {trial_name} failed.")
            return True
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a profiling experiment.")
    parser.add_argument(
        "-e",
        "--expr_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-f",
        "--trial_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--trace",
        action="store_true",
    )

    args = parser.parse_args()
    main(args)
