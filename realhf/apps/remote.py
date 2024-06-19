import argparse
import functools
import json
import multiprocessing
import os
import pickle
import re
import socket
import subprocess

from omegaconf import OmegaConf
import torch

multiprocessing.set_start_method("spawn", force=True)

from realhf.api.quickstart.entrypoint import (
    QUICKSTART_CONFIG_CLASSES,
    QUICKSTART_EXPR_CACHE_PATH,
)
from realhf.base import (
    cluster,
    gpu_utils,
    importing,
    logging,
    name_resolve,
    names,
)

RAY_HEAD_WAIT_TIME = 500
logger = logging.getLogger("Main-Workers")


def _patch_external_impl(exp_name, trial_name):
    import realhf.api.core.system_api as system_api

    if os.path.exists(QUICKSTART_EXPR_CACHE_PATH):
        for exp_cache in os.listdir(QUICKSTART_EXPR_CACHE_PATH):
            target_cache_name = f"{exp_name}_{trial_name}.json"
            if exp_cache != target_cache_name:
                continue
            cache_file = os.path.join(
                QUICKSTART_EXPR_CACHE_PATH, target_cache_name
            )
            with open(cache_file, "r") as f:
                cache = json.load(f)
            usercode_path = cache["usercode_path"]
            exp_cls_args = OmegaConf.create(cache["args"])
            config_name = cache["config_name"]
            # Import user code to register quickstart experiments.
            importing.import_usercode(usercode_path, "_realhf_user_code")
            # Register the internal experiment.
            exp_cls = QUICKSTART_CONFIG_CLASSES[config_name]
            system_api.register_experiment(
                exp_name, functools.partial(exp_cls, **exp_cls_args)
            )


def main_worker(args):
    import realhf.base.constants as constants

    constants.set_experiment_trial_names(args.experiment_name, args.trial_name)
    _patch_external_impl(args.experiment_name, args.trial_name)

    worker_index_start = (
        args.jobstep_id * args.wprocs_per_jobstep + args.wproc_offset
    )
    worker_index_end = min(
        worker_index_start + args.wprocs_per_jobstep,
        args.wprocs_in_job + args.wproc_offset,
    )

    group_name = f"{args.worker_type}-{args.worker_submission_index}"
    logger.debug(
        f"{args.worker_type} group id {args.jobstep_id}, "
        f"worker index {worker_index_start}:{worker_index_end}"
    )

    # Isolate within the same slurm job, among different jobsteps.
    if torch.cuda.is_initialized():
        raise RuntimeError(
            "CUDA already initialized before isolating CUDA devices. This should not happen."
        )
    gpu_utils.isolate_cuda_device(
        group_name,
        args.jobstep_id,
        args.n_jobsteps,
        args.experiment_name,
        args.trial_name,
    )
    if os.environ.get("CUDA_VISIBLE_DEVICES", None):
        logger.debug(
            "CUDA_VISIBLE_DEVICES: %s", os.environ["CUDA_VISIBLE_DEVICES"]
        )

    # NOTE: Importing these will initialize DeepSpeed/CUDA devices.
    # profiler.import_profiler_registers()
    import realhf.impl.dataset
    import realhf.impl.model
    import realhf.system

    logger.debug(f"Run {args.worker_type} worker with args: %s", args)
    assert not args.experiment_name.startswith(
        "/"
    ), f'Invalid experiment_name "{args.experiment_name}" starts with "/"'
    if args.wprocs_per_jobstep == 1:
        realhf.system.run_worker(
            worker_type=args.worker_type,
            experiment_name=args.experiment_name,
            trial_name=args.trial_name,
            worker_name=f"{args.worker_type}/{worker_index_start}",
            worker_server_type="zmq",
        )
    else:
        workers = []
        for wid in range(worker_index_start, worker_index_end):
            worker_args = dict(
                worker_type=args.worker_type,
                experiment_name=args.experiment_name,
                trial_name=args.trial_name,
                worker_name=f"{args.worker_type}/{wid}",
                worker_server_type="zmq",
            )
            p = multiprocessing.Process(
                target=realhf.system.run_worker, kwargs=worker_args
            )
            p.name = f"{args.worker_type}/{wid}"
            p.start()
            workers.append(p)

        logger.info(
            f"Waiting for {args.wprocs_per_jobstep} {args.worker_type} workers of group id {args.jobstep_id}."
        )
        worker_exit_code = 0
        while not worker_exit_code:
            for p in workers:
                if p.exitcode is None:
                    p.join(timeout=5)
                elif p.exitcode == 0:
                    pass
                else:
                    logger.error(f"{p.name} exitcode: {p.exitcode}")
                    worker_exit_code = p.exitcode
                    break
        for p in workers:
            if p.is_alive():
                p.kill()
        exit(worker_exit_code)


def main_controller(args):
    """
    Args:
        args: argparse result including:
            experiment_name:
            trial_name:
            config_index: the index of experiment configuration (experiment may return multiple configurations)
            ignore_worker_error: bool, if False, stop the experiment when any worker(s) fail.
    """
    import realhf.api.core.system_api as system_api
    import realhf.base.constants as constants
    import realhf.system as system

    constants.set_experiment_trial_names(args.experiment_name, args.trial_name)
    _patch_external_impl(args.experiment_name, args.trial_name)

    logger.debug("Running controller with args: %s", args)
    assert not args.experiment_name.startswith("/"), args.experiment_name
    if args.type == "ray":
        # launch ray cluster head
        ray_flags = [
            f"--num-cpus=0",
            f"--num-gpus=0",
            f"--port={args.ray_port}",
            "--head",
        ]
        cmd = f"ray start {' '.join(ray_flags)}"
        output = subprocess.check_output(cmd, shell=True).decode("ascii")
        logger.info("Successfully launched ray cluster head.")

        pattern = r"ray start --address='(\d+\.\d+\.\d+\.\d+:\d+)'"
        match = re.search(pattern, output)
        if match:
            addr = match.group(1)
            logger.info("Found ray address: '%s'", addr)
        else:
            raise RuntimeError(
                f"Address not found in ray start output: {output}."
            )
        ray_addr_name = names.ray_cluster(
            args.experiment_name, args.trial_name, "address"
        )
        name_resolve.add(
            ray_addr_name,
            addr,
            delete_on_exit=True,
            keepalive_ttl=RAY_HEAD_WAIT_TIME,
        )

    controller = system.make_controller(
        type_=args.type,
        experiment_name=args.experiment_name,
        trial_name=args.trial_name,
    )
    experiment = system_api.make_experiment(args.experiment_name)
    controller.start(
        experiment=experiment,
        ignore_worker_error=args.ignore_worker_error,
    )

    if args.type == "ray":
        subprocess.check_output(f"ray stop", shell=True)


def main_ray(args):
    ray_addr_name = names.ray_cluster(
        args.experiment_name, args.trial_name, "address"
    )
    try:
        address = name_resolve.wait(ray_addr_name, timeout=RAY_HEAD_WAIT_TIME)
    except TimeoutError:
        raise TimeoutError("Timeout waiting for ray cluster head address.")
    ray_flags = [f"--address={address}"]

    cmd = f"ray start {' '.join(ray_flags)}"
    _ = subprocess.check_output(cmd, shell=True).decode("ascii")
    logger.info(
        f"Successfully launched nodes for {args.worker_type} in Ray cluster."
    )

    host_ip = socket.gethostbyname(socket.gethostname())
    name_resolve.add(
        names.ray_cluster(
            args.experiment_name,
            args.trial_name,
            f"{args.worker_type}/{args.jobstep_id}",
        ),
        host_ip,
        delete_on_exit=True,
        keepalive_ttl=300,
    )

    while True:
        try:
            ray_exiting_name = names.ray_cluster(
                args.experiment_name, args.trial_name, "exiting"
            )
            try:
                name_resolve.wait(ray_exiting_name, timeout=10)
                break
            except TimeoutError:
                pass
        except KeyboardInterrupt:
            break

    subprocess.check_output(f"ray stop", shell=True)


def main():
    parser = argparse.ArgumentParser(prog="marl")
    subparsers = parser.add_subparsers(dest="cmd", help="sub-command help")
    subparsers.required = True

    subparser = subparsers.add_parser(
        "controller", help="run a controller of experiment"
    )
    subparser.add_argument("--experiment_name", "-e", type=str, required=True)
    subparser.add_argument("--trial_name", "-f", type=str, required=True)
    subparser.add_argument("--ignore_worker_error", action="store_true")
    subparser.add_argument(
        "--raise_worker_error", dest="ignore_worker_error", action="store_false"
    )
    subparser.add_argument("--type", type=str, default="zmq")
    subparser.add_argument("--ray_port", type=int, default=8777)
    subparser.set_defaults(feature=False)
    subparser.set_defaults(func=main_controller)

    subparser = subparsers.add_parser("worker", help="run a standalone worker")
    subparser.add_argument("--experiment_name", "-e", type=str, required=True)
    subparser.add_argument("--trial_name", "-f", type=str, required=True)
    subparser.add_argument("--worker_type", "-w", type=str, required=True)
    subparser.add_argument(
        "--jobstep_id",
        "-i",
        type=int,
        required=True,
        help="jobstep/task ID in a slurm job.",
    )
    subparser.add_argument(
        "--n_jobsteps",
        "-g",
        type=int,
        required=True,
        help="`--ntasks` of `srun`, aka SLURM_NPROCS.",
    )
    subparser.add_argument(
        "--worker_submission_index",
        "-r",
        type=int,
        required=True,
        help="Submission index to slurm for this worker. Used for locating job name and logs.",
    )
    subparser.add_argument(
        "--wprocs_per_jobstep",
        "-p",
        type=int,
        required=True,
        help="Number of worker processes launched by multiprocessing in this script.",
    )
    subparser.add_argument(
        "--wprocs_in_job",
        "-j",
        type=int,
        required=True,
        help="Number of worker processes in this slurm job.",
    )
    subparser.add_argument(
        "--wproc_offset",
        "-o",
        type=int,
        required=True,
        help="Offset of worker processes of this slurm job. "
        "For example, we may allocate 4 type `A` workers with 1 GPU each and 2 with 0.5 GPU each. "
        "This launches 2 jobs, the former with 4 job steps and the latter with 2 job steps. "
        "The offset is 0 for the 1st job and 4 for the 2nd job.",
    )
    subparser.set_defaults(func=main_worker)

    subparser = subparsers.add_parser(
        "ray", help="launch ray cluster write ray address to name_resolve"
    )
    subparser.add_argument("--experiment_name", "-e", type=str, required=True)
    subparser.add_argument("--trial_name", "-f", type=str, required=True)
    subparser.add_argument("--worker_type", "-w", type=str, required=True)
    subparser.add_argument("--jobstep_id", "-i", type=int, required=True)
    subparser.add_argument("--n_jobsteps", "-g", type=int, required=True)
    subparser.set_defaults(func=main_ray)

    args = parser.parse_args()

    args.func(args)


if __name__ == "__main__":
    main()
