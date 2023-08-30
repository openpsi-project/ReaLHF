import argparse
import logging
import os
import re
import subprocess
import time

from base.constants import DATE_FORMAT, LOG_FORMAT
import base.gpu_utils
import base.name_resolve
import base.names

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
    system.run_worker(
        worker_type=args.worker_type,
        experiment_name=args.experiment_name,
        trial_name=args.trial_name,
        worker_name=f"{args.worker_type}/{args.group_id}",
        worker_server_type='zmq',
    )


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
    controller = system.make_controller(type_=args.type,
                                        experiment_name=args.experiment_name,
                                        trial_name=args.trial_name)
    experiment = api.config.make_experiment(args.experiment_name)
    controller.start(
        experiment=experiment,
        ignore_worker_error=args.ignore_worker_error,
    )


def main_ray(args):
    ray_flags = [
        f"--num-cpus={0 if args.head else args.cpu}",
        f"--num-gpus={0 if args.head else args.gpu}",
        f"--memory={int(args.mem)}",
        f"--object-store-memory={int(args.obj_store_mem)}",
    ]
    ray_addr_name = base.names.ray_cluster(args.experiment_name, args.trial_name, "address")
    if args.head:
        ray_flags += [
            f"--port={args.port}",
            "--head",
        ]
    else:
        try:
            address = base.name_resolve.wait(ray_addr_name, timeout=60)
        except TimeoutError:
            raise TimeoutError("Timeout waiting for ray cluster head address.")
        ray_flags += [f"--address={address}"]

    cmd = f"ray start {' '.join(ray_flags)}"
    output = subprocess.check_output(cmd, shell=True).decode('ascii')

    if args.head:
        pattern = r"ray start --address='(\d+\.\d+\.\d+\.\d+:\d+)'"
        match = re.search(pattern, output)
        if match:
            addr = match.group(1)
            logger.debug("Found ray address: '%s'", addr)
        else:
            raise RuntimeError(f"Address not found in ray start output: {output}.")
        base.name_resolve.add(ray_addr_name, addr, delete_on_exit=True, keepalive_ttl=60)

    else:
        _name = base.names.ray_cluster(args.experiment_name, args.trial_name, args.worker_type)
        base.name_resolve.add_subentry(_name, args.group_id, delete_on_exit=True, keepalive_ttl=60)

    while True:
        try:
            ray_exiting_name = base.names.ray_cluster(args.experiment_name, args.trial_name, "exiting")
            try:
                base.name_resolve.wait(ray_exiting_name, timeout=10)
                break
            except TimeoutError:
                pass
        except KeyboardInterrupt:
            break

    subprocess.check_output(f"ray stop", shell=True)


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
    subparser.add_argument('--type', type=str, default='zmq')
    subparser.set_defaults(feature=False)
    subparser.set_defaults(func=main_controller)

    subparser = subparsers.add_parser("worker", help="run a standalone worker")
    subparser.add_argument("--worker_type", '-w', type=str, required=True)
    subparser.add_argument("--experiment_name", "-e", type=str, required=True)
    subparser.add_argument("--trial_name", "-f", type=str, required=True)
    subparser.add_argument("--group_id", "-i", type=int, required=True)
    subparser.add_argument("--group_size", "-g", type=int, required=True)
    subparser.set_defaults(func=main_worker)

    subparser = subparsers.add_parser("reset_name_resolve", help="reset name resolve repo for a trial")
    subparser.add_argument("--experiment_name", "-e", type=str, required=True)
    subparser.add_argument("--trial_name", "-f", type=str, required=True)
    subparser.set_defaults(func=main_reset_name_resolve)

    subparser = subparsers.add_parser("ray", help='launch ray cluster write ray address to name_resolve')
    subparser.add_argument("--experiment_name", "-e", type=str, required=True)
    subparser.add_argument("--trial_name", "-f", type=str, required=True)
    subparser.add_argument("--worker_type", '-w', type=str, required=True)
    subparser.add_argument("--group_id", "-i", type=int, required=True)
    subparser.add_argument("--group_size", "-g", type=int, required=True)
    subparser.add_argument("--port", type=int, default=8777)
    subparser.add_argument("--mem", type=int, default=int(20e9), help='in bytes')
    subparser.add_argument("--obj_store_mem", type=int, default=int(20e9), help='in bytes')
    subparser.add_argument("--head", action='store_true')
    subparser.add_argument("--cpu", type=int, default=0, help='only used on non-head nodes')
    subparser.add_argument("--gpu", type=int, default=0, help='only used on non-head nodes')
    subparser.set_defaults(func=main_ray)

    args = parser.parse_args()

    args.func(args)


if __name__ == "__main__":
    main()
