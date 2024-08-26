import argparse
import datetime
import getpass
import pathlib
import re
import sys

import hydra

from realhf.api.quickstart.entrypoint import QUICKSTART_FN
from realhf.base.cluster import spec as cluster_spec
from realhf.base.importing import import_module

# NOTE: Register all implemented experiments inside ReaL.
import_module(
    str(pathlib.Path(__file__).resolve().parent.parent / "experiments" / "common"),
    re.compile(r".*_exp\.py$"),
)
import realhf.experiments.benchmark.profile_exp


def main():
    parser = argparse.ArgumentParser(prog="ReaL Quickstart")
    subparsers = parser.add_subparsers(dest="cmd", help="sub-command help")
    subparsers.required = True
    for k, v in QUICKSTART_FN.items():
        subparser = subparsers.add_parser(k)
        subparser.set_defaults(func=v)
    args = parser.parse_known_args()[0]

    launch_hydra_task(args.cmd, QUICKSTART_FN[args.cmd])


def launch_hydra_task(name: str, func: hydra.TaskFunction):
    # Disable hydra logging.
    if not any("hydra/job_logging=disabled" in x for x in sys.argv):
        sys.argv += ["hydra/job_logging=disabled"]

    if any("experiment_name=" in x for x in sys.argv):
        experiment_name = next(x for x in sys.argv if "experiment_name=" in x).split(
            "="
        )[1]
        if "_" in experiment_name:
            raise RuntimeError("experiment_name should not contain `_`.")
    else:
        experiment_name = f"quickstart-{name}"
        print(f"Experiment name not manually set. Default to {experiment_name}.")
        sys.argv += [f"experiment_name={experiment_name}"]

    if (
        "--multirun" in sys.argv
        or "hydra.mode=MULTIRUN" in sys.argv
        or "-m" in sys.argv
    ):
        raise NotImplementedError("Hydra multi-run is not supported.")
    # non-multirun mode, add trial_name and hydra run dir
    if any("trial_name=" in x for x in sys.argv):
        trial_name = next(x for x in sys.argv if "trial_name=" in x).split("=")[1]
    else:
        trial_name = f"run{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        sys.argv += [f"trial_name={trial_name}"]
    if "_" in trial_name:
        raise RuntimeError("trial_name should not contain `_`.")
    sys.argv += [
        f"hydra.run.dir={cluster_spec.fileroot}/logs/{getpass.getuser()}/"
        f"{experiment_name}/{trial_name}/hydra-outputs/"
    ]

    sys.argv.pop(1)

    func()


if __name__ == "__main__":
    main()
