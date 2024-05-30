from typing import Callable, Optional
import argparse
import dataclasses
import datetime
import functools
import getpass
import os
import pickle
import subprocess
import sys

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
import hydra

from reallm.base.cluster import spec as cluster_spec
from reallm.base.constants import LOG_ROOT, MODEL_SAVE_ROOT, QUICKSTART_EXPR_CACHE_PATH
from reallm.experiments.common import DPOConfig, PPOConfig, RWConfig, SFTConfig
import reallm.api.core.system_api as system_api

cs = ConfigStore.instance()


@dataclasses.dataclass
class _MainStartArgs:
    experiment_name: str
    trial_name: str
    mode: str
    debug: bool = True
    partition: str = "dev"
    wandb_mode: str = "disabled"
    image_name: Optional[str] = None
    ignore_worker_error: bool = False
    remote_reset: bool = False


def kind_reminder(config_name, logger, args):
    logger.info(f"Running {config_name} experiment.")
    logger.info(f"Logs will be dumped to {os.path.join(LOG_ROOT, args.experiment_name, args.trial_name)}")
    logger.info(
        f"Model checkpoints will be saved to {os.path.join(MODEL_SAVE_ROOT, args.experiment_name, args.trial_name)}"
    )
    # for k, v in args.items():
    #     if hasattr(v, "parallel") and (v.parallel.pipeline_parallel_size > 1
    #                                    or v.parallel.model_parallel_size > 1):
    #         logger.warning(f"Detected model named '{k}' enables pipeline parallel or model parallel. "
    #                        "Please ensure that (1) there are enough GPUs for your experiment "
    #                        "and (2) the model checkpoint has been converted into "
    #                        "shards using scripts/transform_to_pipe_ckpt.py.")
    #     if hasattr(v, "parallel") and v.base_model_path is None:
    #         logger.warning(
    #             f"Detected `base_model_path` of model named '{k}' is not specified. Using `path` as `base_model_path`."
    #         )
    #         v.base_model_path = v.path
    #     if hasattr(v, "parallel") and v.tokenizer_path is None:
    #         logger.warning(
    #             f"Detected `tokenizer_path` of model named '{k}' is not specified. Using `base_model_path` as `tokenizer_path`."
    #         )
    #         v.tokenizer_path = v.base_model_path

    slurm_available = (int(
        subprocess.run(
            "squeue",
            shell=True,
            stdout=open(os.devnull, "wb"),
            stderr=open(os.devnull, "wb"),
        ).returncode) == 0)
    if slurm_available:
        logger.warning("Slurm is available. You probably run the system on ctrl nodes. "
                       "Using slurm to launch remote workers.")
    else:
        logger.warning("Slurm is not available. Using local mode.")
    mode = "slurm" if slurm_available else "local"
    return mode


def build_quickstart_entry_point(config_name: str, exp_cls: Callable):

    @hydra.main(version_base=None, config_name=config_name)
    def run(args):
        # NOTE: we import logging here to avoid hydra logging overwrite
        import reallm.base.logging as logging

        logger = logging.getLogger("quickstart", "colored")

        exp_name = args.experiment_name
        if args.trial_name == MISSING:
            args.trial_name = trial_name = (f"run{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
        else:
            trial_name = args.trial_name
        from reallm.apps.main import main_start, main_stop

        mode = kind_reminder(config_name, logger, args)

        exp_fn = functools.partial(exp_cls, **args)

        # FIXME: use uuid to avoid name conflict
        os.makedirs(os.path.dirname(QUICKSTART_EXPR_CACHE_PATH), exist_ok=True)
        with open(QUICKSTART_EXPR_CACHE_PATH, "wb") as f:
            pickle.dump((exp_name, exp_fn), f)
        system_api.register_experiment(exp_name, exp_fn)

        try:
            main_start(_MainStartArgs(exp_name, trial_name, mode, debug=True))
        except Exception as e:
            main_stop(_MainStartArgs(exp_name, trial_name, mode, debug=True))
            logger.warning("Exception occurred. Stopping all workers.")
            raise e

    cs.store(name=config_name, node=exp_cls)
    return run


run_sft = build_quickstart_entry_point("sft", SFTConfig)
run_rw = build_quickstart_entry_point("rw", RWConfig)
run_dpo = build_quickstart_entry_point("dpo", DPOConfig)
run_ppo = build_quickstart_entry_point("ppo", PPOConfig)


def main():
    parser = argparse.ArgumentParser(prog="ReaL Quickstart")
    subparsers = parser.add_subparsers(dest="cmd", help="sub-command help")
    subparsers.required = True

    subparser = subparsers.add_parser("sft", help="run supervised-finetuning")
    subparser.set_defaults(func=run_sft)
    subparser = subparsers.add_parser("rw", help="run reward modeling")
    subparser.set_defaults(func=run_rw)
    subparser = subparsers.add_parser("ppo", help="run PPO RLHF")
    subparser.set_defaults(func=run_ppo)
    subparser = subparsers.add_parser("dpo", help="run direct preference optimization")
    subparser.set_defaults(func=run_dpo)
    args = parser.parse_known_args()[0]

    # Disable hydra logging.
    if not any("hydra/job_logging=disabled" in x for x in sys.argv):
        sys.argv += ["hydra/job_logging=disabled"]

    if any("experiment_name=" in x for x in sys.argv):
        experiment_name = next(x for x in sys.argv if "experiment_name=" in x).split("=")[1]
        if "_" in experiment_name:
            raise RuntimeError("experiment_name should not contain `_`.")
    else:
        experiment_name = f"quickstart-{args.cmd}"
        print(f"Experiment name not manually set. Default to {experiment_name}.")
        sys.argv += [f"experiment_name={experiment_name}"]

    if ("--multirun" in sys.argv or "hydra.mode=MULTIRUN" in sys.argv or "-m" in sys.argv):
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
    args.func()


if __name__ == "__main__":
    main()
