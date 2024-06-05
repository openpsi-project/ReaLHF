from typing import Callable, Optional
import datetime
import functools
import os
import pickle
import inspect
import dataclasses
import json
import omegaconf
import subprocess

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
import hydra

from reallm.base.constants import LOG_ROOT, MODEL_SAVE_ROOT, QUICKSTART_EXPR_CACHE_PATH
import reallm.api.core.system_api as system_api


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
    logger.info(
        f"Logs will be dumped to {os.path.join(LOG_ROOT, args.experiment_name, args.trial_name)}"
    )
    logger.info(
        f"Model checkpoints will be saved to {os.path.join(MODEL_SAVE_ROOT, args.experiment_name, args.trial_name)}"
    )

    slurm_available = (
        int(
            subprocess.run(
                "squeue",
                shell=True,
                stdout=open(os.devnull, "wb"),
                stderr=open(os.devnull, "wb"),
            ).returncode
        )
        == 0
    )
    if slurm_available:
        logger.warning(
            "Slurm is available. You probably run the system on ctrl nodes. "
            "Using slurm to launch remote workers."
        )
    else:
        logger.warning("Slurm is not available. Using local mode.")
    mode = "slurm" if slurm_available else "local"
    return mode


cs = ConfigStore.instance()
QUICKSTART_CONFIG_CLASSES = {}
QUICKSTART_USERCODE_PATHS = {}
QUICKSTART_FN = {}

import traceback

def g():
    for line in traceback.format_stack():
        print(line.strip())

def register_quickstart_exp(config_name: str, exp_cls: Callable):
    usercode_path = os.path.abspath(inspect.getfile(inspect.currentframe().f_back))
    # g()

    @hydra.main(version_base=None, config_name=config_name)
    def run(args):
        # NOTE: we import logging here to avoid hydra logging overwrite
        import reallm.base.logging as logging

        logger = logging.getLogger("quickstart", "colored")

        exp_name = args.experiment_name
        if args.trial_name == MISSING:
            args.trial_name = trial_name = (
                f"run{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
            )
        else:
            trial_name = args.trial_name
        from reallm.apps.main import main_start, main_stop

        mode = kind_reminder(config_name, logger, args)

        exp_fn = functools.partial(exp_cls, **args)

        os.makedirs(os.path.dirname(QUICKSTART_EXPR_CACHE_PATH), exist_ok=True)
        cache_file = os.path.join(
            QUICKSTART_EXPR_CACHE_PATH, f"{exp_name}_{trial_name}.json"
        )
        with open(cache_file, "w") as f:
            dict_args = OmegaConf.to_container(args)
            json.dump(dict(args=dict_args, usercode_path=usercode_path, config_name=config_name), f, indent=4, ensure_ascii=False)

        system_api.register_experiment(exp_name, exp_fn)

        try:
            main_start(_MainStartArgs(exp_name, trial_name, mode, debug=True))
        except Exception as e:
            main_stop(_MainStartArgs(exp_name, trial_name, mode, debug=True))
            logger.warning("Exception occurred. Stopping all workers.")
            raise e

    cs.store(name=config_name, node=exp_cls)
    assert config_name not in QUICKSTART_CONFIG_CLASSES
    QUICKSTART_CONFIG_CLASSES[config_name] = exp_cls
    assert config_name not in QUICKSTART_USERCODE_PATHS
    QUICKSTART_USERCODE_PATHS[config_name] = usercode_path
    assert config_name not in QUICKSTART_FN
    QUICKSTART_FN[config_name] = run
    return run
