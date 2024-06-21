import dataclasses
import os
import pickle
from typing import Optional, Set

import realhf.base.constants as constants

RECOVER_INFO_PATH = None


@dataclasses.dataclass
class StepInfo:
    epoch: int
    epoch_step: int
    global_step: int


@dataclasses.dataclass
class RecoverInfo:
    recover_start: StepInfo
    last_step_info: StepInfo
    hash_vals_to_ignore: Set[int] = dataclasses.field(default_factory=set)


def dump_recover_info(recover_info: RecoverInfo):
    global RECOVER_INFO_PATH
    if RECOVER_INFO_PATH is None:
        RECOVER_INFO_PATH = os.path.join(
            constants.RECOVER_ROOT,
            constants.experiment_name(),
            constants.trial_name(),
            "recover_info.pkl",
        )
    with open(RECOVER_INFO_PATH, "wb") as f:
        pickle.dump(recover_info, f)


def load_recover_info() -> Optional[RecoverInfo]:
    global RECOVER_INFO_PATH
    if RECOVER_INFO_PATH is None:
        RECOVER_INFO_PATH = os.path.join(
            constants.RECOVER_ROOT,
            constants.experiment_name(),
            constants.trial_name(),
            "recover_info.pkl",
        )
    try:
        with open(RECOVER_INFO_PATH, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Resume info not found at {RECOVER_INFO_PATH}. "
            f"This should not be a resumed experiment!"
        )
