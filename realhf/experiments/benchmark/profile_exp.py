import copy
import dataclasses
import itertools
import json
import os
from typing import *

from omegaconf import OmegaConf

from realhf.api.core.config import (
    DatasetAbstraction,
    ModelInterfaceAbstraction,
    ModelInterfaceType,
)
from realhf.api.core.dfg import MFCDef
from realhf.api.core.system_api import ExperimentConfig
from realhf.api.quickstart.dataset import PromptOnlyDatasetConfig
from realhf.api.quickstart.device_mesh import MFCConfig
from realhf.api.quickstart.entrypoint import register_quickstart_exp
from realhf.api.quickstart.model import ModelTrainEvalConfig, ParallelismConfig
from realhf.base import constants, logging
from realhf.experiments.common.common import CommonExperimentConfig

logger = logging.getLogger("Profiling Experiment", "system")


def decompose_to_three_factors(n: int) -> List[Tuple[int, int, int]]:
    factors = []
    for i in range(1, int(n ** (1 / 2)) + 1):
        if n % i == 0:
            for j in range(i, int((n // i) ** (1 / 2)) + 1):
                if (n // i) % j == 0:
                    k = (n // i) // j
                    factors += list(set(itertools.permutations([i, j, k])))
    return factors


def default_parallel_config(n_gpus: int, handle_name: str) -> List[Dict[str, Any]]:
    factors = decompose_to_three_factors(n_gpus)
    return [
        {
            "data_parallel_size": dp,
            "model_parallel_size": mp,
            "pipeline_parallel_size": pp,
            "use_sequence_parallel": (mp > 1) and (handle_name != "generate"),
        }
        for dp, mp, pp in factors
    ]


@dataclasses.dataclass
class ProfileConfig(CommonExperimentConfig):
    interface_impl: str = ""
    handle_name: str = ""
    interface_kwargs_json: str = ""
    allocations_jsonl: Optional[str] = None
    n_mbs: Optional[List[int]] = None
    model: ModelTrainEvalConfig = dataclasses.field(
        default_factory=ModelTrainEvalConfig
    )
    dataset: PromptOnlyDatasetConfig = dataclasses.field(
        default_factory=PromptOnlyDatasetConfig
    )

    def __post_init__(self):
        if self.handle_name not in ["train_step", "generate", "inference"]:
            raise NotImplementedError(f"Unknown handle_name: {self.handle_name}")
        if not os.path.exists(self.interface_kwargs_json):
            raise FileNotFoundError(
                f"File not found: {self.interface_kwargs_json}. "
                "It should be a JSON file specifying the arguments "
                "for the interface implementation."
            )
        with open(self.interface_kwargs_json, "r") as f:
            kwargs = json.load(f)
        assert isinstance(kwargs, dict)
        if self.allocations_jsonl is None:
            self.parallel_configs = default_parallel_config(
                self.n_nodes * self.n_gpus_per_node, self.handle_name
            )
        else:
            assert self.allocations_jsonl.endswith(".jsonl")
            assert os.path.exists(self.allocations_jsonl)
            with open(self.allocations_jsonl, "r") as f:
                self.parallel_configs = [json.loads(l) for l in f.readlines()]
        for pcfg in self.parallel_configs:
            assert isinstance(pcfg, dict), type(pcfg)
            assert all(
                k
                in [
                    "data_parallel_size",
                    "model_parallel_size",
                    "pipeline_parallel_size",
                    "use_sequence_parallel",
                ]
                for k in pcfg.keys()
            ), pcfg.keys()
            assert (self.n_nodes * self.n_gpus_per_node) == (
                pcfg.get("data_parallel_size", 1)
                * pcfg.get("model_parallel_size", 1)
                * pcfg.get("pipeline_parallel_size", 1)
            )
        if self.n_mbs is None:
            self.n_mbs = [1]
        else:
            self.n_mbs = OmegaConf.to_container(self.n_mbs)
            assert isinstance(self.n_mbs, list), type(self.n_mbs)
            assert all(isinstance(x, int) for x in self.n_mbs)

    @property
    def max_prompt_len(self):
        return self.dataset.max_prompt_len

    @property
    def models(self):
        return {
            "default": self.model,
        }

    @property
    def rpcs(self):
        if self.handle_name == "train_step":
            interface_type = ModelInterfaceType.TRAIN_STEP
        elif self.handle_name == "inference":
            interface_type = ModelInterfaceType.INFERENCE
        elif self.handle_name == "generate":
            interface_type = ModelInterfaceType.GENERATE
        else:
            raise NotImplementedError(
                f"Unknown which handle to run in the interface: {self.handle_name}"
            )
        with open(self.interface_kwargs_json, "r") as f:
            interface_kwargs = json.load(f)
        rpc = MFCDef(
            n_seqs=self.dataset.train_bs_n_seqs,
            name="default",
            n_mbs=self.allocation.n_mbs,
            interface_type=interface_type,
            interface_impl=ModelInterfaceAbstraction(
                self.interface_impl, args=interface_kwargs
            ),
            model_name="default",
            input_keys=["packed_prompts"],
            log_return_value=False,
            model_type=self.model.type,
            model_path=self.model.path,
            balanced_dp=True,
        )
        return {"default": rpc}

    @property
    def allocations(self):
        return {"default": self.allocation}

    @property
    def datasets(self):
        return [
            DatasetAbstraction(
                "prompt",
                args=dict(
                    max_length=self.dataset.max_prompt_len,
                    dataset_path=self.dataset.path,
                    pad_to_max_length=self.dataset.pad_to_max_length,
                ),
            )
        ]

    @property
    def tokenizer_name_or_path(self):
        return self.model.path

    def initial_setup(self) -> List[ExperimentConfig]:
        self.allocation_mode = "manual"
        setups = []
        allocation_log_path = os.path.join(
            constants.LOG_ROOT,
            self.experiment_name,
            self.trial_name,
            "allocations.jsonl",
        )
        logger.info(
            f"Allocaiton configurations of the profiling experiment will be saved to: {allocation_log_path}"
        )
        with open(allocation_log_path, "w") as f:
            for pcfg, n_mbs in itertools.product(self.parallel_configs, self.n_mbs):
                pcfg["use_sequence_parallel"] = pcfg.get(
                    "use_sequence_parallel", True
                ) & (self.handle_name != "generate")
                self.allocation = MFCConfig(
                    parallel=ParallelismConfig(**pcfg), n_mbs=n_mbs
                )
                f.write(json.dumps(dataclasses.asdict(self.allocation)) + "\n")
                setup = copy.deepcopy(super().initial_setup())
                for m in setup.model_worker:
                    m.profile_mode = True
                setups.append(setup)
        return setups


register_quickstart_exp("profile", ProfileConfig)
