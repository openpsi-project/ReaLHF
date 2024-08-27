import copy
import dataclasses
import math
import pprint
from typing import *

from realhf.api.core.dfg import ParamReallocHook
from realhf.api.core.system_api import ExperimentConfig
from realhf.api.quickstart.entrypoint import register_quickstart_exp
from realhf.apps.quickstart import main
from realhf.experiments.common.ppo_exp import PPOConfig
from realhf.experiments.common.utils import resolve_replica_ids, resolve_rpc_hooks


@dataclasses.dataclass
class PPORefEMAConfig(PPOConfig):
    ref_ema_eta: float = 0.001

    def initial_setup(self) -> ExperimentConfig:
        rpc_allocs = self._get_rpc_allocations()

        resolve_replica_ids(rpc_allocs)
        resolve_rpc_hooks(
            rpc_allocs, self.models
        )  # inplace modify MFCDefs in rpc allocations

        pprint.pprint(rpc_allocs)

        ######### The main difference from normal PPO #########
        def _find_rpc(name):
            return next(alloc.rpc for alloc in rpc_allocs if alloc.rpc.name == name)

        # Remove the offload hook of ref_inf, because
        # we need to receive parameters from peer GPUs and update it immediately.
        ref_inf = _find_rpc("ref_inf")
        ref_inf._post_hooks = []

        # Add an unidirectional parameter reallocation hook.
        actor_train = _find_rpc("actor_train")
        actor_train.add_post_hook(
            ParamReallocHook(
                target=ref_inf.model_name,
                eta=self.ref_ema_eta,
            )
        )
        ######### The main difference from normal PPO #########

        model_worker = self._get_model_worker_configs(rpc_allocs)

        return ExperimentConfig(
            exp_ctrl=self.exp_ctrl,
            model_rpcs=[rpc_alloc.rpc for rpc_alloc in rpc_allocs],
            model_worker=model_worker,
        )


register_quickstart_exp("ppo-ref-ema", PPORefEMAConfig)

if __name__ == "__main__":
    main()
