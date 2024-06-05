from collections import defaultdict
from statistics import mean, stdev
from types import MethodType
import json
import os
import time

import torch
import torch.distributed as dist

from reallm.impl.model.backend.pipe_engine import PipelinableModelRunner
from reallm.impl.model.backend.pipe_engine.instruction import *
from reallm.impl.model.nn.real_llm_generate import GenerationConfig
import reallm.base.constants as constants
import reallm.base.logging as logging
import reallm.impl.model.backend.pipe_engine.static_schedule as schedule

logger = logging.getLogger("Profile", "benchmark")

HOME_PATH = os.path.expanduser("~")
DUMP_PATH = os.path.join(HOME_PATH, "logs/profile_stats")


class ProfileEngine(PipelinableModelRunner):
    # compute_instructions = [OptimizerStep, ForwardPass, BackwardPass]
    # profile_instructions = {
    #     "fwd_gen_0": ForwardPass,
    #     "fwd_gen_1": ForwardPass,
    #     "fwd_inf": ForwardPass,
    #     "fwd_train": ForwardPass,
    #     "bwd_train": BackwardPass,
    #     "reduce_grads": ReduceGrads,
    #     "opt": OptimizerStep
    # }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.profile_gconfig = GenerationConfig(min_new_tokens=1, max_new_tokens=1)
        self._async_p2p = False
        self._async_instruction = False

        self.curr = None
        self.profile_stats = defaultdict(list)

        self.tensor_specs = []  # hidden_dim
        # self.last_exec_time_cost = None
        # self.last_full_time_cost = None

    def __post_init__(self):
        pass

    def _exec_schedule(self, pipe_schedule, terminate_condition=None):
        # overrides the original _exec_schedule method
        # 1. remove burn out
        # 2. record profile stats for each instruction
        self.step_count = 0
        st = time.monotonic()
        for step_cmds in pipe_schedule:
            # For each instruction in the step
            step_id, micro_batch_id, step_cmds = step_cmds
            for cmd in step_cmds:
                if type(cmd) not in self._INSTRUCTION_MAP:
                    raise RuntimeError(
                        f"{self.__class__.__name__} does not understand instruction {repr(cmd)}")

                # Equivalent to: self._exec_forward_pass(buffer_id=0)
                try:
                    self._exec_instr = MethodType(self._INSTRUCTION_MAP[type(cmd)], self)
                    self._exec_instr(*cmd.args)
                except Exception as e:
                    logger.error(f"Rank {self.global_rank} step {self.step_count}, Exception in cmd {cmd}")
                    raise e
                # logger.info(f"rank {self.global_rank} complete cmd: {cmd}")
            self.step_count += 1
        # self.last_exec_time_cost = time.monotonic() - st
        torch.cuda.synchronize()
        logger.info(f"{pipe_schedule} done in {time.monotonic() - st} s")
