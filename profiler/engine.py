from collections import defaultdict
from statistics import mean, stdev
from types import MethodType
import json
import os
import time

import torch
import torch.distributed as dist

from impl.model.backend.pipe_engine import DeepSpeedPipelineEngine
from impl.model.backend.pipe_engine.instruction import *
from impl.model.nn.flash_mqat.flash_generate import GenerationConfig
import base.constants as constants
import base.logging as logging
import impl.model.backend.pipe_engine.static_schedule as schedule

logger = logging.getLogger("Profile", "benchmark")

HOME_PATH = os.path.expanduser("~")
DUMP_PATH = os.path.join(HOME_PATH, "logs/profile_stats")


class ProfileEngine(DeepSpeedPipelineEngine):
    compute_instructions = [OptimizerStep, ForwardPass, BackwardPass]
    profile_instructions = {
        "fwd_gen_0": ForwardPass,
        "fwd_gen_1": ForwardPass,
        "fwd_inf": ForwardPass,
        "fwd_train": ForwardPass,
        "bwd_train": BackwardPass,
        "reduce_grads": ReduceGrads,
        "opt": OptimizerStep
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile_gconfig = GenerationConfig(min_new_tokens=1, max_new_tokens=1)
        self._async_p2p = False
        self._async_instruction = False

        self.curr = None
        self.profile_stats = defaultdict(list)

        self.tensor_specs = []  # hidden_dim

    def __post_init__(self):
        pass

    def _exec_schedule(self, pipe_schedule, terminate_condition=None):
        # overrides the original _exec_schedule method
        # 1. remove burn out
        # 2. record profile stats for each instruction
        self.step_count = 0
        if self._generate_mode:
            self.curr = "fwd_gen_0"
        elif self._inference_mode:
            self.curr = "fwd_inf"
        elif self._train_mode:
            self.curr = "fwd_train"
        for step_cmds in pipe_schedule:
            # For each instruction in the step
            step_id, micro_batch_id, step_cmds = step_cmds
            for cmd in step_cmds:
                if type(cmd) not in self._INSTRUCTION_MAP:
                    raise RuntimeError(
                        f'{self.__class__.__name__} does not understand instruction {repr(cmd)}')

                # Equivalent to: self._exec_forward_pass(buffer_id=0)
                try:
                    st = time.monotonic_ns()
                    self._exec_instr = MethodType(self._INSTRUCTION_MAP[type(cmd)], self)
                    self._exec_instr(*cmd.args)
                    torch.cuda.synchronize()
                    cmd_time = time.monotonic_ns() - st

                    if type(cmd) == self.profile_instructions[self.curr]:
                        self.profile_stats[self.curr].append(cmd_time)
                        if self._generate_mode and self.curr == "fwd_gen_0":
                            self.curr = "fwd_gen_1"
                        elif self._train_mode and self.curr == "fwd_train":
                            self.curr = "bwd_train"
                        elif self._train_mode and self.curr == "bwd_train":
                            self.curr = "reduce_grads"
                        elif self._train_mode and self.curr == "reduce_grads":
                            self.curr = "opt"
                            # to measure correct optimizer time,
                            # torch.distributed.barrier() is required
                            # dist.barrier()
                            dist.barrier()

                    logger.info(f"rank {self.global_rank} step {self.step_count}, cmd {cmd} time: {cmd_time}")

                except Exception as e:
                    logger.error(f"Rank {self.global_rank} step {self.step_count}, Exception in cmd {cmd}")
                    raise e
                # logger.info(f"rank {self.global_rank} complete cmd: {cmd}")
            self.step_count += 1
        dist.barrier()

    def mock_comm_data(self):
        pass

    def profile_p2p(self):
        pass

    def dump_file_name(self):
        pass

    def discard_stats(self):
        self.profile_stats = defaultdict(list)

    def print_stats(self):
        mp_rank = constants.model_parallel_rank()
        s = f"Profile stats for rank {self.global_rank} " + \
            f"(dp, pp, mp) = ({self.dp_id}, {self.stage_id}, {mp_rank}):\n"
        for k, v in self.profile_stats.items():
            v = [vv // int(10e3) for vv in v]
            s += f"Instruction <{k}> len: {len(v)};"
            if len(v) > 0:
                s += f" mean: {mean(v):.2f} micro secs; stdev: {stdev(v):.2f} micro secs\n"
        logger.info(s)

    def dump_stats(self):
        os.makedirs(DUMP_PATH, exist_ok=True)
