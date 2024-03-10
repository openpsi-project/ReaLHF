from collections import defaultdict
from statistics import mean, stdev
from types import MethodType
import json
import os
import time

from deepspeed.runtime.engine import DeepSpeedEngine
from flash_attn.bert_padding import unpad_input
import torch
import torch.distributed as dist

from impl.model.backend.pipe_engine import DeepSpeedPipelineEngine
from impl.model.backend.pipe_engine.instruction import *
from impl.model.nn.flash_mqat.flash_generate import GenerationConfig
from impl.model.nn.flash_mqat.flash_mqat_base import FlashMQATBlock, FlashMQATConfig
from impl.model.utils.data import PipeCacheData, PipeTransferData
import base.constants as constants
import base.logging as logging
import impl.model.backend.pipe_engine.static_schedule as schedule
import impl.model.parallelism.pipeline_parallel.p2p as p2p

logger = logging.getLogger("Profile", "benchmark")

HOME_PATH = os.path.expanduser("~")
DUMP_PATH = os.path.join(HOME_PATH, "logs/profile_stats")


class ProfileLayerEngines:
    pass


class ProfileEngine(DeepSpeedPipelineEngine):

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

    def profile_p2p(self, bs, seqlen):
        h = self.hidden_dim
        # train/inference fwd
        self.profile_forward("fwd", (bs * seqlen, h))
        self.profile_backward("bwd", (bs * seqlen, h))
        # fwd gen
        self.profile_forward("fwd_gen_1", (bs, 1, h))

        # large scale speed test
        # self.profile_forward("2GB", (1, 1024, 1024, 1024)) # 2GB
        # self.profile_offload("2GB", (1, 1024, 1024, 1024)) # 2GB

    def profile_forward(self, name, tensor_shape, dtype=torch.float16):
        send_buf = torch.rand(tensor_shape, dtype=dtype, device=self.device)
        recv_buf = torch.zeros_like(send_buf, device=self.device)

        torch.cuda.synchronize()
        dist.barrier()
        st = time.monotonic_ns()
        if self.stage_id % 2 == 0:
            if not self.is_last_stage():
                p2p.send(send_buf, self.next_stage)
        else:
            if not self.is_first_stage():
                p2p.recv(recv_buf, self.prev_stage)

        torch.cuda.synchronize()
        cost = time.monotonic_ns() - st
        k = "send" if self.stage_id % 2 == 0 else "recv"
        k = k + "_" + name
        self.profile_stats[k].append(cost)

        dist.barrier()

        st = time.monotonic_ns()
        if self.stage_id % 2 == 0:
            if not self.is_first_stage():
                p2p.recv(recv_buf, self.prev_stage)
        else:
            if not self.is_last_stage():
                p2p.send(send_buf, self.next_stage)

        torch.cuda.synchronize()
        cost = time.monotonic_ns() - st
        k = "recv" if self.stage_id % 2 == 0 else "send"
        k = k + "_" + name
        self.profile_stats[k].append(cost)

    def profile_backward(self, name, tensor_shape, dtype=torch.float16):
        send_buf = torch.rand(tensor_shape, dtype=dtype, device=self.device)
        recv_buf = torch.zeros_like(send_buf, device=self.device)

        torch.cuda.synchronize()
        dist.barrier()
        st = time.monotonic_ns()
        if self.stage_id % 2 == 0:
            if not self.is_first_stage():
                p2p.send(send_buf, self.prev_stage)
        else:
            if not self.is_last_stage():
                p2p.recv(recv_buf, self.next_stage)

        torch.cuda.synchronize()
        cost = time.monotonic_ns() - st
        k = "send" if self.stage_id % 2 == 0 else "recv"
        k = k + "_" + name
        self.profile_stats[k].append(cost)

        dist.barrier()

        st = time.monotonic_ns()
        if self.stage_id % 2 == 0:
            if not self.is_last_stage():
                p2p.recv(recv_buf, self.next_stage)
        else:
            if not self.is_first_stage():
                p2p.send(send_buf, self.prev_stage)
        torch.cuda.synchronize()
        cost = time.monotonic_ns() - st
        k = "recv" if self.stage_id % 2 == 0 else "send"
        k = k + "_" + name
        self.profile_stats[k].append(cost)

    def profile_offload(self, name, tensor_shape, dtype=torch.float16):
        dist.barrier()
        buf = torch.rand(tensor_shape, dtype=dtype, device=self.device)

        st = time.monotonic_ns()
        # buf.to("cpu")
        buf = buf.cpu()
        cost = time.monotonic_ns() - st
        k = "store_" + name
        self.profile_stats[k].append(cost)

        dist.barrier()
        st = time.monotonic_ns()
        # buf.to(self.device)
        buf = buf.cuda()
        cost = time.monotonic_ns() - st
        k = "load_" + name
        self.profile_stats[k].append(cost)

    def dump_file_name(self):
        pass

    def discard_stats(self):
        self.profile_stats = defaultdict(list)

    def print_stats(self):
        mp_rank = constants.model_parallel_rank()
        s = f"Profile stats for rank {self.global_rank} " + \
            f"(dp, pp, mp) = ({self.dp_id}, {self.stage_id}, {mp_rank}):\n"
        for k, v in self.profile_stats.items():
            s += str(v) + "\n"
            v = [vv / 1000 for vv in v]
            s += f"Instruction <{k}> len: {len(v)}; v: {v}"
            if len(v) > 0:
                s += f" mean: {mean(v):.2f} micro secs; stdev: {stdev(v):.2f} micro secs\n"
        logger.info(s)

    def dump_stats(self):
        os.makedirs(DUMP_PATH, exist_ok=True)
