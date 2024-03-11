from typing import Dict
import gc
import itertools
import multiprocessing as mp
import os
import queue
import socket
import time

from deepspeed.accelerator import get_accelerator
from flash_attn.bert_padding import unpad_input
import colorama
import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data

from profiler.comm import ProfileCommunication
from profiler.utils import random_sample

from base.monitor import gpu_utilization_monitor, time_mark
from base.topology import PipelineParallelGrid
from impl.model.backend.pipe_engine.stream_pipe_engine import EngineFuture, StreamPipeEngine
import api.config as config
import api.data
import api.model
import base.constants
import base.gpu_utils as gpu_utils
import base.logging as logging
import base.namedarray as namedarray
import base.numpy_utils
import base.seeding as seeding
import base.timeutil
import system.request_reply_stream as request_reply_stream
import system.worker_base as worker_base

# Register all implemented datasets and models.
import impl.model  # isort:skip
import impl.dataset  # isort:skip

logger = logging.getLogger("Model Worker", "colored")
blogger = logging.getLogger("benchmark")


class ProfileCompelte(Exception):

    def __init__(self, message):
        disclaimer = (colorama.Fore.GREEN + "\033[1m" +
                      "<This is not an error. It is just a way to stop the experiment.> ")
        super().__init__(disclaimer + colorama.Style.RESET_ALL + colorama.Fore.YELLOW +
                         colorama.Style.BRIGHT + "\033[1m" + message + colorama.Style.RESET_ALL)


class ProfileWorker(worker_base.Worker):

    def __init__(self, server=None):
        super().__init__(server)
        self.config = None
        self.model_name = None

        self.__ddp_env_resolved = False

    def _configure(self, cfg: config.ProfileWorker):
        self.config = cfg
        self.model_name = cfg.model_name
        self.device_mesh_name = cfg.device_mesh_name

        self.__experiment_name = self.config.worker_info.experiment_name
        self.__trial_name = self.config.worker_info.trial_name

        # NOTE: here worker_index is different from peer/ddp rank
        self.__worker_index = cfg.worker_info.worker_index

        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
        torch.backends.cudnn.deterministic = cfg.cudnn_deterministic

        seeding.set_random_seed(cfg.seed)

        # Reveal DDP identity of this worker to world.
        # NOTE: We include master worker in the process group, so the global rank is model_worker_index + 1
        gpu_utils.reveal_ddp_identity_single_model(self.__experiment_name, self.__trial_name, self.model_name,
                                                   self.__worker_index)
        self.__ddp_env_resolved = False

        r = self.config.worker_info
        r.model_name = cfg.model_name
        return r

    def __lazy_setup(self):
        """Setup pytorch ddp processes, and algorithms."""
        base.constants.set_model_name(self.config.model_name)
        self.__pg_info = gpu_utils.setup_ddp_single_model(
            self.__experiment_name,
            self.__trial_name,
            self.config.model_name,
            self.__worker_index,
        )
        base.constants.set_parallelism_group(
            self.config.model_name,
            dist.group.WORLD,
        )
        base.constants.set_experiment_trial_names(self.__experiment_name, self.__trial_name)

        logger.info(f"SetUp Information - Model worker index {self.__worker_index}"
                    f' type "{self.config.model_name}" located at '
                    f"{socket.gethostname()} GPU {self.__pg_info.local_gpu_id}.")

        # if self.config.backend.type_ in ["ds_train", "ds_inference"]:
        deepspeed.init_distributed()
        self.logger.info("deepspeed init distributed on model worker")
        self.__device = torch.device("cuda:0")

        self.__profile_comm = ProfileCommunication(self.device_mesh_name, self.__device,
                                                   self.__pg_info.local_gpu_id, self.__pg_info.global_rank,
                                                   self.__pg_info.world_size)

        self.profile_rounds = 7
        self.warmup_rounds = 2
        self.current_profile_round = 0
        self.profile_start = None

    def _poll(self):
        if not self.__ddp_env_resolved:
            self.__lazy_setup()
            self.__ddp_env_resolved = True
            self.tracer.start()

        if self.profile_start is None:
            self.profile_start = time.monotonic()

        self.__profile_comm.profile_comm()

        self.current_profile_round += 1
        blogger.debug("Current profile round done: {}".format(self.current_profile_round))
        if self.current_profile_round <= self.warmup_rounds:
            self.__profile_comm.reset_stats()
            logger.info("Warmup round {} done.".format(self.current_profile_round))

        if self.current_profile_round >= self.warmup_rounds + self.profile_rounds:
            self.__profile_comm.print_stats()
            self.__profile_comm.dump_stats()
            raise ProfileCompelte(f"Profile rounds {self.warmup_rounds + self.profile_rounds} complete !!! "
                                  f"total time cost {time.monotonic() - self.profile_start} s.")

        return worker_base.PollResult(0, 0)
