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

from profiler.engine import ProfileEngine
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

        self.__clear_cache_frequency = base.timeutil.FrequencyControl(
            frequency_steps=self.config.cuda_cache_clear_freq)

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

        topo = self.config.topo
        grid = PipelineParallelGrid(
            topology=topo,
            world_size=self.__pg_info.world_size,
            process_group=dist.group.WORLD,
            process_group_offset=0,
        )
        base.constants.set_grid(self.model_name, grid)

        self.__model = api.model.make_model(
            self.config.model,
            name=self.model_name,
            device=self.__device,
        )
        self.__interface = api.model.make_interface(self.config.interface)
        self.__backend = api.model.make_backend(self.config.backend)
        self.__engine = None

        self._mp_rank = base.constants.model_parallel_rank()
        self._pp_rank = base.constants.pipe_parallel_rank()
        self._dp_rank = base.constants.data_parallel_rank()
        self._pp_size = base.constants.pipe_parallel_world_size()

        # DP head will receive data from the master, broadcast to all data parallel peers.
        # It will also return result back to master, while other workers in the data parallel group return None.
        self._is_dp_head = self._mp_rank == 0 and self._pp_rank == self._pp_size - 1

        self.__gpu_util_mp = mp.Process(target=gpu_utilization_monitor,
                                        args=(self.__pg_info.local_gpu_id, 7200))
        self.__gpu_util_mp.start()

        self.batch_size = 32
        self.seq_len = 128
        self.vocab_size = 32000

        self.max_profile_rounds = 7
        self.warmup_rounds = 2
        self.current_profile_round = 0
        self.profile_start = None

    def __model_poll_step(self) -> worker_base.PollResult:
        # initialize
        if self.__engine is None:
            ft_spec = api.model.FinetuneSpec(10, 100, 10, self.batch_size)
            self.__model = self.__backend.initialize(self.__model, ft_spec)
            self.__engine = self.__model.module
            assert isinstance(self.__engine, ProfileEngine)
        data = random_sample(self.batch_size, self.seq_len, self.vocab_size)
        # gen_data = random_sample(self.batch_size * 4, self.seq_len, self.vocab_size)

        # instruction profiles
        self.__interface.inference(self.__model, data)

        self.__interface.train_step(self.__model, data)

        self.__interface.generate(self.__model, data)

        self.__engine.profile_p2p(self.batch_size, self.seq_len)

        # communication profiles
        return worker_base.PollResult(sample_count=1, batch_count=1)

    def _poll(self):
        if not self.__ddp_env_resolved:
            self.__lazy_setup()
            self.__ddp_env_resolved = True
            self.tracer.start()

        if self.profile_start is None:
            self.profile_start = time.monotonic()

        r = self.__model_poll_step()
        assert isinstance(self.__engine, ProfileEngine)

        if self.config.cuda_cache_cleanliness and self.__clear_cache_frequency.check():
            # following huggingface trl # ALWAYS COST 0.3+ SEC
            st = time.monotonic()
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            et = time.monotonic()
            if self._is_dp_head:
                blogger.debug(f"Model worker {self.model_name} cleared cache in {et-st:.4f}s")

        tik = time.perf_counter()
        blogger.debug(("Model worker #{}#: MemAllocated=*{}*GB, MaxMemAllocated=${}$GB".format(
            self.model_name,
            round(get_accelerator().memory_allocated() / 1024**3, 2),
            round(get_accelerator().max_memory_allocated() / 1024**3, 2),
        )))
        blogger.debug(f"monitoring overhead {time.perf_counter()-tik}s")
        if os.environ.get("DLLM_TRACE", "0") == "1":
            self.tracer.save()

        self.current_profile_round += 1
        blogger.debug("Current profile round done: {}".format(self.current_profile_round))
        if self.current_profile_round <= self.warmup_rounds:
            self.__engine.discard_stats()
            logger.info("Warmup round {} done.".format(self.current_profile_round))

        if self.current_profile_round >= self.max_profile_rounds:
            self.__engine.print_stats()
            self.__gpu_util_mp.terminate()
            raise ProfileCompelte(f"Profile rounds {self.max_profile_rounds} complete !!! "
                                  f"total time cost {time.monotonic() - self.profile_start} s.")

        return r
