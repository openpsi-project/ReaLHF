from typing import Any, Optional, Tuple
import datetime
import gc
import logging
import os
import queue
import socket
import threading
import time

import numpy as np
import torch
import torch.utils.data

import api.config as config
import api.data
import api.model
import base.gpu_utils as gpu_utils
import base.namedarray as namedarray
import base.seeding as seeding
import system.request_reply_stream as request_reply_stream
import system.worker_base as worker_base

# some modif


class ModelWorker(worker_base.Worker):

    def __init__(self, server=None):
        super().__init__(server)
        self.config = None
        self.model_name = None

        self.__ddp_env_resolved = False
        self.__ddp_rank = None

        self.__stream = None

    @property
    def is_master(self):
        return self.__ddp_rank == 0

    def _configure(self, cfg: config.ModelWorker):
        self.config = cfg
        self.model_name = cfg.model_name

        self.__experiment_name = self.config.worker_info.experiment_name
        self.__trial_name = self.config.worker_info.trial_name
        self.__worker_index = cfg.worker_info.worker_index
        assert int(self.__worker_index) == int(os.environ['SLURM_PROCID']), (self.__worker_index,
                                                                             os.environ['SLURM_PROCID'])

        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
        torch.backends.cudnn.deterministic = cfg.cudnn_deterministic

        seeding.set_random_seed(cfg.seed)

        self.__stream = request_reply_stream.make_reply_server(cfg.worker_info, cfg.stream)

        # Reveal DDP identity of this worker to world.
        gpu_utils.reveal_ddp_identity(self.__experiment_name, self.__trial_name, self.model_name,
                                      self.__worker_index)
        self.__ddp_env_resolved = False

        r = self.config.worker_info
        r.model_name = cfg.model_name
        return r

    def __lazy_setup(self):
        """Setup pytorch ddp processes, and algorithms."""
        self.__world_size, self.__ddp_rank, local_gpu_id = gpu_utils.setup_ddp(
            self.__experiment_name, self.__trial_name, self.model_name, self.__worker_index)

        self.logger.info(
            f"SetUp Information - Model worker index {self.__worker_index}"
            f" type \"{self.config.model_name}\" located at {socket.gethostname()} GPU {local_gpu_id}.")

        self.__device = torch.device('cuda:0')
        self.__model = api.model.make_model(
            self.config.model,
            name=self.model_name,
            device=self.__device,
        )
        self.__interface = api.model.make_interface(self.config.interface)
        self.__backend = api.model.make_backend(self.config.backend)

        if self.config.eval_datasets is not None and self.config.eval_dataloader is not None:
            eval_dataset = torch.utils.data.ConcatDataset([
                api.data.make_dataset(
                    d,
                    self.config.seed,
                    self.__worker_index,
                    self.__world_size,
                    self.__model.tokenizer,
                    self.config.worker_info.experiment_name,
                    self.config.worker_info.trial_name,
                    cache_root=(None
                                if not self.config.use_dataset_cache else self.config.dataset_cahce_root),
                ) for d in self.config.eval_datasets
            ],)
            eval_dataloader = api.data.make_dataloader(self.config.eval_dataloader, eval_dataset)
        else:
            eval_dataloader = None
        self.__eval_dataloader = eval_dataloader

    def _poll(self):
        if not self.__ddp_env_resolved:
            self.__lazy_setup()
            self.__ddp_env_resolved = True

        request: request_reply_stream.Request = self.__stream.poll_request()
        if request is None:
            return worker_base.PollResult(0, 0)

        tik = time.perf_counter()
        if self.__worker_index == 0:
            self.logger.info(f"Model worker {self.model_name} received request {request.handle_name}.")
        try:
            if request.handle_name == 'initialize':
                self.__model = self.__backend.initialize(self.__model, request.data)
                res = None
            elif request.handle_name == 'save':
                res = self.__interface.save(self.__model, request.data)  # -> None
            elif request.handle_name == 'inference':
                res = self.__interface.inference(self.__model, request.data)  # -> NamedArray
            elif request.handle_name == 'train':
                res = self.__interface.train_step(self.__model, request.data)  # -> Dict
            elif request.handle_name == 'generate':
                res = self.__interface.generate(self.__model, request.data)  # -> NamedArray
            elif request.handle_name == 'evaluate':
                res = self.__interface.evaluate(self.__model, self.__eval_dataloader)  # -> Dict
            else:
                raise NotImplementedError(f"Unknown request type: {request.handle_name}.")
        except RuntimeError as e:
            self.print_monitor_info()
            raise e
        if self.__worker_index == 0:
            self.logger.info(f"Model worker {self.model_name} handle request {request.handle_name}"
                             f" in {time.perf_counter() - tik:.4f}s")
        reply = request_reply_stream.Reply(data=res)

        self.__stream.post_reply(reply)

        if self.config.cuda_cache_cleanliness:
            # following huggingface trl
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()

        sample_count = request.data.length(0) if isinstance(request.data, namedarray.NamedArray) else 0
        return worker_base.PollResult(sample_count=sample_count, batch_count=1)
