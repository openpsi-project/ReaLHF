import gc
import socket
import time

from deepspeed.accelerator import get_accelerator
import deepspeed
import torch
import torch.distributed as dist
import torch.utils.data

from base.monitor import time_mark
from base.topology import PipelineParallelGrid
import api.config as config
import api.data
import api.model
import base.constants
import base.gpu_utils as gpu_utils
import base.logging as logging
import base.namedarray as namedarray
import base.seeding as seeding
import base.timeutil
import system.request_reply_stream as request_reply_stream
import system.worker_base as worker_base

# Register all implemented datasets and models.
import impl.model  # isort:skip
import impl.dataset  # isort:skip

logger = logging.getLogger("Model Worker", "colored")
blogger = logging.getLogger("benchmark")


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
        # NOTE: here worker_index is different from peer/ddp rank
        self.__worker_index = cfg.worker_info.worker_index

        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
        torch.backends.cudnn.deterministic = cfg.cudnn_deterministic

        seeding.set_random_seed(cfg.seed)

        # Reveal DDP identity of this worker to world.
        gpu_utils.reveal_ddp_identity(self.__experiment_name, self.__trial_name, self.model_name,
                                      self.__worker_index)
        self.__ddp_env_resolved = False

        self.__clear_cache_frequency = base.timeutil.FrequencyControl(
            frequency_steps=self.config.cuda_cache_clear_freq)

        r = self.config.worker_info
        r.model_name = cfg.model_name
        return r

    def __lazy_setup(self):
        """Setup pytorch ddp processes, and algorithms."""
        self.__stream = request_reply_stream.make_stream(self.config.worker_info, self.config.stream)

        self.__world_size, self.__ddp_rank, local_gpu_id = gpu_utils.setup_ddp(
            self.__experiment_name, self.__trial_name, self.model_name, self.__worker_index)

        logger.info(f"SetUp Information - Model worker index {self.__worker_index}"
                    f' type "{self.config.model_name}" located at {socket.gethostname()} GPU {local_gpu_id}.')

        if self.config.backend.type_ in ["ds_train", "ds_inference"]:
            self.logger.info("deepspeed init distributed on model worker")
            deepspeed.init_distributed()
        self.__device = torch.device("cuda:0")

        self.__model = api.model.make_model(
            self.config.model,
            name=self.model_name,
            device=self.__device,
        )
        self.__interface = api.model.make_interface(self.config.interface)
        self.__backend = api.model.make_backend(self.config.backend)

        self.__world_group = dist.new_group(ranks=range(self.__world_size))
        self.__grid = PipelineParallelGrid(process_group=self.__world_group, topology=self.config.topo)
        base.constants.set_grid(self.__grid)

        if self.config.eval_datasets is not None and self.config.eval_dataloader is not None:
            eval_datasets = [
                api.data.make_dataset(
                    d,
                    self.config.seed,
                    self.config.dp_rank,
                    self.config.topo.get_dim("data"),
                    self.__model.tokenizer,
                    self.config.worker_info.experiment_name,
                    self.config.worker_info.trial_name,
                    cache_root=(None
                                if not self.config.use_dataset_cache else self.config.dataset_cahce_root),
                ) for d in self.config.eval_datasets
            ]
            if len(eval_datasets) > 1:
                eval_dataset = torch.utils.data.ConcatDataset(eval_datasets)
            else:
                eval_dataset = eval_datasets[0]
            eval_dataloader = api.data.make_dataloader(self.config.eval_dataloader, eval_dataset)
        else:
            eval_dataloader = None
        self.__eval_dataloader = eval_dataloader

    def _poll(self):
        if not self.__ddp_env_resolved:
            self.__lazy_setup()
            self.__ddp_env_resolved = True

        try:
            request: request_reply_stream.Payload = self.__stream.poll()
        except request_reply_stream.NoMessage:
            return worker_base.PollResult(0, 0)

        tik = time.perf_counter()
        if self.is_master:
            logger.info(f"Model worker {self.model_name} received request {request.handle_name}.")
        try:
            worker_identifier = f"{self.model_name}_{self.__ddp_rank}"
            if request.handle_name == "initialize":
                self.__model = self.__backend.initialize(self.__model, request.data)
                res = None
            elif request.handle_name == "save":
                res = self.__interface.save(self.__model, request.data)  # -> None
            elif request.handle_name == "inference":
                time_mark(f"{self.model_name}_inference_start", worker_identifier)
                res = self.__interface.inference(self.__model, request.data)  # -> NamedArray
                time_mark(f"{self.model_name}_inference_end", worker_identifier)
            elif request.handle_name == "train_step":
                time_mark(f"{self.model_name}_train_start", worker_identifier)
                res = self.__interface.train_step(self.__model, request.data)  # -> Dict
                time_mark(f"{self.model_name}_train_end", worker_identifier)
            elif request.handle_name == "generate":
                time_mark(f"{self.model_name}_generate_start", worker_identifier)
                res = self.__interface.generate(self.__model, request.data)  # -> NamedArray
                time_mark(f"{self.model_name}_generate_end", worker_identifier)
            elif request.handle_name == "evaluate":
                res = self.__interface.evaluate(self.__model, self.__eval_dataloader)  # -> Dict
            else:
                raise NotImplementedError(f"Unknown request type: {request.handle_name}.")
        except RuntimeError as e:
            # We may print some info here.
            raise e

        if self.is_master:
            blogger.info(f"Model worker #{self.model_name}# handle request *{request.handle_name}*"
                         f" in ${time.perf_counter() - tik:.4f}$s")

        reply = request_reply_stream.Payload(request_id=request.request_id,
                                             handle_name=request.handle_name,
                                             data=res)
        self.__stream.post(reply)

        if self.config.cuda_cache_cleanliness and self.__clear_cache_frequency.check():
            # following huggingface trl # ALWAYS COST 0.3+ SEC
            st = time.monotonic()
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            et = time.monotonic()
            if self.is_master:
                blogger.debug(f"Model worker {self.model_name} cleared cache in {et-st:.4f}s")

        # logging gpu/cpu stats
        # self.print_monitor_info()
        tik = time.perf_counter()
        blogger.debug(("Model worker #{}#: MemAllocated=*{}*GB, MaxMemAllocated=${}$GB".format(
            self.model_name,
            round(get_accelerator().memory_allocated() / 1024**3, 2),
            round(get_accelerator().max_memory_allocated() / 1024**3, 2),
        )))
        blogger.debug(f"monitoring overhead {time.perf_counter()-tik}s")

        sample_count = (request.data.length(0) if isinstance(request.data, namedarray.NamedArray) else 0)
        return worker_base.PollResult(sample_count=sample_count, batch_count=1)
