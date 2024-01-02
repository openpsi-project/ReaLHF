from collections import defaultdict
from typing import Dict, List, Optional
import asyncio
import itertools
import copy
import getpass
import os
import time
import gc

import colorama
import deepspeed
import numpy as np
import torch
import torch.distributed

from base.cluster import spec as cluster_spec
from base.constants import MODEL_SAVE_ROOT
import api.config as config_pkg
import api.data as data_api
import api.dfg
import api.model as model_api
import base.constants
import base.dataparallel as dataparallel
import base.gpu_utils as gpu_utils
import base.logging as logging
import base.namedarray as namedarray
import base.numpy_utils
import base.timeutil
import base.topology
import base.topology as topology
import system.request_reply_stream as request_reply_stream
import system.worker_base as worker_base

logger = logging.getLogger("master worker", "system")


class ExperimentComplete(Exception):
    def __init__(self, message):
        disclaimer = (
            colorama.Fore.GREEN
            + "\033[1m"
            + "<This is not an error. It is just a way to stop the experiment.> "
        )
        super().__init__(
            disclaimer
            + colorama.Style.RESET_ALL
            + colorama.Fore.YELLOW
            + colorama.Style.BRIGHT
            + "\033[1m"
            + message
            + colorama.Style.RESET_ALL
        )


def request_all(
    streams: List[request_reply_stream.RequestReplyStream],
    handle_type: str,
    datas: List[namedarray.NamedArray],
):
    """Send request of `handle_type` to multiple streams. len(streams)==len(datas)"""
    requests = [
        request_reply_stream.Payload(
            handle_name=handle_type,
            is_tensor=False,
            data=data,
        )
        for data in datas
    ]
    logging.getLogger("benchmark").debug(f"master worker #request_all# *end* time ${time.time_ns()}$")
    tik = time.perf_counter()
    for s, r in zip(streams, requests):
        s.post(r)
    t = time.perf_counter() - tik
    logging.getLogger("benchmark").debug(
        f'Request "{handle_type}" time in total: ' f"{t:.4f}s, {t / len(requests):.4f}s per request"
    )


def gather_all_replies(
    streams: List[request_reply_stream.RequestReplyStream],
) -> List:
    """Collect responses from multiple streams. Blocking method."""
    responses = [s.poll(block=True).data for s in streams]
    logging.getLogger("benchmark").debug(f"master worker #gather_all_replies# *end* time ${time.time_ns()}$")
    return responses


async def _awaitable_response(
    stream: request_reply_stream.RequestReplyStream,
) -> request_reply_stream.Payload:
    while True:
        try:
            return stream.poll(block=False)
        except request_reply_stream.NoMessage:
            await asyncio.sleep(0.01)
            continue


async def model_rpc_func(
    rpc_config: api.dfg.ModelRPC,
    rpc_futures: Dict[str, asyncio.Future],
    parent_rpc_names: List[str],
    data_registry: Dict[str, torch.Tensor],
    streams: List[request_reply_stream.RequestReplyStream],
    mas_dp_head_group: torch.distributed.ProcessGroup,
    scatter_buffer: Dict[str, List[torch.Tensor]],
    gather_buffer: Dict[str, List[torch.Tensor]],
    device: torch.device,
):
    num_dp = len(streams)

    tik = time.perf_counter()
    for parent in parent_rpc_names:
        await rpc_futures[parent]

    tok = time.perf_counter()
    logging.getLogger("benchmark").debug(
        f"RPC name {rpc_config.name} starts running. Wait parents time {tok - tik:.4f}s."
    )

    data = {}
    for k in rpc_config.input_data:
        if k not in rpc_config.input_key_remap:
            data[k] = data_registry[k]
        else:
            data[rpc_config.input_key_remap[k]] = data_registry[k]
    data = namedarray.from_dict(data)
    datas = dataparallel.get_broker(rpc_config.dp_broker_type).scatter_to(data, num_dp)

    dtypes = {k: v.dtype for k, v in datas[0].items()}
    all_shapes = [{k: v.shape for k, v in data.items()} for data in datas]
    buf_shapes = {}
    for k in dtypes:
        buf_shapes[k] = base.numpy_utils.shape_union(*[tuple(s[k]) for s in all_shapes])

    for stream, shapes in zip(streams, all_shapes):
        req = request_reply_stream.Payload(
            handle_name=rpc_config.interface_type.value,
            actual_shapes=shapes,
            buf_shapes=buf_shapes,
            dtypes=dtypes,
            is_tensor=True,
        )
        stream.post(req)

    # Expand scatter buffer if necessary
    _scatter_buffer_changed = False
    for k, buf_shape in buf_shapes.items():
        if k not in scatter_buffer:
            logger.info(f"Create scatter buffer on master worker for {k} with shape {buf_shape}")
            scatter_buffer[k] = [
                torch.empty(buf_shape, dtype=dtypes[k], device=device) for _ in range(num_dp + 1)
            ]
            _scatter_buffer_changed = True
        elif k in scatter_buffer and not base.numpy_utils.shape_leq(buf_shape, scatter_buffer[k][0].shape):
            logger.info(
                f"Resize scatter buffer on master worker for {k}"
                f" from {scatter_buffer[k][0].shape} to {buf_shape}"
            )
            new_x = []
            for x in scatter_buffer[k]:
                padding = tuple(
                    itertools.chain.from_iterable(
                        reversed([(0, s2 - s1) for s1, s2 in zip(x.shape, buf_shape)])
                    )
                )
                new_x.append(torch.nn.functional.pad(x, pad=padding, mode="constant", value=0))
            scatter_buffer[k] = new_x
            _scatter_buffer_changed = True
    if _scatter_buffer_changed:
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    # Put data into scatter buffer
    for i, data in enumerate(datas):
        for k, v in data.items():
            assert len(scatter_buffer[k]) == len(datas) + 1
            s = tuple(slice(0, x) for x in v.shape)
            scatter_buffer[k][i + 1][s] = data[k]

    # Scatter data to DP head model workers.
    for k in scatter_buffer:
        torch.distributed.scatter(
            scatter_buffer[k][0],
            scatter_list=scatter_buffer[k],
            src=0,
            group=mas_dp_head_group,
        )

    responses = await asyncio.gather(*[_awaitable_response(s) for s in streams])

    recv_tik = time.perf_counter()
    if responses[0].is_tensor:
        all_buf_shapes = [response.buf_shapes for response in responses]
        for buf_shapes in all_buf_shapes:
            for k, v in buf_shapes.items():
                assert buf_shapes[k] == all_buf_shapes[0][k]

        # Expand gather buffer if necessary.
        _gather_buffer_changed = False
        for k, buf_shape in buf_shapes.items():
            if k not in gather_buffer:
                logger.info(f"create gather buffer on master worker for {k} with shape {buf_shape}")
                gather_buffer[k] = [
                    torch.empty(buf_shape, dtype=responses[0].dtypes[k], device=device)
                    for _ in range(num_dp + 1)
                ]
                _gather_buffer_changed = True
            elif k in gather_buffer and not base.numpy_utils.shape_leq(buf_shape, gather_buffer[k][0].shape):
                logger.info(
                    f"Resize gather buffer on master worker for {k} from {gather_buffer[k][0].shape} to {buf_shape}"
                )
                new_x = []
                for x in gather_buffer[k]:
                    padding = tuple(
                        itertools.chain.from_iterable(
                            reversed([(0, s2 - s1) for s1, s2 in zip(x.shape, buf_shape)])
                        )
                    )
                    new_x.append(torch.nn.functional.pad(x, pad=padding, mode="constant", value=0))
                gather_buffer[k] = new_x
                _gather_buffer_changed = True
        if _gather_buffer_changed:
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()

        for k in gather_buffer:
            assert len(gather_buffer[k]) == len(datas) + 1
            torch.distributed.gather(
                gather_buffer[k][0],
                gather_list=gather_buffer[k],
                dst=0,
                group=mas_dp_head_group,
            )
        all_res = []
        for i, response in enumerate(responses):
            res_ = {}
            for k, vs in gather_buffer.items():
                assert len(vs) == len(responses) + 1
                v = vs[i + 1]
                shape = response.actual_shapes[k]
                s = tuple(slice(0, x) for x in shape)
                res_[k] = v[s]
            all_res.append(namedarray.from_dict(res_))
        res = dataparallel.get_broker(rpc_config.dp_broker_type).gather_from(all_res)
    else:
        res = dataparallel.get_broker(rpc_config.dp_broker_type).gather_from(
            [response.data for response in responses]
        )
    logger.info(f"Master worker return from model worker time: {time.perf_counter() - recv_tik:.4f}s")

    if rpc_config.log_return_value:
        logger.info(f"RPC name {rpc_config.name} returns {res}")

    for k in rpc_config.output_data:
        assert res[k].is_cuda, f"Output {k} returned by {rpc_config.name} is not on cuda"
        if k in rpc_config.output_key_remap:
            data_registry[rpc_config.output_key_remap[k]] = res[k]
        else:
            data_registry[k] = res[k]

    rpc_futures[rpc_config.name].set_result(1)

    logging.getLogger("benchmark").debug(
        f"Model rpc {rpc_config.name} finished. Run time {time.perf_counter() - tok:.4f}s."
    )


class MasterWorker(worker_base.Worker):
    os.makedirs(MODEL_SAVE_ROOT, exist_ok=True)

    def __init__(self, server=None):
        super().__init__(server)
        self.__initialized = False

        self._data_registry = {}

        self._epoch = -1
        self._epoch_step = self._global_step = 0

        self._ft_spec = None
        self._train_start_time = None

        # for benchmark
        self.e2e_time_history = []
        self.level_time_history = defaultdict(list)

    def _configure(self, config: config_pkg.MasterWorker):
        self.config = config

        self.__model_topos: Dict[str, topology.PipeModelDataParallelTopology] = config.model_topos

        # Build execution graph and initialize concurrency utilities.
        self.__rpc_parents, self.__rpc_edges = api.dfg.build_graph(config.model_rpcs)
        self.__model_rpcs = config.model_rpcs

        self.__event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.__event_loop)

        # Save and eval control.
        self.__total_train_epochs = config.total_train_epochs
        self.__save_ctl = base.timeutil.EpochStepTimeFreqCtl(
            freq_epoch=config.save_frequency_epochs,
            freq_step=config.save_frequency_steps,
            freq_sec=config.save_frequency_seconds,
        )
        self.__eval_ctl = base.timeutil.EpochStepTimeFreqCtl(
            freq_epoch=config.eval_frequency_epochs,
            freq_step=config.eval_frequency_steps,
            freq_sec=config.eval_frequency_seconds,
        )

        self.MODEL_SAVE_ROOT = os.path.join(
            MODEL_SAVE_ROOT,
            config.worker_info.experiment_name,
            config.worker_info.trial_name,
        )
        os.makedirs(self.MODEL_SAVE_ROOT, exist_ok=True)

        gpu_utils.reveal_ddp_identity(
            expr_name=self.config.worker_info.experiment_name,
            trial_name=self.config.worker_info.trial_name,
            worker_index=0,
        )

        # Used only for benchmark
        self.__benchmark_steps = config.benchmark_steps

        return config.worker_info

    def __lazy_init(self):
        # Set up streams.
        self.__model_streams: Dict[
            config_pkg.MasterStreamID, List[request_reply_stream.RequestReplyStream]
        ] = {
            k: request_reply_stream.make_master_stream(
                self.config.worker_info,
                v,
                n_subscribers=self.__model_topos[k.model_name].get_dim("model")
                * self.__model_topos[k.model_name].get_dim("pipe"),
            )
            for k, v in self.config.model_streams.items()
        }
        self.__data_stream: request_reply_stream.RequestReplyStream = request_reply_stream.make_master_stream(
            self.config.worker_info,
            self.config.data_stream,
            n_subscribers=1,
        )

        # Set up the global process group.
        self.__pg_info = gpu_utils.setup_ddp(
            expr_name=self.config.worker_info.experiment_name,
            trial_name=self.config.worker_info.trial_name,
            worker_index=0,
            mw_topos=self.config.mw_topos,
        )
        for model_name_ in self.config.mw_topos:
            base.constants.set_parallelism_group(
                model_name_,
                self.__pg_info.mw_groups[model_name_],
            )
        deepspeed.init_distributed()
        self.logger.info("deepspeed init distributed on master worker")
        self.__device = torch.device("cuda:0")

        offset = 1
        for model_name_, topo_ in self.config.mw_topos.items():
            grid = base.topology.PipelineParallelGrid(
                topology=topo_,
                process_group=self.__pg_info.mw_groups[model_name_],
                world_size=topo_.world_size(),
                process_group_offset=offset,
            )
            base.constants.set_grid(model_name_, grid)
            offset += topo_.world_size()

        # Request training specification from data workers, e.g. batch size and total train steps.
        self.__data_stream.post(request_reply_stream.Payload(handle_name="spec"))
        ft_specs: List[model_api.FinetuneSpec] = [self.__data_stream.poll(block=True).data]
        if len(set(x.steps_per_epoch for x in ft_specs)) != 1:
            raise RuntimeError(
                f"steps_per_epoch not equal among data workers:"
                f" {list(x.steps_per_epoch for x in ft_specs)}. "
                "Consider launching less data workers."
            )
        ft_spec = ft_specs[0]
        ft_spec.total_train_epochs = self.config.total_train_epochs
        ft_spec.total_train_steps = ft_spec.total_train_epochs * ft_spec.steps_per_epoch

        batch_size = ft_spec.batch_size_per_device
        logger.info(
            "\n\n"
            + "=" * 40
            + f"\nTotal train epochs: {ft_spec.total_train_epochs}"
            + f"\nTotal train steps: {ft_spec.total_train_steps}"
            + f"\nSteps per epoch: {ft_spec.steps_per_epoch}"
            + f"\nEffective batch size: {batch_size}\n"
            + "=" * 40
            + "\n"
        )
        logger.info(f"ft_spec = {ft_spec}")

        model_ft_specs = []
        for ms_id in self.__model_streams:
            model_name = ms_id.model_name
            num_dp = self.__model_topos[model_name].get_dim("data")
            model_ft_spec = copy.deepcopy(ft_spec)
            # FIXME: batch size returned by data workers may be the number of tokens, is this correct for deepspeed config?
            model_ft_spec.batch_size_per_device = batch_size // num_dp
            model_ft_specs.append(model_ft_spec)
        request_all(list(self.__model_streams.values()), "initialize", model_ft_specs)
        gather_all_replies(list(self.__model_streams.values()))

        self._ft_spec = ft_spec

        self.__scatter_buffers = {rpc.name: {} for rpc in self.__model_rpcs}
        self.__gather_buffers = {rpc.name: {} for rpc in self.__model_rpcs}

    def _poll(self):
        if not self.__initialized:
            self.__lazy_init()
            self.__initialized = True
            self._train_start_time = time.perf_counter()

        # fetch data from dataloader
        fetch_data_start = time.perf_counter()
        self.__data_stream.post(request_reply_stream.Payload(handle_name="fetch"))
        data_batches: List[data_api.DataBatch] = [self.__data_stream.poll(block=True).data]
        assert len(set(x.epoch for x in data_batches)) == 1
        assert len(set(x.epoch_step for x in data_batches)) == 1
        assert len(set(x.global_step for x in data_batches)) == 1

        should_eval = self.__eval_ctl.check(epochs=int(data_batches[0].epoch > self._epoch), steps=1)
        should_save = self.__save_ctl.check(epochs=int(data_batches[0].epoch > self._epoch), steps=1)

        # Update counters. All starting from 0.
        self._epoch = epoch = data_batches[0].epoch
        self._epoch_step = epoch_step = data_batches[0].epoch_step
        self._global_step = global_step = data_batches[0].global_step
        datas = [x.data for x in data_batches]

        # Manage fetched data. We assume fetched data is a flattened dict.
        sample = dataparallel.ParallelDataBroker.gather_from([namedarray.from_dict(x) for x in datas])
        logging.getLogger("benchmark").debug(
            f"Fetch data time consumption: {time.perf_counter() - fetch_data_start:.3f}s."
        )
        for key, value in sample.items():
            self._data_registry[key] = value.to(self.__device)

        # Evaluate if necessary.
        if should_eval:
            all_model_streams = list(self.__model_streams.values())
            request_all(all_model_streams, "evaluate", [None for _ in all_model_streams])
            eval_stats = dataparallel.ParallelDataBroker.gather_from(gather_all_replies(all_model_streams))
            logger.info(
                f"Evaluation results at epoch {self._epoch + 1} step {self._epoch_step + 1}: {eval_stats}"
            )

        # Save if necessary.
        if should_save:
            dp0streams = {k: v for k, v in self.__model_streams.items() if k.dp_rank == 0}
            assert len(dp0streams) > 0
            model_save_dirs = [os.path.join(self.MODEL_SAVE_ROOT, k.model_name) for k in dp0streams]
            request_all(list(dp0streams.values()), "save", model_save_dirs)
            gather_all_replies(list(dp0streams.values()))

        if self._epoch >= self.__total_train_epochs:
            raise ExperimentComplete(f"Training completes! Yeah!!!")

        # Main execution steps.
        execution_start = time.perf_counter()
        futures = {rpc.name: asyncio.Future(loop=self.__event_loop) for rpc in self.__model_rpcs}
        tasks = []
        for i, rpc in enumerate(self.__model_rpcs):
            concerned_streams = {
                k: v for k, v in self.__model_streams.items() if k.model_name == rpc.model_name
            }
            topo = self.__model_topos[rpc.model_name]
            assert len(concerned_streams) == topo.get_dim("data")

            task = self.__event_loop.create_task(
                model_rpc_func(
                    rpc_config=rpc,
                    rpc_futures=futures,
                    parent_rpc_names=self.__rpc_parents[i],
                    data_registry=self._data_registry,
                    streams=list(concerned_streams.values()),
                    mas_dp_head_group=self.__pg_info.mas_dp_head_groups[rpc.model_name],
                    scatter_buffer=self.__scatter_buffers[rpc.name],
                    gather_buffer=self.__gather_buffers[rpc.name],
                    device=self.__device,
                )
            )
            tasks.append(task)
        self.__event_loop.run_until_complete(asyncio.gather(*tasks, *futures.values()))

        self._data_registry.clear()
        total_time_consumption = time.perf_counter() - self._train_start_time
        time_per_step = total_time_consumption / (global_step + 1)
        e2e_time = time.perf_counter() - execution_start
        self.e2e_time_history.append(e2e_time)
        logger.info(
            f"Epoch {epoch + 1}/{self._ft_spec.total_train_epochs} "
            f"step {epoch_step + 1}/{self._ft_spec.steps_per_epoch} "
            f"(global step {global_step + 1}/{self._ft_spec.total_train_steps}) finishes. "
            f"#End to end# execution time: *{e2e_time:.3f}*s. "
            f"Total time consumption: {total_time_consumption:.3f}s. "
            f"Estimated remaining time: {time_per_step * (self._ft_spec.total_train_steps - global_step - 1):.3f}s."
        )

        bs = sample[list(sample.keys())[0]].shape[0]
        if self.__benchmark_steps is not None and global_step >= self.__benchmark_steps:
            logger.info(
                f"Finished benchmark {self.__benchmark_steps}. Total time consumption {total_time_consumption:.3f}"
            )
            logger.info(f"avg #e2e# time *{np.mean(self.e2e_time_history):.3f}*")
            for i, level_time_history in self.level_time_history.items():
                logger.info(f"avg #level{i+1}# time *{np.mean(level_time_history):.3f}*")
            raise ExperimentComplete(f"Benchmark completes! Yeah!!!")

        return worker_base.PollResult(sample_count=bs, batch_count=1)
