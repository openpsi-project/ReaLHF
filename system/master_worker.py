from collections import defaultdict
from typing import Dict, List
import asyncio
import copy
import getpass
import os
import time

import colorama
import numpy as np
import torch

from base.cluster import spec as cluster_spec
from base.constants import MODEL_SAVE_ROOT
import api.config as config_pkg
import api.data as data_api
import api.dfg
import api.model as model_api
import base.dataparallel as dataparallel
import base.logging as logging
import base.namedarray as namedarray
import base.timeutil
import base.topology as topology
import system.request_reply_stream as request_reply_stream
import system.worker_base as worker_base

logger = logging.getLogger("master worker", "system")


class ExperimentComplete(Exception):

    def __init__(self, message):
        disclaimer = (colorama.Fore.GREEN + "\033[1m" +
                      "<This is not an error. It is just a way to stop the experiment.> ")
        super().__init__(disclaimer + colorama.Style.RESET_ALL + colorama.Fore.YELLOW +
                         colorama.Style.BRIGHT + "\033[1m" + message + colorama.Style.RESET_ALL)


def request_all(
    streams: List[request_reply_stream.RequestReplyStream],
    handle_type: str,
    datas: List[namedarray.NamedArray],
):
    """Send request of `handle_type` to multiple streams. len(streams)==len(datas)"""
    requests = [request_reply_stream.Payload(handle_name=handle_type, data=data) for data in datas]
    logging.getLogger("benchmark").debug(f"master worker #request_all# *end* time ${time.time_ns()}$")
    tik = time.perf_counter()
    for s, r in zip(streams, requests):
        s.post(r)
    t = time.perf_counter() - tik
    logging.getLogger("benchmark").debug(f'Request "{handle_type}" time in total: '
                                         f"{t:.4f}s, {t / len(requests):.4f}s per request")


def gather_all_replies(
    streams: List[request_reply_stream.RequestReplyStream],) -> List[List[namedarray.NamedArray]]:
    """Collect responses from multiple streams. Blocking method."""
    responses = [list([v.data for v in s.poll_all_blocked().values()]) for s in streams]
    logging.getLogger("benchmark").debug(f"master worker #gather_all_replies# *end* time ${time.time_ns()}$")
    return responses


async def parallel_rpc(
    rpc_handle_name: str,
    data: namedarray.NamedArray,
    stream: request_reply_stream.RequestReplyStream,
):
    req = request_reply_stream.Payload(handle_name=rpc_handle_name, data=data)
    stream.post(req)

    all_res = await stream.async_poll_all()
    all_res = list(all_res.values())

    # data contains all res for one data parallel rank
    data = [x.data for x in all_res]
    if len(data) == sum([bool(x) for x in data]):
        assert all([x == data[0] for x in data]), data
        data = data[0]
    elif sum([bool(x) for x in data]) == 1:
        # pipeline parallel, only one result
        data = [x for x in data if bool(x)][0]
    elif sum([bool(x) for x in data]) == 0:
        data = {}
    else:
        # model parallel, multiple same result
        data = [x for x in data if bool(x)][0]
    return data


async def model_rpc_func(
    rpc_config: api.dfg.ModelRPC,
    rpc_futures: Dict[str, asyncio.Future],
    parent_rpc_names: List[str],
    data_registry: Dict[str, torch.Tensor],
    streams: List[request_reply_stream.RequestReplyStream],
):
    num_dp = len(streams)

    tik = time.perf_counter()
    for parent in parent_rpc_names:
        await rpc_futures[parent]

    tok = time.perf_counter()
    logging.getLogger("benchmark").debug(
        f"RPC name {rpc_config.name} starts running. Wait parents time {tok - tik:.4f}s.")

    data = {}
    for k in rpc_config.input_data:
        if k not in rpc_config.input_key_remap:
            data[k] = data_registry[k]
        else:
            data[rpc_config.input_key_remap[k]] = data_registry[k]
    data = namedarray.from_dict(data)
    data = dataparallel.get_broker(rpc_config.dp_broker_type).scatter_to(data, num_dp)

    awaitables = []
    for s, x in zip(streams, data):
        awaitables.append(parallel_rpc(rpc_config.interface_type.value, x, s))
    res = await asyncio.gather(*awaitables)

    res = dataparallel.get_broker(rpc_config.dp_broker_type).gather_from(res)
    if rpc_config.log_return_value:
        logger.info(f"RPC name {rpc_config.name} returns {res}")

    for k in rpc_config.output_data:
        if k in rpc_config.output_key_remap:
            data_registry[rpc_config.output_key_remap[k]] = res[k]
        else:
            data_registry[k] = res[k]

    rpc_futures[rpc_config.name].set_result(1)

    logging.getLogger("benchmark").debug(
        f"Model rpc {rpc_config.name} finished. Run time {time.perf_counter() - tok:.4f}s.")


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

        # Used only for benchmark
        self.__benchmark_steps = config.benchmark_steps

        return config.worker_info

    def _poll(self):
        if not self.__initialized:
            self.__model_streams: Dict[config_pkg.MasterStreamID,
                                       List[request_reply_stream.RequestReplyStream]] = {
                                           k: request_reply_stream.make_master_stream(
                                               self.config.worker_info, v)
                                           for k, v in self.config.model_streams.items()
                                       }
            self.__data_stream: request_reply_stream.RequestReplyStream = (
                request_reply_stream.make_master_stream(self.config.worker_info, self.config.data_stream))
            # Request training specification from data workers, e.g. batch size and total train steps.
            request_all([self.__data_stream], "spec", [None])
            ft_specs: List[model_api.FinetuneSpec] = gather_all_replies([self.__data_stream])[0]
            if len(set(x.steps_per_epoch for x in ft_specs)) != 1:
                raise RuntimeError(f"steps_per_epoch not equal among data workers:"
                                   f" {list(x.steps_per_epoch for x in ft_specs)}. "
                                   "Consider launching less data workers.")
            ft_spec = ft_specs[0]
            ft_spec.total_train_epochs = self.config.total_train_epochs
            ft_spec.total_train_steps = ft_spec.total_train_epochs * ft_spec.steps_per_epoch

            batch_size = len(self.__data_stream.recv_sockets) * ft_spec.batch_size_per_device
            logger.info("\n\n" + "=" * 40 + f"\nTotal train epochs: {ft_spec.total_train_epochs}" +
                        f"\nTotal train steps: {ft_spec.total_train_steps}" +
                        f"\nSteps per epoch: {ft_spec.steps_per_epoch}" +
                        f"\nEffective batch size: {batch_size}\n" + "=" * 40 + "\n")
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

            self.__initialized = True
            self._ft_spec = ft_spec
            self._train_start_time = time.perf_counter()

        # fetch data from dataloader
        fetch_data_start = time.perf_counter()
        request_all([self.__data_stream], "fetch", [None])
        data_batches: List[data_api.DataBatch] = gather_all_replies([self.__data_stream])[0]
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
            f"Fetch data time consumption: {time.perf_counter() - fetch_data_start:.3f}s.")
        for key, value in sample.items():
            self._data_registry[key] = value

        # Evaluate if necessary.
        if should_eval:
            all_model_streams = list(self.__model_streams.values())
            request_all(all_model_streams, "evaluate", [None for _ in all_model_streams])
            eval_stats = dataparallel.ParallelDataBroker.gather_from(gather_all_replies(all_model_streams))
            logger.info(
                f"Evaluation results at epoch {self._epoch + 1} step {self._epoch_step + 1}: {eval_stats}")

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
                k: v
                for k, v in self.__model_streams.items() if k.model_name == rpc.model_name
            }
            topo = self.__model_topos[rpc.model_name]
            assert len(concerned_streams) == topo.get_dim("data")

            task = self.__event_loop.create_task(
                model_rpc_func(
                    rpc,
                    futures,
                    self.__rpc_parents[i],
                    self._data_registry,
                    list(concerned_streams.values()),
                ))
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
