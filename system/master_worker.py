from collections import defaultdict
from typing import Dict, List
import asyncio
import copy
import getpass
import logging
import os
import time

import numpy as np
import torch

from base.cluster import spec as cluster_spec
import api.config as config_pkg
import api.data as data_api
import api.dfg
import api.model as model_api
import base.dataparallel as dataparallel
import base.namedarray as namedarray
import base.timeutil
import base.topology as topology
import system.request_reply_stream as request_reply_stream
import system.worker_base as worker_base

logger = logging.getLogger("master worker")


def request_all(
    streams: List[request_reply_stream.RequestClient],
    handle_type: str,
    datas: List[namedarray.NamedArray],
):
    """Send request of `handle_type` to multiple streams. len(streams)==len(datas)"""
    requests = [request_reply_stream.Request(handle_type, data) for data in datas]
    logger.info(f"master worker #request_all# *end* time ${time.time_ns()}$")
    tik = time.perf_counter()
    for s, r in zip(streams, requests):
        s.post_request(r)
    t = time.perf_counter() - tik
    logger.debug(f"Request \"{handle_type}\" time in total: "
                 f"{t:.4f}s, {t / len(requests):.4f}s per request")


def gather_all_replies(streams: List[request_reply_stream.RequestClient]) -> List[namedarray.NamedArray]:
    """Collect responses from multiple streams. Blocking method."""
    responses = [s.poll_reply(block=True).data for s in streams]
    logger.info(f"master worker #gather_all_replies# *end* time ${time.time_ns()}$")
    return responses


async def parallel_rpc(
    rpc_handle_name: str,
    data: namedarray.NamedArray,
    streams: List[request_reply_stream.RequestClient],
):
    for stream in streams:
        # NOTE: Since post_request change req.data in-place, we need to construct a new request for each stream.
        req = request_reply_stream.Request(rpc_handle_name, data)
        stream.post_request(req)
    all_res = []
    for stream in streams:
        while True:
            try:
                res = stream.poll_reply()
                break
            except request_reply_stream.NoMessage:
                await asyncio.sleep(0.01)
        all_res.append(res)

    data = [res.data for res in all_res]
    if len(data) == sum([bool(x) for x in data]):
        assert all([x == data[0] for x in data]), data
        data = data[0]
    elif sum([bool(x) for x in data]) == 1:
        data = [x for x in data if bool(x)][0]
    elif sum([bool(x) for x in data]) == 0:
        data = {}
    else:
        raise RuntimeError(data)
    return data


async def model_rpc_func(
    rpc_config: api.dfg.ModelRPC,
    rpc_futures: Dict[str, asyncio.Future],
    parent_rpc_names: List[str],
    data_registry: Dict[str, torch.Tensor],
    streams: List[List[request_reply_stream.RequestClient]],
):
    num_dp = len(streams)

    tik = time.perf_counter()
    for parent in parent_rpc_names:
        await rpc_futures[parent]

    tok = time.perf_counter()
    logger.info(f"RPC name {rpc_config.name} starts running. Wait parents time {tok - tik:.4f}s.")

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
    for k in rpc_config.output_data:
        if k in rpc_config.output_key_remap:
            data_registry[rpc_config.output_key_remap[k]] = res[k]
        else:
            data_registry[k] = res[k]

    rpc_futures[rpc_config.name].set_result(1)

    logger.info(f"Model rpc {rpc_config.name} finished. Run time {time.perf_counter() - tok:.4f}s.")


class MasterWorker(worker_base.Worker):
    MODEL_SAVE_ROOT = f"{cluster_spec.fileroot}/checkpoints/{getpass.getuser()}"
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

        # Streams and topos.
        self.__model_streams: Dict[str, List[request_reply_stream.NameResolvingRequestClient]] = {
            k: request_reply_stream.make_request_client(config.worker_info, v)
            for k, v in config.model_streams.items()
        }
        self.__data_streams: List[request_reply_stream.NameResolvingRequestClient] = [
            request_reply_stream.make_request_client(config.worker_info, s) for s in config.data_streams
        ]
        self.__model_topos: Dict[str, topology.PipeModelDataParallelTopology] = config.model_topos

        # Build execution graph and initialize concurrency utilities.
        self.__rpc_parents, self.__rpc_edges = api.dfg.build_graph(config.model_rpcs)
        self.__model_rpcs = config.model_rpcs

        self.__event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.__event_loop)

        # Save and eval control.
        self.__total_train_epochs = config.total_train_epochs
        self.__save_step_freq_ctl = base.timeutil.FrequencyControl(
            frequency_seconds=config.save_frequency_seconds, frequency_steps=config.save_frequency_steps)
        self.__save_epoch_freq_ctl = base.timeutil.FrequencyControl(
            frequency_steps=config.save_frequency_epochs)

        self.__eval_step_freq_ctl = base.timeutil.FrequencyControl(
            frequency_seconds=config.eval_frequency_seconds, frequency_steps=config.eval_frequency_steps)
        self.__eval_epoch_freq_ctl = base.timeutil.FrequencyControl(
            frequency_steps=config.eval_frequency_epochs)

        self.MODEL_SAVE_ROOT = os.path.join(self.MODEL_SAVE_ROOT, config.worker_info.experiment_name,
                                            config.worker_info.trial_name)

        # Used only for benchmark
        self.__benchmark_steps = config.benchmark_steps

        return config.worker_info

    def _poll(self):
        if not self.__initialized:
            # Request training specification from data workers, e.g. batch size and total train steps.
            request_all(self.__data_streams, 'spec', [None for _ in self.__data_streams])
            ft_spec: model_api.FinetuneSpec = gather_all_replies(self.__data_streams)[0]
            ft_spec.total_train_epochs = self.config.total_train_epochs
            ft_spec.total_train_steps = ft_spec.total_train_epochs * ft_spec.steps_per_epoch

            batch_size = len(self.__data_streams) * ft_spec.batch_size_per_device
            self.logger.info("\n\n" + "=" * 40 + f"\nTotal train epochs: {ft_spec.total_train_epochs}" +
                             f"\nTotal train steps: {ft_spec.total_train_steps}" +
                             f"\nSteps per epoch: {ft_spec.steps_per_epoch}" +
                             f"\nEffective batch size: {batch_size}\n" + "=" * 40 + "\n")
            logger.info(f"ft_spec = {ft_spec}")

            model_ft_specs = []
            for model_id in self.__model_streams:
                model_name = model_id.split('@')[0]
                num_dp = self.__model_topos[model_name].get_dim('data')
                model_ft_spec = copy.deepcopy(ft_spec)
                assert batch_size % num_dp == 0, (batch_size, num_dp)
                model_ft_spec.batch_size_per_device = batch_size // num_dp
                model_ft_specs.append(model_ft_spec)
            request_all(list(self.__model_streams.values()), 'initialize', model_ft_specs)
            gather_all_replies(list(self.__model_streams.values()))

            self.__initialized = True
            self._ft_spec = ft_spec
            self._train_start_time = time.perf_counter()

        # fetch data from dataloader
        fetch_data_start = time.perf_counter()
        request_all(self.__data_streams, 'fetch', [None for _ in self.__data_streams])
        data_batches: List[data_api.DataBatch] = gather_all_replies(self.__data_streams)
        assert len(set(x.epoch for x in data_batches)) == 1
        assert len(set(x.epoch_step for x in data_batches)) == 1
        assert len(set(x.global_step for x in data_batches)) == 1

        epoch_should_save = self.__save_epoch_freq_ctl.check(steps=int(data_batches[0].epoch > self._epoch))
        epoch_should_eval = self.__eval_epoch_freq_ctl.check(steps=int(data_batches[0].epoch > self._epoch))

        # Update counters. All starting from 0.
        self._epoch = epoch = data_batches[0].epoch
        self._epoch_step = epoch_step = data_batches[0].epoch_step
        self._global_step = global_step = data_batches[0].global_step
        datas = [x.data for x in data_batches]

        # Manage fetched data. We assume fetched data is a flattened dict.
        sample = dataparallel.ParallelDataBroker.gather_from([namedarray.from_dict(x) for x in datas])
        logger.info(f"Fetch data time consumption: {time.perf_counter() - fetch_data_start:.3f}s.")
        for key, value in sample.items():
            self._data_registry[key] = value

        # Evaluate if necessary.
        step_time_should_eval = self.__eval_step_freq_ctl.check()
        if epoch_should_eval or step_time_should_eval:
            all_model_streams = list(self.__model_streams.values())
            request_all(all_model_streams, 'evaluate', [None for _ in all_model_streams])
            eval_stats = dataparallel.ParallelDataBroker.gather_from(gather_all_replies(all_model_streams))
            self.logger.info(
                f"Evaluation results at epoch {self._epoch + 1} step {self._epoch_step + 1}: {eval_stats}")

        # Save if necessary.
        step_should_save = self.__save_step_freq_ctl.check()
        if epoch_should_save or step_should_save:
            dp0streams = {k: v for k, v in self.__model_streams.items() if 'dp_00' in k.split('@')[1]}
            assert len(dp0streams) > 0
            model_save_dirs = [os.path.join(self.MODEL_SAVE_ROOT, k) for k in dp0streams]
            request_all(list(dp0streams.values()), 'save', model_save_dirs)
            gather_all_replies(list(dp0streams.values()))

        if self._epoch >= self.__total_train_epochs:
            raise RuntimeError(f"Training completes! Yeah!!!")

        # Main execution steps.
        execution_start = time.perf_counter()
        futures = {rpc.name: asyncio.Future(loop=self.__event_loop) for rpc in self.__model_rpcs}
        tasks = []
        for i, rpc in enumerate(self.__model_rpcs):
            concerned_streams = {
                k: v
                for k, v in self.__model_streams.items() if k.startswith(rpc.model_name)
            }
            topo = self.__model_topos[rpc.model_name]
            reorg_streams = []
            for dp_i in range(topo.get_dim('data')):
                dp_i_streams = [
                    v for k, v in concerned_streams.items() if f'dp_{dp_i:02d}' in k.split('@')[1]
                ]
                assert len(dp_i_streams) > 0
                reorg_streams.append(dp_i_streams)

            task = self.__event_loop.create_task(
                model_rpc_func(rpc, futures, self.__rpc_parents[i], self._data_registry, reorg_streams))
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
            raise RuntimeError(f"Benchmark completes! Yeah!!!")

        return worker_base.PollResult(sample_count=bs, batch_count=1)
