from typing import Any, Callable, Dict, get_type_hints, List
import concurrent.futures
import copy
import functools
import getpass
import itertools
import logging
import os
import time

import numpy as np
import torch

from api.ecs import Commands, DataQuery, ModelQuery, RawDataQuery
import api.config as config_pkg
import api.data as data_api
import api.model as model_api
import base.namedarray as namedarray
import base.timeutil
import system.request_reply_stream as request_reply_stream
import system.worker_base as worker_base

logger = logging.getLogger("master worker")


def request_all(streams, handle_type, datas):
    requests = [request_reply_stream.Request(handle_type, data) for data in datas]
    tik = time.perf_counter()
    for s, r in zip(streams, requests):
        s.post_request(r)
    t = time.perf_counter() - tik
    logger.debug(f"Request \"{handle_type}\" time in total: "
                 f"{t:.4f}s, {t / len(requests):.4f}s per request")


def gather_all_replies(streams):
    responses = [s.poll_reply().data for s in streams]
    return responses


def model_rpc_call(data: namedarray.NamedArray, request_type, streams):
    datas = namedarray.split(data, len(streams))
    for x in datas:
        x.register_metadata(**data.metadata)
    start = time.perf_counter()
    request_all(streams, request_type, datas)
    replies = gather_all_replies(streams)
    logger.debug(f"RPC call \"{request_type}\" time consumption: {time.perf_counter() - start:.3f}s.")
    if isinstance(replies[0], Dict):
        return {k: np.mean([r[k] for r in replies]) for k in replies[0].keys()}
    elif isinstance(replies[0], namedarray.NamedArray):
        return namedarray.recursive_aggregate(replies, lambda x: torch.cat(x, dim=0))
    else:
        raise NotImplementedError()


def _build_find_stream_fn(model_type_hint):

    def find_stream_fn(master_worker: MasterWorker):
        return master_worker.model_streams[model_type_hint().name]

    return find_stream_fn


def _build_find_data_fn(data_type_hint):

    def find_data_fn(master_worker: MasterWorker):
        return master_worker.data_registry[data_type_hint().name]

    return find_data_fn


def _build_find_commands_fn():

    def find_commands_fn(master_worker: MasterWorker):
        return master_worker.commands

    return find_commands_fn


def wrap_func(
    func: Callable,
    rpc_call_fn: Callable[[Any, str, List, namedarray.NamedArray], namedarray.NamedArray],
):
    type_hints = get_type_hints(func)

    operating_fns = []
    for type_hint in type_hints.values():
        if isinstance(type_hint(), ModelQuery):
            fn = _build_find_stream_fn(type_hint)
        elif isinstance(type_hint(), (DataQuery, RawDataQuery)):
            fn = _build_find_data_fn(type_hint)
        elif isinstance(type_hint(), Commands):
            fn = _build_find_commands_fn()
        operating_fns.append(fn)

    def wrapped_func(master_worker):
        arguments = []
        for type_hint, fn in zip(type_hints.values(), operating_fns):
            if isinstance(type_hint(), ModelQuery):
                streams = fn(master_worker)

                class DuckModel:
                    pass

                DuckModel.generate = functools.partial(rpc_call_fn, request_type='generate', streams=streams)
                DuckModel.__call__ = functools.partial(rpc_call_fn, request_type='inference', streams=streams)
                DuckModel.train_step = functools.partial(rpc_call_fn, request_type='train', streams=streams)
                DuckModel.evaluate = functools.partial(rpc_call_fn, request_type='evaluate', streams=streams)
                arg = DuckModel()
            elif isinstance(type_hint(), (DataQuery, RawDataQuery, Commands)):
                arg = fn(master_worker)
            arguments.append(arg)
        return func(*arguments)

    return wrapped_func


class MasterWorker(worker_base.Worker):
    # MODEL_SAVE_ROOT = f"/data/aigc/llm/{getpass.getuser()}/checkpoints"
    MODEL_SAVE_ROOT = f"/data/aigc/llm/checkpoints/{getpass.getuser()}"
    os.makedirs(MODEL_SAVE_ROOT, exist_ok=True)

    def __init__(self, server=None):
        super().__init__(server)
        self.__initialized = False
        self.__commands = Commands()

        self._epoch = -1
        self._epoch_step = self._global_step = 0
        self._ft_spec = None
        self._train_start_time = None

    @property
    def commands(self):
        return self.__commands

    @property
    def model_streams(self):
        return self.__model_streams

    @property
    def data_registry(self):
        return self.commands.data_registry

    def _configure(self, config: config_pkg.MasterWorker):
        self.config = config
        self.__model_streams: Dict[str, List[request_reply_stream.NameResolvingRequestClient]] = {
            model_name: [
                request_reply_stream.make_request_client(config.worker_info, s) for s in this_model_streams
            ]
            for model_name, this_model_streams in config.model_streams.items()
        }
        self.__data_streams: List[request_reply_stream.NameResolvingRequestClient] = [
            request_reply_stream.make_request_client(config.worker_info, s) for s in config.data_streams
        ]

        self.__levels, exec_funcs = config.leveled_exec_funcs.levels, config.leveled_exec_funcs.funcs
        exec_funcs = [[wrap_func(func, model_rpc_call) for func in funcs] for funcs in exec_funcs]
        logger.info(f"Task levels resolved by ECS: {self.__levels}.")

        max_concurrency = max(len(tasks) for tasks in exec_funcs)
        logger.info(f"Thread pool max concurrency: {max_concurrency}")
        self.__thread_pool_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency)
        self.__exec_funcs = exec_funcs

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

        return config.worker_info

    def _poll(self):
        if not self.__initialized:
            request_all(self.__data_streams, 'spec', [None for _ in self.__data_streams])
            ft_spec: model_api.FinetuneSpec = gather_all_replies(self.__data_streams)[0]
            ft_spec.total_train_epochs = self.config.total_train_epochs
            ft_spec.total_train_steps = ft_spec.total_train_epochs * ft_spec.steps_per_epoch

            batch_size = len(self.__data_streams) * ft_spec.batch_size_per_device
            self.logger.info("\n\n" + "=" * 40 + f"\nTotal train epochs: {ft_spec.total_train_epochs}" +
                             f"\nTotal train steps: {ft_spec.total_train_steps}" +
                             f"\nSteps per epoch: {ft_spec.steps_per_epoch}" +
                             f"\nEffective batch size: {batch_size}\n" + "=" * 40 + "\n")

            for model_name, model_streams in self.__model_streams.items():
                model_ft_spec = copy.deepcopy(ft_spec)
                assert batch_size % len(model_streams) == 0, (batch_size, len(model_streams))
                model_ft_spec.batch_size_per_device = batch_size // len(model_streams)
                request_all(model_streams, 'initialize', [model_ft_spec for _ in model_streams])
            all_model_streams = list(itertools.chain.from_iterable(self.model_streams.values()))
            gather_all_replies(all_model_streams)

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
        self.commands._update_counter(epoch, epoch_step, global_step)
        datas = [x.data for x in data_batches]

        # Manage fetched data. We assume fetched data is a flattened dict.
        sample = {}
        for k in datas[0].keys():
            if isinstance(datas[0][k], torch.Tensor):
                if len(datas[0][k].shape) < 2:
                    raise RuntimeError(
                        f"Data {k} is not batched. Expect the first dimension to be batch size."
                        f"For packed inputs, please unsqueeze the first dimension and pad the second dimension to be the same."
                    )
                sample[k] = torch.cat([x[k] for x in datas], dim=0)
            else:
                # There may be other metadata, e.g. pad token id.
                assert all(datas[0][k] == x[k] for x in datas)
                sample[k] = datas[0][k]
        logger.info(f"Fetch data time consumption: {time.perf_counter() - fetch_data_start:.3f}s.")
        for key, value in sample.items():
            self.commands.set_data(key, value)

        # Evaluate if necessary.
        step_time_should_eval = self.__eval_step_freq_ctl.check()
        if epoch_should_eval or step_time_should_eval:
            all_model_streams = list(itertools.chain.from_iterable(self.model_streams.values()))
            request_all(all_model_streams, 'evaluate', [None for _ in all_model_streams])
            eval_replies = gather_all_replies(all_model_streams)

            eval_cnt, eval_stats = {}, {}
            for reply in eval_replies:
                for k, v in reply.items():
                    eval_cnt[k] = eval_cnt.get(k, 0) + 1
                    eval_stats[k] = eval_stats.get(k, 0) + v
            eval_stats = {
                k: v / cnt
                for k, v, cnt in zip(eval_stats.keys(), eval_stats.values(), eval_cnt.values())
            }

            self.logger.info(
                f"Evaluation results at epoch {self._epoch + 1} step {self._epoch_step + 1}: {eval_stats}")

        # Save if necessary.
        step_should_save = self.__save_step_freq_ctl.check()
        if epoch_should_save or step_should_save:
            head_model_streams = [v[0] for v in self.model_streams.values()]
            model_types = list(self.model_streams.keys())
            model_save_dirs = [os.path.join(self.MODEL_SAVE_ROOT, model_type) for model_type in model_types]
            request_all(head_model_streams, 'save', model_save_dirs)
            gather_all_replies(head_model_streams)

        if self._epoch >= self.__total_train_epochs:
            raise RuntimeError(f"Training completes! Yeah!!!")

        # Main execution steps.
        execution_start = time.perf_counter()
        for i, (tasks, task_names) in enumerate(zip(self.__exec_funcs, self.__levels)):
            logger.info(f"Executing tasks level {i + 1}, task names {task_names}...")
            tik = time.perf_counter()
            futures = [self.__thread_pool_executor.submit(task, self) for task in tasks]
            [future.result() for future in futures]
            logger.info(f"Execute tasks level {i + 1} in {time.perf_counter() - tik:.3f}s.")
        self.data_registry.clear()
        total_time_consumption = time.perf_counter() - self._train_start_time
        time_per_step = total_time_consumption / (global_step + 1)
        logger.info(
            f"Epoch {epoch + 1}/{self._ft_spec.total_train_epochs} "
            f"step {epoch_step + 1}/{self._ft_spec.steps_per_epoch} "
            f"(global step {global_step + 1}/{self._ft_spec.total_train_steps}) finishes. "
            f"Execution time consumption: {time.perf_counter() - execution_start:.3f}s. "
            f"Total time consumption: {total_time_consumption:.3f}s. "
            f"Estimated remaining time: {time_per_step * (self._ft_spec.total_train_steps - global_step - 1):.3f}s."
        )

        bs = sample[list(sample.keys())[0]].shape[0]
        return worker_base.PollResult(sample_count=bs, batch_count=1)

    def exit(self):
        self.__thread_pool_executor.shutdown()
        super().exit()
