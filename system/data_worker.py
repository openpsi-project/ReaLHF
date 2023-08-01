import logging

import torch.utils.data

import api.config as config_pkg
import api.data as data_api
import api.model as model_api
import base.seeding as seeding
import system.request_reply_stream as request_reply_stream
import system.worker_base as worker_base

logger = logging.getLogger("Data Worker")

DATA_WORKER_NAME = "_data_worker"


class DataWorker(worker_base.Worker):

    def __init__(self, server=None):
        super().__init__(server)
        self.__initialized = False
        self.__epoch = -1
        self.__global_step = self.__epoch_step = 0

        self.__dataset = None
        self.__dataloader = None

        self.__dict_sample = None

    def _configure(self, config: config_pkg.DataWorker):
        self.config = config
        self.__worker_count = self.config.worker_info.worker_count
        self.__worker_index = self.config.worker_info.worker_index
        seeding.set_random_seed(self.config.seed)
        self.__stream = request_reply_stream.make_reply_server(config.worker_info, config.stream)
        return config.worker_info

    def __setup_datasets(self):
        # initialize data sets
        self.__dataset = torch.utils.data.ConcatDataset([
            data_api.make_dataset(
                d,
                self.config.seed,
                self.__worker_index,
                self.__worker_count,
                self.config.tokenizer_name_or_path,
                self.config.worker_info.experiment_name,
                self.config.worker_info.trial_name,
                cache_root=(None if not self.config.use_dataset_cache else self.config.dataset_cahce_root),
            ) for d in self.config.datasets
        ],)
        self.__dataloader = data_api.make_dataloader(self.config.dataloader, self.__dataset)
        self.__data_generator = enumerate([])

    def _poll(self):
        # first poll: load dataset
        if not self.__initialized:
            self.__setup_datasets()
            self.__initialized = True

        # fetch data from dataloader
        if self.__dict_sample is None:
            try:
                self.__epoch_step, self.__dict_sample = next(self.__data_generator)
            except StopIteration:
                self.__epoch += 1
                self.__data_generator = enumerate(self.__dataloader)
                self.__epoch_step, self.__dict_sample = next(self.__data_generator)

        request: request_reply_stream.Request = self.__stream.poll_request()

        if request.handle_name == 'fetch':
            res = data_api.DataBatch(data=self.__dict_sample,
                                     epoch=self.__epoch,
                                     epoch_step=self.__epoch_step,
                                     global_step=self.__global_step)
            sample_count = self.__dict_sample[list(self.__dict_sample.keys())[0]].shape[0]
            batch_count = 1
            self.__dict_sample = None
            self.__global_step += 1

        elif request.handle_name == "spec":
            res = model_api.FinetuneSpec(
                total_train_epochs=-1,  # place-holder, to be filled by master worker
                total_train_steps=-1,  # place-holder, to be filled by master worker
                steps_per_epoch=len(self.__dataloader),
                batch_size_per_device=self.__dataloader.batch_size,
            )
            sample_count = batch_count = 0
        else:
            raise NotImplementedError(f"Unknown request type: {request.handle_name}.")

        reply = request_reply_stream.Reply(data=res)

        self.__stream.post_reply(reply)

        return worker_base.PollResult(sample_count=sample_count, batch_count=batch_count)
