# model level generation test
# dp_size = pp_size = 2
import os
import sys

sys.path.append("../")
import argparse
import multiprocessing as mp
import random
import time

import deepspeed
import torch

mp.set_start_method('spawn', force=True)

from base.namedarray import NamedArray
from impl.model.pipeline_parallel.engine import PipelineEngine
from impl.model.pipeline_parallel.topology import PipeDataParallelTopology
import api.config as config_package
import api.model
import api.model_spec
import base.gpu_utils
import base.name_resolve as name_resolve
import base.names as names

model_path = "/lustre/meizy/backup_zy/model_saves/four_layer_starcoder"


def setup_gpu(worker_index):
    name_resolve.clear_subtree(names.trainer_ddp_peer("test", "test", "pipedatamodel"))
    time.sleep(1)
    base.gpu_utils.reveal_ddp_identity("test", "test", "pipedatamodel", worker_index)
    time.sleep(1)
    world_size, ddp_rank, local_gpu_id = base.gpu_utils.setup_ddp("test", "test", "pipedatamodel",
                                                                  worker_index)
    device = torch.device('cuda', local_gpu_id)
    return device


def setup_gpu_deepspeed_cli():
    parser = argparse.ArgumentParser(description='My training script.')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    deepspeed.init_distributed()
    return args.local_rank, device


def get_model(model_path, device):
    config_file = os.path.join(model_path, "config.json")
    config = api.model_spec.TransformerConfig.from_huggingface_config(config_file)
    topology = PipeDataParallelTopology(num_pp=2, num_dp=2)
    model_config = config_package.Model(type_="mqa_transformer_pipe",
                                        args=dict(
                                            model_name_or_path=model_path,
                                            config=config,
                                            topology=topology,
                                        ))
    model = api.model.make_model(model_config, name="pipedatamodel", device=device)
    return model


def get_interface():
    return api.model.make_interface(config_package.ModelInterface(type_="simple", args=dict()))


def get_backend():
    return api.model.make_backend(
        config_package.ModelBackend(type_='ds_train',
                                    args=dict(optimizer_name='adam',
                                              optimizer_config=dict(lr=1e-5,
                                                                    weight_decay=0.0,
                                                                    betas=(0.9, 0.95)),
                                              warmup_steps_proportion=0.0,
                                              min_lr_ratio=0.0,
                                              zero_stage=1,
                                              dp_size=2,
                                              engine_type="pipe")))


def get_input(tokenizer, device, s):
    prompts = tokenizer(s, return_tensors="pt", padding=True)

    input_ids, attention_mask = prompts["input_ids"], prompts["attention_mask"]
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    return input_ids, attention_mask


def random_sentence(min_len=1, max_len=20):
    words = ["the", "quick", "brown", "fox", "jumped", "over", "the", "lazy", "dog"]
    sentence_length = random.randint(min_len, max_len)
    return " ".join(random.choices(words, k=sentence_length))


def get_example_batch(tokenizer, device, batch_size, dp_rank, dp_worldsize, seed=1):
    random.seed(seed)
    whole_batch = [random_sentence() for _ in range(batch_size)]
    dp_batch = whole_batch[batch_size // dp_worldsize * dp_rank:batch_size // dp_worldsize * (dp_rank + 1)]
    print(f"global rank {torch.distributed.get_rank()}, dp rank {dp_rank} batch: {dp_batch}")
    return get_input(tokenizer, device, dp_batch)


def main(worker_index=0):
    device = setup_gpu(worker_index)
    # worker_index, device = setup_gpu_deepspeed_cli()
    model = get_model(model_path, device)
    interface = get_interface()
    backend = get_backend()

    finetune_spec = api.model.FinetuneSpec(total_train_epochs=1,
                                           total_train_steps=10,
                                           steps_per_epoch=10,
                                           batch_size_per_device=4,
                                           gradient_accumulation_steps=2)
    dp_worldsize = 2
    dp_rank = torch.distributed.get_rank() % 2

    model = backend.initialize(model, finetune_spec)
    # test generate
    print(f"rank {worker_index}: model initialized")
    input_ids, attention_mask = get_example_batch(model.tokenizer, device, 8, dp_rank, dp_worldsize)
    data = NamedArray(
        prompts=input_ids,
        prompt_att_mask=attention_mask,
    )
    print(f"rank {worker_index}: begin generate")
    outputs = interface.generate(model, data)
    print(f"rank {worker_index}: end generate")
    print(f"rank {worker_index}: generate outputs: {outputs}")

    # test inference
    data = NamedArray(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    print(f"rank {worker_index}: begin inference")
    outputs = interface.inference(model, data)
    print(f"rank {worker_index}: end inference")
    print(f"rank {worker_index}: inference outputs: {outputs}")

    # test train


if __name__ == "__main__":
    # main()
    ps = [mp.Process(target=main, args=(i,)) for i in range(4)]

    for p in ps:
        p.start()

    for p in ps:
        p.join()
