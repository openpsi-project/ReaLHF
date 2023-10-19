import random

import torch

from impl.model.backend.ds_pipe_engine import PipeDataParallelTopology
import api.config as config_package
import api.model

model_path = "/lustre/meizy/backup_zy/model_saves/pipe_4l_starcoder"
# model_path = "/lustre/meizy/backup_zy/model_saves/four_layers_starcoder"
EXPR_NAME = "test"
TRIAL_NAME = "test"
MODEL_NAME = "pipedatamodel"
MODEL_TYPE = "model_worker"
PIPE_DEGREE = 4
DATA_DEGREE = 1


def get_simple_interface():
    return api.model.make_interface(config_package.ModelInterface(type_="pipe_flash_sft", args=dict()))


def get_finetune_spec(bs_per_device=4):
    finetune_spec = api.model.FinetuneSpec(
        total_train_epochs=1,
        total_train_steps=10,
        steps_per_epoch=10,
        batch_size_per_device=bs_per_device,
    )
    return finetune_spec


def get_pipe_backend():
    return api.model.make_backend(
        config_package.ModelBackend(type_='ds_train',
                                    args=dict(optimizer_name='adam',
                                              optimizer_config=dict(lr=1e-5,
                                                                    weight_decay=0.0,
                                                                    betas=(0.9, 0.95)),
                                              warmup_steps_proportion=0.0,
                                              min_lr_ratio=0.0,
                                              zero_stage=1,
                                              engine_type="pipe",
                                              num_pipeline_stages=PIPE_DEGREE)))


def get_pipe_model(model_path, device):
    topology = PipeDataParallelTopology(num_pp=PIPE_DEGREE, num_dp=DATA_DEGREE)
    model_config = config_package.Model(type_="starcoder_flash_mqat_pipe",
                                        args=dict(
                                            model_path=model_path,
                                            num_pp=PIPE_DEGREE,
                                            num_dp=DATA_DEGREE,
                                            load_from_full_ckpt=False,
                                            dtype=torch.float16,
                                        ))
    model = api.model.make_model(model_config, name=MODEL_NAME, device=device)
    return topology, model


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
