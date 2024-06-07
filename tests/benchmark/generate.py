from typing import *
import gc
import math
import os
import shutil
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import transformers

from reallm.api.core import data_api
from reallm.api.core.config import ModelFamily
from reallm.api.core.model_api import (HF_MODEL_FAMILY_REGISTRY, load_hf_tokenizer, Model, ModelName,
                                       ReaLModelConfig)
from reallm.base import constants, logging
from reallm.base.testing import clear_name_resolve, init_global_constants, LocalMultiProcessTest
from reallm.impl.dataset.prompt_dataset import PromptDataset
from reallm.impl.model.nn.real_llm_api import add_helper_functions, make_real_model, ReaLModel
from reallm.impl.model.nn.real_llm_generate import GenerationConfig

logger = logging.getLogger("tests.test_saveload")

MODEL_FAMILY_TO_PATH = {
    ModelFamily("llama", 7, is_critic=False): "/lustre/public/pretrained_model_weights/Llama-2-7b-hf",
    ModelFamily("llama", 13, is_critic=False): "/lustre/public/pretrained_model_weights/Llama-2-13b-hf",
}


@torch.no_grad()
def single_gpu_generate(model_families: List[ModelFamily]):
    dist.init_process_group("nccl", rank=0, world_size=1, init_method="tcp://localhost:7777")
    import deepspeed

    model_class = []
    model_size = []
    bs_list = []
    time_cost = []
    num_warmup = 1
    num_profile = 10
    prompt_len = 1024
    max_seqlen = min_seqlen = 128
    initial_bs = 1
    max_bs = 1024

    deepspeed.init_distributed()
    model_name = "benchmark"
    init_global_constants(1, 1, 1, False, False, model_name=model_name, max_prompt_len=1024)
    assert dist.get_world_size() == 1, dist.get_world_size()

    torch.cuda.manual_seed_all(3)
    torch.manual_seed(3)

    with constants.model_scope(model_name):
        for model_family in model_families:
            if model_family not in MODEL_FAMILY_TO_PATH:
                logger.warning(f"Skipping test for {model_family} due to unknown path.")
                return
            hf_path = MODEL_FAMILY_TO_PATH[model_family]

            model: Model = make_real_model(
                model_name,
                device="cuda",
                is_critic=False,
                init_critic_from_actor=False,
                model_path=hf_path,
                dtype="fp16",
                hf_model_family=model_family._class,
            )
            module: ReaLModel = model.module
            module.instantiate()
            module.eval()

            tokenizer = load_hf_tokenizer(hf_path)
            hf_config = transformers.AutoConfig.from_pretrained(hf_path)
            mconfig: ReaLModelConfig = getattr(ReaLModel, f"config_from_{model_family._class}")(hf_config)

            # util = data_api.DatasetUtility(tokenizer=tokenizer, ddp_rank=0, world_size=1, seed=1)
            # dataset = PromptDataset(
            #     util,
            #     max_prompt_len=1024,
            #     pad_to_max_length=False,
            #     dataset_path="/lustre/fw/datasets/antropic-hh/ppo_prompt_only.jsonl",
            # )

            bs = initial_bs
            while bs <= max_bs:
                try:
                    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs)
                    # NOTE: this test will not always pass, so we run multiple trials
                    gconfig = GenerationConfig(
                        min_new_tokens=min_seqlen,
                        max_new_tokens=max_seqlen,
                        greedy=True,
                    )
                    # for i in range(num_warmup + num_profile):
                    # for i, x in enumerate(dataloader):
                    #     if i > num_profile + num_warmup:
                    #         break
                    for i in range(num_warmup + num_profile):
                        input_ids = torch.randint(0, mconfig.vocab_size, (bs, prompt_len), dtype=torch.long)
                        st = time.monotonic()
                        # print(x)
                        res = module.generate(
                            input_ids=input_ids.cuda(),
                            tokenizer=tokenizer,
                            gconfig=gconfig,
                        )
                        tt = time.monotonic() - st
                        print(f"bs {bs} sequences shape {res.sequences.shape}, time cost {tt} s")
                        if i >= num_warmup:
                            model_class.append(model_family._class)
                            model_size.append(model_family.size)
                            bs_list.append(bs)
                            time_cost.append(tt)

                    gc.collect()
                    torch.cuda.empty_cache()
                    gc.collect()
                    # print(i, res)
                except Exception as e:
                    print(e)
                    break
                bs = bs * 2

    import pandas as pd

    df = pd.DataFrame({
        "model_class": model_class,
        "model_size": model_size,
        "batch_size": bs_list,
        "time_cost": time_cost,
    })
    print(df)
    model_sizes = df["model_size"].unique()
    model_classes = df["model_class"].unique()

    # avg token throughput
    for model_class, model_size in zip(model_classes, model_sizes):
        for bs in df["batch_size"].unique():
            df_bs = df[(df["model_class"] == model_class)
                       & (df["model_size"] == model_size)
                       & (df["batch_size"] == bs)]
            time_cost = df_bs["time_cost"].mean()
            print(f"Model: {(model_class, model_size)} batch_size: {bs}; "
                  f"time_cost: {time_cost:.4f}; throughput: {bs*max_seqlen/time_cost:.4f} tokens/s")

    import os
    import pickle

    store_path = os.environ.get("HOME", ".")
    store_path = os.path.join(store_path, "logs", "benchmark", "generate_result.pkl")
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    with open(store_path, "wb") as f:
        pickle.dump(df, f)


@torch.no_grad()
def huggingface_generate(model_families: List[ModelFamily]):
    model_class = []
    model_size = []
    bs_list = []
    time_cost = []

    num_warmup = 1
    num_profile = 5
    prompt_len = 1024
    max_seqlen = min_seqlen = 128
    initial_bs = 1
    max_bs = 1024

    for model_family in model_families:
        hf_path = MODEL_FAMILY_TO_PATH[model_family]
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(hf_path)
        model = AutoModelForCausalLM.from_pretrained(hf_path, attn_implementation="flash_attention_2")

        model.to("cuda", dtype=torch.float16)

        bs = initial_bs
        while bs <= max_bs:
            try:
                for i in range(num_warmup + num_profile):
                    st = time.monotonic()
                    input_ids = torch.randint(0, tokenizer.vocab_size, (bs, prompt_len),
                                              dtype=torch.long).cuda()
                    res = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=max_seqlen,
                        min_new_tokens=min_seqlen,
                        num_beams=1,
                        do_sample=False,
                    )
                    tt = time.monotonic() - st
                    print(f"bs {bs} sequences shape {res.shape}, time cost {tt} s")
                    if i >= num_warmup:
                        model_class.append(model_family._class)
                        model_size.append(model_family.size)
                        bs_list.append(bs)
                        time_cost.append(tt)

                    gc.collect()
                    torch.cuda.empty_cache()
                    gc.collect()
                    # print(i, res)
            except Exception as e:
                print(e)
                break
            bs = bs * 2

    import pandas as pd

    df = pd.DataFrame({
        "model_class": model_class,
        "model_size": model_size,
        "batch_size": bs_list,
        "time_cost": time_cost,
    })
    print(df)
    model_sizes = df["model_size"].unique()
    model_classes = df["model_class"].unique()

    # avg token throughput
    for model_class, model_size in zip(model_classes, model_sizes):
        for bs in df["batch_size"].unique():
            df_bs = df[(df["model_class"] == model_class)
                       & (df["model_size"] == model_size)
                       & (df["batch_size"] == bs)]
            time_cost = df_bs["time_cost"].mean()
            print(f"Model: {(model_class, model_size)} batch_size: {bs}; "
                  f"time_cost: {time_cost:.4f}; throughput: {bs*max_seqlen/time_cost:.4f} tokens/s")

    import os
    import pickle

    store_path = os.environ.get("HOME", ".")
    store_path = os.path.join(store_path, "logs", "benchmark", "huggingface_result.pkl")
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    with open(store_path, "wb") as f:
        pickle.dump(df, f)


if __name__ == "__main__":
    model_families = [
        ModelFamily("llama", 7, is_critic=False),
        # ModelFamily("llama", 13, is_critic=False),
    ]
    single_gpu_generate(model_families)
    # huggingface_generate(model_families)
