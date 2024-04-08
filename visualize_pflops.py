from typing import *
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import transformers

from api.config.config_base import MODEL_TYPE_TO_PATH, ModelType
from api.config.config_flash_model import FLASH_MODEL_CONFIG_CONVERTER
from base.monitor import caculuate_llama_forward_flops, calculate_llama_gen_flops, calculate_llama_train_flops


def compute_rlhf_pflops(
    actor_size: int,
    critic_size: int,
    ref_size: int,
    rw_size: int,
    batch_size: int,
    prompt_len: int,
    gen_len: int,
    avg_time: float,
):
    mconfigs = {}
    for name, model_size in [
        ("actor", actor_size),
        ("critic", critic_size),
        ("ref", ref_size),
        ("rw", rw_size),
    ]:
        hf_model_type = "llama" if model_size != 34 else "codellama"
        path = MODEL_TYPE_TO_PATH[ModelType(hf_model_type, model_size, True)]
        hf_config = transformers.AutoConfig.from_pretrained(path)
        mconfig = FLASH_MODEL_CONFIG_CONVERTER[hf_model_type](hf_config)
        mconfigs[name] = mconfig
    assert (prompt_len + gen_len) * batch_size == 2**17, (batch_size, prompt_len, gen_len)
    flops = 0
    flops += calculate_llama_gen_flops(
        batch_size,
        [prompt_len] * batch_size,
        gen_len,
        num_layers=mconfigs["actor"].n_layers,
        hidden_size=mconfigs["actor"].hidden_dim,
        intermediate_size=mconfigs["actor"].intermediate_dim,
        vocab_size=mconfigs["actor"].vocab_size,
    )
    for name in ["critic", "ref", "rw"]:
        flops += caculuate_llama_forward_flops(
            batch_size,
            [prompt_len + gen_len] * batch_size,
            num_layers=mconfigs[name].n_layers,
            hidden_size=mconfigs[name].hidden_dim,
            intermediate_size=mconfigs[name].intermediate_dim,
            vocab_size=mconfigs[name].vocab_size,
        )
    for name in ["actor", "critic"]:
        flops += calculate_llama_train_flops(
            4,
            batch_size,
            [prompt_len + gen_len] * batch_size,
            num_layers=mconfigs[name].n_layers,
            hidden_size=mconfigs[name].hidden_dim,
            intermediate_size=mconfigs[name].intermediate_dim,
            vocab_size=mconfigs[name].vocab_size,
        )
    return flops / 1e15 / avg_time


def amend_baseline_data(all_data: List, baseline_name: str):
    if baseline_name == "dschat":
        os.system(
            "python3 ../DeepSpeedExamples/sosp_parselog.py --max --no_print --dump_to_file /tmp/dschat.pkl")
        with open("/tmp/dschat.pkl", "rb") as f:
            data: pd.DataFrame = pickle.load(f)
    elif baseline_name == "openrlhf":
        os.system("python3 ../OpenRLHF/sosp_parselog.py --max --dump_to_file /tmp/openrlhf.pkl")
        with open("/tmp/openrlhf.pkl", "rb") as f:
            data: pd.DataFrame = pickle.load(f)
    else:
        raise NotImplementedError()

    # case 1-3, increased actor/ref size, fixed rew/critic size, seqlen=128,384,896
    for i, actor_size in enumerate([7, 13]):
        for gpu_scale_factor in [1, 2, 4, 8][:4 - i]:
            ref_size = actor_size
            critic_size = rew_size = 7
            seqlen = 384
            bs = 2**17 // (seqlen + 128)
            df = data[(data["actor_size"] == actor_size)
                      & (data["critic_size"] == critic_size)
                      & (data["seqlen"] == seqlen)
                      & (data["gpu_scale_factor"] == gpu_scale_factor)]
            assert len(df) == 1, df
            d = df.to_dict(orient="records")[0]
            p = compute_rlhf_pflops(
                actor_size,
                critic_size,
                ref_size,
                rew_size,
                batch_size=bs,
                prompt_len=128,
                gen_len=seqlen,
                avg_time=d["avg_time"],
            )
            all_data.append(
                dict(
                    case=1 if seqlen == 128 else 2 if seqlen == 384 else 3,
                    ngpus=(8 if actor_size == 7 else
                           16 if actor_size == 13 else 32 if actor_size == 34 else 64) * gpu_scale_factor,
                    pflops=p,
                    system=baseline_name,
                ))
            print(
                dict(
                    case=1 if seqlen == 128 else 2 if seqlen == 384 else 3,
                    ngpus=(8 if actor_size == 7 else
                           16 if actor_size == 13 else 32 if actor_size == 34 else 64) * gpu_scale_factor,
                    pflops=p,
                    system=baseline_name,
                ))
    return all_data


if __name__ == "__main__":
    all_data = []
    all_data = amend_baseline_data(all_data, "dschat")
    print(pd.DataFrame(all_data).to_string(index=False))
