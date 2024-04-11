
from typing import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import transformers
import pickle

from base.monitor import calculate_llama_gen_flops, calculate_llama_train_flops, caculuate_llama_forward_flops
from api.config.config_base import ModelType, MODEL_TYPE_TO_PATH
from api.config.config_flash_model import FLASH_MODEL_CONFIG_CONVERTER


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
        os.system("python3 ../DeepSpeedExamples/sosp_parselog.py --max --dump_to_file /tmp/dschat.pkl")
        with open("/tmp/dschat.pkl", "rb") as f:
            data: pd.DataFrame = pickle.load(f)
    elif baseline_name == "openrlhf":
        os.system("python3 ../OpenRLHF/sosp_parselog.py --max --dump_to_file /tmp/openrlhf.pkl")
        with open("/tmp/openrlhf.pkl", "rb") as f:
            data: pd.DataFrame = pickle.load(f)
    else:
        raise NotImplementedError()

    for i, actor_size in enumerate([7, 13, 34]):
        gpu_scale_factors = [1,2,4,8][:4-i]
        ref_size = actor_size
        critic_size = rew_size = 7
        seqlen = 384
        bs = 2**17 // (seqlen + 128)
        for gpu_scale_factor in gpu_scale_factors:
            df = data[
                (data["actor_size"] == actor_size)
                & (data["critic_size"] == critic_size)
                & (data["seqlen"] == seqlen)
                & (data["gpu_scale_factor"] == gpu_scale_factor)
            ]
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
                    ngpus=(
                        8 if actor_size == 7 else 16 if actor_size == 13 else 32 if actor_size == 34 else 64
                    ) * gpu_scale_factor,
                    pflops=p,
                    system=baseline_name,
                    model_size=actor_size,
                )
            )
    return all_data


def main():
    all_data = []
    all_data = amend_baseline_data(all_data, "dschat")
    # all_data = amend_baseline_data(all_data, "openrlhf")

    # Convert data to DataFrame
    df = pd.DataFrame(all_data)

    # Set style
    sns.set_style("whitegrid")

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 3), sharex=True, sharey=True)

    # Plot for each seqlen setting
    for i, ((case, group), ax) in enumerate(zip(df.groupby("system"), axes.flatten())):
        sns.lineplot(x="ngpus", y="pflops", data=group, ax=ax, hue="model_size")

        ax.set_title(case)
        ax.set_xlabel("Number of GPUs")
        ax.set_ylabel("Throughput PetaFLOP/s")
        # ax.set_xscale("log")
        # ax.set_xticks([8, 16, 32, 64])
        # ax.set_xlim(6, 80)
        # ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    # Adjust layout
    plt.tight_layout()

    plt.savefig("vss.png")


if __name__ == "__main__":
    main()
