import math
import os
import shutil
from typing import *

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformers

import realhf.base.testing as testing
from realhf.api.core.config import ModelFamily
from realhf.api.core.model_api import ReaLModelConfig
from realhf.base import constants, logging
from realhf.impl.model.nn.real_llm_api import add_helper_functions

logger = logging.getLogger("tests.test_hf_consistency")


def get_llama_hf_config():
    return transformers.LlamaConfig(
        vocab_size=200,
        hidden_size=128,
        intermediate_size=200,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        hidden_act="silu",
        rms_norm_eps=1e-5,
    )


def get_qwen2_hf_config():
    return transformers.Qwen2Config(
        vocab_size=200,
        hidden_size=128,
        intermediate_size=200,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        hidden_act="silu",
        rms_norm_eps=1e-5,
    )


def get_gemma_hf_config():
    return transformers.GemmaConfig(
        **{
            "attention_bias": False,
            "attention_dropout": 0.0,
            "hidden_act": "gelu",
            "hidden_size": 128,
            "intermediate_size": 200,
            "max_position_embeddings": 8192,
            "num_attention_heads": 8,
            "num_hidden_layers": 2,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-06,
            "rope_theta": 10000.0,
            "vocab_size": 200,
        }
    )


def get_gpt_config():
    return transformers.GPT2Config(
        vocab_size=200,
        n_positions=200,
        n_embd=128,
        n_layer=2,
        n_head=8,
        n_inner=200,
        activation_function="gelu",
    )


def hf_config_factory(model_family_name: str):
    if model_family_name == "llama":
        return get_llama_hf_config()
    elif model_family_name == "qwen2":
        return get_qwen2_hf_config()
    elif model_family_name == "gemma":
        return get_gemma_hf_config()
    elif model_family_name == "gpt2":
        return get_gpt_config()
    else:
        raise NotImplementedError(model_family_name)


dist.init_process_group(
    "gloo",
    rank=0,
    world_size=1,
    init_method="tcp://localhost:7777",
)
testing.init_global_constants(
    num_dp=1,
    num_mp=1,
    num_pp=1,
    sequence_parallel=False,
    max_prompt_len=128,
)
assert dist.get_world_size() == 1, dist.get_world_size()


@pytest.mark.parametrize("model_family_name", ["qwen2", "llama", "gemma", "gpt2"])
@torch.no_grad()
def test_consistency(tmp_path, model_family_name: str):
    # NOTE: import here to avoid initializing CUDA context in the main process
    from realhf.impl.model import _HF_REGISTRIES
    from realhf.impl.model.nn.real_llm_api import ReaLModel

    # NOTE: we run CPU float32 test instead of GPU test,
    # because GPU inherently has non-deterministic behavior

    with constants.model_scope(testing.MODEL_NAME):

        hf_config = hf_config_factory(model_family_name)
        mconfig: ReaLModelConfig = getattr(
            ReaLModel, f"config_from_{model_family_name}"
        )(hf_config)

        # convert back to HF config
        hf_config = getattr(ReaLModel, f"config_to_{model_family_name}")(mconfig)

        # initialize model
        model = ReaLModel(mconfig, dtype=torch.float32, device="cpu")
        add_helper_functions(model)
        model.instantiate()

        real_sd = model.state_dict()

        save_dir = tmp_path / "model"
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        getattr(ReaLModel, f"to_{model_family_name}")(model, None, save_dir)

        hf_sd = _HF_REGISTRIES[model_family_name].sd_to_hf_converter(real_sd, mconfig)
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(save_dir)

        # 2. test HF -> ReaL state dict conversion
        real_sd_ = _HF_REGISTRIES[model_family_name].sd_from_hf_converter(
            hf_model.state_dict(), mconfig
        )
        for k in real_sd:
            assert torch.allclose(real_sd[k], real_sd_[k], atol=1e-5), (
                k,
                real_sd[k],
                real_sd_[k],
            )

        # 3. test forward pass
        max_seqlen = 32
        bs = 4
        input_ids = torch.randint(
            0, mconfig.vocab_size, (bs, max_seqlen), dtype=torch.long
        )
        input_lens = torch.randint(1, max_seqlen, (bs,), dtype=torch.int32)
        # input_lens = torch.full((bs,), max_seqlen, dtype=torch.int32)
        attention_mask = torch.arange(max_seqlen)[None, :] < input_lens[:, None]

        hf_model.eval()
        logits1 = hf_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits * attention_mask.unsqueeze(-1)

        model.eval()
        logits2 = model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits * attention_mask.unsqueeze(-1)
        assert torch.allclose(logits1, logits2, atol=1e-5), (
            model_family_name,
            (logits1 - logits2).abs().max(),
        )

        print("Passed", model_family_name)


if __name__ == "__main__":
    from pathlib import Path

    test_consistency(Path("/tmp"), "gpt")
