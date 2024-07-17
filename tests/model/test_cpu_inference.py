import dataclasses
from typing import *

import pytest
import torch
import torch.distributed as dist
import transformers

from realhf.base import constants, logging, testing
from realhf.impl.model.nn.real_llm_api import add_helper_functions

logger = logging.getLogger("tests.test_cpu")


# NOTE: To run test for a new model class, please implement and register `real_config_maker`
# in realhf.api.from_hf.<your_model_class_name> and add the model class name to the
# `model_class` fixture in this file.
@pytest.fixture(params=["llama", "gpt2", "qwen2", "gemma", "mistral", "mixtral"])
def model_class(request):
    return request.param


def maybe_prepare_cpu_env(max_prompt_len: int):
    if not dist.is_initialized():
        # for parametrized runs
        dist.init_process_group(
            "gloo", rank=0, world_size=1, init_method="tcp://localhost:7777"
        )
        import deepspeed

        deepspeed.init_distributed()
        testing.init_global_constants(
            num_dp=1,
            num_mp=1,
            num_pp=1,
            sequence_parallel=False,
            max_prompt_len=max_prompt_len,
        )
        assert dist.get_world_size() == 1, dist.get_world_size()


@pytest.fixture
def mconfig(model_class):
    from realhf.impl.model.nn.real_llm_api import ReaLModel

    mconfig = getattr(ReaLModel, f"make_{model_class}_config")()
    return mconfig


@pytest.fixture
def save_path(tmpdir_factory: pytest.TempdirFactory):
    return tmpdir_factory.mktemp("save_path")


@pytest.fixture
def cpu_real_model(model_class, mconfig, save_path):
    max_prompt_len = mconfig.n_positions
    maybe_prepare_cpu_env(max_prompt_len)
    with constants.model_scope(testing.MODEL_NAME):
        from realhf.impl.model.nn.real_llm_api import ReaLModel

        model = ReaLModel(mconfig, dtype=torch.float32, device="cpu")
        add_helper_functions(model)
        model.instantiate()
        model.eval()
        getattr(model, f"to_{model_class}")(None, save_path)
    return model


@pytest.fixture
def cpu_hf_model(save_path):
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(save_path).to(
        torch.float32
    )
    hf_model.eval()
    return hf_model


@torch.no_grad()
def test_inference_cpu_consistency(cpu_real_model, cpu_hf_model, model_class, mconfig):
    max_prompt_len = mconfig.n_positions
    with constants.model_scope(testing.MODEL_NAME):
        bs = 10
        torch.manual_seed(1)
        input_ids = torch.randint(
            0, mconfig.vocab_size, (bs, max_prompt_len), dtype=torch.long
        )
        input_lens = torch.full((bs,), max_prompt_len, dtype=torch.int32)
        attention_mask = torch.arange(max_prompt_len)[None, :] < input_lens[:, None]

        logits1 = cpu_hf_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits * attention_mask.unsqueeze(-1)
        logits2 = cpu_real_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits * attention_mask.unsqueeze(-1)

        assert torch.allclose(logits1, logits2, atol=1e-4), (
            model_class,
            (logits1 - logits2).abs().max(),
        )
