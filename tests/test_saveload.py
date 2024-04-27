from typing import *
import os

import pytest
import torch
import torch.distributed as dist
import transformers

from reallm.api.core.config import ModelFamily
from reallm.api.core.model_api import HF_MODEL_FAMILY_REGISTRY, MODEL_FAMILY_TO_PATH, ReaLModelConfig
from reallm.base import constants, logging
from reallm.impl.model.nn.real_llm_api import ReaLModel
from tests.utils import init_global_constants, LocalMultiProcessTest

logger = logging.getLogger("tests.test_saveload")


def _shrink_mconfig(mconfig: ReaLModelConfig):
    mconfig.hidden_dim = 128
    mconfig.head_dim = 32
    mconfig.n_kv_heads = 1
    mconfig.intermediate_dim = 512
    mconfig.n_layers = 4
    return mconfig


def _save_then_load(model_family_name: str, is_critic: bool):
    model_name = "saveload_test"
    init_global_constants(2, 2, 2, model_name="saveload_test")
    assert dist.get_world_size() == 8, dist.get_world_size()
    save_path = "/tmp/ReaL-saveload-test"
    os.makedirs(save_path, exist_ok=True)
    with constants.model_scope(model_name):
        key = ModelFamily(model_family_name, 0, is_critic)
        if key not in MODEL_FAMILY_TO_PATH:
            logger.warning(
                f"Skipping test for {model_family_name} due to the absence of zero-size model path.")
            return
        hf_path = MODEL_FAMILY_TO_PATH[key]
        tokenizer = transformers.AutoTokenizer.from_pretrained(hf_path)
        hf_config = transformers.AutoConfig.from_pretrained(hf_path)
        config_converter = HF_MODEL_FAMILY_REGISTRY[model_family_name]["config_converter"]
        mconfig: ReaLModelConfig = config_converter(hf_config)
        mconfig = _shrink_mconfig(mconfig)

        model = ReaLModel(mconfig, dtype=torch.float16, device="cuda")
        getattr(model, f"to_{model_family_name}")(tokenizer, save_path)


@pytest.mark.parametrize("model_family_name", ["llama"])
@pytest.mark.parametrize("is_critic", [True, False])
def test_save_then_load(model_family_name: str, is_critic: bool):
    test_impl = LocalMultiProcessTest(
        world_size=8,
        func=_save_then_load,
        args=(model_family_name, is_critic),
        expr_name="saveload_test",
        trial_name="test",
    )
    test_impl.launch()
