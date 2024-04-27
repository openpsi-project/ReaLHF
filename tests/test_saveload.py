from typing import *
import os

import pytest
import torch
import torch.distributed as dist
import transformers

from reallm.api.core.config import ModelFamily
from reallm.api.core.model_api import HF_MODEL_FAMILY_REGISTRY, MODEL_FAMILY_TO_PATH, ReaLModelConfig
from reallm.base import constants, logging
from tests.utils import clear_name_resolve, init_global_constants, LocalMultiProcessTest

logger = logging.getLogger("tests.test_saveload")


def _shrink_mconfig(mconfig: ReaLModelConfig):
    mconfig.hidden_dim = 128
    mconfig.head_dim = 32
    mconfig.n_kv_heads = 1
    mconfig.intermediate_dim = 512
    mconfig.n_layers = 4
    return mconfig


def _save_then_load(model_family_name: str, is_critic: bool):
    # NOTE: import here to avoid initializing CUDA context in the main process
    from reallm.impl.model.nn.real_llm_api import ReaLModel

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
        mconfig: ReaLModelConfig = getattr(ReaLModel, f"config_from_{model_family_name}")(hf_config)
        mconfig = _shrink_mconfig(mconfig)

        model = ReaLModel(mconfig, dtype=torch.float16, device="cuda")
        model.instantiate()
        getattr(model, f"to_{model_family_name}")(tokenizer, save_path)


def test_save_then_load(model_family_name: str, is_critic: bool):
    expr_name = "saveload_test"
    trial_name = "test"
    clear_name_resolve(expr_name=expr_name, trial_name=trial_name)
    test_impl = LocalMultiProcessTest(
        world_size=8,
        func=_save_then_load,
        expr_name=expr_name,
        trial_name=trial_name,
        model_family_name=model_family_name,
        is_critic=is_critic,
    )
    test_impl.launch()


if __name__ == "__main__":
    test_save_then_load("llama", False)
