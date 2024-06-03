from typing import *
import math
import os
import shutil

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import transformers

from reallm.api.core.config import ModelFamily
from reallm.api.core.model_api import HF_MODEL_FAMILY_REGISTRY, MODEL_FAMILY_TO_PATH, ReaLModelConfig
from reallm.base import constants, logging
from reallm.base.testing import clear_name_resolve, init_global_constants, LocalMultiProcessTest
from reallm.impl.model.nn.real_llm_api import add_helper_functions

logger = logging.getLogger("tests.test_saveload")


def _shrink_mconfig(mconfig: ReaLModelConfig):
    mconfig.hidden_dim = 256
    mconfig.head_dim = 32
    mconfig.n_kv_heads = 1
    mconfig.intermediate_dim = 512
    mconfig.n_layers = 4
    return mconfig


@torch.no_grad()
def test_consistency(model_family_names: List[str]):
    # NOTE: import here to avoid initializing CUDA context in the main process
    from reallm.impl.model.nn.real_llm_api import ReaLModel

    # NOTE: we run CPU float32 test instead of GPU test, because GPU inherently has non-deterministic behavior
    dist.init_process_group("gloo", rank=0, world_size=1, init_method="tcp://localhost:7777")
    import deepspeed

    deepspeed.init_distributed()
    model_name = "consistency_test"
    init_global_constants(1, 1, 1, model_name=model_name)
    assert dist.get_world_size() == 1, dist.get_world_size()
    save_path = "/tmp/ReaL-consistency-test"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    with constants.model_scope(model_name):
        for model_family_name in model_family_names:
            key = ModelFamily(model_family_name, 0, is_critic=False)
            if key not in MODEL_FAMILY_TO_PATH:
                logger.warning(
                    f"Skipping test for {model_family_name} due to the absence of zero-size model path.")
                return
            hf_path = MODEL_FAMILY_TO_PATH[key]
            tokenizer = transformers.AutoTokenizer.from_pretrained(hf_path)
            hf_config = transformers.AutoConfig.from_pretrained(hf_path)
            mconfig: ReaLModelConfig = getattr(ReaLModel, f"config_from_{model_family_name}")(hf_config)
            mconfig = _shrink_mconfig(mconfig)

            # initialize model
            model = ReaLModel(mconfig, dtype=torch.float32, device="cpu")
            add_helper_functions(model)
            model.instantiate()

            # save
            getattr(model, f"to_{model_family_name}")(tokenizer, save_path)

            # load
            getattr(model, f"from_{model_family_name}")(save_path, False)

            # load hf
            hf_model = transformers.AutoModelForCausalLM.from_pretrained(save_path).to(torch.float32)
            hf_model.eval()

            max_seqlen = 128
            bs = 10
            input_ids = torch.randint(0, mconfig.vocab_size, (bs, max_seqlen), dtype=torch.long)
            # input_lens = torch.randint(1, max_seqlen, (bs,), dtype=torch.int32)
            input_lens = torch.full((bs,), max_seqlen, dtype=torch.int32)
            attention_mask = torch.arange(max_seqlen)[None, :] < input_lens[:, None]

            logits1 = hf_model(input_ids=input_ids,
                               attention_mask=attention_mask).logits * attention_mask.unsqueeze(-1)

            model.eval()
            logits2 = model(input_ids=input_ids,
                            attention_mask=attention_mask).logits * attention_mask.unsqueeze(-1)
            assert torch.allclose(logits1, logits2, atol=1e-5), (
                model_family_name,
                (logits1 - logits2).abs().max(),
            )


if __name__ == "__main__":
    test_consistency(["llama"])
