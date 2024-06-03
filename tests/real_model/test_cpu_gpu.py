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
from reallm.api.core.model_api import HF_MODEL_FAMILY_REGISTRY, load_hf_tokenizer, ReaLModelConfig
from reallm.base import constants, logging
from reallm.base.testing import clear_name_resolve, init_global_constants, LocalMultiProcessTest
from reallm.impl.model.nn.real_llm_api import add_helper_functions
from reallm.impl.model.nn.real_llm_generate import GenerationConfig

logger = logging.getLogger("tests.test_saveload")

MODEL_FAMILY_TO_PATH = {
    ModelFamily("llama", 0, is_critic=False): "/lustre/public/pretrained_model_weights/testOnly/llama-2-16l",
}


def _shrink_mconfig(mconfig: ReaLModelConfig):
    mconfig.hidden_dim = 128
    mconfig.head_dim = 16
    mconfig.n_kv_heads = 1
    mconfig.intermediate_dim = 256
    mconfig.n_layers = 2
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
    init_global_constants(1, 1, 1, False, False, model_name=model_name, max_prompt_len=128)
    assert dist.get_world_size() == 1, dist.get_world_size()
    save_path = "/tmp/ReaL-consistency-test"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    torch.cuda.manual_seed_all(3)
    torch.manual_seed(3)

    with constants.model_scope(model_name):
        for model_family_name in model_family_names:
            key = ModelFamily(model_family_name, 0, is_critic=False)
            if key not in MODEL_FAMILY_TO_PATH:
                logger.warning(
                    f"Skipping test for {model_family_name} due to the absence of zero-size model path.")
                return
            hf_path = MODEL_FAMILY_TO_PATH[key]
            tokenizer = load_hf_tokenizer(hf_path)
            hf_config = transformers.AutoConfig.from_pretrained(hf_path)
            mconfig: ReaLModelConfig = getattr(ReaLModel, f"config_from_{model_family_name}")(hf_config)
            mconfig = _shrink_mconfig(mconfig)

            # initialize model
            cpu_model = ReaLModel(mconfig, dtype=torch.float32, device="cpu")
            add_helper_functions(cpu_model)
            cpu_model.instantiate()
            cpu_model.eval()

            gpu_model = ReaLModel(mconfig, dtype=torch.float16, device="cuda")
            add_helper_functions(gpu_model)
            gpu_model.instantiate()
            gpu_model.load_state_dict(cpu_model.state_dict())
            gpu_model.eval()

            max_seqlen = 128
            bs = 10
            input_ids = torch.randint(0, mconfig.vocab_size, (bs, max_seqlen), dtype=torch.long)
            # input_lens = torch.randint(1, max_seqlen, (bs,), dtype=torch.int32)
            input_lens = torch.full((bs,), max_seqlen, dtype=torch.int32)
            attention_mask = torch.arange(max_seqlen)[None, :] < input_lens[:, None]

            logits1 = cpu_model(input_ids=input_ids,
                                attention_mask=attention_mask).logits * attention_mask.unsqueeze(-1)

            logits2 = gpu_model(
                input_ids=input_ids.cuda(),
                attention_mask=attention_mask.cuda()).logits.float().cpu() * attention_mask.unsqueeze(-1)
            assert torch.allclose(logits1, logits2, atol=5e-3), (
                model_family_name,
                (logits1 - logits2).abs().max(),
            )

            bs = 4
            max_seqlen = 5

            # NOTE: this test will not always pass, so we run multiple trials
            n_trials = 10
            gconfig = GenerationConfig(min_new_tokens=max_seqlen, max_new_tokens=max_seqlen, greedy=True)
            for i in range(n_trials):
                input_ids = torch.randint(0, mconfig.vocab_size, (bs, max_seqlen), dtype=torch.long)

                scores1 = cpu_model.generate(input_ids=input_ids, tokenizer=tokenizer, gconfig=gconfig).scores
                scores2 = (gpu_model.generate(input_ids=input_ids.cuda(),
                                              tokenizer=tokenizer,
                                              gconfig=gconfig).scores.float().cpu())

                try:
                    assert torch.allclose(scores1, scores2, atol=5e-2), (
                        model_family_name,
                        (scores1 - scores2).abs().max(),
                    )
                    break
                except AssertionError as e:
                    if i == n_trials - 1:
                        raise e
                    else:
                        print(f"Trial {i} failed: {e}")


if __name__ == "__main__":
    test_consistency(["llama"])
