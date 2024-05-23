from typing import *
import os
import shutil

import torch
import torch.distributed as dist
import transformers

from reallm.api.core.config import ModelFamily
from reallm.api.core.model_api import ReaLModelConfig
from reallm.base import constants, logging
from tests.utils import clear_name_resolve, init_global_constants, LocalMultiProcessTest

logger = logging.getLogger("tests.test_saveload")


def _shrink_mconfig(mconfig: ReaLModelConfig):
    mconfig.hidden_dim = 4096
    mconfig.head_dim = 128
    mconfig.n_kv_heads = 32
    mconfig.intermediate_dim = 11008
    mconfig.n_layers = 32
    return mconfig


def _save_then_load(model_family_name: str, is_critic: bool, pp_dp_mp: Tuple, tokenizer_path: str):
    # NOTE: import here to avoid initializing CUDA context in the main process
    from reallm.impl.model.nn.real_llm_api import ReaLModel

    # os.environ["REAL_SAVE_MAX_SHARD_SIZE_BYTE"] = str(int(1e6))

    model_name = "saveload_test"
    num_pp, num_dp, num_mp = pp_dp_mp
    init_global_constants(num_dp=num_dp, num_mp=num_mp, num_pp=num_pp, model_name="saveload_test")
    assert dist.get_world_size() == 8, dist.get_world_size()
    save_path = "/tmp/ReaL-saveload-test"
    save_path2 = "/tmp/ReaL-saveload-test2"
    if dist.get_rank() == 0:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        if os.path.exists(save_path2):
            shutil.rmtree(save_path2)
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(save_path2, exist_ok=True)
    dist.barrier()
    with constants.model_scope(model_name):
        hf_path = tokenizer_path
        tokenizer = transformers.AutoTokenizer.from_pretrained(hf_path)
        hf_config = transformers.AutoConfig.from_pretrained(hf_path)
        mconfig: ReaLModelConfig = getattr(ReaLModel, f"config_from_{model_family_name}")(hf_config)
        mconfig = _shrink_mconfig(mconfig)

        # initialize model
        model = ReaLModel(mconfig, dtype=torch.float16, device="cuda")
        model.instantiate()
        dist.barrier()

        # save
        getattr(model, f"to_{model_family_name}")(tokenizer, save_path)
        dist.barrier()
        file_size = 0
        for fn in os.listdir(save_path):
            if fn.endswith(".bin"):
                file_size += os.path.getsize(os.path.join(save_path, fn))

        # load
        getattr(model, f"from_{model_family_name}")(save_path, is_critic)
        dist.barrier()

        # load hf
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(save_path)
        dist.barrier()

        # save again, check size
        getattr(model, f"to_{model_family_name}")(tokenizer, save_path2)
        dist.barrier()
        file_size2 = 0
        for fn in os.listdir(save_path2):
            if fn.endswith(".bin"):
                file_size2 += os.path.getsize(os.path.join(save_path2, fn))
        assert file_size2 == file_size, (file_size, file_size2)


def test_save_then_load(model_family_name: str, is_critic: bool, pp_dp_mp: Tuple, tokenizer_path: str):
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
        pp_dp_mp=pp_dp_mp,
        tokenizer_path=tokenizer_path,
    )
    test_impl.launch()


if __name__ == "__main__":
    for i, pp_dp_mp in enumerate([(2, 2, 2)]):
        print(">" * 10 + f" testing with pp_dp_mp={pp_dp_mp} " + "<" * 10)
        for model_family_name, path in [("llama", "/lustre/public/pretrained_model_weights/Llama-2-7b-hf/")]:
            test_save_then_load(model_family_name, True, pp_dp_mp, path)
            test_save_then_load(model_family_name, False, pp_dp_mp, path)
    # for i, pp_dp_mp in enumerate([(2, 4, 1), (4, 1, 2), (8, 1, 1), (1, 8, 1)]):
    #     print(">" * 10 + f" testing with pp_dp_mp={pp_dp_mp} " + "<" * 10)
    #     for model_family_name in ["starcoder", "llama"]:
    #         test_save_then_load(model_family_name, True, pp_dp_mp)
    #         test_save_then_load(model_family_name, False, pp_dp_mp)
