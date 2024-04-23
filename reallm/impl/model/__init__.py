try:
    import torch
    if torch.cuda.is_available():
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(True)
        torch._C._debug_set_autodiff_subgraph_inlining(False)
except ModuleNotFoundError:
    pass

import os

try:
    import transformer_engine.pytorch as te

    TE_ENABLED = True
except ImportError:
    TE_ENABLED = False
USE_TE_BACKEND = TE_ENABLED and os.getenv("FLASH_MQAT_USE_TE") == "1"

import reallm.impl.model.backend.deepspeed
import reallm.impl.model.backend.pipe_inf
import reallm.impl.model.interface.dpo_flash_interface
import reallm.impl.model.interface.ppo_flash_interface
import reallm.impl.model.interface.rw_flash_interface
import reallm.impl.model.interface.sft_flash_interface
import reallm.impl.model.nn.from_hf_impl
import reallm.impl.model.nn.real_llm_generate
import reallm.impl.model.nn.real_llm_api
import reallm.impl.model.nn.real_llm_base
import reallm.impl.model.nn.real_llm_parallel
