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

import impl.model.backend.deepspeed
import impl.model.backend.pipe_inf
import impl.model.interface.dpo_flash_interface
# import impl.model.interface.flash.gen_scoring_flash_interface
import impl.model.interface.ppo_flash_interface
import impl.model.interface.rw_flash_interface
import impl.model.interface.sft_flash_interface
import impl.model.nn.basic_nn
import impl.model.nn.flash_mqat.flash_from_hf_impl
import impl.model.nn.flash_mqat.flash_generate
import impl.model.nn.flash_mqat.flash_mqat_api
import impl.model.nn.flash_mqat.flash_mqat_base
import impl.model.nn.flash_mqat.flash_mqat_parallel
import impl.model.nn.lora
