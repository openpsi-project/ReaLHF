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

import impl.model.backend.deepspeed
import impl.model.interface.chat
import impl.model.interface.dpo_interface
import impl.model.interface.flash.dpo_flash_interface
# import impl.model.interface.flash.gen_scoring_flash_interface
import impl.model.interface.flash.ppo_flash_interface
import impl.model.interface.flash.rw_flash_interface
import impl.model.interface.flash.sft_flash_interface
import impl.model.interface.simple_interface
import impl.model.interface.wps_ac_interface
import impl.model.nn.basic_nn
import impl.model.nn.flash_mqat.flash_from_hf_impl
import impl.model.nn.flash_mqat.flash_generate
import impl.model.nn.flash_mqat.flash_mqat_api
import impl.model.nn.flash_mqat.flash_mqat_base
import impl.model.nn.flash_mqat.flash_mqat_parallel
import impl.model.nn.lora
# import impl.model.nn.model_parallel_nn
import impl.model.nn.stream_pipe_nn
