import torch

# Instantiate all registered HuggingFace models.
from reallm.api.core.model_api import HF_MODEL_FAMILY_REGISTRY
from reallm.impl.model.conversion.hf_registry import HFModelRegistry
from reallm.impl.model.nn.real_llm_api import ReaLModel
# FIXME: automatic import
import reallm.impl.model.backend.deepspeed
import reallm.impl.model.backend.pipe_inf
import reallm.impl.model.interface.dpo_flash_interface
import reallm.impl.model.interface.ppo_flash_interface
import reallm.impl.model.interface.rw_flash_interface
import reallm.impl.model.interface.sft_flash_interface
import reallm.impl.model.nn.real_llm_api
import reallm.impl.model.nn.real_llm_base
import reallm.impl.model.nn.real_llm_generate
import reallm.impl.model.nn.real_llm_parallel

if torch.cuda.is_available():
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(True)
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._jit_set_nvfuser_enabled(True)
    torch._C._debug_set_autodiff_subgraph_inlining(False)

for name, helpers in HF_MODEL_FAMILY_REGISTRY.items():
    r = HFModelRegistry(**helpers)

    def _load_from_hf(model: ReaLModel, load_dir: str, init_critic_from_actor: bool):
        r.load(model, load_dir, init_critic_from_actor)

    def _save_to_hf(model: ReaLModel, tokenizer, save_dir: str, epoch, epoch_step, global_step):
        r.save(model, tokenizer, save_dir, epoch, epoch_step, global_step)

    setattr(ReaLModel, f"from_{name}", _load_from_hf)
    setattr(ReaLModel, f"to_{name}", _save_to_hf)
