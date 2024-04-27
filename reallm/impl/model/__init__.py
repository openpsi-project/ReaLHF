from pathlib import Path
# Instantiate all registered HuggingFace models.
import importlib
import os

import torch

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

# Import all existing HuggingFace model registries.
hf_impl_path = Path(os.path.dirname(__file__)).parent.parent / "api" / "from_hf"
for x in os.listdir(hf_impl_path.absolute()):
    if not x.endswith(".py"):
        continue
    importlib.import_module(f"reallm.api.from_hf.{x.strip('.py')}")

for name, helpers in HF_MODEL_FAMILY_REGISTRY.items():
    r = HFModelRegistry(**helpers)

    def _load_from_hf(model: ReaLModel, load_dir: str, init_critic_from_actor: bool):
        r.load(model, load_dir, init_critic_from_actor)

    def _save_to_hf(model: ReaLModel,
                    tokenizer,
                    save_dir: str,
                    epoch=None,
                    epoch_step=None,
                    global_step=None):
        r.save(model, tokenizer, save_dir, epoch, epoch_step, global_step)

    @staticmethod
    def _config_from_hf(hf_config=None, model_path=None, is_critic=False):
        return r.config_from_hf(hf_config, model_path, is_critic)

    @staticmethod
    def _config_to_hf(config):
        return r.config_to_hf(config)

    setattr(ReaLModel, f"from_{name}", _load_from_hf)
    setattr(ReaLModel, f"to_{name}", _save_to_hf)
    setattr(ReaLModel, f"config_from_{name}", _config_from_hf)
    setattr(ReaLModel, f"config_to_{name}", _config_to_hf)
