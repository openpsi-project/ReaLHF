from pathlib import Path
import functools
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
    if "starcoder" in x.strip('.py'):
        # HACK: StarCoder seems to have a bug with transformers v0.39.1:
        # load_state_dict does not work on this model,
        # and the weights are not changed after loading. Skip this model temporarily.
        continue
    importlib.import_module(f"reallm.api.from_hf.{x.strip('.py')}")

_HF_REGISTRIES = {}


def _load_from_hf(model: ReaLModel, registry_name, load_dir: str, init_critic_from_actor: bool):
    r = _HF_REGISTRIES[registry_name]
    setattr(model, "save_to_hf", functools.partial(_save_to_hf, model, registry_name))
    return r.load(model, load_dir, init_critic_from_actor)


def _save_to_hf(model: ReaLModel,
                registry_name,
                tokenizer,
                save_dir: str,
                epoch=None,
                epoch_step=None,
                global_step=None):
    r = _HF_REGISTRIES[registry_name]
    r.save(model, tokenizer, save_dir, epoch, epoch_step, global_step)


def _config_from_hf(registry_name, hf_config=None, model_path=None, is_critic=False):
    r = _HF_REGISTRIES[registry_name]
    return r.config_from_hf(hf_config, model_path, is_critic)


def _config_to_hf(registry_name, config):
    r = _HF_REGISTRIES[registry_name]
    return r.config_to_hf(config)


for name, helpers in HF_MODEL_FAMILY_REGISTRY.items():
    _HF_REGISTRIES[name] = r = HFModelRegistry(**helpers)

    _load_from_hf_ = functools.partialmethod(_load_from_hf, name)
    setattr(ReaLModel, f"from_{name}", _load_from_hf_)

    _save_to_hf_ = functools.partialmethod(_save_to_hf, name)
    setattr(ReaLModel, f"to_{name}", _save_to_hf_)

    _config_from_hf_ = functools.partial(_config_from_hf, name)
    setattr(ReaLModel, f"config_from_{name}", staticmethod(_config_from_hf_))

    _config_to_hf_ = functools.partial(_config_to_hf, name)
    setattr(ReaLModel, f"config_to_{name}", staticmethod(_config_to_hf_))
