from base.monitor import process_memory_mb
from impl.model.nn.flash_mqat.flash_mqat_base import FlashMQATModel
from impl.model.nn.flash_mqat.flash_mqat_parallel import ModelParallelModule
import api.huggingface
import api.model
import base.logging as logging

logger = logging.getLogger("model_parallel_nn")


def model_parallel_wrap_fn(
    model_path: str,
    is_critic: bool,
    init_critic_from_actor: bool = False,
    init_from_scratch: bool = False,
):

    def model_parallel_wrap_fn_(model: api.model.Model) -> api.model.Model:
        if not isinstance(model.module, FlashMQATModel):
            raise RuntimeError(f"Only FlashMQAT models can be wrapped as "
                               f"pipeline module, provided type {type(model.module)}")
        config = model.module.config
        module = ModelParallelModule(model.module, config, device=model.device)
        if not init_from_scratch:
            process_memory_mb("before_load")
            module.load(model_path, init_critic_from_actor=init_critic_from_actor)
            process_memory_mb("after_load")
        model.module = module
        return model

    return model_parallel_wrap_fn_


api.model.register_wrapper("model_parallel", model_parallel_wrap_fn)
