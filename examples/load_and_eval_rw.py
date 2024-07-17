import torch
import torch.distributed as dist
import transformers

from realhf.api.core.config import ModelName
from realhf.api.core.model_api import ReaLModelConfig
from realhf.base import constants
from realhf.base.testing import init_global_constants


def load_and_use_single_process(path: str, model_family_name: str):
    # Initialize distributed environment.
    dist.init_process_group(
        "nccl", rank=0, world_size=1, init_method="tcp://localhost:7777"
    )
    model_name = ModelName("default", 0)
    init_global_constants(
        num_dp=1,
        num_mp=1,
        num_pp=1,
        sequence_parallel=False,
        model_name=model_name,
    )

    # NOTE: import here to avoid CUDA re-initialization
    from realhf.impl.model.nn.real_llm_api import ReaLModel, add_helper_functions

    # Call a method like `config_from_llama` to get the config.
    mconfig: ReaLModelConfig = getattr(ReaLModel, f"config_from_{model_family_name}")(
        transformers.AutoConfig.from_pretrained(path)
    )
    # IMPORTANT: Set the critic flag to True.
    # Since the output head and the token embedding no long have the same shape,
    # We set tied_embedding to be False.
    mconfig.is_critic = True
    mconfig.tied_embedding = False

    with constants.model_scope(model_name):
        # Construct the model.
        model = ReaLModel(mconfig, dtype=torch.float16, device="cuda")
        model.instantiate()

        # Load the reward checkpoint
        # Since the checkpoint is already critic model, we set
        # init_critic_from_actor to be False.
        model = getattr(model, f"from_{model_family_name}")(
            path, init_critic_from_actor=False
        )
        # Add helper functions to make the model like HuggingFace models.
        add_helper_functions(model)

        # Use the model.
        bs = 10
        seqlen = 256
        input_ids = torch.randint(
            0, mconfig.vocab_size, (bs, seqlen), dtype=torch.long, device="cuda"
        )
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        # The final dimension of the output scores is 1.
        scores = model(input_ids, attention_mask).logits
        assert scores.shape == (bs, seqlen, 1), scores.shape


if __name__ == "__main__":
    path = "/lustre/aigc/llm/checkpoints/fw/quickstart-rw/llama-ray-manual/default/epoch1epochstep10globalstep10/"
    model_family_name = "llama"
    load_and_use_single_process(path, model_family_name)
