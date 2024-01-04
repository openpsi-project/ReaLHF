from typing import *
import json

from api.config import *


def get_flash_mqat_model_config(
    from_type: str,
    model_path: str,
    hf_model_type: str,
    tokenizer_path: str,
    use_pipe: bool,
    dtype: Optional[str] = None,
    # model parallelism optimization
    sequence_parallel: bool = False,
    gradient_accumulation_fusion: bool = False,
    # pipeline partition method
    partition_method: Optional[str] = "parameters_balanced",
    # LoRA config
    lora_dim: Optional[int] = None,
    lora_scaling: Optional[float] = None,
    is_sft_lora: bool = False,
    sft_lora_path: Optional[str] = None,
    is_rew_lora: bool = False,
    rew_lora_path: Optional[str] = None,
    use_lora: bool = False,
):
    """Make a configuration to build model.

    Possible values of `from_type`:
        > hf_as_actor: build actor (decoder-only LLM) from huggingface models
        > hf_as_critic: build critic (transformer that outputs values instead of logits) from huggingface models
        > actor_as_critic: build critic from actor, replace the head with a new one, whether using pipeline depends on `use_pipe`
        > pp_actor_as_critic: build critic from pipelined actor, whose state dict should be remapped
        > pp_self: build non-pipeline actor/critic from pipelined actor/critic
        > random_actor: build a randomly initialized non-pipeline actor
        > random_critic build a randomly initialized non-pipeline critic
        > self: build a actor/critic from itself, whether using pipeline depends on `use_pipe`
            Note that it may not be built successfully if `use_pipe` is not consistent with the saved checkpoint
    """
    if gradient_accumulation_fusion:
        raise RuntimeError("gradient_accumulation_fusion is not supported yet")
    if (use_lora or is_sft_lora or is_rew_lora) and use_pipe:
        raise NotImplementedError("LORA is not supported in pipeline model")

    if use_pipe:
        pipe_init_from_scratch = from_type == "random_actor" or from_type == "random_critic"
        pipe_init_critic_from_actor = from_type == "actor_as_critic"
        with open(os.path.join(model_path, "flash_mqat_config.json"), "r") as f:
            is_critic = json.load(f)["is_critic"]
        from_type = "empty_critic" if is_critic else "empty_actor"

    model = Model(
        "flash_mqat",
        args=dict(
            model_path=model_path,
            from_type=from_type,
            dtype=dtype,
            hf_model_type=hf_model_type,
            tokenizer_path=tokenizer_path,
            sequence_parallel=sequence_parallel,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
        ),
    )
    if is_sft_lora:
        model.wrappers += [
            ModelWrapper(
                "lora",
                args=dict(
                    lora_module_kwargs=dict(
                        lora_dim=lora_dim,
                        lora_scaling=lora_scaling,
                    ),
                    lora_keys_to_replace=["c_attn.linear", "c_proj."],
                    load_lora_path=sft_lora_path,
                    lora_op_after_creation="squash",
                ),
            ),
        ]
    if is_rew_lora:
        model.wrappers += [
            ModelWrapper(
                "lora",
                args=dict(
                    lora_module_kwargs=dict(
                        lora_dim=lora_dim,
                        lora_scaling=lora_scaling,
                    ),
                    lora_keys_to_replace=["c_attn.linear", "c_proj."],
                    load_lora_path=rew_lora_path,
                    lora_op_after_creation="squash",
                ),
            ),
        ]
    if use_lora:
        model.wrappers += [
            ModelWrapper(
                "lora",
                args=dict(
                    lora_module_kwargs=dict(
                        lora_dim=lora_dim,
                        lora_scaling=lora_scaling,
                    ),
                    lora_keys_to_replace=["c_attn.linear", "c_proj."],
                    additional_module_names_to_opt=["v_head"],
                ),
            ),
        ]
    if use_pipe:
        model.wrappers += [
            ModelWrapper(
                "pipe",
                args=dict(
                    model_path=model_path,
                    partition_method=partition_method,
                    init_from_scatch=pipe_init_from_scratch,
                    init_critic_from_actor=pipe_init_critic_from_actor,
                ),
            )
        ]
    return model
