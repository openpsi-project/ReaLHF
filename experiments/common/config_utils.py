from typing import *

from api.config import *


def get_flash_mqat_model_config(
    model_path: str,
    from_model_type: str,
    tokenizer_path: str,
    pp_size: int,
    dp_size: int,
    is_critic: bool,
    use_lora: bool,
    lora_dim: Optional[int] = None,
    lora_scaling: Optional[float] = None,
    is_sft_lora: bool = False,
    sft_lora_path: Optional[str] = None,
    is_rew_lora: bool = False,
    rew_lora_path: Optional[str] = None,
    init_from_scratch: bool = False,
    v_head_path: Optional[str] = None,
    reward_scaling: float = 1.0,
    reward_bias: float = 0.0,
):
    model = Model(
        "flash_mqat_clm_hf" if not is_critic else "flash_mqat_critic",
        args=dict(
            model_path=model_path,
            from_type=from_model_type,
            tokenizer_path=tokenizer_path,
            init_from_scratch=(init_from_scratch or pp_size > 1),
        ),
    )
    if is_critic:
        model.args["v_head_path"] = v_head_path
        model.args["output_scaling"] = reward_scaling
        model.args["output_bias"] = reward_bias

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
    if pp_size > 1:
        model.wrappers += [
            ModelWrapper(
                "pipe",
                args=dict(
                    model_path=model_path,
                    num_pp=pp_size,
                    num_dp=dp_size,
                    is_critic=is_critic,
                ),
            )
        ]
    return model
