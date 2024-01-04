from typing import *

from api.config import *


def get_flash_mqat_model_config(
    model_path: str,
    from_model_type: str,
    tokenizer_path: str,
    pp_size: int,
    mp_size: int,
    dp_size: int,
    is_critic: bool,
    use_lora: bool,
    sequence_parallel: bool = False,
    gradient_accumulation_fusion: bool = False,
    dtype: Optional[str] = None,
    lora_dim: Optional[int] = None,
    lora_scaling: Optional[float] = None,
    is_sft_lora: bool = False,
    sft_lora_path: Optional[str] = None,
    is_rew_lora: bool = False,
    rew_lora_path: Optional[str] = None,
    init_from_scratch: bool = False,
    init_critic_from_actor: bool = False,
    v_head_path: Optional[str] = None,
    partition_method: Optional[str] = "parameters",
):
    if gradient_accumulation_fusion:
        raise RuntimeError("gradient_accumulation_fusion is not supported yet")
    if (use_lora or is_sft_lora or is_rew_lora) and pp_size > 1:
        raise NotImplementedError("LORA is not supported in pipeline model")
    model = Model(
        "flash_mqat_actor" if not is_critic else "flash_mqat_critic",
        args=dict(
            model_path=model_path,
            from_type=from_model_type,
            tokenizer_path=tokenizer_path,
            init_from_scratch=(init_from_scratch or pp_size > 1 or mp_size > 1),
            no_param_instantiation=(pp_size > 1 or mp_size > 1),
            dtype=dtype,
        ),
    )
    if is_critic:
        model.args["v_head_path"] = v_head_path
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
    if pp_size > 1 and mp_size == 1:
        model.wrappers += [
            ModelWrapper(
                "pipe",
                args=dict(
                    model_path=model_path,
                    num_pp=pp_size,
                    num_dp=dp_size,
                    is_critic=is_critic,
                    partition_method=partition_method,
                    init_critic_from_actor=init_critic_from_actor,
                ),
            )
        ]
    elif mp_size > 1 and pp_size == 1:
        model.wrappers += [
            ModelWrapper(
                "model_parallel",
                args=dict(
                    model_path=model_path,
                    is_critic=is_critic,
                    sequence_parallel=sequence_parallel,
                    gradient_accumulation_fusion=gradient_accumulation_fusion,
                    init_critic_from_actor=init_critic_from_actor,
                    init_from_scratch=init_from_scratch,
                ),
            )
        ]
    elif pp_size > 1 and mp_size > 1:
        model.wrappers += [
            ModelWrapper(
                "model_pipe_parallel",
                args=dict(
                    model_path=model_path,
                    num_pp=pp_size,
                    num_mp=mp_size,
                    num_dp=dp_size,
                    sequence_parallel=sequence_parallel,
                    gradient_accumulation_fusion=gradient_accumulation_fusion,
                    is_critic=is_critic,
                    partition_method=partition_method,
                    init_critic_from_actor=init_critic_from_actor,
                    init_from_scratch=init_from_scratch,
                ),
            )
        ]
    return model
