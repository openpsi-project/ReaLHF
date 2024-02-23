import os
from api.config import *
import functools
from experiments.common.ppo_exp import PPOConfig, PPOHyperparmeters
from experiments.common import ModelConfig, ModelBackend, ParallelismConfig, OptimizerConfig
from .pposys_exp import PPOSysExperiment
from copy import deepcopy


def build_llama2_model(
    model_size: str,
    dp_size: int,
    zero_stage: int,
    pp_size: int = 1,
    pp_mbs: int = None,
    gradient_checkpointing: bool = True,
    offload_model: bool = False,
    offload_opt: bool = False,
    use_hybrid_engine: bool = False,
    num_inf_pipeline_mbs: int = None,
):
    assert model_size in ["7b", "13b", "34b", "70b"]
    if model_size in ["7b", "13b", "70b"]:
        model_dir_prefix = f"Llama-2-{model_size}-hf"
        model_type_ = "llama"
    else:
        model_dir_prefix = "CodeLlama-34b-hf"
        model_type_ = "codellama"
    if pp_size > 1:
        ckpt_base_dir = "/lustre/public/pretrained_model_weights/sharded_new/"
        model_path = os.path.join(ckpt_base_dir, f"{model_dir_prefix}_{pp_size}pp_1mp")
    else:
        ckpt_base_dir = "/lustre/public/pretrained_model_weights/"
        model_path = os.path.join(ckpt_base_dir, model_dir_prefix)
    parallel = ParallelismConfig(
        model_parallel_size=1,
        pipeline_parallel_size=pp_size,
        data_parallel_size=dp_size,
        use_sequence_parallel=False,
        num_pipeline_micro_batches=pp_mbs,
        num_inf_pipeline_mbs=num_inf_pipeline_mbs,
    )
    optimizer = OptimizerConfig(
        zero_stage=zero_stage if pp_size == 1 else 1,
        offload=offload_opt,
        use_hybrid_engine=use_hybrid_engine,
    )
    return ModelConfig(
        type=model_type_,
        path=model_path,
        base_model_path=model_path,
        tokenizer_path=model_path,
        gradient_checkpointing=gradient_checkpointing,
        enable_bf16=False,
        enable_fp16=True,
        offload=offload_model,
        parallel=parallel,
        optimizer=optimizer,
    )


critic = build_llama2_model(
    model_size="7b",
    dp_size=8,
    zero_stage=2,
)
rew = build_llama2_model(
    model_size="7b",
    dp_size=4,
    zero_stage=0,
)


def register_sosp_experiments_with_fixed_bs(model_size: str, dp_size: int):
    """The following experiments have a fixed batch size 512, max prompt len 1024, and max new tokens 1024.
    
    Used for:
    1. Identifying the fastest data+pipe parallelism configuration;
    2. Getting an initial benchmark result for comparing with other systems.
    """
    if model_size in ["7b", "13b"]:
        n_actor_gpus = 16
        n_ref_gpus = 4
        ref_dp_size = 2
        master_nodelist = "QH-com16"
        actor_nodelist = "QH-com[17-18]"
        critic_nodelist = "QH-com19"
        ref_nodelist = "QH-com15"
        rew_nodelist = "QH-com15"
    elif model_size == "34b":
        n_actor_gpus = 32
        n_ref_gpus = 8
        ref_dp_size = 2
        master_nodelist = "QH-com25"
        actor_nodelist = "QH-com[30-33]"
        critic_nodelist = "QH-com34"
        ref_nodelist = "QH-com36"
        rew_nodelist = "QH-com37"
    elif model_size == "70b":
        n_actor_gpus = 64
        n_ref_gpus = 16
        ref_dp_size = 2
        master_nodelist = "QH-com49"
        actor_nodelist = "QH-com[25-28,30-33]"
        critic_nodelist = "QH-com39"
        ref_nodelist = "QH-com[36-37]"
        rew_nodelist = "QH-com01"
    else:
        raise NotImplementedError()
    if dp_size > n_actor_gpus:
        return
    pp_size = n_actor_gpus // dp_size
    ref_pp_size = n_ref_gpus // ref_dp_size
    # by default, n_mbs = 2 * pp_size
    actor = build_llama2_model(
        model_size=model_size,
        dp_size=dp_size,
        pp_size=pp_size,
        pp_mbs=pp_size * 2,
        zero_stage=1 if pp_size > 1 else 2,
    )
    ref = build_llama2_model(
        model_size=model_size,
        dp_size=ref_dp_size,
        pp_size=ref_pp_size,
        zero_stage=0,
    )
    model_size_int = int(model_size.split("b")[0])
    register_experiment(
        f"sosp-baseline-a{model_size_int}-{dp_size}x{pp_size}-c7r7",
        functools.partial(
            PPOSysExperiment,
            master_nodelist=master_nodelist,
            actor_nodelist=actor_nodelist,
            critic_nodelist=critic_nodelist,
            ref_nodelist=ref_nodelist,
            rew_nodelist=rew_nodelist,
            actor=actor,
            critic=critic,
            ref=ref,
            rew=rew,
        ),
    )

    if pp_size == 1:
        return
    # register experiments with n_mbs = pp_size
    actor = deepcopy(actor)
    actor.parallel.num_pipeline_micro_batches = pp_size
    register_experiment(
        f"sosp-baseline-a{model_size_int}-{dp_size}x{pp_size}-c7r7-mb1",
        functools.partial(
            PPOSysExperiment,
            master_nodelist=master_nodelist,
            actor_nodelist=actor_nodelist,
            critic_nodelist=critic_nodelist,
            ref_nodelist=ref_nodelist,
            rew_nodelist=rew_nodelist,
            actor=actor,
            critic=critic,
            ref=ref,
            rew=rew,
        ),
    )

    # register experiments with gen n_mbs = pp_size * 2 while train n_mbs = pp_size
    actor = deepcopy(actor)
    actor.parallel.num_pipeline_micro_batches = 2 * pp_size
    actor.parallel.num_inf_pipeline_mbs = pp_size
    register_experiment(
        f"sosp-baseline-a{model_size_int}-{dp_size}x{pp_size}-c7r7-mb1gen",
        functools.partial(
            PPOSysExperiment,
            master_nodelist=master_nodelist,
            actor_nodelist=actor_nodelist,
            critic_nodelist=critic_nodelist,
            ref_nodelist=ref_nodelist,
            rew_nodelist=rew_nodelist,
            actor=actor,
            critic=critic,
            ref=ref,
            rew=rew,
        ),
    )

def register_sosp_experiments_with_full_gpu_mem(model_size: str, dp_size: int):
    if model_size == "7b":
        batch_size = 320
        n_actor_gpus = 8
        n_ref_gpus = 2
        ref_dp_size = 2
        master_nodelist = "QH-com16"
        actor_nodelist = "QH-com[17-18]"
        critic_nodelist = "QH-com19"
        ref_nodelist = "QH-com15"
        rew_nodelist = "QH-com15"
    elif model_size in ["13b"]:
        batch_size = 512
        n_actor_gpus = 16
        n_ref_gpus = 4
        ref_dp_size = 2
        master_nodelist = "QH-com16"
        actor_nodelist = "QH-com[17-18]"
        critic_nodelist = "QH-com19"
        ref_nodelist = "QH-com15"
        rew_nodelist = "QH-com15"
    elif model_size == "34b":
        batch_size = 512
        n_actor_gpus = 32
        n_ref_gpus = 8
        ref_dp_size = 2
        master_nodelist = "QH-com25"
        actor_nodelist = "QH-com[30-33]"
        critic_nodelist = "QH-com34"
        ref_nodelist = "QH-com36"
        rew_nodelist = "QH-com37"
    elif model_size == "70b":
        batch_size = 512
        n_actor_gpus = 64
        n_ref_gpus = 16
        ref_dp_size = 2
        master_nodelist = "QH-com49"
        actor_nodelist = "QH-com[25-28,30-33]"
        critic_nodelist = "QH-com39"
        ref_nodelist = "QH-com[36-37]"
        rew_nodelist = "QH-com01"
    else:
        raise NotImplementedError()
    if dp_size > n_actor_gpus:
        return
    pp_size = n_actor_gpus // dp_size
    ref_pp_size = n_ref_gpus // ref_dp_size
    # by default, n_mbs = 2 * pp_size
    actor = build_llama2_model(
        model_size=model_size,
        dp_size=dp_size,
        pp_size=pp_size,
        zero_stage=1 if pp_size > 1 else 2,
    )
    actor.parallel.num_pipeline_micro_batches = 2 * pp_size
    actor.parallel.num_inf_pipeline_mbs = pp_size
    ref = build_llama2_model(
        model_size=model_size,
        dp_size=ref_dp_size,
        pp_size=ref_pp_size,
        zero_stage=0,
    )
    model_size_int = int(model_size.split("b")[0])
    register_experiment(
        f"sosp-baseline-a{model_size_int}-{dp_size}x{pp_size}-c7r7-fullmem",
        functools.partial(
            PPOSysExperiment,
            master_nodelist=master_nodelist,
            actor_nodelist=actor_nodelist,
            critic_nodelist=critic_nodelist,
            ref_nodelist=ref_nodelist,
            rew_nodelist=rew_nodelist,
            actor=actor,
            critic=critic,
            ref=ref,
            rew=rew,
            batch_size=batch_size,
        ),
    )


for dp_size in map(lambda x: 2**x, range(7)):
    for model_size in ["7b", "13b", "34b", "70b"]:
        register_sosp_experiments_with_fixed_bs(model_size, dp_size)
        register_sosp_experiments_with_full_gpu_mem(model_size, dp_size)
