import os
from api.config import *
import functools
from experiments.common.ppo_exp import PPOConfig, PPOHyperparmeters
from experiments.common import ModelConfig, ModelBackend, ParallelismConfig, OptimizerConfig
from .pposys_exp import PPOSysExperiment


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
):
    assert model_size in ["7b", "13b", "34b", "70b"]
    if model_size in ["7b", "34b"]:
        model_dir_prefix = f"Llama-2-{model_size}-hf"
        model_type_ = "codellama"
    elif model_size == "13b":
        model_dir_prefix = "codellama-13B"
        model_type_ = "codellama"
    elif model_size == "70b":
        model_dir_prefix = "Llama-2-70b-hf"
        model_type_ = "llama"
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

############################### 7b experiment begin ###############################
def register_7b_dp8pp2_experiment():
    actor = build_llama2_model(
        model_size="7b",
        dp_size=8,
        pp_size=2,
        pp_mbs=4,
        zero_stage=1,
    )
    ref = build_llama2_model(
        model_size="7b",
        dp_size=4,
        zero_stage=0,
    )
    register_experiment(
        f"sosp-baseline-a7c7r7",
        functools.partial(
            PPOSysExperiment,
            master_nodelist="QH-com16",
            actor_nodelist="QH-com[17-18]",
            critic_nodelist="QH-com19",
            ref_nodelist="QH-com20",
            rew_nodelist="QH-com20",
            actor=actor,
            critic=critic,
            ref=ref,
            rew=rew,
        ),
    )

def register_7b_hybrid_dp8_experiment():
    actor = build_llama2_model(
        model_size="7b",
        dp_size=8,
        zero_stage=3,
        use_hybrid_engine=True,
    )
    ref = build_llama2_model(
        model_size="7b",
        dp_size=4,
        zero_stage=0,
    )
    register_experiment(
        f"sosp-baseline-a7c7r7-he",
        functools.partial(
            PPOSysExperiment,
            master_nodelist="QH-com16",
            actor_nodelist="QH-com[17-18]",
            critic_nodelist="QH-com19",
            ref_nodelist="QH-com20",
            rew_nodelist="QH-com20",
            actor=actor,
            critic=critic,
            ref=ref,
            rew=rew,
        ),
    )
############################### 7b experiment end ###############################

register_7b_dp8pp2_experiment()
register_7b_hybrid_dp8_experiment()
