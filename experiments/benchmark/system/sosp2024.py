from copy import deepcopy
import functools
import os

from .pposys_exp import PPOSysExperiment
from api.config import *
from experiments.common import ModelBackend, ModelConfig, OptimizerConfig, ParallelismConfig
from experiments.common.ppo_exp import PPOConfig, PPOHyperparmeters


def build_llama2_model(
    model_size: str,
    dp_size: int,
    zero_stage: int,
    mp_size: int = 1,
    pp_size: int = 1,
    gradient_checkpointing: bool = True,
    offload_model: bool = False,
    offload_opt: bool = False,
    use_hybrid_engine: bool = False,
):
    assert model_size in ["7b", "13b", "34b", "70b"]
    if model_size in ["7b", "13b", "70b"]:
        model_dir_prefix = f"Llama-2-{model_size}-hf"
        model_type_ = "llama"
    else:
        model_dir_prefix = "CodeLlama-34b-hf"
        model_type_ = "codellama"
    if pp_size > 1 or mp_size > 1:
        ckpt_base_dir = "/lustre/public/pretrained_model_weights/sharded_new/"
        model_path = os.path.join(ckpt_base_dir, f"{model_dir_prefix}_{pp_size}pp_{mp_size}mp")
    else:
        ckpt_base_dir = "/lustre/public/pretrained_model_weights/"
        model_path = os.path.join(ckpt_base_dir, model_dir_prefix)
    parallel = ParallelismConfig(
        model_parallel_size=mp_size,
        pipeline_parallel_size=pp_size,
        data_parallel_size=dp_size,
        use_sequence_parallel=False,  # FIXME: use sequence parallelism during training, but not generate
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


def register_sosp_experiments_with_fixed_bs(model_size: str, dp_size: int):
    """The following experiments have a fixed batch size 512, max prompt len 1024, and max new tokens 1024.

    Used for:
    1. Identifying the fastest data+pipe parallelism configuration;
    2. Getting an initial benchmark result for comparing with other systems.
    """
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


def _get_heuristic_device_partition(model_size: int):
    # actor, critic, ref, rew
    if model_size == 7:
        # TODO: the following partition is obviously not optimal, we should colocate actor & critic in this case
        device_partition = (3, 2, 1, 1)
        ngpus = 8
        nodelist = "QH-com09"
    elif model_size == 13:
        device_partition = (8, 4, 2, 1)
        ngpus = 16
        nodelist = "QH-com[13-14]"
    elif model_size == 34:
        device_partition = (24, 2, 4, 1)
        ngpus = 32
        nodelist = "QH-com[19-22]"
    elif model_size == 70:
        device_partition = (48, 6, 8, 1)
        ngpus = 64
        nodelist = "QH-com[36-43]"
    assert sum(device_partition) == ngpus - 1, (device_partition, ngpus, model_size)
    return ngpus, device_partition, nodelist


interested_parallel_strategies = [
    # mp, dp, pp
    dict(model_size=7, actor=(1, 3, 1), ref=(1, 1, 1)),
    dict(model_size=13, actor=(2, 4, 1), ref=(1, 2, 1)),
    dict(model_size=34, actor=(2, 4, 3), ref=(2, 2, 1)),
    dict(model_size=70, actor=(4, 4, 3), ref=(4, 2, 1)),
]


def register_sosp_experiments_with_full_gpu_mem(
    model_size: str,
    actor_parallel_strategy: Tuple[int, int, int],
    ref_parallel_strategy: Tuple[int, int, int],
    gen_bs: int,
    max_answer_len: int,
):
    model_size_int = int(model_size.split("b")[0])
    ngpus, device_partition, nodelist = _get_heuristic_device_partition(model_size_int)

    n_actor_gpus = device_partition[0]
    n_critic_gpus = device_partition[1]
    n_ref_gpus = device_partition[2]
    n_rew_gpus = device_partition[3]

    # build critic model
    # since critic training/inference is usually fast, we use maximum pp_size for maximum batch size
    if model_size_int == 7:
        # in this case critic and actor have the same size, use dp
        critic_dp_size = n_critic_gpus
        critic_pp_size = 1
    else:
        critic_dp_size = 1
        critic_pp_size = n_critic_gpus
    critic = build_llama2_model(
        model_size="7b",
        dp_size=critic_dp_size,
        pp_size=critic_pp_size,
        zero_stage=1 if critic_pp_size > 1 else 2,
        offload_opt=True,
    )

    # build reward model
    rew = build_llama2_model(
        model_size="7b",
        dp_size=n_rew_gpus,
        zero_stage=0,
    )

    mp_size, dp_size, pp_size = actor_parallel_strategy
    assert mp_size * dp_size * pp_size == n_actor_gpus, (mp_size, dp_size, pp_size, n_actor_gpus, model_size)
    # by default, n_mbs = 2 * pp_size
    actor = build_llama2_model(
        model_size=model_size,
        dp_size=dp_size,
        mp_size=mp_size,
        pp_size=pp_size,
        zero_stage=1 if pp_size > 1 else 2,
        offload_opt=True,
    )

    ref_mp_size, ref_dp_size, ref_pp_size = ref_parallel_strategy
    assert ref_mp_size * ref_dp_size * ref_pp_size == n_ref_gpus
    ref = build_llama2_model(
        model_size=model_size,
        dp_size=ref_dp_size,
        mp_size=ref_mp_size,
        pp_size=ref_pp_size,
        zero_stage=0,
    )

    register_experiment(
        f"sba-a{model_size_int}-{mp_size}x{dp_size}x{pp_size}-ref{ref_mp_size}x{ref_dp_size}x{ref_pp_size}-c7r7-{'-'.join(map(str, device_partition))}-s{max_answer_len}g{gen_bs}",
        functools.partial(
            PPOSysExperiment,
            master_nodelist=nodelist,
            actor_nodelist=nodelist,
            critic_nodelist=nodelist,
            ref_nodelist=nodelist,
            rew_nodelist=nodelist,
            actor=actor,
            critic=critic,
            ref=ref,
            rew=rew,
            batch_size=int(n_actor_gpus * gen_bs * 4),
            max_answer_len=max_answer_len,
        ),
    )


for dp_size in map(lambda x: 2**x, range(7)):
    for model_size in ["7b", "13b", "34b", "70b"]:
        register_sosp_experiments_with_fixed_bs(model_size, dp_size)

for max_answer_len in [256, 512, 1024]:
    for bs in range(1, 129):
        for setting in interested_parallel_strategies:
            model_size = f"{setting['model_size']}b"
            register_sosp_experiments_with_full_gpu_mem(
                model_size=model_size,
                actor_parallel_strategy=setting["actor"],
                ref_parallel_strategy=setting["ref"],
                gen_bs=bs,
                max_answer_len=max_answer_len,
            )
