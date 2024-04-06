from typing import List
import copy
import dataclasses
import functools

from .device_mapping import auto_device_mapping as auto
from api.config.config_dataset import DatasetType, PromptOnlyDatasetConfig
from api.config.config_device_mesh import ClusterDeviceMesh, RPCAllocation
from api.config.config_system import _LLM_ENVVARS, ExperimentSaveEvalControl, register_experiment
from api.config.dfg import ModelInterface, ModelInterfaceType, ModelRPC, ModelType
from base.topology import PipeModelDataParallelTopology
from experiments.common.ppo_exp import PPOHyperparameters
import base.logging as logging

logger = logging.getLogger("Auto PPO exp", "colored")


def register_auto_ppo_experiment(
    size: int,
    gen_bs: int,
    train_bs: int,
    seqlen: int,
):
    assert size in [7, 13, 34, 70]
    if size == 7:
        n_nodes = 1
        nodelist = "QH-com25"
    elif size == 13:
        n_nodes = 2
        nodelist = "QH-com[25-26]"
    elif size == 34:
        n_nodes = 4
        nodelist = "QH-com[25-28]"
    elif size == 70:
        n_nodes = 8
        nodelist = "QH-com[23-30]"

    model_class = "llama" if size != 34 else "codellama"

    @auto(n_nodes=n_nodes, nodelist=nodelist)
    @dataclasses.dataclass
    class AutoPPOExperiment:
        seed: int = 1
        exp_ctrl: ExperimentSaveEvalControl = dataclasses.field(default_factory=functools.partial(
            ExperimentSaveEvalControl,
            benchmark_steps=10,
        ),)
        ppo: PPOHyperparameters = dataclasses.field(default_factory=functools.partial(
            PPOHyperparameters,
            max_new_tokens=seqlen,
            min_new_tokens=seqlen,
        ))

        @property
        def dataset(self) -> DatasetType:
            return PromptOnlyDatasetConfig(
                max_prompt_len=128,
                n_tokens_per_batch=1048576,
                path="/lustre/fw/datasets/antropic-hh/ppo_prompt_only.jsonl",
                # path="/lustre/fw/datasets/imdb/rl/ppo_prompt.jsonl",
            )

        @property
        def rpcs(self) -> List[ModelRPC]:
            ppo_kwargs = dict(
                n_minibatches=self.ppo.ppo_n_minibatches,
                kl_ctl=self.ppo.kl_ctl,
                discount=self.ppo.discount,
                gae_lambda=self.ppo.gae_lambda,
                eps_clip=self.ppo.eps_clip,
                value_eps_clip=self.ppo.value_eps_clip,
                max_reward_clip=self.ppo.max_reward_clip,
                adaptive_kl_ctl=self.ppo.use_adaptive_kl_ctl,
                value_norm=self.ppo.value_norm,
                value_norm_type=self.ppo.value_norm_type,
                value_norm_beta=self.ppo.value_norm_beta,
                value_norm_eps=self.ppo.value_norm_eps,
            )
            generation_kwargs = dict(
                max_new_tokens=self.ppo.max_new_tokens,
                min_new_tokens=self.ppo.min_new_tokens,
                greedy=self.ppo.greedy,
                top_p=self.ppo.top_p,
                top_k=self.ppo.top_k,
                temperature=self.ppo.temperature,
            )
            actor_interface = ModelInterface(
                "flash_actor",
                args={
                    **copy.deepcopy(ppo_kwargs),
                    "generation_config": generation_kwargs,
                    "early_stop_imp_ratio": self.ppo.early_stop_imp_ratio,
                    "force_no_logits_mask": True,
                    "adv_norm": self.ppo.adv_norm,
                },
            )
            ref_interface = copy.deepcopy(actor_interface)
            ref_interface.args["enable_save"] = False

            critic_interface = ModelInterface(
                "flash_critic",
                args=copy.deepcopy(ppo_kwargs),
            )
            rw_interface = ModelInterface(
                "flash_paired_rw",
                args=dict(
                    enable_save=False,
                    output_scaling=self.ppo.reward_output_scaling,
                    output_bias=self.ppo.reward_output_bias,
                ),
            )
            return [
                ModelRPC(
                    model_name="actor",
                    model_type=ModelType(model_class, size, is_critic=False),
                    interface_type=ModelInterfaceType.GENERATE,
                    interface_impl=actor_interface,
                    input_data=["packed_prompts", "prompt_cu_seqlens"],
                    output_data=[
                        "seq_no_eos_mask",
                        "packed_seq",
                        "cu_seqlens",
                        "packed_logprobs",
                        "prompt_mask",
                    ],
                    balanced_dp=True,
                    min_n_seqs=gen_bs,
                    max_n_seqs=gen_bs,
                    max_n_tokens=gen_bs * 128,
                ),
                ModelRPC(
                    model_name="reward",
                    model_type=ModelType("llama", 7, is_critic=True),
                    interface_type=ModelInterfaceType.INFERENCE,
                    interface_impl=rw_interface,
                    input_data=["packed_seq", "cu_seqlens"],
                    input_key_remap={"packed_seq": "packed_input_ids"},
                    output_data=["scores"],
                    output_key_remap={"scores": "rewards"},
                    min_n_seqs=gen_bs,
                    max_n_seqs=gen_bs,
                    max_n_tokens=gen_bs * (128 + seqlen),
                ),
                ModelRPC(
                    model_name="ref",
                    model_type=ModelType(model_class, size, is_critic=False),
                    interface_type=ModelInterfaceType.INFERENCE,
                    interface_impl=ref_interface,
                    input_data=[
                        "packed_seq",
                        "cu_seqlens",
                    ],
                    output_data=["logprobs"],
                    output_key_remap={"logprobs": "packed_ref_logprobs"},
                    min_n_seqs=gen_bs,
                    max_n_seqs=gen_bs,
                    max_n_tokens=gen_bs * (128 + seqlen),
                ),
                ModelRPC(
                    model_name="critic",
                    model_type=ModelType("llama", 7, is_critic=True),
                    interface_type=ModelInterfaceType.INFERENCE,
                    interface_impl=critic_interface,
                    input_data=["packed_seq", "cu_seqlens", "seq_no_eos_mask"],
                    output_data=["scores"],
                    output_key_remap={"scores": "values"},
                    min_n_seqs=gen_bs,
                    max_n_seqs=gen_bs,
                    max_n_tokens=gen_bs * (128 + seqlen),
                ),
                ModelRPC(
                    model_name="actor",
                    model_type=ModelType(model_class, size, is_critic=False),
                    interface_type=ModelInterfaceType.TRAIN_STEP,
                    interface_impl=actor_interface,
                    input_data=[
                        "packed_seq",
                        "cu_seqlens",
                        "packed_logprobs",
                        "packed_ref_logprobs",
                        "rewards",
                        "values",
                        "prompt_mask",
                        "seq_no_eos_mask",
                    ],
                    log_return_value=True,
                    min_n_seqs_per_dp=self.ppo.ppo_n_minibatches,
                    min_n_seqs=train_bs,
                    max_n_seqs=train_bs,
                    balanced_dp=True,
                    max_n_tokens=train_bs * (128 + seqlen),
                ),
                ModelRPC(
                    model_name="critic",
                    interface_type=ModelInterfaceType.TRAIN_STEP,
                    model_type=ModelType("llama", 7, is_critic=True),
                    interface_impl=critic_interface,
                    input_data=[
                        "packed_seq",
                        "cu_seqlens",
                        "packed_logprobs",
                        "packed_ref_logprobs",
                        "rewards",
                        "values",
                        "prompt_mask",
                        "seq_no_eos_mask",
                    ],
                    log_return_value=True,
                    min_n_seqs_per_dp=self.ppo.ppo_n_minibatches,
                    min_n_seqs=train_bs,
                    max_n_seqs=train_bs,
                    balanced_dp=True,
                    max_n_tokens=train_bs * (128 + seqlen),
                ),
            ]

    register_experiment(f"sosp-a{size}s{seqlen}g{gen_bs}t{train_bs}", AutoPPOExperiment)


for size in [7, 13, 34, 70]:
    for gen_bs in [16, 32, 48, 64, 80, 100, 128, 160, 200, 240, 256, 288, 320, 360, 400, 512, 640, 1024]:
        # for seqlen in [256, 512, 1024]:
        for seqlen in [128, 384, 896]:
            train_bs = gen_bs
            register_auto_ppo_experiment(size, gen_bs, train_bs, seqlen)
