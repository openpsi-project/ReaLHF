import argparse
from typing import *

from realhf.api.core.config import ModelInterfaceAbstraction, ModelName
from realhf.api.core.dfg import MFCDef, ModelInterfaceType, build_graph

_dump_impl = ModelInterfaceAbstraction("")


def _get_ppo_rpcs() -> List[MFCDef]:
    """The most popular PPO algorithm proposed by InstructGPT."""
    actor_gen = MFCDef(
        n_seqs=1,
        name="actorGen",
        interface_type=ModelInterfaceType.GENERATE,
        interface_impl=_dump_impl,
        model_name="actor",
        input_keys=["prompt"],
        output_keys=["input_ids", "oldlogp"],
    )
    rew_inf = MFCDef(
        n_seqs=1,
        name="rewInf",
        interface_type=ModelInterfaceType.INFERENCE,
        interface_impl=_dump_impl,
        model_name="rew",
        input_keys=["input_ids"],
        output_keys=["rew"],
    )
    ref_inf = MFCDef(
        n_seqs=1,
        name="refInf",
        interface_type=ModelInterfaceType.INFERENCE,
        interface_impl=_dump_impl,
        model_name="ref",
        input_keys=["input_ids"],
        output_keys=["reflogp"],
    )
    critic_inf = MFCDef(
        n_seqs=1,
        name="criticInf",
        interface_type=ModelInterfaceType.INFERENCE,
        interface_impl=_dump_impl,
        model_name=ModelName("critic", 0),
        input_keys=["input_ids"],
        output_keys=["oldvalue"],
    )
    actor_train = MFCDef(
        n_seqs=1,
        name="actorTrain",
        interface_type=ModelInterfaceType.TRAIN_STEP,
        interface_impl=_dump_impl,
        model_name=ModelName("actor", 1),
        input_keys=["input_ids", "oldlogp", "reflogp", "rew", "oldvalue"],
    )
    critic_train = MFCDef(
        n_seqs=1,
        name="criticTrain",
        interface_type=ModelInterfaceType.TRAIN_STEP,
        interface_impl=_dump_impl,
        model_name=ModelName("critic", 1),
        input_keys=["input_ids", "oldlogp", "reflogp", "rew", "oldvalue"],
    )
    return [actor_gen, rew_inf, ref_inf, critic_inf, actor_train, critic_train]


def _get_reinforce_rpcs():
    """A REINFORCE algorithm that uses the rewards of greedy generations as the
    baseline."""
    actor_gen = MFCDef(
        n_seqs=1,
        name="greedyGen",
        interface_type=ModelInterfaceType.GENERATE,
        interface_impl=_dump_impl,
        model_name="actor",
        input_keys=["prompt"],
        output_keys=["greedy_input_ids"],
    )
    actor_sample = MFCDef(
        n_seqs=1,
        name="sampleGen",
        interface_type=ModelInterfaceType.GENERATE,
        interface_impl=_dump_impl,
        model_name="actor",
        input_keys=["prompt"],
        output_keys=["sampled_input_ids"],
    )
    greedy_inf = MFCDef(
        n_seqs=1,
        name="greedy_inf",
        interface_type=ModelInterfaceType.INFERENCE,
        interface_impl=_dump_impl,
        model_name="rew",
        input_keys=["greedy_input_ids"],
        output_keys=["greedy_rew"],
    )
    sample_inf = MFCDef(
        n_seqs=1,
        name="sample_inf",
        interface_type=ModelInterfaceType.INFERENCE,
        interface_impl=_dump_impl,
        model_name="rew",
        input_keys=["sampled_input_ids"],
        output_keys=["sample_rew"],
    )
    actor_train = MFCDef(
        n_seqs=1,
        name="actorTrain",
        interface_type=ModelInterfaceType.TRAIN_STEP,
        interface_impl=_dump_impl,
        model_name=ModelName("actor", 1),
        input_keys=["sampled_input_ids", "greedy_rew", "sample_rew"],
    )
    return [actor_gen, actor_train, greedy_inf, actor_sample, sample_inf]


def _get_grpo_rpcs():
    """In terms of dataflow, GRPO is PPO without the critic model."""
    rollout = MFCDef(
        name=f"actor_gen",
        model_name="actor",
        interface_type=ModelInterfaceType.GENERATE,
        interface_impl=_dump_impl,
        input_keys=["packed_prompts"],
        output_keys=[
            f"packed_input_ids",
            f"packed_logprobs",
        ],
        n_seqs=1,
    )

    inf_reward = MFCDef(
        name=f"rew_inf",
        model_name="reward",
        interface_type=ModelInterfaceType.INFERENCE,
        interface_impl=_dump_impl,
        input_keys=[f"packed_input_ids"],
        output_keys=[f"rewards"],
        n_seqs=1,
    )

    inf_ref_logits = MFCDef(
        name=f"ref_inf",
        model_name="ref",
        n_mbs=1,
        interface_type=ModelInterfaceType.INFERENCE,
        interface_impl=_dump_impl,
        input_keys=[f"packed_input_ids"],
        output_keys=[f"packed_ref_logprobs"],
        n_seqs=1,
    )

    train_actor = MFCDef(
        name="actor_train",
        model_name="actor",
        n_mbs=1,
        interface_type=ModelInterfaceType.TRAIN_STEP,
        interface_impl=_dump_impl,
        input_keys=[
            f"packed_input_ids",
            f"packed_logprobs",
            f"packed_ref_logprobs",
            f"rewards",
            f"prompt_mask",
            f"packed_logits_mask",
        ],
        n_seqs=1,
    )
    return [rollout, inf_reward, inf_ref_logits, train_actor]


def _get_dpo_rpcs():
    ref_inf = MFCDef(
        name="ref_inf",
        model_name=ModelName("ref", 0),
        interface_type=ModelInterfaceType.INFERENCE,
        interface_impl=_dump_impl,
        input_keys=[
            "packed_input_ids",
            "prompt_lens",
        ],
        output_keys=["seqlogp"],
        n_seqs=1,
    )
    dpo = MFCDef(
        name="actor_train",
        model_name=ModelName("actor", 0),
        interface_type=ModelInterfaceType.TRAIN_STEP,
        interface_impl=_dump_impl,
        input_keys=[
            "packed_input_ids",
            "seqlogp",
            "prompt_lens",
        ],
        log_return_value=True,
        n_seqs=1,
    )
    return [ref_inf, dpo]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visualize Dataflow Graph")
    parser.add_argument("--algo", "-a", type=str, default="ppo")
    args = parser.parse_args()
    if args.algo == "ppo":
        rpcs = _get_ppo_rpcs()
    elif args.algo == "reinforce":
        rpcs = _get_reinforce_rpcs()
    elif args.algo == "grpo":
        rpcs = _get_grpo_rpcs()
    elif args.algo == "dpo":
        rpcs = _get_dpo_rpcs()
    else:
        raise ValueError(f"Invalid algorithm: {args.algo}")

    build_graph(
        rpcs, verbose=True, graph_path=f"docs/source/images/dfg/{args.algo}.svg"
    )
