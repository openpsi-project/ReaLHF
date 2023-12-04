from typing import Optional
import argparse
import dataclasses
import datetime
import functools
import getpass
import os
import pickle
import subprocess
import sys

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
import hydra

from base.cluster import spec as cluster_spec
from base.constants import LOG_ROOT, MODEL_SAVE_ROOT, QUICKSTART_EXPR_CACHE_PATH
import api.config

SUPPORTED_MODELS = ["starcoder", "llama", "gpt2", "saved"]

cs = ConfigStore.instance()


@dataclasses.dataclass
class ParallelismConfig:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1

    def __post_init__(self):
        if self.tensor_parallel_size != 1:
            raise ValueError("Tensor parallelism is not supported yet")


@dataclasses.dataclass
class ModelConfig:
    """Model configuration.

    We use customized model class, i.e., impl.nn.model.flash_mqat, instead of HuggingFace's.
    This class enables 3D parallelism and flash-attention for better scalibility.
    The model uses no-pad flash-attn to save GPU memory, i.e., input sequences are packed into
    a single 1D tensor. The price is that we need to convert each HuggingFace model of interest
    to this customized model manually.

    If you find that the model of your interest is not supported, please reach out @fuwei for help.

    Args:
        type (str): Model type. Please check SUPPORTED_MODELS. `saved` means loading a saved customized model,
            while others mean converting a HuggingFace model to our customized model.
        path (str): Model path, the directory instead of the file.
        lora (bool): Whether to use LoRA.
        lora_dim (int): LoRA dimension.
        lora_scaling (float): LoRA scaling factor.
        gradient_checkpointing (bool): Whether to use gradient checkpointing of MLP inside each block.
        enable_fp16 (bool): Whether to use fp16.
        parallel (ParallelismConfig): Parallelism configuration.
    """

    type: str = dataclasses.field(
        metadata={"choices": SUPPORTED_MODELS},
        default="llama",
    )
    path: str = "/lustre/fw/pretrained/llama-7b/"
    lora: bool = False
    lora_dim: int = 32
    lora_scaling: float = 32.0
    gradient_checkpointing: bool = False
    enable_fp16: bool = True
    parallel: ParallelismConfig = dataclasses.field(default_factory=ParallelismConfig)
    base_model_path: Optional[str] = None


@dataclasses.dataclass
class OptimizerConfig:
    """Optimizer configuration.

    Args:
        type (str): Optimizer type. Currently only adam optimizer is supported.
        lr (float): Learning rate.
        weight_decay (float): Weight decay.
        beta1 (float): Adam beta1.
        beta2 (float): Adam beta2.
        eps (float): Adam epsilon in the denominator.
        zero_stage (int): Stage of DeepSpeed ZeRO optimization. Should be one of 0, 1, 2, 3.
            If pipeline parallelism is used, this should be at most 1.
        min_lr_ratio (float): Minimum learning rate ratio after learning rate annealing.
            Should be in the interval of [0.0, 1.0].
        lr_scheduler_type (str): Learning rate scheduler type.
        warmup_steps_proportion (float): Proportion of total training steps to warm up.
            Should be in the interval of [0.0, 1.0].
    """

    type: str = dataclasses.field(
        metadata={"choices": ["adam"]},
        default="adam",
    )
    lr: float = 1e-6
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-5
    zero_stage: int = dataclasses.field(
        metadata={"choices": [0, 1, 2, 3]},
        default=2,
    )
    min_lr_ratio: float = 0.0
    lr_scheduler_type: str = dataclasses.field(
        metadata={"choices": ["linear", "cosine"]},
        default="cosine",
    )
    warmup_steps_proportion: float = 0.02


@dataclasses.dataclass
class PromptAnswerDatasetConfig:
    """Datasets used for SFT.

    The raw data must be a json or jsonl file, where each piece of data is a dictionary
    with keys `prompt` and `answer`. Both `prompt` and `answer` are strings.
    Check `impl/dataset/common/packed_prompt_answer_dataset.py` or inspect the example
    dataset for more details.

    Sequences will be packed into an 1D tensor without padding, so the batch size is determined
    by the number of tokens. The user has the responsibility to control the number of sequences
    in each batch by adjusting `max_seqlen` and `train_tokens_per_batch`, such that it is greater
    than or equal to the data parallelism degree. For example, if `model.parallel.data_parallel_size`
    is set to 4, then the number of sequences in each batch should be at least 4.

    Args:
        max_seqlen (str): Maximum sequence length (prompt + answer).
            Sequences longer than this will be truncated.
        train_tokens_per_batch (int): Number of tokens in each batch during training.
        valid_tokens_per_batch (int): Number of tokens in each batch during evaluation.
        train_path (str): Path to the training dataset.
        valid_path (str): Path to the evaluation dataset.
    """

    max_seqlen: int = 1024
    train_tokens_per_batch: int = 8192
    valid_tokens_per_batch: int = 8192
    train_path: str = "/lustre/fw/datasets/imdb/rl/sft_pos-train.jsonl"
    valid_path: str = "/lustre/fw/datasets/imdb/rl/sft_pos-valid.jsonl"


@dataclasses.dataclass
class PairedComparisonDatasetConfig:
    """Datasets used for paired-comparison reward modeling.

    The raw data must be a json or jsonl file, where each piece of data is a dictionary
    with keys `prompt`, `pos_answers`, and `neg_answers`. `prompt` is a string.
    `pos_answers` and `neg_answers` are lists of strings. They must have the same size.
    Check `impl/dataset/common/packed_rw_paired_dataset.py` or inspect the example
    dataset for more details.

    Answer pairs of the same prompt will be sampled in the same batch. Hence, the number of sequences
    in each batch must be even, in the form of [P1A1+, P1A1-, P1A2+, P1A2-, P2A1+, P2A1-, P2A2+, P2A2-, ...],
    where `P` means prompt, `A` means answer, `+` means positive, and `-` means negative.

    Sequences will be packed into an 1D tensor without padding, so the batch size is determined
    by the number of tokens. The user has the responsibility to control the number of *answer pairs*
    in each batch by adjusting `max_seqlen` and `train_tokens_per_batch`, such that it is greater
    than or equal to the data parallelism degree.

    The raw dataset may contain multiple answer pairs for each prompt. We will randomly sample
    `max_pairs_per_prompt` answer pairs for each prompt.

    Args:
        max_pairs_per_prompt (int): Maximum number of answer pairs per prompt.
        max_seqlen (str): Maximum sequence length (prompt + answer).
            Sequences longer than this will be truncated.
        train_tokens_per_batch (int): Number of tokens in each batch during training.
        valid_tokens_per_batch (int): Number of tokens in each batch during evaluation.
        train_path (str): Path to the training dataset.
        valid_path (str): Path to the evaluation dataset.
    """

    max_pairs_per_prompt: int = 2
    max_seqlen: int = 1024
    train_tokens_per_batch: int = 32768
    valid_tokens_per_batch: int = 32768
    train_path: str = "/lustre/fw/datasets/imdb/rl/rm_paired-train.jsonl"
    valid_path: str = "/lustre/fw/datasets/imdb/rl/rm_paired-valid.jsonl"


@dataclasses.dataclass
class PromptOnlyDatasetConfig:
    """Datasets used for PPO RLHF.

    The raw data must be a json or jsonl file, where each piece of data is a dictionary
    with a single key called `prompt`, which is a string.
    Check `impl/dataset/common/prompt_dataset.py` or inspect the example
    dataset for more details.

    Sampled prompts will be left-padded to `max_prompt_len` for generation.

    Args:
        max_prompt_len (int): Maximum prompt length. Prompts shorter than this will be left-padded
            and prompts longer than this will be truncated.
        batch_size (int): Number of prompts in each batch.
        path (str): Path to the dataset.
    """

    max_prompt_len: int = 256
    batch_size: int = 256
    path: str = "/data/aigc/llm/datasets/llama/train.jsonl"


@dataclasses.dataclass
class SFTConfig:
    """Experiment configuration for supervised-finetuning (SFT).

    Args:
        experiment_name (str): Experiment name. **This will be automatically filled**.
        train_epochs (int): Number of training epochs.
        eval_freq (int): Evaluation frequency in terms of *epochs*.
        save_freq (int): Checkpoint saving frequency in terms of *training steps*.
        seed (int): Random seed.
        model (ModelConfig): Model configuration.
        optimizer (OptimizerConfig): Optimizer configuration.
        dataset (PromptAnswerDatasetConfig): Dataset configuration.
    """

    experiment_name: str = MISSING
    train_epochs: int = 1
    eval_freq: Optional[int] = 1
    save_freq: Optional[int] = 50
    seed: int = 42
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
    dataset: PromptAnswerDatasetConfig = dataclasses.field(default_factory=PromptAnswerDatasetConfig)


@dataclasses.dataclass
class RWConfig:
    """Experiment configuration for paired-comparison reward modeling.

    Args:
        experiment_name (str): Experiment name. **This will be automatically filled**.
        train_epochs (int): Number of training epochs.
        eval_freq (int): Evaluation frequency in terms of *epochs8.
        save_freq (int): Checkpoint saving frequency in terms of *training steps*.
        seed (int): Random seed.
        model (ModelConfig): Model configuration. Should be initialized with a SFT model.
        optimizer (OptimizerConfig): Optimizer configuration.
        dataset (PromptAnswerDatasetConfig): Dataset configuration.
    """

    experiment_name: str = MISSING
    train_epochs: int = 1
    eval_freq: Optional[int] = 1
    save_freq: Optional[int] = 50
    seed: int = 42
    is_sft_lora: bool = False
    sft_lora_path: Optional[str] = None
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
    dataset: PairedComparisonDatasetConfig = dataclasses.field(default_factory=PairedComparisonDatasetConfig)


@dataclasses.dataclass
class PPOConfig:
    """Experiment configuration for PPO RLHF.

    Args:
        experiment_name (str): Experiment name. **This will be automatically filled**.
        train_epochs (int): Number of training epochs.
        eval_freq (int): Evaluation frequency in terms of *epochs8.
        save_freq (int): Checkpoint saving frequency in terms of *training steps*.
        seed (int): Random seed.
        actor (ModelConfig): Actor model configuration. Should be initialized with a SFT model.
        critic (ModelConfig): Critic model configuration. Should be initialized with a RW model.
        ref (ModelConfig): Reference model configuration. The SFT model should be loaded and freezed.
        rew (ModelConfig): Reward model configuration. The RW model should be loaded and freezed.
        dataset (PromptAnswerDatasetConfig): Dataset configuration.
        actor_optimizer (OptimizerConfig): Actor optimizer configuration.
        critic_optimizer (OptimizerConfig): Critic optimizer configuration.
        max_new_tokens (int): Maximum number of new tokens to generate in each iteration.
        min_new_tokens (int): Minimum number of new tokens to generate in each iteration.
        greedy (bool): Whether to use greedy decoding. PPO may not work if set to True.
        top_p (float): Top-p sampling ratio.
        top_k (float): Top-k sampling ratio.
        temperature (float): Sampling temperature.
        n_minibatches (int): Number of minibatches in each PPO update.
        kl_ctl (float): Coefficient of KL divergence rewards.
        discount (float): Discount factor.
        gae_lambda (float): Lambda factor in GAE.
        eps_clip (float): PPO clipping factor.
        value_eps_clip (float): PPO value clipping factor.
        max_reward_clip (float): Maximum reward value.
        reward_output_scaling (float): Scaling factor of the reward model output.
        reward_output_bias (float): Bias of the reward model output.
            The number outputed by the reward model will be
            CLIP((x - bias) * scaling, -max_reward_clip, max_reward_clip).
        early_stop_imp_ratio (float): PPO update will be early stopped if importance ratio
            exceeds this maximum value.
    """

    experiment_name: str = MISSING
    train_epochs: int = 1
    eval_freq: Optional[int] = 1
    save_freq: Optional[int] = 50
    seed: int = 42
    actor: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    critic: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    ref: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    rew: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    dataset: PromptOnlyDatasetConfig = dataclasses.field(default_factory=PromptOnlyDatasetConfig)
    actor_optimizer: OptimizerConfig = dataclasses.field(default_factory=functools.partial(
        OptimizerConfig,
        lr=9.65e-6,
        weight_decay=0.0,
        eps=1e-5,
        lr_scheduler_type="linear",
        warmup_steps_proportion=0.075,
    ))
    critic_optimizer: OptimizerConfig = dataclasses.field(default_factory=functools.partial(
        OptimizerConfig,
        lr=5e-6,
        weight_decay=0.0,
        eps=1e-5,
        lr_scheduler_type="linear",
        warmup_steps_proportion=0.075,
    ))
    max_new_tokens: int = 512
    min_new_tokens: int = 10
    greedy: bool = False
    top_p: float = 1.0
    top_k: float = 200
    temperature: float = 1.0
    n_minibatches: int = 8
    kl_ctl: float = 0.1
    discount: float = 1.0
    gae_lambda: float = 1.0
    eps_clip: float = 0.2
    value_eps_clip: float = 0.2
    max_reward_clip = 20.0
    reward_output_scaling: float = 1.0
    reward_output_bias: float = 0.0
    early_stop_imp_ratio: float = 5.0


@dataclasses.dataclass
class DPOConfig:
    """Experiment configuration for direct preference optimization.

    Args:
        experiment_name (str): Experiment name. **This will be automatically filled**.
        train_epochs (int): Number of training epochs.
        eval_freq (int): Evaluation frequency in terms of *epochs8.
        save_freq (int): Checkpoint saving frequency in terms of *training steps*.
        seed (int): Random seed.
        actor (ModelConfig): Actor model configuration. Should be initialized with a SFT model.
        ref (ModelConfig): Reference model configuration. The SFT model should be loaded and freezed.
        optimizer (OptimizerConfig): Optimizer configuration.
        dataset (PromptAnswerDatasetConfig): Dataset configuration.
        beta (float): KL coefficient in the DPO paper. The same meaning as `kl_ctl` in PPO config.
    """

    experiment_name: str = MISSING
    train_epochs: int = 1
    eval_freq: Optional[int] = 1
    save_freq: Optional[int] = 50
    seed: int = 42
    actor: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    ref: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    dataset: PairedComparisonDatasetConfig = dataclasses.field(default_factory=PairedComparisonDatasetConfig)
    optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
    beta: float = 0.1


cs.store(name="sft", node=SFTConfig)
cs.store(name="rw", node=RWConfig)
cs.store(name="ppo", node=PPOConfig)
cs.store(name="dpo", node=DPOConfig)


@dataclasses.dataclass
class _MainStartArgs:
    experiment_name: str
    trial_name: str
    mode: str
    debug: bool = True
    partition: str = "dev"
    wandb_mode: str = "disabled"
    image_name: Optional[str] = None
    ignore_worker_error: bool = False
    remote_reset: bool = False
    trace: bool = False


@hydra.main(version_base=None, config_name="sft")
def run_sft(args: SFTConfig):
    import base.logging as logging

    logger = logging.getLogger("quickstart")

    exp_name = args.experiment_name
    trial_name = f"run{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    from apps.main import main_start
    from experiments.common.sft_exp import SFTExperiment

    logger.info("Running supervised-finetuning experiment.")
    logger.info("Logs will be dumped to %s", os.path.join(LOG_ROOT, exp_name, trial_name))
    logger.info("Model checkpoints will be saved to %s", os.path.join(MODEL_SAVE_ROOT, exp_name, trial_name))

    if args.model.parallel.pipeline_parallel_size > 1:
        logger.warning(
            "Pipeline parallel is enabled. Please ensure that (1) there are enough GPUs for your experiment "
            "and (2) the model checkpoint has been converted into shards using scripts/transform_to_pipe_ckpt.py."
        )

    exp_fn = functools.partial(
        SFTExperiment,
        seed=args.seed,
        total_train_epochs=args.train_epochs,
        model_type=args.model.type,
        model_path=args.model.path,
        dp_size=args.model.parallel.data_parallel_size,
        pp_size=args.model.parallel.pipeline_parallel_size,
        use_lora=args.model.lora,
        lora_scaling=args.model.lora_scaling,
        lora_dim=args.model.lora_dim,
        enable_fp16=args.model.enable_fp16,
        gradient_checkpointing=args.model.gradient_checkpointing,
        max_seqlen=args.dataset.max_seqlen,
        train_dataset_path=args.dataset.train_path,
        valid_dataset_path=args.dataset.valid_path,
        train_tokens_per_batch=args.dataset.train_tokens_per_batch,
        valid_tokens_per_batch=args.dataset.valid_tokens_per_batch,
        lr=args.optimizer.lr,
        weight_decay=args.optimizer.weight_decay,
        adam_betas=(args.optimizer.beta1, args.optimizer.beta2),
        lr_scheduler_type=args.optimizer.lr_scheduler_type,
        warmup_proportion=args.optimizer.warmup_steps_proportion,
        adam_eps=args.optimizer.eps,
        min_lr_ratio=args.optimizer.min_lr_ratio,
        zero_stage=args.optimizer.zero_stage,
        save_freq_steps=args.save_freq,
        eval_freq_epochs=args.eval_freq,
    )

    os.makedirs(os.path.dirname(QUICKSTART_EXPR_CACHE_PATH), exist_ok=True)
    with open(QUICKSTART_EXPR_CACHE_PATH, "wb") as f:
        pickle.dump((exp_name, exp_fn), f)
    api.config.register_experiment(exp_name, exp_fn)

    slurm_available = int(subprocess.run("squeue", shell=True, stdout=open(os.devnull, "wb")).returncode) == 0
    mode = "slurm" if slurm_available else "local"

    main_start(_MainStartArgs(exp_name, trial_name, mode, debug=True))


@hydra.main(version_base=None, config_name="rw")
def run_rw(args: RWConfig):
    import base.logging as logging

    logger = logging.getLogger("quickstart")
    exp_name = args.experiment_name
    trial_name = f"run{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    from apps.main import main_start
    from experiments.common.rw_exp import PairedRWExperiment

    logger.info("Running paired comparison reward modeling experiment.")
    logger.info("Logs will be dumped to %s", os.path.join(LOG_ROOT, exp_name, trial_name))
    logger.info("Model checkpoints will be saved to %s", os.path.join(MODEL_SAVE_ROOT, exp_name, trial_name))

    if args.model.parallel.pipeline_parallel_size > 1:
        logger.warning(
            "Pipeline parallel is enabled. Please ensure that (1) there are enough GPUs for your experiment "
            "and (2) the model checkpoint has been converted into shards using scripts/transform_to_pipe_ckpt.py."
        )
    if args.model.base_model_path is None:
        raise ValueError("model.base_model_path must be specified for RW experiment.")

    exp_fn = functools.partial(
        PairedRWExperiment,
        model_path=args.model.path,
        tokenizer_path=args.model.base_model_path,
        seed=args.seed,
        total_train_epochs=args.train_epochs,
        save_freq_steps=args.save_freq,
        eval_freq_epochs=args.eval_freq,
        is_sft_lora=args.is_sft_lora,
        base_model_type=args.model.type,
        sft_lora_path=args.sft_lora_path,
        dp_size=args.model.parallel.data_parallel_size,
        pp_size=args.model.parallel.pipeline_parallel_size,
        use_lora=args.model.lora,
        lora_scaling=args.model.lora_scaling,
        lora_dim=args.model.lora_dim,
        enable_fp16=args.model.enable_fp16,
        gradient_checkpointing=args.model.gradient_checkpointing,
        max_pairs_per_prompt=args.dataset.max_pairs_per_prompt,
        max_seqlen=args.dataset.max_seqlen,
        train_dataset_path=args.dataset.train_path,
        valid_dataset_path=args.dataset.valid_path,
        train_tokens_per_batch=args.dataset.train_tokens_per_batch,
        valid_tokens_per_batch=args.dataset.valid_tokens_per_batch,
        lr=args.optimizer.lr,
        weight_decay=args.optimizer.weight_decay,
        adam_betas=(args.optimizer.beta1, args.optimizer.beta2),
        lr_scheduler_type=args.optimizer.lr_scheduler_type,
        warmup_proportion=args.optimizer.warmup_steps_proportion,
        adam_eps=args.optimizer.eps,
        min_lr_ratio=args.optimizer.min_lr_ratio,
        zero_stage=args.optimizer.zero_stage,
    )

    os.makedirs(os.path.dirname(QUICKSTART_EXPR_CACHE_PATH), exist_ok=True)
    with open(QUICKSTART_EXPR_CACHE_PATH, "wb") as f:
        pickle.dump((exp_name, exp_fn), f)
    api.config.register_experiment(exp_name, exp_fn)

    slurm_available = int(subprocess.run("squeue", shell=True, stdout=open(os.devnull, "wb")).returncode) == 0
    mode = "slurm" if slurm_available else "local"

    main_start(_MainStartArgs(exp_name, trial_name, mode, debug=True))


@hydra.main(version_base=None, config_name="ppo")
def run_ppo(args):
    import base.logging as logging

    logger = logging.getLogger("quickstart")
    # TODO: implement this
    trial_name = f"run{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    print(args)


@hydra.main(version_base=None, config_name="dpo")
def run_dpo(args):
    import base.logging as logging

    logger = logging.getLogger("quickstart")
    # TODO: implement this
    trial_name = f"run{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    print(args)


def main():
    parser = argparse.ArgumentParser(prog="distributed_llm")
    subparsers = parser.add_subparsers(dest="cmd", help="sub-command help")
    subparsers.required = True

    subparser = subparsers.add_parser("sft", help="run supervised-finetuning")
    subparser.set_defaults(func=run_sft)
    subparser = subparsers.add_parser("rw", help="run reward modeling")
    subparser.set_defaults(func=run_rw)
    subparser = subparsers.add_parser("ppo", help="run PPO RLHF")
    subparser.set_defaults(func=run_ppo)
    subparser = subparsers.add_parser("dpo", help="run direct preference optimization")
    subparser.set_defaults(func=run_dpo)
    args = parser.parse_known_args()[0]

    experiment_name = f"quickstart-{args.cmd}"
    if any("experiment_name" in x for x in sys.argv):
        raise ValueError("experiment_name should not be specified in the command line")
    sys.argv += [f"experiment_name={experiment_name}", "hydra/job_logging=disabled"]
    if "--multirun" not in sys.argv and "hydra.mode=MULTIRUN" not in sys.argv and "-m" not in sys.argv:
        trial_name = f"run{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        sys.argv += [
            f"hydra.run.dir={cluster_spec.fileroot}/logs/{getpass.getuser()}/"
            f"{experiment_name}/{trial_name}/hydra-outputs/"
            f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        ]
    else:
        sys.argv += [
            f"hydra.sweep.dir={cluster_spec.fileroot}/logs/{getpass.getuser()}/"
            f"{experiment_name}/hydra-sweep-outputs/"
            f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        ]

    sys.argv.pop(1)
    args.func()


if __name__ == "__main__":
    main()
