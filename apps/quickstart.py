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

SUPPORTED_MODELS = ["starcoder", "llama", "gpt2", "deepseek", "codellama"]

cs = ConfigStore.instance()


@dataclasses.dataclass
class ParallelismConfig:
    model_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1


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
        partition_method (str): Partition method for modules using pipeline parallel. 
                                Support "uniform", "parameters" and "parameters_balanced".
    """

    type: str = dataclasses.field(
        metadata={"choices": SUPPORTED_MODELS},
        default="llama",
    )
    path: str = "/lustre/fw/pretrained/llama-7b/"
    base_model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    lora: bool = False
    lora_dim: int = 32
    lora_scaling: float = 32.0
    gradient_checkpointing: bool = False
    enable_fp16: bool = True
    enable_bf16: bool = False
    parallel: ParallelismConfig = dataclasses.field(default_factory=ParallelismConfig)
    partition_method: Optional[str] = "parameters"

    def __post_init__(self):
        if self.enable_bf16 and self.enable_fp16:
            raise ValueError("enable_bf16 and enable_fp16 cannot be both True.")


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
    offload: bool = False


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
    path: str = "/lustre/fw/datasets/imdb/rl/ppo_prompt.jsonl"


@dataclasses.dataclass
class SFTConfig:
    """Experiment configuration for supervised-finetuning (SFT).

    Args:
        experiment_name (str): Experiment name. **This will be automatically filled**.
        trial_name (str): Trial name. **This will be automatically filled**.
        trace (bool): Whether to enable viztracer tracing.
        train_epochs (int): Number of training epochs.
        eval_freq (int): Evaluation frequency in terms of *epochs*.
        save_freq (int): Checkpoint saving frequency in terms of *training steps*.
        seed (int): Random seed.
        model (ModelConfig): Model configuration.
        optimizer (OptimizerConfig): Optimizer configuration.
        dataset (PromptAnswerDatasetConfig): Dataset configuration.
    """

    experiment_name: str = MISSING
    trial_name: str = MISSING
    trace: bool = False
    train_epochs: int = 1
    eval_freq: Optional[int] = 1
    save_freq: Optional[int] = 50
    seed: int = 42
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
    dataset: PromptAnswerDatasetConfig = dataclasses.field(default_factory=PromptAnswerDatasetConfig)
    _configuration_name: str = "Supervised-Finetuning"


@dataclasses.dataclass
class RWConfig:
    """Experiment configuration for paired-comparison reward modeling.

    Args:
        experiment_name (str): Experiment name. **This will be automatically filled**.
        trial_name (str): Trial name. **This will be automatically filled**.
        trace (bool): Whether to enable viztracer tracing.
        train_epochs (int): Number of training epochs.
        eval_freq (int): Evaluation frequency in terms of *epochs8.
        save_freq (int): Checkpoint saving frequency in terms of *training steps*.
        seed (int): Random seed.
        model (ModelConfig): Model configuration. Should be initialized with a SFT model.
        optimizer (OptimizerConfig): Optimizer configuration.
        dataset (PromptAnswerDatasetConfig): Dataset configuration.
    """

    experiment_name: str = MISSING
    trial_name: str = MISSING
    trace: bool = False
    train_epochs: int = 1
    eval_freq: Optional[int] = 1
    save_freq: Optional[int] = 50
    seed: int = 42
    is_sft_lora: bool = False
    sft_lora_path: Optional[str] = None
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
    dataset: PairedComparisonDatasetConfig = dataclasses.field(default_factory=PairedComparisonDatasetConfig)
    _configuration_name: str = "Paired Comparison Reward Modeling"


@dataclasses.dataclass
class PPOConfig:
    """Experiment configuration for PPO RLHF.

    Args:
        experiment_name (str): Experiment name. **This will be automatically filled**.
        trial_name (str): Trial name. **This will be automatically filled**.
        trace (bool): Whether to enable viztracer tracing.
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
    trial_name: str = MISSING
    trace: bool = False
    train_epochs: int = 1
    save_freq: Optional[int] = 50
    seed: int = 42
    is_sft_lora: bool = False
    is_sft_pipe: bool = False
    sft_lora_path: Optional[str] = None
    is_rew_lora: bool = False
    is_rew_pipe: bool = False
    rew_lora_path: Optional[str] = None
    rew_head_path: Optional[str] = None
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
    top_k: int = 200
    temperature: float = 1.0
    ppo_n_minibatches: int = 4
    kl_ctl: float = 0.1
    adv_norm: bool = False
    discount: float = 1.0
    gae_lambda: float = 1.0
    eps_clip: float = 0.2
    value_eps_clip: float = 0.2
    max_reward_clip: float = 20.0
    reward_output_scaling: float = 1.0
    reward_output_bias: float = 0.0
    early_stop_imp_ratio: float = 5.0
    use_adaptive_kl_ctl: bool = False
    value_norm: bool = False
    value_norm_type: str = dataclasses.field(metadata={"choices": ["exp", "ma"]}, default="exp")
    value_norm_beta: float = 0.99995
    value_norm_eps: float = 1e-5
    _configuration_name: str = "Proximal Policy Optimization"

    def __post_init__(self):
        if (self.actor.path != self.ref.path or self.actor.base_model_path != self.ref.base_model_path
                or self.actor.type != self.ref.type):
            raise ValueError("actor and ref must be the same model.")
        if (self.critic.path != self.rew.path or self.critic.base_model_path != self.rew.base_model_path
                or self.critic.type != self.rew.type):
            raise ValueError("critic and rew must be the same model.")
        if self.actor.tokenizer_path != self.critic.tokenizer_path:
            raise ValueError(
                f"`actor` and `critic` must use the same tokenizer. "
                "It is possible that you are using the same base model with different sizes "
                "(e.g., LLaMa 13b as the actor and 7b as the critic). They have the same "
                "tokenizer but different model paths. Please specify the tokenizer path manually.")


@dataclasses.dataclass
class DPOConfig:
    """Experiment configuration for direct preference optimization.

    Args:
        experiment_name (str): Experiment name. **This will be automatically filled**.
        trial_name (str): Trial name. **This will be automatically filled**.
        trace (bool): Whether to enable viztracer tracing.
        train_epochs (int): Number of training epochs.
        eval_freq (int): Evaluation frequency in terms of *epochs8.
        save_freq (int): Checkpoint saving frequency in terms of *training steps*.
        seed (int): Random seed.
        model (ModelConfig): Model model configuration. Should be initialized with a SFT model.
            It is also used as the reference model.
        optimizer (OptimizerConfig): Optimizer configuration.
        dataset (PromptAnswerDatasetConfig): Dataset configuration.
        beta (float): KL coefficient in the DPO paper. The same meaning as `kl_ctl` in PPO config.
    """

    experiment_name: str = MISSING
    trial_name: str = MISSING
    trace: bool = False
    train_epochs: int = 1
    save_freq: Optional[int] = 50
    seed: int = 42
    is_sft_pipe: bool = False
    is_sft_lora: bool = False
    sft_lora_path: Optional[str] = None
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    dataset: PairedComparisonDatasetConfig = dataclasses.field(default_factory=PairedComparisonDatasetConfig)
    optimizer: OptimizerConfig = dataclasses.field(default_factory=OptimizerConfig)
    beta: float = 0.1
    _configuration_name: str = "Direct Preference Optimization"


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


def kind_reminder(logger, args):
    logger.info(f"Running {args._configuration_name} experiment.")
    logger.info(f"Logs will be dumped to {os.path.join(LOG_ROOT, args.experiment_name, args.trial_name)}")
    logger.info(
        f"Model checkpoints will be saved to {os.path.join(MODEL_SAVE_ROOT, args.experiment_name, args.trial_name)}"
    )

    slurm_available = int(subprocess.run("squeue", shell=True, stdout=open(os.devnull, "wb")).returncode) == 0
    if slurm_available:
        logger.warning(
            "Slurm is available. You probably run the system on ctrl nodes. Using slurm to launch remote workers."
        )
    else:
        logger.warning("Slurm is not available. Using local mode.")
    mode = "slurm" if slurm_available else "local"
    return mode


@hydra.main(version_base=None, config_name="sft")
def run_sft(args: SFTConfig):
    # NOTE: we import logging here to avoid hydra logging overwrite
    import base.logging as logging

    logger = logging.getLogger("quickstart", "colored")

    exp_name = args.experiment_name
    if args.trial_name == MISSING:
        args.trial_name = trial_name = f"run{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    else:
        trial_name = args.trial_name
    from apps.main import main_start, main_stop
    from experiments.common.sft_exp import SFTExperiment

    mode = kind_reminder(logger, args)

    if args.model.parallel.pipeline_parallel_size > 1:
        logger.warning(
            "Pipeline parallel is enabled. Please ensure that (1) there are enough GPUs for your experiment "
            "and (2) the model checkpoint has been converted into shards using scripts/transform_to_pipe_ckpt.py."
        )
    if args.model.parallel.model_parallel_size > 1:
        logger.warning(
            "Model parallel is enabled. Please ensure that (1) there are enough GPUs for your experiment "
            "and (2) the model checkpoint has been converted into shards using scripts/transform_to_pipe_ckpt.py."
        )

    exp_fn = functools.partial(SFTExperiment,
                               seed=args.seed,
                               total_train_epochs=args.train_epochs,
                               model_type=args.model.type,
                               model_path=args.model.path,
                               dp_size=args.model.parallel.data_parallel_size,
                               mp_size=args.model.parallel.model_parallel_size,
                               pp_size=args.model.parallel.pipeline_parallel_size,
                               use_lora=args.model.lora,
                               lora_scaling=args.model.lora_scaling,
                               lora_dim=args.model.lora_dim,
                               enable_fp16=args.model.enable_fp16,
                               enable_bf16=args.model.enable_bf16,
                               offload_optimizer=args.optimizer.offload,
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
                               partition_method=args.model.partition_method)

    os.makedirs(os.path.dirname(QUICKSTART_EXPR_CACHE_PATH), exist_ok=True)
    with open(QUICKSTART_EXPR_CACHE_PATH, "wb") as f:
        pickle.dump((exp_name, exp_fn), f)
    api.config.register_experiment(exp_name, exp_fn)

    try:
        main_start(_MainStartArgs(exp_name, trial_name, mode, debug=True, trace=args.trace))
    except Exception as e:
        main_stop(_MainStartArgs(exp_name, trial_name, mode, debug=True, trace=args.trace))
        logger.warning("Exception occurred. Stopping all workers.")
        raise e


@hydra.main(version_base=None, config_name="rw")
def run_rw(args: RWConfig):
    # NOTE: we import logging here to avoid hydra logging overwrite
    import base.logging as logging

    logger = logging.getLogger("quickstart", "colored")

    exp_name = args.experiment_name
    if args.trial_name == MISSING:
        args.trial_name = trial_name = f"run{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    else:
        trial_name = args.trial_name
    from apps.main import main_start, main_stop
    from experiments.common.rw_exp import PairedRWExperiment

    mode = kind_reminder(logger, args)

    if args.model.base_model_path is None:
        logger.warning("`base_model_path` is not specified. Using `path` as `base_model_path`.")
        args.model.base_model_path = args.model.path
    if args.model.tokenizer_path is None:
        logger.warning("`tokenizer_path` is not specified. Using `base_model_path` as `tokenizer_path`.")
        args.model.tokenizer_path = args.model.base_model_path

    if args.model.parallel.pipeline_parallel_size > 1:
        logger.critical(
            "Pipeline parallel is enabled when training the reward model. **This is usually unnecessary.** "
            "The reward model should not be large and using DeepSpeed ZeRO-2 data parallel is usually sufficient. "
            "If you insist in using PP, please ensure that (1) there are enough GPUs for your experiment "
            "and (2) the model checkpoint has been converted into shards using scripts/transform_to_pipe_ckpt.py."
        )
    if args.model.parallel.model_parallel_size > 1:
        logger.critical(
            "Model parallel is enabled when training the reward model. **This is usually unnecessary.** "
            "The reward model should not be large and using DeepSpeed ZeRO-2 data parallel is usually sufficient. "
            "If you insist in using PP, please ensure that (1) there are enough GPUs for your experiment "
            "and (2) the model checkpoint has been converted into shards using scripts/transform_to_pipe_ckpt.py."
        )
    if args.is_sft_lora and (args.model.base_model_path == args.model.path or args.sft_lora_path is None):
        raise ValueError(
            "model.base_model_path and sft_lora_path must be specified for RW experiment if SFT was trained with LoRA."
            " `path` is the path of saved LoRA weights and `base_model_path` is the path of the base model.")

    exp_fn = functools.partial(PairedRWExperiment,
                               model_path=args.model.path,
                               tokenizer_path=args.model.tokenizer_path,
                               seed=args.seed,
                               total_train_epochs=args.train_epochs,
                               save_freq_steps=args.save_freq,
                               eval_freq_epochs=args.eval_freq,
                               is_sft_lora=args.is_sft_lora,
                               base_model_type=args.model.type,
                               sft_lora_path=args.sft_lora_path,
                               dp_size=args.model.parallel.data_parallel_size,
                               mp_size=args.model.parallel.model_parallel_size,
                               pp_size=args.model.parallel.pipeline_parallel_size,
                               use_lora=args.model.lora,
                               lora_scaling=args.model.lora_scaling,
                               lora_dim=args.model.lora_dim,
                               enable_fp16=args.model.enable_fp16,
                               enable_bf16=args.model.enable_bf16,
                               offload_optimizer=args.optimizer.offload,
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
                               partition_method=args.model.partition_method)

    os.makedirs(os.path.dirname(QUICKSTART_EXPR_CACHE_PATH), exist_ok=True)
    with open(QUICKSTART_EXPR_CACHE_PATH, "wb") as f:
        pickle.dump((exp_name, exp_fn), f)
    api.config.register_experiment(exp_name, exp_fn)

    try:
        main_start(_MainStartArgs(exp_name, trial_name, mode, debug=True, trace=args.trace))
    except Exception as e:
        main_stop(_MainStartArgs(exp_name, trial_name, mode, debug=True, trace=args.trace))
        logger.warning("Exception occurred. Stopping all workers.")
        raise e


@hydra.main(version_base=None, config_name="ppo")
def run_ppo(args: PPOConfig):
    # NOTE: we import logging here to avoid hydra logging overwrite
    import base.logging as logging

    logger = logging.getLogger("quickstart", "colored")

    exp_name = args.experiment_name
    if args.trial_name == MISSING:
        args.trial_name = trial_name = f"run{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    else:
        trial_name = args.trial_name
    from apps.main import main_start, main_stop
    from experiments.common.ppo_exp import PPOExperiment

    mode = kind_reminder(logger, args)

    if args.actor.base_model_path is None:
        logger.warning("`base_model_path` is not specified. Using `path` as `base_model_path`.")
        args.actor.base_model_path = args.actor.path
    if args.actor.tokenizer_path is None:
        logger.warning("`tokenizer_path` is not specified. Using `base_model_path` as `tokenizer_path`.")
        args.actor.tokenizer_path = args.actor.base_model_path

    if args.critic.base_model_path is None:
        logger.warning("`base_model_path` is not specified. Using `path` as `base_model_path`.")
        args.critic.base_model_path = args.critic.path
    if args.critic.tokenizer_path is None:
        logger.warning("`tokenizer_path` is not specified. Using `base_model_path` as `tokenizer_path`.")
        args.critic.tokenizer_path = args.critic.base_model_path

    if args.actor.parallel.pipeline_parallel_size > 1:
        logger.warning(
            "Pipeline parallel of the actor model is enabled. Please ensure that (1) there are enough GPUs for your experiment "
            "and (2) the model checkpoint has been converted into shards using scripts/transform_to_pipe_ckpt.py."
        )
    if args.actor.parallel.model_parallel_size > 1:
        logger.warning(
            "Model parallel of the actor model is enabled. Please ensure that (1) there are enough GPUs for your experiment "
            "and (2) the model checkpoint has been converted into shards using scripts/transform_to_pipe_ckpt.py."
        )
    if args.critic.parallel.pipeline_parallel_size > 1:
        logger.critical(
            "Pipeline parallel of the critic model is enabled. **This is usually unnecessary.** "
            "The reward model should not be large and using DeepSpeed ZeRO-2 data parallel is usually sufficient. "
            "If you insist in using PP, please ensure that (1) there are enough GPUs for your experiment "
            "and (2) the model checkpoint has been converted into shards using scripts/transform_to_pipe_ckpt.py."
        )
    if args.critic.parallel.model_parallel_size > 1:
        logger.critical(
            "Model parallel of the critic model is enabled. **This is usually unnecessary.** "
            "The reward model should not be large and using DeepSpeed ZeRO-2 data parallel is usually sufficient. "
            "If you insist in using PP, please ensure that (1) there are enough GPUs for your experiment "
            "and (2) the model checkpoint has been converted into shards using scripts/transform_to_pipe_ckpt.py."
        )
    if args.ref.parallel.pipeline_parallel_size > 1:
        logger.warning(
            "Pipeline parallel of the reference model is enabled. Please ensure that (1) there are enough GPUs for your experiment "
            "and (2) the model checkpoint has been converted into shards using scripts/transform_to_pipe_ckpt.py."
        )
    if args.ref.parallel.model_parallel_size > 1:
        logger.warning(
            "Model parallel of the reference model is enabled. Please ensure that (1) there are enough GPUs for your experiment "
            "and (2) the model checkpoint has been converted into shards using scripts/transform_to_pipe_ckpt.py."
        )
    if args.is_sft_lora and (args.actor.base_model_path == args.actor.path or args.sft_lora_path is None):
        raise ValueError(
            "sft_lora_path and actor.base_model_path must be specified if the SFT model was trained with LoRA."
            " `path` is the path of saved LoRA weights and `base_model_path` is the path of the base model.")
    if args.is_rew_lora and (args.critic.base_model_path == args.critic.path or args.rew_lora_path is None
                             or args.rew_head_path is None):
        raise ValueError(
            "rew_lora_path and critic.base_model_path must be specified for RW experiment."
            " `path` is the path of saved LoRA weights and `base_model_path` is the path of the base model.")

    exp_fn = functools.partial(
        PPOExperiment,
        sft_model_path=args.actor.path,
        rew_model_path=args.critic.path,
        ref_model_path=args.ref.path,
        tokenizer_path=args.actor.tokenizer_path,
        seed=args.seed,
        total_train_epochs=args.train_epochs,
        save_freq_steps=args.save_freq,
        # sft lora
        is_sft_lora=args.is_sft_lora,
        is_sft_pipe=args.is_sft_pipe,
        sft_base_model_type=args.actor.type,
        sft_lora_path=args.sft_lora_path,
        # rew lora
        is_rew_lora=args.is_rew_lora,
        is_rew_pipe=args.is_rew_pipe,
        rew_base_model_type=args.critic.type,
        rew_lora_path=args.rew_lora_path,
        rew_head_path=args.rew_head_path,
        # actor
        actor_dp_size=args.actor.parallel.data_parallel_size,
        actor_mp_size=args.actor.parallel.model_parallel_size,
        actor_pp_size=args.actor.parallel.pipeline_parallel_size,
        actor_use_lora=args.actor.lora,
        actor_lora_scaling=args.actor.lora_scaling,
        actor_lora_dim=args.actor.lora_dim,
        actor_enable_fp16=args.actor.enable_fp16,
        actor_enable_bf16=args.actor.enable_bf16,
        offload_actor_optimizer_state=args.actor_optimizer.offload,
        actor_gradient_checkpointing=args.actor.gradient_checkpointing,
        actor_partition_method=args.actor.partition_method,
        # critic
        critic_dp_size=args.critic.parallel.data_parallel_size,
        critic_mp_size=args.critic.parallel.model_parallel_size,
        critic_pp_size=args.critic.parallel.pipeline_parallel_size,
        critic_use_lora=args.critic.lora,
        critic_lora_scaling=args.critic.lora_scaling,
        critic_lora_dim=args.critic.lora_dim,
        critic_enable_fp16=args.critic.enable_fp16,
        critic_enable_bf16=args.critic.enable_bf16,
        offload_critic_optimizer_state=args.critic_optimizer.offload,
        critic_gradient_checkpointing=args.critic.gradient_checkpointing,
        critic_partition_method=args.critic.partition_method,
        # rew & ref
        ref_dp_size=args.ref.parallel.data_parallel_size,
        ref_pp_size=args.ref.parallel.pipeline_parallel_size,
        ref_mp_size=args.ref.parallel.model_parallel_size,
        rew_dp_size=args.rew.parallel.data_parallel_size,
        ref_enable_bf16=args.ref.enable_bf16,
        rew_enable_bf16=args.rew.enable_bf16,
        # dataset
        max_prompt_len=args.dataset.max_prompt_len,
        batch_size=args.dataset.batch_size,
        dataset_path=args.dataset.path,
        # actor optim
        actor_lr=args.actor_optimizer.lr,
        actor_weight_decay=args.actor_optimizer.weight_decay,
        actor_adam_betas=(args.actor_optimizer.beta1, args.actor_optimizer.beta2),
        actor_lr_scheduler_type=args.actor_optimizer.lr_scheduler_type,
        actor_warmup_proportion=args.actor_optimizer.warmup_steps_proportion,
        actor_adam_eps=args.actor_optimizer.eps,
        actor_min_lr_ratio=args.actor_optimizer.min_lr_ratio,
        actor_zero_stage=args.actor_optimizer.zero_stage,
        # critic optim
        critic_lr=args.critic_optimizer.lr,
        critic_weight_decay=args.critic_optimizer.weight_decay,
        critic_adam_betas=(args.critic_optimizer.beta1, args.critic_optimizer.beta2),
        critic_lr_scheduler_type=args.critic_optimizer.lr_scheduler_type,
        critic_warmup_proportion=args.critic_optimizer.warmup_steps_proportion,
        critic_adam_eps=args.critic_optimizer.eps,
        critic_min_lr_ratio=args.critic_optimizer.min_lr_ratio,
        critic_zero_stage=args.critic_optimizer.zero_stage,
        # ppo
        rew_output_scaling=args.reward_output_scaling,
        rew_output_bias=args.reward_output_bias,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        greedy=args.greedy,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
        ppo_n_minibatches=args.ppo_n_minibatches,
        kl_ctl=args.kl_ctl,
        adv_norm=args.adv_norm,
        discount=args.discount,
        gae_lambda=args.gae_lambda,
        eps_clip=args.eps_clip,
        value_eps_clip=args.value_eps_clip,
        max_reward_clip=args.max_reward_clip,
        early_stop_imp_ratio=args.early_stop_imp_ratio,
        use_adaptive_kl_ctl=args.use_adaptive_kl_ctl,
        value_norm=args.value_norm,
        value_norm_type=args.value_norm_type,
        value_norm_beta=args.value_norm_beta,
        value_norm_eps=args.value_norm_eps)

    os.makedirs(os.path.dirname(QUICKSTART_EXPR_CACHE_PATH), exist_ok=True)
    with open(QUICKSTART_EXPR_CACHE_PATH, "wb") as f:
        pickle.dump((exp_name, exp_fn), f)
    api.config.register_experiment(exp_name, exp_fn)

    try:
        main_start(_MainStartArgs(exp_name, trial_name, mode, debug=True, trace=args.trace))
    except Exception as e:
        main_stop(_MainStartArgs(exp_name, trial_name, mode, debug=True, trace=args.trace))
        logger.warning("Exception occurred. Stopping all workers.")
        raise e


@hydra.main(version_base=None, config_name="dpo")
def run_dpo(args: DPOConfig):
    # NOTE: we import logging here to avoid hydra logging overwrite
    import base.logging as logging

    logger = logging.getLogger("quickstart", "colored")

    exp_name = args.experiment_name
    if args.trial_name == MISSING:
        args.trial_name = trial_name = f"run{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    else:
        trial_name = args.trial_name
    from apps.main import main_start, main_stop
    from experiments.common.dpo_exp import DPOExperiment

    mode = kind_reminder(logger, args)

    if args.model.base_model_path is None:
        logger.warning("`base_model_path` is not specified. Using `path` as `base_model_path`.")
        args.model.base_model_path = args.model.path
    if args.model.tokenizer_path is None:
        logger.warning("`tokenizer_path` is not specified. Using `base_model_path` as `tokenizer_path`.")
        args.model.tokenizer_path = args.model.base_model_path

    if args.model.parallel.pipeline_parallel_size > 1:
        logger.warning(
            "Pipeline parallel is enabled. Please ensure that (1) there are enough GPUs for your experiment "
            "and (2) the model checkpoint has been converted into shards using scripts/transform_to_pipe_ckpt.py."
        )
    if args.model.parallel.model_parallel_size > 1:
        logger.warning(
            "Model parallel is enabled. Please ensure that (1) there are enough GPUs for your experiment "
            "and (2) the model checkpoint has been converted into shards using scripts/transform_to_pipe_ckpt.py."
        )
    if args.is_sft_lora and (args.sft_lora_path == args.model.path or args.model.base_model_path is None):
        raise ValueError(
            "sft_lora_path and model.base_model_path must be specified for DPO experiment."
            " `path` is the path of saved LoRA weights and `base_model_path` is the path of the base model.")
    if args.dataset.valid_path is not None:
        logger.warning(
            "DPO does not support validation because we can't compute reference logps during training.")

    exp_fn = functools.partial(DPOExperiment,
                               model_path=args.model.path,
                               tokenizer_path=args.model.tokenizer_path,
                               seed=args.seed,
                               total_train_epochs=args.train_epochs,
                               save_freq_steps=args.save_freq,
                               is_sft_pipe=args.is_sft_pipe,
                               is_sft_lora=args.is_sft_lora,
                               base_model_type=args.model.type,
                               sft_lora_path=args.sft_lora_path,
                               dp_size=args.model.parallel.data_parallel_size,
                               mp_size=args.model.parallel.model_parallel_size,
                               pp_size=args.model.parallel.pipeline_parallel_size,
                               use_lora=args.model.lora,
                               lora_scaling=args.model.lora_scaling,
                               lora_dim=args.model.lora_dim,
                               enable_fp16=args.model.enable_fp16,
                               enable_bf16=args.model.enable_bf16,
                               offload_optimizer=args.optimizer.offload,
                               gradient_checkpointing=args.model.gradient_checkpointing,
                               max_pairs_per_prompt=args.dataset.max_pairs_per_prompt,
                               max_seqlen=args.dataset.max_seqlen,
                               train_dataset_path=args.dataset.train_path,
                               train_tokens_per_batch=args.dataset.train_tokens_per_batch,
                               lr=args.optimizer.lr,
                               weight_decay=args.optimizer.weight_decay,
                               adam_betas=(args.optimizer.beta1, args.optimizer.beta2),
                               lr_scheduler_type=args.optimizer.lr_scheduler_type,
                               warmup_proportion=args.optimizer.warmup_steps_proportion,
                               adam_eps=args.optimizer.eps,
                               min_lr_ratio=args.optimizer.min_lr_ratio,
                               zero_stage=args.optimizer.zero_stage,
                               beta=args.beta,
                               partition_method=args.model.partition_method)

    os.makedirs(os.path.dirname(QUICKSTART_EXPR_CACHE_PATH), exist_ok=True)
    with open(QUICKSTART_EXPR_CACHE_PATH, "wb") as f:
        pickle.dump((exp_name, exp_fn), f)
    api.config.register_experiment(exp_name, exp_fn)

    try:
        main_start(_MainStartArgs(exp_name, trial_name, mode, debug=True, trace=args.trace))
    except Exception as e:
        main_stop(_MainStartArgs(exp_name, trial_name, mode, debug=True, trace=args.trace))
        logger.warning("Exception occurred. Stopping all workers.")
        raise e


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

    # Disable hydra logging.
    if not any("hydra/job_logging=disabled" in x for x in sys.argv):
        sys.argv += ["hydra/job_logging=disabled"]

    if any("experiment_name=" in x for x in sys.argv):
        experiment_name = next(x for x in sys.argv if "experiment_name=" in x).split("=")[1]
        if "_" in experiment_name:
            raise RuntimeError("experiment_name should not contain `_`.")
    else:
        experiment_name = f"quickstart-{args.cmd}"
        sys.argv += [f"experiment_name={experiment_name}"]

    if "--multirun" not in sys.argv and "hydra.mode=MULTIRUN" not in sys.argv and "-m" not in sys.argv:
        # non-multirun mode, add trial_name and hydra run dir
        if any("trial_name=" in x for x in sys.argv):
            trial_name = next(x for x in sys.argv if "trial_name=" in x).split("=")[1]
        else:
            trial_name = f"run{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
            sys.argv += [f"trial_name={trial_name}"]
        if "_" in trial_name:
            raise RuntimeError("trial_name should not contain `_`.")
        sys.argv += [
            f"hydra.run.dir={cluster_spec.fileroot}/logs/{getpass.getuser()}/"
            f"{experiment_name}/{trial_name}/hydra-outputs/"
        ]
    else:
        # Multi-run mode, add hydra sweep dir. Trial names will be automatically generated.
        sys.argv += [
            f"hydra.sweep.dir={cluster_spec.fileroot}/logs/{getpass.getuser()}/"
            f"{experiment_name}/hydra-sweep-outputs/"
            f"{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        ]

    sys.argv.pop(1)
    args.func()


if __name__ == "__main__":
    main()
