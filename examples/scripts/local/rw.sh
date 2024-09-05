# MODEL_FAMILY specifies how the pretrained checkpoint is loaded, e.g., as a LLaMA model or a GPT model.
MODEL_FAMILY=llama

# PRETRAINED_PATH is the HuggingFace checkpoint or the saved SFT checkpoint.
# The latter is the common practice.
# ReaL saves checkpoints with the same format as HuggingFace,
# so you don't need to convert or split checkpoints explicitly.
# HF pretrained checkpoint
PRETRAINED_PATH=/lustre/public/pretrained_model_weights/Llama-2-7b-hf
# or SFT checkpoint
PRETRAINED_PATH=/lustre/aigc/llm/checkpoints/fw/quickstart-sft/llama-local-manual/default/epoch7epochstep5globalstep50/

# Option 1: The experiment runs locally with subprocesses.
MODE=local
# Option 2: The experiment runs in a Ray cluster
# MODE=ray
# Option 3: The experiment runs in a SLURM + pyxis cluster
# Using the slurm mode requires a cluster spec file
# and setting CLUSTER_SPEC_PATH to the path of it.
# MODE=slurm

# `experiment_name` and `trial_name` can be arbitrary.
# Logs and saved checkpoints will be indexed by them.
EXP_NAME=quickstart-rw
TRIAL_NAME=$MODEL_FAMILY-$MODE-manual

# We use the "manual" allocation mode here to manually specify the parallelism strategy,
# which is pipeline=2, tensor-model=2, and data=2, using in total of 8 GPUs.

# The `rw` subcommand specifies that this is a reward modeling experiment.
# The reward modeling experiment converges very fast, so we set a smaller
# `total_train_epochs` and `save_freq_steps` for demonstration.
# Note that we set `model.type.is_critic=True` to initialize a reward model from the LLM
# by re-initializing the LM head.
python3 -m realhf.apps.quickstart rw \
    mode=$MODE \
    experiment_name=$EXP_NAME \
    trial_name=$TRIAL_NAME \
    exp_ctrl.total_train_epochs=2 \
    exp_ctrl.save_freq_steps=10 \
    exp_ctrl.eval_freq_epochs=1 \
    model.optimizer.type=adam \
    model.optimizer.lr_scheduler_type=cosine \
    model.optimizer.lr=1e-5 \
    model.optimizer.warmup_steps_proportion=0.02 \
    model.type._class=$MODEL_FAMILY \
    model.type.is_critic=True \
    model.path=$PRETRAINED_PATH \
    dataset.train_path=.data/rm_paired-train.jsonl \
    dataset.valid_path=.data/rm_paired-valid.jsonl \
    dataset.max_seqlen=1024 \
    dataset.train_bs_n_seqs=512 \
    dataset.valid_bs_n_seqs=512 \
    allocation_mode=manual \
    n_nodes=1 \
    allocation.parallel.pipeline_parallel_size=2 \
    allocation.parallel.model_parallel_size=2 \
    allocation.parallel.data_parallel_size=2 \
    allocation.parallel.use_sequence_parallel=True