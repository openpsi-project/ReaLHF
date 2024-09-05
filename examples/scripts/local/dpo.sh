# MODEL_FAMILY specifies how the pretrained checkpoint is loaded, e.g., as a LLaMA model or a GPT model.
MODEL_FAMILY=gpt2

# PRETRAINED_PATH is the HuggingFace checkpoint or the saved SFT checkpoint.
# The latter is the common practice.
# ReaL saves checkpoints with the same format as HuggingFace,
# so you don't need to convert or split checkpoints explicitly.
PRETRAINED_PATH=/lustre/aigc/llm/checkpoints/fw/quickstart-sft/$MODEL_FAMILY/default/epoch7epochstep5globalstep50/

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
EXP_NAME=quickstart-dpo
TRIAL_NAME=$MODEL_FAMILY-$MODE-manual

# We use the "manual" allocation mode here to manually specify the parallelism strategy of training
# and inference. The parallel strategy for training prefers tensor-model parallelism while the
# inference prefers pipeline parallelism, which are more efficient for their corresponding workloads.

# The `dpo` subcommand specifies that this is a DPO experiment.
# The `save_freq_steps` is set to `null` to disable saving checkpoints.
# Enable it if you want to save checkpoints.
python3 -m realhf.apps.quickstart dpo \
    mode=$MODE \
    experiment_name=$EXP_NAME \
    trial_name=$TRIAL_NAME \
    exp_ctrl.total_train_epochs=2 \
    exp_ctrl.save_freq_steps=null \
    n_nodes=1 \
    allocation_mode=manual \
    actor.type._class=$MODEL_FAMILY \
    actor.path=$PRETRAINED_PATH \
    actor_train.parallel.pipeline_parallel_size=2 \
    actor_train.parallel.model_parallel_size=1 \
    actor_train.parallel.data_parallel_size=4 \
    actor_train.parallel.use_sequence_parallel=True \
    ref.type._class=$MODEL_FAMILY \
    ref.path=$PRETRAINED_PATH \
    ref_inf.parallel.pipeline_parallel_size=4 \
    ref_inf.parallel.model_parallel_size=1 \
    ref_inf.parallel.data_parallel_size=2 \
    ref_inf.parallel.use_sequence_parallel=True \
    dataset.train_path=.data/rm_paired-train.jsonl \
    dataset.max_pairs_per_prompt=2 \
    dataset.max_seqlen=1024 \
    dataset.train_bs_n_seqs=512 \
    dataset.valid_bs_n_seqs=512