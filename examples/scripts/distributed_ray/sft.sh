# MODEL_FAMILY specifies how the pretrained checkpoint is loaded, e.g., as a LLaMA model or a GPT model.
MODEL_FAMILY=llama

# PRETRAINED_PATH is the HuggingFace checkpoint.
PRETRAINED_PATH=/lustre/public/pretrained_model_weights/Llama-2-7b-hf

# Option 1: The experiment runs locally with subprocesses.
# MODE=local
# Option 2: The experiment runs in a Ray cluster
MODE=ray
# Option 3: The experiment runs in a SLURM + pyxis cluster
# Using the slurm mode requires a cluster spec file
# and setting CLUSTER_SPEC_PATH to the path of it.
# MODE=slurm

# `experiment_name` and `trial_name` can be arbitrary.
# Logs and saved checkpoints will be indexed by them.
EXP_NAME=quickstart-sft
TRIAL_NAME=$MODEL_FAMILY-$MODE-manual

# We use the "manual" allocation mode here to manually specify the parallelism strategy,
# which is pipeline=2, tensor-model=2, and data=2, using in total of 8 GPUs.

# The `sft` subcommand specifies that this is a supervised fine-tuning experiment.
python3 -m realhf.apps.quickstart sft \
    mode=$MODE \
    experiment_name=$EXP_NAME \
    trial_name=$TRIAL_NAME \
    exp_ctrl.total_train_epochs=8 \
    exp_ctrl.save_freq_steps=50 \
    exp_ctrl.eval_freq_epochs=1 \
    model.optimizer.type=adam \
    model.optimizer.lr_scheduler_type=cosine \
    model.optimizer.lr=1e-5 \
    model.optimizer.warmup_steps_proportion=0.02 \
    model.type._class=$MODEL_FAMILY \
    model.path=$PRETRAINED_PATH \
    dataset.train_path=.data/sft_pos-train.jsonl \
    dataset.valid_path=.data/sft_pos-train.jsonl \
    dataset.max_seqlen=1024 \
    dataset.train_bs_n_seqs=512 \
    dataset.valid_bs_n_seqs=512 \
    allocation_mode=manual \
    n_nodes=4 \
    allocation.parallel.pipeline_parallel_size=2 \
    allocation.parallel.model_parallel_size=4 \
    allocation.parallel.data_parallel_size=4 \
    allocation.parallel.use_sequence_parallel=True