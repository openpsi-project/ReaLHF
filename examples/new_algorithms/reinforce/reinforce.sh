# MODEL_FAMILY specifies how the pretrained checkpoint is loaded, e.g., as a LLaMA model or a GPT model.
# You can specify different model families for the SFT and the RW model, but you need to
# re-tokenize the sequences if necessary.
MODEL_FAMILY=gpt2

# SFT_MODEL_PATH and RW_MODEL_PATH are the saved SFT and RW checkpoints.
# ReaL saves checkpoints with the same format as HuggingFace,
# so you don't need to convert or split checkpoints explicitly.
# You can also directly use the pre-trained HuggingFace checkpoint, but this
# will not ensure the optimal algorithm performance.
SFT_MODEL_PATH=/lustre/aigc/llm/checkpoints/fw/quickstart-sft/$MODEL_FAMILY/default/epoch7epochstep5globalstep50/
RW_MODEL_PATH=/lustre/aigc/llm/checkpoints/fw/quickstart-rw/$MODEL_FAMILY/default/epoch1epochstep15globalstep15/

# Option 1: The experiment runs locally with subprocesses.
# MODE=local
# Option 2: The experiment runs in a Ray cluster
MODE=local
# Option 3: The experiment runs in a SLURM + pyxis cluster
# Using the slurm mode requires a cluster spec file
# and setting CLUSTER_SPEC_PATH to the path of it.
# MODE=slurm

# `experiment_name` and `trial_name` can be arbitrary.
# Logs and saved checkpoints will be indexed by them.
EXP_NAME=quickstart-reinforce
TRIAL_NAME=$MODEL_FAMILY-$MODE-manual

# When using the "manual" allocation mode, the user should specify the device allocation
# and parallel strategies for each model function calls.
# The number of GPUs is `n_nodes` * `n_gpus_per_node` (not set explictly here, defaults to 8).
# We provide a template in the following command and the user can modify it according to
# the specific model and the available GPUs.

# The following command shows an example of manual allocation on two nodes,
# but it can be modified according to the specific model and the available GPUs.
unset CLUSTER_SPEC_PATH
python3 examples/new_algorithms/reinforce/reinforce_exp.py reinforce \
    mode=$MODE \
    experiment_name=$EXP_NAME \
    trial_name=$TRIAL_NAME \
    exp_ctrl.total_train_epochs=8 \
    exp_ctrl.save_freq_steps=null \
    actor.type._class=$MODEL_FAMILY \
    actor.path=$SFT_MODEL_PATH \
    actor.optimizer.lr=1e-4 \
    actor.optimizer.lr_scheduler_type=constant \
    rew.type._class=$MODEL_FAMILY \
    rew.type.is_critic=True \
    rew.path=$RW_MODEL_PATH \
    dataset.path=.data/ppo_prompt.jsonl \
    dataset.max_prompt_len=128 \
    dataset.train_bs_n_seqs=512 \
    gen.max_new_tokens=512 \
    gen.min_new_tokens=512 \
    gen.use_cuda_graph=True \
    gen.top_p=0.9 gen.top_k=5000 \
    allocation_mode=manual \
    n_nodes=1 \
    nodelist=\'NODE01\' \
    actor_train.device_mesh=\'NODE01:0,1,2,3,4,5,6,7\' \
    actor_train.parallel.data_parallel_size=4 \
    actor_train.parallel.model_parallel_size=1 \
    actor_train.parallel.pipeline_parallel_size=2 \
    sample_gen.device_mesh=\'NODE01:0,1,2,3\' \
    sample_gen.parallel.data_parallel_size=2 \
    sample_gen.parallel.model_parallel_size=1 \
    sample_gen.parallel.pipeline_parallel_size=2 \
    sample_rew_inf.device_mesh=\'NODE01:0,1,2,3\' \
    sample_rew_inf.parallel.data_parallel_size=4 \
    sample_rew_inf.parallel.model_parallel_size=1 \
    sample_rew_inf.parallel.pipeline_parallel_size=1 \
    greedy_gen.device_mesh=\'NODE01:4,5,6,7\' \
    greedy_gen.parallel.data_parallel_size=2 \
    greedy_gen.parallel.model_parallel_size=1 \
    greedy_gen.parallel.pipeline_parallel_size=2 \
    greedy_rew_inf.device_mesh=\'NODE01:4,5,6,7\' \
    greedy_rew_inf.parallel.data_parallel_size=4 \
    greedy_rew_inf.parallel.model_parallel_size=1 \
    greedy_rew_inf.parallel.pipeline_parallel_size=1
