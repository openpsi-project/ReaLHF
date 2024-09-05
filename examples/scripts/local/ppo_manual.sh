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
MODE=local
# Option 2: The experiment runs in a Ray cluster
# MODE=ray
# Option 3: The experiment runs in a SLURM + pyxis cluster
# Using the slurm mode requires a cluster spec file
# and setting CLUSTER_SPEC_PATH to the path of it.
# MODE=slurm

# `experiment_name` and `trial_name` can be arbitrary.
# Logs and saved checkpoints will be indexed by them.
EXP_NAME=quickstart-ppo
TRIAL_NAME=$MODEL_FAMILY-$MODE-manual

# When using the "manual" allocation mode, the user should specify the device allocation
# and parallel strategies for each model function calls.
# The number of GPUs is `n_nodes` * `n_gpus_per_node` (not set explictly here, defaults to 8).
# We provide a template in the following command and the user can modify it according to
# the specific model and the available GPUs.

# The `ppo` subcommand specifies that this is a PPO experiment.
# The `save_freq_steps` is set to `null` to disable saving checkpoints.
# Enable it if you want to save checkpoints.
# The `ppo` option is used to control the generation and PPO algorithm hyperparameters.
# Note that the performance of PPO is sensitive to the the pre-trained model and hyperparameters.
# It's the user's responsibility to tune them appropriately.
# The allocation of model function calls is specified by a pattern `hostname:gpu_id1,gpu_id2,...`
# for slicing GPUS of a single node, and `hostname1,hostname2` for multiple nodes.
# Only 1, 2, 4, 8 GPUs on a single node or multiple complete nodes (e.g., 16, 24) are supported.
# If the CLUSTER_SPEC_PATH is not set, `hostname`s are NODE01, NODE02, etc, otherwise it's the
# hostname specified in this file. The `gpu_id`s are the GPU indices on the host,
# from 0 to `n_gpus_per_node` (defaults to 8, can be changed) - 1.
# Once allocations are all set, parallel strategies can be specified as long as the world size
# equals to the number of GPUs in the allocation.

# The following command shows an example of manual allocation on two nodes,
# but it can be modified according to the specific model and the available GPUs.
unset CLUSTER_SPEC_PATH
python3 -m realhf.apps.quickstart ppo \
    mode=$MODE \
    experiment_name=$EXP_NAME \
    trial_name=$TRIAL_NAME \
    exp_ctrl.total_train_epochs=1 \
    exp_ctrl.save_freq_steps=null \
    actor.type._class=$MODEL_FAMILY \
    actor.path=$SFT_MODEL_PATH \
    actor.optimizer.lr_scheduler_type=constant \
    actor.optimizer.lr=1e-4 \
    actor.optimizer.warmup_steps_proportion=0.0 \
    critic.type._class=$MODEL_FAMILY \
    critic.type.is_critic=True \
    critic.path=$RW_MODEL_PATH \
    ref.type._class=$MODEL_FAMILY \
    ref.path=$SFT_MODEL_PATH \
    rew.type._class=$MODEL_FAMILY \
    rew.type.is_critic=True \
    rew.path=$RW_MODEL_PATH \
    dataset.path=.data/ppo_prompt.jsonl \
    dataset.max_prompt_len=128 \
    dataset.train_bs_n_seqs=128 \
    ppo.gen.max_new_tokens=512 \
    ppo.gen.min_new_tokens=512 \
    ppo.gen.top_p=0.9 ppo.gen.top_k=1000 \
    ppo.gen.use_cuda_graph=True \
    ppo.ppo_n_minibatches=4 \
    ppo.kl_ctl=0.1 \
    ppo.value_eps_clip=0.2 \
    ppo.reward_output_scaling=10.0 \
    ppo.adv_norm=True ppo.value_norm=True \
    allocation_mode=manual \
    n_nodes=1 \
    nodelist=\'NODE01\' \
    actor_train.device_mesh=\'NODE01:0,1,2,3\' \
    actor_train.parallel.data_parallel_size=2 \
    actor_train.parallel.model_parallel_size=1 \
    actor_train.parallel.pipeline_parallel_size=2 \
    actor_gen.device_mesh=\'NODE01:0,1,2,3,4,5,6,7\' \
    actor_gen.parallel.data_parallel_size=4 \
    actor_gen.parallel.model_parallel_size=1 \
    actor_gen.parallel.pipeline_parallel_size=2 \
    critic_train.device_mesh=\'NODE01:4,5,6,7\' \
    critic_train.parallel.data_parallel_size=2 \
    critic_train.parallel.model_parallel_size=1 \
    critic_train.parallel.pipeline_parallel_size=2 \
    critic_inf.device_mesh=\'NODE01:0,1\' \
    critic_inf.parallel.data_parallel_size=2 \
    critic_inf.parallel.model_parallel_size=1 \
    critic_inf.parallel.pipeline_parallel_size=1 \
    rew_inf.device_mesh=\'NODE01:2,3\' \
    rew_inf.parallel.data_parallel_size=1 \
    rew_inf.parallel.model_parallel_size=1 \
    rew_inf.parallel.pipeline_parallel_size=2 \
    ref_inf.device_mesh=\'NODE01:4,5,6,7\' \
    ref_inf.parallel.data_parallel_size=1 \
    ref_inf.parallel.model_parallel_size=1 \
    ref_inf.parallel.pipeline_parallel_size=4
