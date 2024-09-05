# MODEL_FAMILY specifies how the pretrained checkpoint is loaded, e.g., as a LLaMA model or a GPT model.
# You can specify different model families for the SFT and the RW model, but you need to
# re-tokenize the sequences if necessary.
MODEL_FAMILY=llama

# SFT_MODEL_PATH and RW_MODEL_PATH are the saved SFT and RW checkpoints.
# ReaL saves checkpoints with the same format as HuggingFace,
# so you don't need to convert or split checkpoints explicitly.
# You can also directly use the pre-trained HuggingFace checkpoint, but this
# will not ensure the optimal algorithm performance.
SFT_MODEL_PATH=/lustre/aigc/llm/checkpoints/fw/quickstart-sft/$MODEL_FAMILY-local-manual/default/epoch7epochstep5globalstep50/
RW_MODEL_PATH=/lustre/aigc/llm/checkpoints/fw/quickstart-rw/$MODEL_FAMILY-ray-manual/default/epoch1epochstep10globalstep10/

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
TRIAL_NAME=$MODEL_FAMILY-$MODE-heuristic

# We use the "heuristic" allocation mode here to automatically determine the parallelism strategy
# for each model function call, i.e., actor generation, critic inference, actor train, etc.
# The number of GPUs is `n_nodes` * `n_gpus_per_node` (not set explictly here, defaults to 8).
# ReaL will make full use of these available GPUs to design allocations.
# This does not ensure the optimal throughput, but it is a good starting point.

# The `heuristic` allocation mode is not ensured to run with every model configurations.
# For example, if the vocabulary size is an odd number, the model parallelism may not work.
# In these cases, you can use the `ppo_manual.sh` to specify the parallelism strategy manually.

# The `ppo` subcommand specifies that this is a PPO experiment.
# The `save_freq_steps` is set to `null` to disable saving checkpoints.
# Enable it if you want to save checkpoints.
# The `ppo` option is used to control the generation and PPO algorithm hyperparameters.
# Note that the performance of PPO is sensitive to the the pre-trained model and hyperparameters.
# It's the user's responsibility to tune them appropriately.
python3 -m realhf.apps.quickstart ppo \
    mode=$MODE \
    experiment_name=$EXP_NAME \
    trial_name=$TRIAL_NAME \
    exp_ctrl.total_train_epochs=1 \
    exp_ctrl.save_freq_steps=null \
    n_nodes=1 \
    allocation_mode=heuristic \
    actor.type._class=$MODEL_FAMILY \
    actor.path=$SFT_MODEL_PATH \
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
    dataset.train_bs_n_seqs=1024 \
    ppo.gen.max_new_tokens=512 \
    ppo.gen.min_new_tokens=512 \
    ppo.gen.use_cuda_graph=True \
    ppo.gen.force_no_logits_mask=True \
    ppo.gen.top_p=0.9 ppo.gen.top_k=1000 \
    ppo.ppo_n_minibatches=4 \
    ppo.kl_ctl=0.1 \
    ppo.value_eps_clip=0.2 \
    ppo.reward_output_scaling=1.0 \
    ppo.adv_norm=True ppo.value_norm=True \
    actor_gen.n_mbs=2 \
    actor_train.n_mbs=4 \
    critic_inf.n_mbs=4 \
    critic_train.n_mbs=4 \
    rew_inf.n_mbs=2 \
    ref_inf.n_mbs=8