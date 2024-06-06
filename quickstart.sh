SFT_MODEL_PATH=/lustre/aigc/llm/checkpoints/fw/quickstart-sft-debug/20240603-1/default/epoch7epochstep5globalstep50/
# SFT_MODEL_PATH=/lustre/public/pretrained_model_weights/Llama-2-13b-hf
RW_MODEL_PATH=/lustre/aigc/llm/checkpoints/fw/quickstart-rw-debug/20240603-1/default/epoch1epochstep40globalstep40/

# python3 -m reallm.apps.quickstart sft experiment_name=quickstart-sft-debug trial_name=20240528 \
#     allocation_mode=pipe_model \
#     total_train_epochs=8 \
#     save_freq_steps=50 eval_freq_epochs=1 \
#     model.optimizer.lr_scheduler_type=cosine \
#     model.optimizer.lr=1e-5 \
#     model.optimizer.warmup_steps_proportion=0.02 \
#     model.type._class=llama \
#     model.type.size=7 \
#     model.type.is_critic=False \
#     model.path=/lustre/public/pretrained_model_weights/Llama-2-7b-hf \
#     model.gradient_checkpointing=True \
#     model.optimizer.offload=False \
#     model.optimizer.type=adam \
#     dataset.train_path=/lustre/fw/datasets/imdb/rl/sft_pos-train.jsonl \
#     dataset.valid_path=/lustre/fw/datasets/imdb/rl/sft_pos-valid.jsonl \
#     dataset.max_seqlen=1024 \
#     dataset.train_bs_n_seqs=512 \
#     dataset.valid_bs_n_seqs=512

# python3 -m reallm.apps.quickstart rw experiment_name=quickstart-rw-debug trial_name=20240604-0 \
#     allocation_mode=pipe_model \
#     total_train_epochs=2 \
#     save_freq_steps=20 eval_freq_epochs=1 \
#     model.type._class=llama \
#     model.type.size=7 \
#     model.type.is_critic=True \
#     model.path=$SFT_MODEL_PATH \
#     model.gradient_checkpointing=True \
#     dataset.train_path=/lustre/fw/datasets/imdb/rl/rm_paired-train.jsonl \
#     dataset.valid_path=/lustre/fw/datasets/imdb/rl/rm_paired-valid.jsonl \
#     dataset.max_pairs_per_prompt=2 \
#     dataset.max_seqlen=1024 \
#     dataset.train_bs_n_seqs=512 \
#     dataset.valid_bs_n_seqs=512

# python3 -m reallm.apps.quickstart dpo experiment_name=quickstart-dpo-debug trial_name=20240605-0 \
#     allocation_mode=pipe_model \
#     n_nodes=1 \
#     recover_mode=disabled \
#     total_train_epochs=2 \
#     save_freq_steps=5 \
#     actor.type._class=llama \
#     actor.type.size=7 \
#     actor.type.is_critic=False \
#     actor.path=$SFT_MODEL_PATH \
#     actor.gradient_checkpointing=True \
#     ref.type._class=llama \
#     ref.type.size=7 \
#     ref.type.is_critic=False \
#     ref.path=$SFT_MODEL_PATH \
#     dataset.train_path=/lustre/fw/datasets/imdb/rl/rm_paired-train-lite.jsonl \
#     dataset.max_pairs_per_prompt=2 \
#     dataset.max_seqlen=256 \
#     dataset.train_bs_n_seqs=256 \
#     dataset.valid_bs_n_seqs=256

python3 -m reallm.apps.quickstart ppo experiment_name=debug-recover trial_name=20240606-0 \
    allocation_mode=heuristic \
    n_nodes=1 \
    nodelist=\'QH-com49\'\
    recover_mode=disabled \
    recover_retries=1 \
    save_freq_steps=null \
    actor.type._class=llama \
    actor.type.size=7 \
    actor.type.is_critic=False \
    actor.path=$SFT_MODEL_PATH \
    actor.gradient_checkpointing=True \
    critic.type._class=llama \
    critic.type.size=7 \
    critic.type.is_critic=True \
    critic.path=$RW_MODEL_PATH \
    critic.gradient_checkpointing=True \
    ref.type._class=llama \
    ref.type.size=7 \
    ref.type.is_critic=False \
    ref.path=$SFT_MODEL_PATH \
    rew.type._class=llama \
    rew.type.size=7 \
    rew.type.is_critic=True \
    rew.path=$RW_MODEL_PATH \
    dataset.path=/lustre/meizy/data/antropic-hh/ppo_prompt_only_short.jsonl \
    dataset.max_prompt_len=256 \
    dataset.train_bs_n_seqs=256 \
    ppo.max_new_tokens=256 \
    ppo.min_new_tokens=256 \
    ppo.ppo_n_minibatches=4 \
    ppo.kl_ctl=0.1 \
    ppo.value_eps_clip=0.5 \
    ppo.reward_output_scaling=1.0 \
    ppo.adv_norm=True ppo.value_norm=True \
    ppo.top_p=0.9 ppo.top_k=1024

    # actor_train_allocation.parallel.model_parallel_size=8 \
    # actor_train_allocation.parallel.pipeline_parallel_size=1 \
    # actor_train_allocation.parallel.data_parallel_size=1 \
    # actor_gen_allocation.parallel.model_parallel_size=1 \
    # actor_gen_allocation.parallel.pipeline_parallel_size=4 \
    # actor_gen_allocation.parallel.data_parallel_size=2 \
    # critic_train_allocation.parallel.model_parallel_size=8 \
    # critic_train_allocation.parallel.pipeline_parallel_size=1 \
    # critic_train_allocation.parallel.data_parallel_size=1 \
    # critic_inf_allocation.parallel.model_parallel_size=4 \
    # critic_inf_allocation.parallel.pipeline_parallel_size=2 \
    # critic_inf_allocation.parallel.data_parallel_size=1 \
    # ref_inf_allocation.parallel.model_parallel_size=1 \
    # ref_inf_allocation.parallel.pipeline_parallel_size=4 \
    # ref_inf_allocation.parallel.data_parallel_size=2 \
    # rew_inf_allocation.parallel.model_parallel_size=1 \
    # rew_inf_allocation.parallel.pipeline_parallel_size=4 \
    # rew_inf_allocation.parallel.data_parallel_size=2 

