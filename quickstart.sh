# SFT_MODEL_PATH=/lustre/aigc/llm/checkpoints/fw/quickstart-sft-debug/20240603-1/default/epoch7epochstep5globalstep50/

# python3 -m reallm.apps.quickstart sft \
#     experiment_name=quickstart-sft-debug trial_name=20240528 \
#     allocation_mode=manual \
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
#     allocation.parallel.pipeline_parallel_size=1 \
#     allocation.parallel.model_parallel_size=2 \
#     allocation.parallel.data_parallel_size=4 \
#     dataset.train_bs_n_seqs=512 \
#     dataset.valid_bs_n_seqs=512

# SFT_MODEL_PATH=/lustre/aigc/llm/checkpoints/fw/quickstart-sft-debug/20240603-1/default/epoch7epochstep11globalstep50/
# SFT_MODEL_PATH=/lustre/public/pretrained_model_weights/testOnly/llama-2-16l/
# python3 -m reallm.apps.quickstart rw \
#     experiment_name=quickstart-rw-debug trial_name=20240603-1 \
#     mode=local allocation_mode=manual \
#     total_train_epochs=1 \
#     save_freq_steps=5 eval_freq_epochs=1 \
#     model.type._class=llama \
#     model.type.size=7 \
#     model.type.is_critic=True \
#     model.path=$SFT_MODEL_PATH \
#     allocation.parallel.pipeline_parallel_size=2 \
#     allocation.parallel.model_parallel_size=2 \
#     allocation.parallel.data_parallel_size=2 \
#     allocation.parallel.use_sequence_parallel=True \
#     model.gradient_checkpointing=True \
#     model.optimizer.lr_scheduler_type=cosine \
#     model.optimizer.lr=1e-5 \
#     model.optimizer.warmup_steps_proportion=0.0 \
#     dataset.train_path=/lustre/fw/datasets/imdb/rl/rm_paired-train.jsonl \
#     dataset.valid_path=/lustre/fw/datasets/imdb/rl/rm_paired-valid.jsonl \
#     dataset.max_pairs_per_prompt=2 \
#     dataset.max_seqlen=1024 \
#     dataset.train_bs_n_seqs=512 \
#     dataset.valid_bs_n_seqs=512

# python3 -m reallm.apps.quickstart dpo experiment_name=quickstart-dpo-debug trial_name=20240605-0 \
#     allocation_mode=manual \
#     mode=local \
#     total_train_epochs=2 \
#     save_freq_steps=5 \
#     actor.type._class=llama \
#     actor.type.size=7 \
#     actor.type.is_critic=False \
#     actor.path=$SFT_MODEL_PATH \
#     actor_train.parallel.pipeline_parallel_size=1 \
#     actor_train.parallel.model_parallel_size=4 \
#     actor_train.parallel.data_parallel_size=2 \
#     actor_train.parallel.use_sequence_parallel=True \
#     ref.type._class=llama \
#     ref.type.size=7 \
#     ref.type.is_critic=False \
#     ref.path=$SFT_MODEL_PATH \
#     ref_inf.parallel.pipeline_parallel_size=1 \
#     ref_inf.parallel.model_parallel_size=2 \
#     ref_inf.parallel.data_parallel_size=4 \
#     ref_inf.parallel.use_sequence_parallel=True \
#     dataset.train_path=/lustre/fw/datasets/imdb/rl/rm_paired-train.jsonl \
#     dataset.max_pairs_per_prompt=2 \
#     dataset.max_seqlen=256 \
#     dataset.train_bs_n_seqs=256 \
#     dataset.valid_bs_n_seqs=256
SFT_MODEL_PATH=/lustre/aigc/llm/checkpoints/fw/quickstart-sft-debug/20240603-1/default/epoch7epochstep11globalstep50/
RW_MODEL_PATH=/lustre/aigc/llm/checkpoints/fw/quickstart-rw-debug/20240603-1/default/epoch1epochstep15globalstep15/
CLUSTER_SPEC_PATH=/lustre/aigc/llm/cluster/qh.json python3 -m reallm.apps.quickstart ppo \
    experiment_name=debug-quickstart-ppo trial_name=20240617-1 \
    total_train_epochs=4 \
    allocation_mode=heuristic \
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
    dataset.path=/lustre/fw/datasets/imdb/rl/ppo_prompt.jsonl \
    dataset.max_prompt_len=256 \
    dataset.train_bs_n_seqs=128 \
    ppo.max_new_tokens=256 \
    ppo.min_new_tokens=256 \
    ppo.ppo_n_minibatches=4 \
    ppo.kl_ctl=0.1 \
    ppo.force_no_logits_mask=False \
    ppo.value_eps_clip=0.2 \
    ppo.reward_output_scaling=10.0 \
    ppo.adv_norm=True ppo.value_norm=True \
    ppo.top_p=0.9 ppo.top_k=1000 
    # actor_train.parallel.model_parallel_size=8 \
    # actor_train.parallel.pipeline_parallel_size=1 \
    # actor_train.parallel.data_parallel_size=1 \
    # actor_gen.parallel.model_parallel_size=1 \
    # actor_gen.parallel.pipeline_parallel_size=2 \
    # actor_gen.parallel.data_parallel_size=4 \
    # critic_train.parallel.model_parallel_size=8 \
    # critic_train.parallel.pipeline_parallel_size=1 \
    # critic_train.parallel.data_parallel_size=1 \
    # critic_inf.parallel.model_parallel_size=4 \
    # critic_inf.parallel.pipeline_parallel_size=2 \
    # critic_inf.parallel.data_parallel_size=1 \
    # ref_inf.parallel.model_parallel_size=4 \
    # ref_inf.parallel.pipeline_parallel_size=2 \
    # ref_inf.parallel.data_parallel_size=1 \
    # rew_inf.parallel.model_parallel_size=1 \
    # rew_inf.parallel.pipeline_parallel_size=4 \
    # rew_inf.parallel.data_parallel_size=2 

