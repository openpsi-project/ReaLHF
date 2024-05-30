# python3 -m reallm.apps.quickstart sft experiment_name=quickstart-sft-debug trial_name=20240523-1 \
#     total_train_epochs=8 \
#     save_freq_steps=50 eval_freq_epochs=1 \
#     model.type._class=llama \
#     model.type.size=7 \
#     model.type.is_critic=False \
#     model.path=/lustre/public/pretrained_model_weights/Llama-2-7b-hf \
#     model.parallel.pipeline_parallel_size=2 \
#     model.parallel.model_parallel_size=2 \
#     model.parallel.data_parallel_size=2 \
#     model.gradient_checkpointing=True \
#     model.parallel.use_sequence_parallel=True \
#     model.optimizer.offload=False \
#     model.optimizer.type=adam \
#     dataset.train_path=/lustre/fw/datasets/imdb/rl/sft_pos-train.jsonl \
#     dataset.valid_path=/lustre/fw/datasets/imdb/rl/sft_pos-valid.jsonl \
#     dataset.max_seqlen=1024 \
#     dataset.train_bs_n_seqs=512 \
#     dataset.valid_bs_n_seqs=512

# python3 -m reallm.apps.quickstart rw experiment_name=quickstart-rw-debug trial_name=20240523 \
#     total_train_epochs=2 \
#     save_freq_steps=20 eval_freq_epochs=1 \
#     model.type._class=llama \
#     model.type.size=7 \
#     model.type.is_critic=True \
#     model.path=/lustre/aigc/llm/checkpoints/fw/quickstart-sft-debug/20240523/default/epoch7epochstep6globalstep100/ \
#     model.parallel.pipeline_parallel_size=1 \
#     model.parallel.model_parallel_size=2 \
#     model.parallel.data_parallel_size=2 \
#     model.gradient_checkpointing=True \
#     model.parallel.use_sequence_parallel=True \
#     dataset.train_path=/lustre/fw/datasets/imdb/rl/rm_paired-train.jsonl \
#     dataset.valid_path=/lustre/fw/datasets/imdb/rl/rm_paired-valid.jsonl \
#     dataset.max_pairs_per_prompt=2 \
#     dataset.max_seqlen=1024 \
#     dataset.train_bs_n_seqs=128 \
#     dataset.valid_bs_n_seqs=128

# python3 -m reallm.apps.quickstart dpo experiment_name=quickstart-dpo-debug trial_name=20240523 \
#     total_train_epochs=2 \
#     save_freq_steps=5 \
#     actor.type._class=llama \
#     actor.type.size=7 \
#     actor.type.is_critic=False \
#     actor.path=/lustre/aigc/llm/checkpoints/fw/quickstart-sft-debug/20240523/default/epoch7epochstep6globalstep100/ \
#     actor.parallel.pipeline_parallel_size=2 \
#     actor.parallel.model_parallel_size=2 \
#     actor.parallel.data_parallel_size=2 \
#     actor.gradient_checkpointing=True \
#     actor.parallel.use_sequence_parallel=True \
#     ref.type._class=llama \
#     ref.type.size=7 \
#     ref.type.is_critic=False \
#     ref.parallel.pipeline_parallel_size=1 \
#     ref.parallel.model_parallel_size=2 \
#     ref.parallel.data_parallel_size=4 \
#     ref.parallel.use_sequence_parallel=True \
#     ref.path=/lustre/aigc/llm/checkpoints/fw/quickstart-sft-debug/20240523/default/epoch7epochstep6globalstep100/ \
#     dataset.train_path=/lustre/fw/datasets/imdb/rl/rm_paired-train-lite.jsonl \
#     dataset.max_pairs_per_prompt=2 \
#     dataset.max_seqlen=512 \
#     dataset.train_bs_n_seqs=512 \
#     dataset.valid_bs_n_seqs=512

SFT_MODEL_PATH=/lustre/aigc/llm/checkpoints/fw/quickstart-sft-debug/20240523/default/epoch7epochstep6globalstep100/
RW_MODEL_PATH=/lustre/aigc/llm/checkpoints/fw/quickstart-rw-debug/20240523/default/epoch1epochstep80globalstep80/
python3 -m reallm.apps.quickstart ppo experiment_name=quickstart-ppo-debug trial_name=20240523 \
    total_train_epochs=2 \
    save_freq_steps=null \
    global_train_bs=256 \
    global_gen_bs=256 \
    actor.type._class=llama \
    actor.type.size=7 \
    actor.type.is_critic=False \
    actor.path=$SFT_MODEL_PATH \
    actor.parallel.pipeline_parallel_size=2 \
    actor.parallel.model_parallel_size=1 \
    actor.parallel.data_parallel_size=2 \
    actor.gradient_checkpointing=True \
    actor.parallel.use_sequence_parallel=True \
    actor_gen_parallel.pipeline_parallel_size=2 \
    actor_gen_parallel.model_parallel_size=1 \
    actor_gen_parallel.data_parallel_size=4 \
    critic.type._class=llama \
    critic.type.size=7 \
    critic.type.is_critic=True \
    critic.path=$RW_MODEL_PATH \
    critic.parallel.pipeline_parallel_size=1 \
    critic.parallel.model_parallel_size=2 \
    critic.parallel.data_parallel_size=2 \
    critic.gradient_checkpointing=True \
    critic.parallel.use_sequence_parallel=True \
    critic_inf_parallel.pipeline_parallel_size=2 \
    critic_inf_parallel.data_parallel_size=1 \
    ref.type._class=llama \
    ref.type.size=7 \
    ref.type.is_critic=False \
    ref.path=$SFT_MODEL_PATH  \
    ref.parallel.data_parallel_size=2 \
    ref.parallel.pipeline_parallel_size=2 \
    rew.type._class=llama \
    rew.type.size=7 \
    rew.type.is_critic=True \
    rew.path=$RW_MODEL_PATH \
    rew.parallel.data_parallel_size=1 \
    rew.parallel.pipeline_parallel_size=2 \
    dataset.path=/lustre/fw/datasets/imdb/rl/ppo_prompt.jsonl \
    dataset.max_prompt_len=256 \
    dataset.train_bs_n_seqs=512 \
    ppo.max_new_tokens=256 \
    ppo.min_new_tokens=256 \
    ppo.ppo_n_minibatches=4 \
    ppo.adv_norm=True ppo.value_norm=True \
    ppo.top_p=0.9 ppo.top_k=1024
