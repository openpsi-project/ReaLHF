# python3 -m reallm.apps.quickstart sft experiment_name=quickstart-sft-debug trial_name=20240523 \
#     total_train_epochs=2 \
#     save_freq_steps=10 eval_freq_epochs=1 \
#     model.type._class=llama \
#     model.type.size=7 \
#     model.type.is_critic=False \
#     model.path=/lustre/public/pretrained_model_weights/Llama-2-7b-hf \
#     model.parallel.pipeline_parallel_size=2 \
#     model.parallel.model_parallel_size=2 \
#     model.parallel.data_parallel_size=2 \
#     model.gradient_checkpointing=True \
#     model.parallel.use_sequence_parallel=False \
#     model.optimizer.offload=False \
#     model.optimizer.type=adam \
#     dataset.train_path=/lustre/fw/datasets/imdb/rl/sft_pos-train.jsonl \
#     dataset.valid_path=/lustre/fw/datasets/imdb/rl/sft_pos-valid.jsonl \
#     dataset.max_seqlen=1024 \
#     dataset.train_tokens_per_batch=131072 \
#     dataset.valid_tokens_per_batch=131072

# python3 -m reallm.apps.quickstart rw experiment_name=quickstart-rw-debug trial_name=20240523 \
#     total_train_epochs=2 \
#     save_freq_steps=10 eval_freq_epochs=1 \
#     model.type._class=llama \
#     model.type.size=7 \
#     model.type.is_critic=True \
#     model.path=/lustre/aigc/llm/checkpoints/fw/quickstart-sft-debug/20240523/default/epoch1epochstep10globalstep10/ \
#     model.parallel.pipeline_parallel_size=2 \
#     model.parallel.model_parallel_size=2 \
#     model.parallel.data_parallel_size=1 \
#     model.gradient_checkpointing=True \
#     model.parallel.use_sequence_parallel=False \
#     model.optimizer.offload=False \
#     model.optimizer.type=adam \
#     dataset.train_path=/lustre/fw/datasets/imdb/rl/rm_paired-train-lite.jsonl \
#     dataset.valid_path=/lustre/fw/datasets/imdb/rl/rm_paired-valid-lite.jsonl \
#     dataset.max_pairs_per_prompt=2 \
#     dataset.max_seqlen=512 \
#     dataset.train_tokens_per_batch=65536 \
#     dataset.valid_tokens_per_batch=65536

python3 -m reallm.apps.quickstart dpo experiment_name=quickstart-dpo-debug trial_name=20240523 \
    total_train_epochs=2 \
    save_freq_steps=5 \
    actor.type._class=llama \
    actor.type.size=7 \
    actor.type.is_critic=False \
    actor.path=/lustre/aigc/llm/checkpoints/fw/quickstart-sft-debug/20240523/default/epoch1epochstep10globalstep10/ \
    actor.parallel.pipeline_parallel_size=2 \
    actor.parallel.model_parallel_size=2 \
    actor.parallel.data_parallel_size=2 \
    actor.gradient_checkpointing=True \
    actor.parallel.use_sequence_parallel=False \
    ref.type._class=llama \
    ref.type.size=7 \
    ref.type.is_critic=False \
    ref.parallel.pipeline_parallel_size=4 \
    ref.parallel.model_parallel_size=1 \
    ref.parallel.data_parallel_size=2 \
    ref.path=/lustre/aigc/llm/checkpoints/fw/quickstart-sft-debug/20240523/default/epoch1epochstep10globalstep10/ \
    dataset.train_path=/lustre/fw/datasets/imdb/rl/rm_paired-train-lite.jsonl \
    dataset.max_pairs_per_prompt=2 \
    dataset.max_seqlen=512 \
    dataset.train_tokens_per_batch=65536 \
    dataset.valid_tokens_per_batch=65536

# python3 -m reallm.apps.quickstart sft experiment_name=quickstart-debug trial_name=20240522 \
#     trace=False \
#     actor.type=llama \
#     actor.path=$model_path \
#     actor.parallel.pipeline_parallel_size=2 \
#     actor.parallel.model_parallel_size=2 \
#     actor.parallel.data_parallel_size=1 \
#     actor.gradient_checkpointing=True \
#     actor.parallel.use_sequence_parallel=False \
#     actor.enable_async_p2p=True \
#     actor.optimizer.offload=False \
#     actor.optimizer.type=adam \
#     critic.type=llama \
#     critic.path=$model_path \
#     critic.parallel.pipeline_parallel_size=4 \
#     critic.parallel.model_parallel_size=1 \
#     critic.parallel.data_parallel_size=1 \
#     critic.gradient_checkpointing=True \
#     critic.parallel.use_sequence_parallel=False \
#     critic.optimizer.offload=False \
#     critic.optimizer.type=adam \
#     ref.type=llama \
#     ref.path=$model_path  \
#     ref.parallel.data_parallel_size=2 \
#     ref.parallel.pipeline_parallel_size=2 \
#     rew.type=llama \
#     rew.path=$model_path \
#     rew.parallel.data_parallel_size=1 \
#     rew.parallel.pipeline_parallel_size=4 \
#     save_freq_steps=null \
#     dataset.max_prompt_len=256 \
#     dataset.n_tokens_per_batch=8192 \
#     actor_per_device_generate_batch_size=8 \
#     actor_per_device_train_batch_size=8 \
#     ppo.max_new_tokens=256 \
#     ppo.min_new_tokens=256 \
#     ppo.ppo_n_minibatches=4 \
#     ppo.adv_norm=True ppo.value_norm=True \
#     ppo.top_p=0.9 ppo.top_k=1024 ppo.actor_as_critic=True \
#     ppo.use_stream_pipe_engine=False
