# /lustre/public/pretrained_model_weights/sharded/Llama-2-70b-hf_8pp_3s

model_path="/lustre/public/pretrained_model_weights/Llama-2-7b-hf"

python3 -m apps.quickstart ppo experiment_name=quickstart-debug trial_name=20240227 \
    trace=False \
    actor.type=llama \
    actor.path=$model_path \
    actor.parallel.pipeline_parallel_size=2 \
    actor.parallel.model_parallel_size=2 \
    actor.parallel.data_parallel_size=1 \
    actor.gradient_checkpointing=True \
    actor.parallel.use_sequence_parallel=False \
    actor.enable_async_p2p=True \
    actor.optimizer.offload=False \
    actor.optimizer.type=adam \
    critic.type=llama \
    critic.path=$model_path \
    critic.parallel.pipeline_parallel_size=4 \
    critic.parallel.model_parallel_size=1 \
    critic.parallel.data_parallel_size=1 \
    critic.gradient_checkpointing=True \
    critic.parallel.use_sequence_parallel=False \
    critic.optimizer.offload=False \
    critic.optimizer.type=adam \
    ref.type=llama \
    ref.path=$model_path  \
    ref.parallel.data_parallel_size=2 \
    ref.parallel.pipeline_parallel_size=2 \
    rew.type=llama \
    rew.path=$model_path \
    rew.parallel.data_parallel_size=1 \
    rew.parallel.pipeline_parallel_size=4 \
    save_freq_steps=null \
    dataset.max_prompt_len=256 \
    dataset.n_tokens_per_batch=8192 \
    actor_per_device_generate_batch_size=8 \
    actor_per_device_train_batch_size=8 \
    ppo.max_new_tokens=256 \
    ppo.min_new_tokens=256 \
    ppo.ppo_n_minibatches=4 \
    ppo.adv_norm=True ppo.value_norm=True \
    ppo.top_p=0.9 ppo.top_k=1024 ppo.actor_as_critic=True \
    ppo.use_stream_pipe_engine=False