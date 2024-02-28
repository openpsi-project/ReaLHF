# /lustre/public/pretrained_model_weights/sharded/Llama-2-70b-hf_8pp_3s

actor_pp_size=2
base_model_path="/lustre/public/pretrained_model_weights/Llama-2-7b-hf"
actor_model_path="/lustre/public/pretrained_model_weights/sharded_new/Llama-2-7b-hf_2pp_2mp/"
critic_model_path="/lustre/public/pretrained_model_weights/Llama-2-7b-hf"

python3 -m apps.quickstart ppo experiment_name=quickstart-debug trial_name=20240227 \
    trace=False \
    actor.type=llama \
    actor.path=$actor_model_path \
    actor.base_model_path=$actor_model_path \
    actor.parallel.pipeline_parallel_size=2 \
    actor.parallel.model_parallel_size=2 \
    actor.parallel.data_parallel_size=1 \
    actor.gradient_checkpointing=True \
    actor.parallel.use_sequence_parallel=False \
    actor.enable_async_p2p=True \
    actor.optimizer.offload=True \
    critic.type=llama \
    critic.path=$critic_model_path \
    critic.base_model_path=$critic_model_path \
    critic.parallel.pipeline_parallel_size=1 \
    critic.parallel.model_parallel_size=1 \
    critic.parallel.data_parallel_size=1 \
    critic.gradient_checkpointing=True \
    critic.parallel.use_sequence_parallel=False \
    critic.optimizer.offload=True \
    ref.type=llama \
    ref.path=$actor_model_path  \
    ref.base_model_path=$actor_model_path \
    ref.parallel.pipeline_parallel_size=1 \
    rew.type=llama \
    rew.path=$critic_model_path \
    rew.base_model_path=$critic_model_path \
    rew.parallel.data_parallel_size=1 \
    save_freq_steps=null \
    dataset.max_prompt_len=256 \
    dataset.n_tokens_per_batch=8192 \
    ppo.max_new_tokens=256 \
    ppo.ppo_n_minibatches=4 \
    ppo.adv_norm=True ppo.value_norm=True \
    ppo.top_p=0.9 ppo.top_k=1024 ppo.actor_as_critic=True 

# SFT
# python3 -m apps.quickstart sft experiment_name=debug-quickstart-sft trial_name=pp2mp2 \
#     model.type=llama \
#     model.path=/lustre/public/pretrained_model_weights/sharded_new/Llama-2-7b-hf_1pp_8mp \
#     model.parallel.pipeline_parallel_size=1 \
#     model.parallel.model_parallel_size=8 \
#     model.parallel.data_parallel_size=1 \
#     model.parallel.use_sequence_parallel=True \
#     model.gradient_checkpointing=True \
#     model.optimizer.offload=False \
#     model.optimizer.lr=2e-5 \
#     save_freq_steps=null \
#     eval_freq_epochs=null \
#     total_train_epochs=2 \
#     dataset.max_seqlen=4096 \
#     dataset.train_tokens_per_batch=65536 \
#     dataset.valid_tokens_per_batch=65536

# RW
# python3 -m apps.quickstart rw experiment_name=debug-quickstart-rw trial_name=dp-pp \
#     model.type=llama \
#     model.path=/lustre/aigc/llm/checkpoints/fw/debug-quickstart-sft/pp2mp2/default/epoch0epochstep9globalstep9/ \
#     model.base_model_path=/lustre/public/pretrained_model_weights/Llama-2-7b-hf \
#     model.parallel.pipeline_parallel_size=2 \
#     model.parallel.model_parallel_size=2 \
#     model.parallel.data_parallel_size=1 \
#     model.gradient_checkpointing=True \
#     model.optimizer.lr=2e-5 \
#     save_freq_steps=5 \
#     eval_freq_epochs=1 \
#     total_train_epochs=1 \
#     dataset.max_seqlen=4096 \
#     dataset.train_tokens_per_batch=65536 \
#     dataset.valid_tokens_per_batch=65536

# normal DP DPO
# python3 -m apps.quickstart dpo experiment_name=debug-quickstart-dpo trial_name=test \
#     actor.type=llama \
#     actor.path=/lustre/aigc/llm/checkpoints/fw/debug-quickstart-sft/pp2mp2/default/epoch0epochstep9globalstep9/ \
#     actor.base_model_path=/lustre/public/pretrained_model_weights/Llama-2-7b-hf \
#     actor.parallel.pipeline_parallel_size=1 \
#     actor.parallel.model_parallel_size=2 \
#     actor.parallel.data_parallel_size=2 \
#     actor.gradient_checkpointing=True \
#     actor.parallel.use_sequence_parallel=True \
#     actor.optimizer.lr=2e-5 \
#     ref.type=llama \
#     ref.path=/lustre/aigc/llm/checkpoints/fw/debug-quickstart-sft/pp2mp1/default/epoch0epochstep9globalstep9/ \
#     ref.base_model_path=/lustre/public/pretrained_model_weights/Llama-2-7b-hf \
#     ref.parallel.pipeline_parallel_size=1 \
#     ref.parallel.model_parallel_size=1 \
#     ref.parallel.data_parallel_size=2 \
#     save_freq_steps=10 \
#     dataset.max_seqlen=2048 \
#     dataset.train_tokens_per_batch=32768 \
#     dataset.valid_tokens_per_batch=32768

# ppo.use_stream_pipe_engine=True \

# PP PPO
# python3 -m apps.quickstart ppo experiment_name=debug-quickstart-ppo trial_name=dp-pp \
#     is_sft_pipe=True \
#     is_rew_pipe=True \
#     actor.type=llama \
#     actor.path=/lustre/aigc/llm/checkpoints/fw/stream-sft-llama7b-dp2-pp2-mp2/run20231217/default/epoch1step71/ \
#     actor.base_model_path=/lustre/public/pretrained_model_weights/sharded/Llama-2-7b-hf_2pp_2mp_3s/ \
#     actor.parallel.pipeline_parallel_size=2 \
#     actor.parallel.model_parallel_size=2 \
#     actor.parallel.data_parallel_size=2 \
#     actor.gradient_checkpointing=True \
#     actor.parallel.use_sequence_parallel=True \
#     critic.type=llama \
#     critic.path=/lustre/aigc/llm/checkpoints/fw/stream-rw-llama7b-dp2-pp2-mp2/run20231217/default/epoch0step101/ \
#     critic.base_model_path=/lustre/public/pretrained_model_weights/sharded/Llama-2-7b-hf_2pp_2mp_3s/ \
#     critic.parallel.pipeline_parallel_size=2 \
#     critic.parallel.model_parallel_size=2 \
#     critic.parallel.data_parallel_size=2 \
#     critic.gradient_checkpointing=True \
#     critic.parallel.use_sequence_parallel=False \
#     ref.type=llama \
#     ref.path=/lustre/aigc/llm/checkpoints/fw/stream-sft-llama7b-dp2-pp2-mp2/run20231217/default/epoch1step71/ \
#     ref.base_model_path=/lustre/public/pretrained_model_weights/sharded/Llama-2-7b-hf_2pp_2mp_3s/ \
#     rew.type=llama \
#     rew.path=/lustre/aigc/llm/checkpoints/fw/stream-rw-llama7b-dp2-pp2-mp2/run20231217/default/epoch0step101/ \
#     rew.base_model_path=/lustre/public/pretrained_model_weights/sharded/Llama-2-7b-hf_2pp_2mp_3s/ \
#     save_freq=5 \
#     dataset.max_prompt_len=1024 \
#     dataset.batch_size=256 \
#     ppo_n_minibatches=8 \
#     max_new_tokens=1024 \
#     adv_norm=True value_norm=True \
#     top_p=1.0 top_k=1000000

# python3 -m apps.quickstart ppo experiment_name=debug-quickstart-ppo trial_name=pp2 \
#     is_sft_pipe=True \
#     is_rew_pipe=True \
#     actor.type=llama \
#     actor.path=/lustre/aigc/llm/checkpoints/fw/llama7b-2pp-sft/ \
#     actor.base_model_path=/lustre/public/pretrained_model_weights/sharded/Llama-2-7b-hf_2pp_3s \
#     actor.parallel.pipeline_parallel_size=2 \
#     actor.parallel.model_parallel_size=1 \
#     actor.parallel.data_parallel_size=1 \
#     actor_optimizer.offload=True \
#     actor.gradient_checkpointing=True \
#     critic.type=llama \
#     critic_optimizer.offload=True \
#     critic.path=/lustre/aigc/llm/checkpoints/fw/llama7b-2pp-rw \
#     critic.base_model_path=/lustre/public/pretrained_model_weights/sharded/Llama-2-7b-hf_2pp_3s \
#     critic.parallel.pipeline_parallel_size=2 \
#     critic.parallel.model_parallel_size=1 \
#     critic.parallel.data_parallel_size=1 \
#     critic.gradient_checkpointing=True \
#     ref.type=llama \
#     ref.path=/lustre/aigc/llm/checkpoints/fw/llama7b-2pp-sft/ \
#     ref.base_model_path=/lustre/public/pretrained_model_weights/sharded/Llama-2-7b-hf_2pp_3s \
#     rew.type=llama \
#     rew.path=/lustre/aigc/llm/checkpoints/fw/llama7b-2pp-rw \
#     rew.base_model_path=/lustre/public/pretrained_model_weights/sharded/Llama-2-7b-hf_2pp_3s \
#     save_freq=null \
#     dataset.max_prompt_len=256 \
#     dataset.batch_size=128 \
#     ppo_n_minibatches=4 \
#     max_new_tokens=256 \
#     adv_norm=True value_norm=True \
#     top_p=1.0 top_k=1000000
