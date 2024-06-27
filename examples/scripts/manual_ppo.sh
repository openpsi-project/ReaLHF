SFT_MODEL_PATH=/lustre/public/pretrained_model_weights/opt-125m/
# NOTE: this example will initialize a reward model from a pretrained LLM with a random projection head.
# This reward is meaningless and training with this reward will probably not work. For test only.
# Loading pre-trained models as critics also requires init_critic_from_actor=True.
RW_MODEL_PATH=/lustre/public/pretrained_model_weights/opt-125m/
python3 -m realhf.apps.quickstart ppo \
    experiment_name=quickstart-ppo trial_name=manual \
    n_nodes=1 \
    total_train_epochs=1 \
    allocation_mode=manual \
    save_freq_steps=null \
    actor.type._class=opt \
    actor.type.size=7 \
    actor.type.is_critic=False \
    actor.path=$SFT_MODEL_PATH \
    actor.gradient_checkpointing=True \
    critic.type._class=opt \
    critic.type.size=7 \
    critic.type.is_critic=True \
    critic.path=$RW_MODEL_PATH \
    critic.gradient_checkpointing=True \
    critic.init_critic_from_actor=True \
    ref.type._class=opt \
    ref.type.size=7 \
    ref.type.is_critic=False \
    ref.path=$SFT_MODEL_PATH \
    rew.type._class=opt \
    rew.type.size=7 \
    rew.type.is_critic=True \
    rew.path=$RW_MODEL_PATH \
    rew.init_critic_from_actor=True \
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
    ppo.top_p=0.9 ppo.top_k=1000 \
    actor_train.device_mesh=\'NODE01:0,1,2,3\' \
    actor_train.parallel.data_parallel_size=2 \
    actor_train.parallel.model_parallel_size=2 \
    actor_train.parallel.pipeline_parallel_size=1 \
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
    rew_inf.parallel.data_parallel_size=2 \
    rew_inf.parallel.model_parallel_size=1 \
    rew_inf.parallel.pipeline_parallel_size=1 \
    ref_inf.device_mesh=\'NODE01:4,5,6,7\' \
    ref_inf.parallel.data_parallel_size=2 \
    ref_inf.parallel.model_parallel_size=2 \
    ref_inf.parallel.pipeline_parallel_size=1
