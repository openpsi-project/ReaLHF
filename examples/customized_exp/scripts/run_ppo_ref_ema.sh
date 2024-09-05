MODEL_FAMILY=gpt2

SFT_MODEL_PATH=/lustre/aigc/llm/checkpoints/fw/quickstart-sft/$MODEL_FAMILY/default/epoch7epochstep5globalstep50/
RW_MODEL_PATH=/lustre/aigc/llm/checkpoints/fw/quickstart-rw/$MODEL_FAMILY/default/epoch1epochstep15globalstep15/

MODE=local
EXP_NAME=quickstart-ppo
TRIAL_NAME=$MODEL_FAMILY-$MODE-manual

unset CLUSTER_SPEC_PATH
python3 examples/customized_exp/ppo_ref_ema.py ppo-ref-ema \
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
