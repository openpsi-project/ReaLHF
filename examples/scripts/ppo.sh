SFT_MODEL_PATH=/lustre/aigc/llm/checkpoints/fw/quickstart-sft-debug/20240603-1/default/epoch7epochstep11globalstep50/
RW_MODEL_PATH=/lustre/aigc/llm/checkpoints/fw/quickstart-rw-debug/20240603-1/default/epoch1epochstep15globalstep15/
CLUSTER_SPEC_PATH=/lustre/aigc/llm/cluster/qh.json python3 -m realrlhf.apps.quickstart ppo \
    experiment_name=quickstart-ppo trial_name=release \
    n_nodes=1 \
    total_train_epochs=1 \
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