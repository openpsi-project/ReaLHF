MODEL_FAMILY=llama

SFT_MODEL_PATH=/lustre/aigc/llm/checkpoints/fw/quickstart-sft/$MODEL_FAMILY-local-manual/default/epoch7epochstep5globalstep50/
RW_MODEL_PATH=/lustre/aigc/llm/checkpoints/fw/quickstart-rw/$MODEL_FAMILY-ray-manual/default/epoch1epochstep10globalstep10/

MODE=local

EXP_NAME=quickstart-grpo
TRIAL_NAME=$MODEL_FAMILY-$MODE-manual

python3 examples/new_algorithms/grpo/grpo_exp.py grpo \
    mode=$MODE \
    experiment_name=$EXP_NAME \
    trial_name=$TRIAL_NAME \
    exp_ctrl.total_train_epochs=8 \
    exp_ctrl.save_freq_steps=null \
    actor.type._class=$MODEL_FAMILY \
    actor.path=$SFT_MODEL_PATH \
    actor.optimizer.lr=1e-4 \
    actor.optimizer.lr_scheduler_type=constant \
    rew.type._class=$MODEL_FAMILY \
    rew.type.is_critic=True \
    rew.path=$RW_MODEL_PATH \
    ref.type._class=$MODEL_FAMILY \
    ref.path=$SFT_MODEL_PATH \
    dataset.path=.data/ppo_prompt.jsonl \
    dataset.max_prompt_len=128 \
    dataset.train_bs_n_seqs=32 \
    allocation_mode=heuristic \
    n_nodes=1 \
    ppo.gen.max_new_tokens=512 \
    ppo.gen.min_new_tokens=512 \
    ppo.gen.use_cuda_graph=True \
    ppo.gen.top_p=0.9 ppo.gen.top_k=1000 \
    ppo.ppo_n_minibatches=4 \
    ppo.reward_output_scaling=1.0 ppo.adv_norm=False
