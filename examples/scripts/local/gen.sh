MODEL_FAMILY=llama
SFT_MODEL_PATH=/lustre/aigc/llm/checkpoints/fw/quickstart-sft/$MODEL_FAMILY-local-manual/default/epoch7epochstep5globalstep50/

MODE=local

EXP_NAME=quickstart-gen
TRIAL_NAME=$MODEL_FAMILY-$MODE

python3 -m realhf.apps.quickstart gen \
    mode=$MODE \
    experiment_name=$EXP_NAME \
    trial_name=$TRIAL_NAME \
    exp_ctrl.total_train_epochs=1 \
    exp_ctrl.save_freq_steps=null \
    n_nodes=1 \
    allocation_mode=manual \
    model.type._class=$MODEL_FAMILY \
    model.path=$SFT_MODEL_PATH \
    dataset.path=.data/ppo_prompt.jsonl \
    dataset.max_prompt_len=1024 \
    dataset.train_bs_n_seqs=100 \
    allocation.parallel.pipeline_parallel_size=1 \
    allocation.parallel.model_parallel_size=2 \
    allocation.parallel.data_parallel_size=4 \
    gen.max_new_tokens=1024 \
    gen.min_new_tokens=1024 \
    gen.use_cuda_graph=True \
    gen.top_p=0.9 gen.top_k=1000