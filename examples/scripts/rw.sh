SFT_MODEL_PATH=/lustre/aigc/llm/checkpoints/fw/quickstart-sft/release/default/epoch7epochstep5globalstep50/
CLUSTER_SPEC_PATH=/lustre/aigc/llm/cluster/qh.json python3 -m realrlhf.apps.quickstart rw \
    experiment_name=quickstart-rw trial_name=release \
    allocation_mode=manual \
    total_train_epochs=1 \
    save_freq_steps=5 eval_freq_epochs=1 \
    model.type._class=llama \
    model.type.size=7 \
    model.type.is_critic=True \
    model.path=$SFT_MODEL_PATH \
    n_nodes=1 \
    allocation.parallel.pipeline_parallel_size=2 \
    allocation.parallel.model_parallel_size=2 \
    allocation.parallel.data_parallel_size=2 \
    allocation.parallel.use_sequence_parallel=True \
    model.gradient_checkpointing=True \
    model.optimizer.lr_scheduler_type=cosine \
    model.optimizer.lr=1e-5 \
    model.optimizer.warmup_steps_proportion=0.0 \
    dataset.train_path=/lustre/fw/datasets/imdb/rl/rm_paired-train.jsonl \
    dataset.valid_path=/lustre/fw/datasets/imdb/rl/rm_paired-valid.jsonl \
    dataset.max_pairs_per_prompt=2 \
    dataset.max_seqlen=1024 \
    dataset.train_bs_n_seqs=512 \
    dataset.valid_bs_n_seqs=512