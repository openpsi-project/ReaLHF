LLAMA_PATH=/lustre/public/pretrained_model_weights/Llama-2-7b-hf
CLUSTER_SPEC_PATH=/lustre/aigc/llm/cluster/qh.json python3 -m realrlhf.apps.quickstart sft \
    experiment_name=quickstart-sft trial_name=release \
    allocation_mode=manual \
    total_train_epochs=8 \
    save_freq_steps=50 eval_freq_epochs=1 \
    model.optimizer.lr_scheduler_type=cosine \
    model.optimizer.lr=1e-5 \
    model.optimizer.warmup_steps_proportion=0.02 \
    model.type._class=llama \
    model.type.size=7 \
    model.type.is_critic=False \
    model.path=$LLAMA_PATH \
    model.gradient_checkpointing=True \
    model.optimizer.offload=False \
    model.optimizer.type=adam \
    dataset.train_path=.data/sft_pos-train.jsonl \
    dataset.valid_path=.data/sft_pos-train.jsonl \
    dataset.max_seqlen=1024 \
    n_nodes=1 \
    allocation.parallel.pipeline_parallel_size=1 \
    allocation.parallel.model_parallel_size=2 \
    allocation.parallel.data_parallel_size=4 \
    allocation.parallel.use_sequence_parallel=True \
    dataset.train_bs_n_seqs=512 \
    dataset.valid_bs_n_seqs=512