MODEL_FAMILY=opt
PRETRAINED_PATH=/lustre/public/pretrained_model_weights/opt-125m
CLUSTER_SPEC_PATH=/lustre/aigc/llm/cluster/qh.json python3 -m realhf.apps.quickstart sft \
    experiment_name=quickstart-sft trial_name=$MODEL_FAMILY \
    allocation_mode=manual \
    total_train_epochs=8 \
    save_freq_steps=50 eval_freq_epochs=1 \
    model.optimizer.lr_scheduler_type=cosine \
    model.optimizer.lr=1e-5 \
    model.optimizer.warmup_steps_proportion=0.02 \
    model.type._class=$MODEL_FAMILY \
    model.type.size=0 \
    model.type.is_critic=False \
    model.path=$PRETRAINED_PATH \
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