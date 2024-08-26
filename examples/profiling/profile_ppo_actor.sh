# The model to profile and its path.
MODEL_FAMILY=llama
SFT_MODEL_PATH=/lustre/public/pretrained_model_weights/Llama-2-7b-hf

EXP_NAME=profile-example
TRIAL_NAME=test

export CLUSTER_SPEC_PATH="/lustre/aigc/llm/cluster/qh.json"

# Setting REAL_DUMP_KERNEL_TIME=1 enables kernel time profiling.
# Time of compute and communication kernels will be summed up and reported.

# Setting REAL_DUMP_TRACE=1 enables execution trace provided by PyTorch.

# Setting REAL_DUMP_MEMORY=1 enables memory profiling provided by PyTorch.

# The dataset content doesn't matter, as long as it is a prompt-only dataset.
# Each entry in the dataset should contain two keys "id" and "prompt".
# By default we pad the prompt to the maximum length in the batch for accurate system-wise benchmark.
# The loaded data will be processed by the "_mock_${handle_name}" method in the interface
# to create mock data suited for the exact interface handle.

# "handle_name" can be "inference", "generate", or "train_step",
# and the "interface_impl" specifies which registered interface implementation to run.
# "interface_kwargs_json" is a JSON configuration of the interface.

# "allocations_jsonl" is a JSONL file that specifies the parallel strategies to profile.
# If not specified, all parallel strategies under the given world size will be profiled.

# "n_mbs" specifies the number of micro-batches to profile.

# The total number of runs will be the product of the number of micro-batches and the number of parallel strategies,
# all within the same experiment_name and trial_name. Instead of re-launching the whole experiment, workers will
# be paused and reconfigured to run the next experiment setup.

REAL_DUMP_KERNEL_TIME=0 REAL_DUMP_TRACE=0 REAL_DUMP_MEMORY=0 \
    python3 -m realhf.apps.quickstart profile \
    mode=local \
    experiment_name=$EXP_NAME \
    trial_name=$TRIAL_NAME \
    exp_ctrl.benchmark_steps=5 \
    exp_ctrl.save_freq_steps=null \
    exp_ctrl.eval_freq_steps=null \
    n_nodes=1 \
    model.type._class=$MODEL_FAMILY \
    model.path=$SFT_MODEL_PATH \
    dataset.path=/lustre/fw/datasets/imdb/rl/ppo_prompt.jsonl \
    dataset.max_prompt_len=1024 \
    dataset.train_bs_n_seqs=64 \
    dataset.pad_to_max_length=True \
    handle_name=train_step \
    interface_impl=ppo_actor \
    'n_mbs=[1, 2, 4]' \
    interface_kwargs_json=./examples/profiling/interfaces/ppo_actor.json \
    allocations_jsonl=./examples/profiling/allocations/local.jsonl
