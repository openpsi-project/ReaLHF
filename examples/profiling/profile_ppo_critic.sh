# The model to profile and its path.
MODEL_FAMILY=llama
SFT_MODEL_PATH=/lustre/public/pretrained_model_weights/Llama-2-7b-hf

EXP_NAME=profile-example
TRIAL_NAME=test

export CLUSTER_SPEC_PATH="/lustre/aigc/llm/cluster/qh.json"

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

# Changing `model.init_critic_from_actor` and `model.type.is_critic` is importance for profiling the critic.

REAL_DUMP_TRACE=1 REAL_DUMP_MEMORY=1 \
    python3 -m realhf.apps.quickstart profile \
    mode=local \
    experiment_name=$EXP_NAME \
    trial_name=$TRIAL_NAME \
    exp_ctrl.benchmark_steps=3 \
    exp_ctrl.save_freq_steps=null \
    exp_ctrl.eval_freq_steps=null \
    n_nodes=1 \
    model.type._class=$MODEL_FAMILY \
    model.path=$SFT_MODEL_PATH \
    dataset.path=/lustre/fw/datasets/imdb/rl/ppo_prompt.jsonl \
    dataset.max_prompt_len=4096 \
    dataset.train_bs_n_seqs=256 \
    dataset.pad_to_max_length=True \
    handle_name=train_step \
    interface_impl=ppo_critic \
    model.init_critic_from_actor=True \
    model.type.is_critic=True \
    'n_mbs=[2]' \
    interface_kwargs_json=./examples/profiling/interfaces/ppo_critic.json
