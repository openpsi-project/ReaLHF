import functools

from api.config.config_system import register_experiment
from experiments.common.sft_exp import SFTExperiment

seeds = range(1, 64)
for s in seeds:
    # TODO: hyperparameters here is not correct
    exp_name = f"imdb-sft-pos-neg-s{s}"
    register_experiment(
        exp_name,
        functools.partial(
            SFTExperiment,
            seed=s,
            model_type="gpt2",
            model_path="/lustre/fw/pretrained/gpt2/",
            train_dataset_path="/lustre/fw/datasets/imdb/rl/sft_pos_neg-train.jsonl",
            valid_dataset_path="/lustre/fw/datasets/imdb/rl/sft_pos_neg-valid.jsonl",
            use_lora=False,
            dp_size=8,
            total_train_epochs=4,
        ),
    )

    exp_name = f"imdb-sft-pos-s{s}"
    register_experiment(
        exp_name,
        functools.partial(
            SFTExperiment,
            seed=s,
            model_type="gpt2",
            model_path="/lustre/fw/pretrained/gpt2/",
            train_dataset_path="/lustre/fw/datasets/imdb/rl/sft_pos-train.jsonl",
            valid_dataset_path="/lustre/fw/datasets/imdb/rl/sft_pos-valid.jsonl",
            use_lora=False,
            dp_size=8,
            total_train_epochs=4,
        ),
    )