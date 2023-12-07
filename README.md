# A Distributed System for LLM RLHF

## Quickstart

Run a SFT experiment!

```python
python3 -m apps.quickstart sft experiment_name=my-cool-experiment trial_name=dp-pp-20231207 \
    model.type=llama \
    model.path=/lustre/public/pretrained_model_weights/testOnly/llama-2-4l_4pp_3s/ \
    model.parallel.pipeline_parallel_size=4 \
    model.parallel.data_parallel_size=2 \
    model.gradient_checkpointing=False \
    save_freq=10 \
    eval_freq=1 \
    optimizer.lr=2e-5 \
    dataset.max_seqlen=2048 \
    dataset.train_tokens_per_batch=65536 \
    dataset.valid_tokens_per_batch=65536
```

We use [hydra](https://hydra.cc/) for CLI hyper-parameter configuration. Supported experiment types of quickstart include `sft` (Supervised Fine-tuning), `rw` (Reward Modeling), `ppo` (Proximal Policy Optimization), and `dpo` (Direct Preference Optimization).

To enable pipeline parallelism, you need to first partition weights of the (HuggingFace) base model with `scripts/transform_to_pipe_ckpt.py`:

```python
python3 -m scripts.transform_to_pipe_ckpt --model_dir /lustre/public/pretrained_model_weights/Llama-2-13b-hf/ \
                                          --model_type llama --num_stages 4 \
                                          --output_dir /lustre/public/pretrained_model_weights/Llama-2-13b-hf_pp4/
```

Then, pass the `output_dir` to `model.path` in quickstart command and set `model.parallel.pipeline_parallel_size` to be the number of stages you partitioned.

## Contributing

Before reading the code, keep in mind that:
1. Everything in [api/config.py](api/config.py) are configurations. They are used to configurate your experiment.
2. Other classes in [api directory](api/) are abstract methods. They represent necessary components for the system to run.
3. Classes in [api/config.py](api/config.py) and other scripts in [api directory](api/) may have same class names. 

To run the code, see `docs/user_guide/00_quickstart.md`.

To understand difference between `model`, `model backend` and `model interface`, read [this doc](docs/user_guide/02_model.md).

If you want to contribute to the codebase (e.g., new datasets/models/algorithms), please open an MR and @fuwei for code review.
