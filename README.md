# A Distributed System for LLM RLHF

## Installation

Under the git repository, run:

```bash
python3 -m build -n
```

## Quickstart

The following command is used to run an SFT experiment:

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

We use [hydra](https://hydra.cc/) for CLI hyper-parameter configuration. Please check, e.g., the `SFTConfig` class for available CLI configurations.
Note that the configuration is [structured](https://hydra.cc/docs/tutorials/structured_config/intro/) and nested options can be configured with recursive dot-field access, as shown in the above example.
As a kind reminder, the passed-in value can be `null` to represent `None` in python.

Supported experiment types of quickstart include `sft` (Supervised Fine-tuning), `rw` (Reward Modeling), `ppo` (Proximal Policy Optimization), and `dpo` (Direct Preference Optimization).
They have separate CLI configurations. Please check `apps/quickstart.py` for detailed information.

To enable pipeline parallelism, you need to first partition weights of the (HuggingFace) base model with `scripts/transform_to_pipe_ckpt.py`:

```python
python3 -m scripts.transform_to_pipe_ckpt --model_dir /lustre/public/pretrained_model_weights/Llama-2-13b-hf/ \
                                          --model_type llama --num_stages 4 \
                                          --output_dir /lustre/public/pretrained_model_weights/Llama-2-13b-hf_pp4/
```

Then, pass the `output_dir` to `model.path` in quickstart command and set `model.parallel.pipeline_parallel_size` to be the number of stages you partitioned.
Run the command and take off!

Quickstart reallm.experiments will run slurm jobs by default. When slurm is not available, it will launch subprocesses locally (e.g. in a `srun` bash container).

The scheduling configuration is not changable in quickstart reallm.experiments. Users cannot specify nodelist/exclude and the number of GPUs used by each model.
Please consider launching customized reallm.experiments if you must do so.

## Launch Customized Experiments

We provide another way to launch customized reallm.experiments via `python3 -m apps.main start` for advanced users and developers.
An example is the system benchmark [here](https://github.com/garrett4wade/distributed_llm/blob/main/experiments/benchmark/system/rlhf_benchmark.py#L182), which can be launched via

```python
python3 -m apps.main start -e sysb-llama-7b -f test20231207 --mode slurm
```

where `-f` specifies a trial name to distinguish different runs of the same experiment name.

Experiments are basically the composition of model, dataset, parallelism, and scheduling configurations. We implement each experiment configuration as a `Experiment` class.
These classes are registered via `register_experiment` and registered ones can be launched by `python3 -m apps.main start -e ${registered_exp_name}`.
Quickstart also internally launches registered reallm.experiments, but these reallm.experiments are dynamically registered and configurable via the command line.

`experiment_name` and the registered class have static one-to-one relationship. In other words,
it sets an alias for a special combination of dataset, model, and hyper-parameters, which is especially useful when we need to freeze hyper-parameters for the sake of reproducibility,
e.g. algorithm and system benchmarks.

## Contributing

Before reading the code, keep in mind that:
1. Everything in [api/config.py](api/config.py) are configurations. They are used to configurate the `Experiment` class.
2. Other classes in [api directory](api/) are abstract methods. They represent necessary components for the system to run.
3. Classes in [api/config.py](api/config.py) and other scripts in [api directory](api/) may have same class names. 

To run the code, see `docs/user_guide/00_quickstart.md`.

To understand difference between `model`, `model backend` and `model interface`, read [this doc](docs/user_guide/02_model.md).

If you want to contribute to the codebase (e.g., new datasets/models/algorithms), please open an MR and @fuwei for code review.
