# How to Launch An Existing Experiment

```
python3 -m apps.main start -e ${registered_experiment_name} -f ${your_trial_name} --mode ${anyof[slurm,local]}
```

+ `-e ${registered_experiment_name}`: The experiment name registered via `api.config.register_experiment`. Usually, user will write an experiment configuration file inside the `experiments` folder, which calls `api.config.register_experiment`, and import this file in `experiments/__init__.py`。
+ `-f ${your_trial_name}`: Any string without "_".
+ `--mode ${anyof[slurm,local]}`: The slurm mode launchs a distributed experiment via the slurm scheduler. The local mode launches a local experiment inside the container of srun (used for debugging).

# How to Write An Experiment Configuration

The experiment configuration is a class that must implement two methods: `scheduling_setup` and `initial_setup`. See the function signatures in `api.config.Experiment`.

`scheduling_setup` returns an `api.config.ExperimentScheduling` object that specifies the count and resources of each worker. This object will be used for slurm. In the local mode, only count and the number of GPUs are considered.

`inital_setup` returns an `api.config.ExperimentConfig` object that contains the global and all workers' configurations. See `docs/system_components` for how to configure each worker.

# API and Concepts

## `Dataset`

Basically a subclass of `torch.utils.data.Dataset` that implements `__len__` and `__getitem__` methods.

`__len__` must returns a integer and `__getitem__` must return a dictionary of tensors.

An `api.data.DatasetUtility` object is required as an argument in the `__init__` method of datasets implemented in this system, which contains several useful handles such as rank, world_size, and the tokenizer.

## `Dataloader`

Basically a `torch.utils.data.DataLoader`. Different dataloaders only differs in arguments passed into this class, such as some task-specific data collators.

Dataloaders should also produce a dictionary of tensors upon each `next` call, with batch size as the first dimension.

Ususally, the dataloader can be used like this:

```python
for data in dataloder:
    data: Dict[torch.Tensor]
    # {"input_ids": torch.IntegerTensor[Size(batch_size, seq_len)],
    #  "attention_mask" torch.BoolTensor[Size(batch_size, seq_len)]}
    ...
```

## `NamedArray`

`NamedArray` is an object we use in model RPCs. It is inherited from the previous SRL project.

Named array extends numpy array in the following ways.
1. Each NamedArray aggregates multiple numpy arrays, possibly of different shapes.
2. Each numpy array is given a name, providing a user-friendly way of indexing to the corresponding data.
3. Named arrays can be nested. (Although is should *not* be nested in this system)

Users can regard it as a nested dictionary of arrays, except that indexing a `NamedArray` results in *slicing every hosted arrays* (again, we don't use this feature in this project).

We use this class because it is convenient to serialize and deserialze `NamedArray` objects with different levels of compression. This feature is only used in data transfer of request-reply streams.

For a complete starter guid of `NamedArray`, check https://10.210.14.2/marl/distributed_marl/-/blob/main/distributed/docs/user_guide/named_array.md.


## `Model`

`Model` primarily contains a neural network (e.g. `torch.nn.Module`, `deepspeed.DeepSpeedEngine`) and a huggingface tokenizer.

It also contains several useful attributes like `device` and `version`. A `Model` object will be passed into `ModelInterface` to deal with RPCs.

## `ModelInterface`

`ModelInterface` encapsules five RPCs, namely "save", "evaluate", "inference", "generate", and "train_step".

Each method takes a `Model` and a `NamedArray` object as the input and produces corresponding model output, such as reference logits or training statistics. See signatures in `api/model.py` for details.

Users can just implement some of these interfaces in a specific algorithm.

## `ModelBackend`

`ModelBackend` exposes an `initialize` function to model workers and handles the backend initialization.

This must be implemented for each model.

Currently we only support the DeepSpeed backend and backend initialization is basically calling `deepspeed.initialize` with different arguments.
