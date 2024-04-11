# How to Launch An Existing Experiment

## Command
```
python3 -m apps.main start -e ${registered_experiment_name} -f ${your_trial_name} --mode ${anyof[slurm,local]}
```

+ `-e ${registered_experiment_name}`: The experiment name registered via `api.config.register_experiment`. Usually, user will write an experiment configuration file inside the `experiments` folder, which calls `api.config.register_experiment`, and import this file in `experiments/__init__.py`.
+ `-f ${your_trial_name}`: Any string without "_".
+ `--mode ${anyof[slurm,local]}`: The slurm mode launchs a distributed experiment via the slurm scheduler. The local mode launches a local experiment inside the container of srun (used for debugging).

## Examples

We use `experiments/packed_dpo_exp.py` as an example. In this configuration file, we can see `register_experiment` at the end of this file.
The registered names are `flash-dpo-${seed}` where `seed` can be `[1-5],42`.
To run this experiment, type `python3 -m apps.main start -e flash-dpo-s1 -f test20231020 --mode slurm`.

## How to Change Datasets and Models in Exisiting Experiments to Customized Ones

The most common usage of this system is to run algorithms on customized datasets/models.
Datasets/models used for existing experiments can be found in `impl/dataset` and `impl/model` respectively.
The experiment configuration will select an existing implementation of dataset/model by specifying a config, either `api.config.Dataset` or `api.config.Model`. For example, we can find out in `experiments/packed_dpo_exp.py` that the used dataset is "packed_rw_pair" and the model is "flash_mqat_clm_hf". These are implemented in `api/dataset/common/packed_rw_paired_dataset` and `api/modle/nn/flash_mqat.py`.

Dataset implementations are usually general enough and require a ".json" or ".jsonl" text file for parsing.
There can be required keys in this file. For example, "packed_rw_pair" dataset requires a key "prompt", which is a string, a key "pos_answers", which is a list of strings, and a key "neg_answers", which is a list of strings with the same length as "pos_answers". **Deploying a new dataset requires the user to convert the dataset into such a format, and then change the corresponding path to the customized dataset path.**

Existing model implemementations include two main types, HuggingFace and `api/modle/nn/flash_mqat.py`. The former is just models implemented in huggingface transformers (see `impl/model/nn/basic_nn.py`). The latter is an internel model implemented by ourselves, `impl/model/nn/flash_mqat.py`. Benefits of this model include (1) it uses flash attention for all model calls (generation, inference, training), which is faster and much more memory-efficient (no PAD tokens, longer context length); (2) it is a stateless model and all operations are transparent; (3) it is designed to be pipeline-parallelism compatible. However, this model has a limitation that it does not automatically support all models from huggingface transformers, e.g., LLaMa. Overall, the advantages outweight the inconvinience. **In general, we should use `impl/model/nn/flash_mqat.py` for all experiments. If you want to use a specific model that has not been supported by the current implementation, please reach out @fuwei for help.**

## How to Write New Algorithms

The current interface is a little bit complicated. Please first read `docs/user_guide/02_model.md` for the interface structure of models. To implement a new algorithm, users should first implement the corresponding `ModelInterface` as examplified in `impl/model/interface/`. Then, if the algorithm involves multiple models and dataflow management among them, users need to implement a model RPC graph about how model interfaces are called. See `experiments/packed_ppo_exp.py` for the example of RLHF.

# How to Write An Experiment Configuration

The experiment configuration is a class that must implement two methods: `scheduling_setup` and `initial_setup`. See the function signatures in `api.config.Experiment`.

`scheduling_setup` returns an `api.config.ExperimentScheduling` object that specifies the count and resources of each worker. This object will be used for slurm. In the local mode, only count and the number of GPUs are considered.

`inital_setup` returns an `api.config.ExperimentConfig` object that contains the global and all workers' configurations. See `docs/system_components` for how to configure each worker.

# API and Concepts

## `Dataset`

Basically a subclass of `torch.utils.data.Dataset` that implements `__len__` and `__getitem__` methods or a subclass of `torch.utils.data.IterableDataset` that implements `__len__` and `__iter__` methods, which is essentially pytorch datasets.
Please refer to PyTorch documentation for details.

`torch.utils.data.Dataset` is the common indexing-based dataset, i.e., if the dataloader calls `dataset[i]` the dataset will load the `i`-th element. `torch.utils.data.IterableDataset` is a dataset used for dynamic batching. If we use flash attention, there's no pad token in each batch, so the batch size (i.e., number of tokens in batch) is variable across different train steps. To balance the batch size, we pre-allocate batches at the beginning of training. The dataloader then iteratively loads batches from it.

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

Note that when the dataset is a `torch.utils.data.IterableDataset`, the `batch_size` argument to dataloader should be `None`. Dataloader is just a python iterator helper in this case.

## `NamedArray`

`NamedArray` is an object we use in model RPCs. It is inherited from the previous SRL project.

Named array extends numpy array in the following ways.
1. NamedArray aggregates multiple numpy arrays, possibly of different shapes.
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
