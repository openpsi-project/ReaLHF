# Examples

See `impl/dataset/common/prompt_dataset.py`.

# Dataset

Datasets used in this system are subclasses of `torch.utils.data.Dataset`. Besides, an `api.data.DatasetUtility` object must be one of the arguments of the `__init__` function of the implemtned dataset.

To implement a new dataset, uses should
1. define a class by inheriting `torch.utils.data.Dataset`;
2. implement the `__init__` function, with `util: data_api.DatasetUtility` as the first argument, and arbitrarily many other arguments;
3. implement the `__len__` function and the `__getitem__` methods just like ordinary PyTorch dataset implementations (or `__len__` and `__iter__` for iterable datasets);
4. register the dataset by calling `data_api.register_dataset("my_cool_dataset", MyDataset)`.

In configuration, users can define `api.config.Dataset` like this

```python
dataset_cfg = realrlhf.api.config.Dataset(type_="my_cool_dataset",
                                 args=dict(**other_arguments))
```

# Dataloader

Dataloaders used in this system are basically `torch.utils.data.DataLoader`. This object is an iterator which accepts the `next` call and returns a dictionary of `torch.Tensor`s at each training step.

In `api/data.py`, we have defined several default dataloaders. To implement and register a new one, users can do something like

```python
def RLHFDataLoader(dataset, max_token_len, *args, **kwargs):
    collator = DataCollatorRLHF(max_token_len)
    return torch.utils.data.DataLoader(dataset, *args, collate_fn=collator, **kwargs)
data_api.register_dataset("excel_prompt", ExcelPromptDataset)
```

where we overwrite the arguments of `torch.utils.data.DataLoader` with a customized data collator.

In configuration, users can define `api.config.DataLoader` like
```python
dataloader = realrlhf.api.config.DataLoader(
    'excel_propmt',
    args=dict(
        max_token_len=max_prompt_len,
        shuffle=True,
        drop_last=True,
        batch_size=batch_size,
        ...
    ),
)
```

or simply

```python
dataloader = 'default'
```
