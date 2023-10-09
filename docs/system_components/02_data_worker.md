# Data Worker

### TL;DR: 根据要用的模型更改`tokenizer_name_or_path`，自行实现需要应用的数据集（和`dataloader`，但`dataloader`基本上可以用默认的，改参数就行），实现方法请看user_guide

一个问题是为啥要在系统中设计多个data worker/data loader，原因是目前我们会直接将json格式的数据集加载到内存中，做tokenization时在某些机器上有可能爆内存。
所以我们设计N个data worker，每个worker负责加载1/N的数据，可以部分解决

## The Poll Step

他会处理两种请求，一种是fetch，即从数据集中加载数据，另一种是spec，即获取数据集的统计信息。

## How to Configure

```python
@dataclasses.dataclass
class DataWorker:
    tokenizer_name_or_path: str
    datasets: List[Union[str, Dataset]]
    stream: Union[str, RequestReplyStream]
    # dataset cache
    dataloader: Union[str, DataLoader] = "default"
    seed: int = 1
    use_dataset_cache: bool = False
    dataset_cahce_root: str = DATASET_CACHE_PATH
    worker_info: Optional[WorkerInformation] = None
```

+ `tokenizer_name_or_path`: huggingface的model path，里面包含了`tokenizer_config.json`和`tokenizer.json`等文件。
+ `datasets`: 由用户自行实现并注册的数据集。请看user_guide中关于数据集的实现方式。
+ `stream`：和master worker相连的RPC stream，与master worker configuration中的stream对应。
+ `dataloader`：由用户自行实现并注册的dataloader。请看user_guide中关于数据集的实现方式。
+ `seed`：随机种子，数据集shuffle的结果与之相关。
+ `use_dataset_cache`：是否要存dataset cache，以便今后实验创建数据集时直接从cache加载。
+ `dataset_cache_root`：dataset cache的路径。
