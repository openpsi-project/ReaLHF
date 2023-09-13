# Model Worker

### TL;DR: 请看user_guide如何实现自己所需要的`Model`和`ModelInterface`，其他保持默认或者适当修改参数。

## The Poll Step

作为master worker的RPC server处理请求，请求包括以下六种：
+ "initialize": 实现在`api.model.ModelBackend`中，初始化模型的后端框架，目前用的是DeepSpeed。需要利用RPC进行初始化的原因是，master worker需要传过来一些global information例如需要训练的epoch数量、每个epoch的step数量等等，这些信息对于后端初始化是有用的。每个模型都必须实现。
+ "save": 实现在`api.model.ModelInterface`中，保存模型参数，不返回结果。可以不实现，不实现的模型在处理这个RPC时什么都不做。
+ "inference": 实现在`api.model.ModelInterface`中，进行模型推理并返回推理结果。
+ "train": 实现在`api.model.ModelInterface`中，给定数据进行训练，可以是一次gradient step也可以是将大batch分解为minibatch后的多个gradient step。
+ "genearte": 实现在`api.model.ModelInterface`中，给定prompt进行多次推理生成回复。
+ "evaluate": 实现在`api.model.ModelInterface`中，给定模型和数据集进行evaluation。如果调用这个API，必须在模型configuration中传入eval datasets和eval dataloader。

## How to Configure

```python
@dataclasses.dataclass
class ModelWorker:
    seed: int
    model: Model
    interface: ModelInterface
    backend: ModelBackend
    model_name: str
    # stream
    stream: Union[str, RequestReplyStream]
    # evaluation
    eval_datasets: Optional[List[Dataset]] = None
    eval_dataloader: Optional[DataLoader] = None
    use_dataset_cache: bool = False
    dataset_cahce_root: str = DATASET_CACHE_PATH
    # cuda & cudnn config
    cudnn_benchmark: bool = False
    cudnn_deterministic: bool = False
    cuda_cache_cleanliness: bool = False  # i.e., clear cuda cache after each RPC call
    worker_info: Optional[WorkerInformation] = None
```

+ `seed`：随机种子，决定模型初始化。
+ `model`：需要用户实现的模型，包括了神经网络和tokenizer。请看user_guide。
+ `interface`：需要用户实现的模型接口，和模型、tokenizer以及算法数据流都有关，是系统的核心部分。具体请看user_guide。
+ `backend`：模型的后端，目前仅实现了DeepSpeed。DeepSpeed支持什么我们就能支持什么，可以在`impl.model.backend.deepspeeed`中查看和修改套皮方式。
+ `model_name`：模型唯一的identifier。目前系统只支持data parallel，data parallel的model名字都一样。
+ `stream`：和master worker相连的RPC stream。
+ `eval_datasets`： 用来做evaluation的数据集，和data worker一样。实现方式请看user_guide。
+ `eval_dataloader`：evaluation dataloader，和data worker一样。实现方式请看user_guide。
+ `use_dataset_cache`：和data worker一样。
+ `dataset_cache_root`：和data worker一样。
+ `cudnn_benchmark`：历史原因保留的参数，设为True可以加速CNN。基本不用。
+ `cudnn_deterministic`：设为True可以让随机种子产生确定性的结果，便于重复实验。
+ `cuda_cache_cleanliness`：设为True会在每次model RPC后清理cuda cache，但非常花时间。基本不用。