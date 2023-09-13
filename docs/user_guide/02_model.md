# Model

The definition of `api.model.Model` is very simple:

```python
@dataclasses.dataclass
class Model:
    name: str
    module: Union[deepspeed.DeepSpeedEngine, transformers.PreTrainedModel, torch.nn.Module]
    tokenizer: transformers.PreTrainedTokenizerFast
    device: Union[str, torch.device]
    ...
```

As its core components, it encapsules a neural network and a tokenizer.

To implement a new model, users should write a function with signature

```python
def foo(name: str, device: Union[str, torch.device], **model_kwargs) -> api.model.Model:
    ...
```

i.e., `name` and `device` must be arguments of this function. It's the user's freedom how to construct the neural network and how to load the tokenizer.

Don't forget the register this model via

```python
api.model.register_model("my_model", foo)
```


To configure such a model, the configuration is
```python
model_cfg = api.config.Model("my_model", dict=model_kwargs)
```

See `impl/model/nn/basic.py` for examples.

# Model Interface

`ModelInterface` is a collection of functions that define the RPC handler of model worker. It has signature:

```python
class ModelInterface(abc.ABC):

    def save(self, model: Model, save_dir: str):
        pass

    def evaluate(self, model: Model, eval_dataloader: torch.utils.data.DataLoader) -> Dict:
        return {}

    def inference(self, model: Model, data: NamedArray) -> NamedArray:
        raise NotImplementedError()

    def generate(self, model: Model, data: NamedArray) -> NamedArray:
        raise NotImplementedError()

    def train_step(self, model: Model, data: NamedArray) -> Dict:
        raise NotImplementedError()
```

Note that the input/output of "inference" and "generation" are `NamedArray` objects, which contain PyTorch CPU tensors.

Do anything you want to implement this following this signature! 

See `impl/model/interface/wps_actor_critic.py` for examples.

# Model Backend

`ModelBackend` implements a `initialize` method that initializes the backend of the model.

Specifically for the deepspeed backend, we wrap the `torch.nn.Module` to a `deepspeed.DeepSpeedEngine` that sets up ZeRO optimizer (and potentially `HybridEngine`, offload, or any other things implemented by DeepSpeed).

See `impl/model/backend/deepspeed.py` for examples.