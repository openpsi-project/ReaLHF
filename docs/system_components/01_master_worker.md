# Master Worker

### TL;DR: 只需要按照需求改`total_train_epochs`以及save和eval的frequency，ECS数据流和stream不用改。

## The Poll Step

+ 在实验开始时，master worker首先会对所有的model worker发initialize backend的request，目前backend用的是deepspeed，会启动ZeROOptimizer。
+ 向data worker发送加载数据的请求，获得数据和对应的统计信息（epoch、epoch_step等），保存数据在本地。
+ 如果达到了evaluation的间隔，发送evaluate的RPC，等待evaluation的统计信息发回并log出来。
+ 如果达到了save的间隔，发送save的RPC，等待save结束。
+ 执行ECS的函数，也就是RLHF的整个workflow，包含多个顺序或同时发布的model RPC。

## ECS

为了能够灵活地设置workflow，我们参考了Entity Component System（ECS）的设计模式。RLHF算法的工作流由很多个子任务组成，比如actor generate、reward inference这些我们都看作为一个子任务（这里子任务可以理解为一次model RPC，但是子任务也可以是顺序的多次model RPC）。我们只要知道了这些子任务都需要什么数据、需要哪些模型，就可以将这个子任务抽取出来写一个函数来完成。

例如，actor model根据prompt生成回复的子任务可以写成
```python
def rollout(
    commands: Commands,
    model: ModelQuery['actor'],
    prompts: RawDataQuery['prompts'],
    prompt_att_mask: RawDataQuery['prompt_att_mask'],
):
    inputs = commands.build_model_inputs(
        prompts=prompts,
        prompt_att_mask=prompt_att_mask,
    )
    # This is a model RPC. ModelInterface should implement `generate` in this case.
    # Input data to the `generate` interface is a `NamedArray` with two fields: prompts and prompt_att_mask.
    res = model.generate(inputs)
    # Data returned by this RPC is a `NamedArray` containing (at least) the following four fields.
    commands.set_data('seq', res['seq'])
    commands.set_data('logp', res['logp'])
    commands.set_data('attention_mask', res['attention_mask'])
    commands.set_data('logits_ignoring_mask', res['logits_ignoring_mask'])
```

接下来，我们只要把整个算法的所有子任务写成的函数都送给`api.ecs.MasterWorkerECS`，它就会resolve出来子任务之间的dependency graph。现在实现的是将这个dependency graph做拓扑排序，形成多个level，每个level包含多个同时的model RPC，这些RPC被执行时会被放到threadpool中同时执行。这样的实现是有问题的，不过对于普通的RLHF workflow足够用了。


## How to Configure

### **在`Experiment`的`initial_setup`函数中，用户不用设置master worker，而是设置`ExperimentConfig`，系统会自动将`ExperimentConfig`的参数转化为master worker的参数。**

```python
@dataclasses.dataclass
class MasterWorker:
    total_train_epochs: int
    # save control
    save_frequency_epochs: int
    save_frequency_steps: int
    save_frequency_seconds: int
    # eval control
    eval_frequency_epochs: int
    eval_frequency_steps: int
    eval_frequency_seconds: int
    # main components
    data_streams: List[Union[str, RequestReplyStream]]
    model_streams: Dict[str, List[Union[str, RequestReplyStream]]]
    leveled_exec_funcs: api.ecs.MasterWorkerExecutable
    worker_info: Optional[WorkerInformation] = None
```

+ `total_train_epochs`: 训练需要迭代数据集几次，超过这个数量会raise error终止训练。
+ `save_frequency_*`: 训练间隔多长时间后需要保存模型，间隔会独立计算。每当达到任意一个间隔后，会向所有的model发送save的RPC，只有实现save interface的model会执行模型保存，其他模型默认会无视这个request。如果设为None则不会计算这个间隔。
+ `eval_frequency_*`: 训练间隔多长时间后需要评估模型，间隔会独立计算。每当达到任意一个间隔后，会向所有的model发送evaluate的RPC，只有实现evaluate interface并且在config中设置evaluation dataset的model会执行模型评估，其他模型默认会无视这个request并返回空字典。如果设为None则不会计算这个间隔。
+ `data_streams`: 和data worker相连的RPC stream。
+ `model_streams`: 和model worker相连的RPC stream。
+ `leveled_exec_funcs`: ECS resolve出来的dependency graph。