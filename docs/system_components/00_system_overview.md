# Motivation

我们希望follow InstructGPT的训练流程实现LLM的RLHF。LLM RLHF每个step的训练过程是这样的：

1. （data loading）从数据集加载数据，数据也就是一些任务相关的prompt；
2. （generation）actor model用这些prompt生成对应的回复，并在生成过程中记录生成序列的logits；
3. （inference）将prompt和生成的回复送给reference model、reward model和critic model分别做推理得到reference logits、rewards和values；
4. （training）将以上所有加载或生成的数据打包分别发给actor和critic做训练，更新它们的参数。

从上面的流程中可以看出，RLHF的流程需要四个模型的参与：actor、critic、reference和reward。
在简单的单机实现中，这四个模型会被分配到同一个GPU上，并且上面所有流程都需要序列化进行（包括不同模型的inference、不同模型的训练）。
这种实现的缺点是一旦模型或者数据batch size增大很容易爆显存，并且并行度不高，例如不同模型的inference、不同模型的训练是完全可以并行的。

# System Overview

我们的系统可以对不同种类的模型做独立调度，将四个参与训练的模型作为worker分配到各自独占的GPU上。
我们可以根据模型在训练任务中的workload调整每个模型需要的GPU数量，比如reference和reward只会做inference，那么他们会占用更少的GPU。

一种设计系统的选择是去中心化，根据任务设计worker，就像我们之前设计的SRL系统一样：actor和critic参与训练，那么将他们设计为trainer worker，另外两个设计为inference worker，他们之前通过REQ-REP和PUSH-PULL的stream相连接。但是，我们发现这样设计出来的系统中的stream会对RLHF的workflow非常specific，将来如果做tensor parallism或pipeline parallism也没有很好的扩展性。

另外一种选择是中心化，除了model之外有一个master worker维护算法整体的工作流，负责给model调度数据和需要做的工作。model此时就仅仅作为RPC server，负责对master worker传来的request做回复。这样做的好处是，如果model worker实现了tensor parallism或者pipeline parallism，那么它只维持一份model shard，只要master worker根据rank解析出来到底需要往这份shard送什么数据收回来什么数据，整体的数据流是和vanilla的实现完全一样的。

我们现在实现的系统由master worker、data worker和model worker组成。他们的职责分别是这样的：

+ master worker：定义了算法的整体工作流程，它可以被一个有向无环图表示，每个节点是一次对model worker的RPC，它的运行依赖于所有父节点生成的数据。源节点必然是data worker加载数据的操作，终止节点必然是对某个model worker的训练操作。
+ data worker：定义了dataloader，仅仅负责处理从数据集中加载数据的请求。
+ model worker：定义了某种类型的模型（可能是actor/critic/reward/reference，或是它们的shard）及其RPC interface，负责处理master worker发过来的RPC请求。interface可能包括：initialize、generate、inference、train、evaluation和save。根据数据流的不同，master worker可能发来的只是这些请求中的一部分，因此并不是所有模型都要实现所有的功能。

# Limitations

+ 没有实现tensor/pipeline parallism；
+ 每次master worker的RPC会把存在本地的数据发出去，并将生成的数据收回来存在本地，有固定的communication cost；
+ master worker的configuration写的不完善；
+ ...
