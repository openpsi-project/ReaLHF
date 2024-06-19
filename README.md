# ***ReaL***: Efficient RLHF Training for LLMs with Parameter Reallocation

## Introduction

***ReaL*** (the abbreviation for *<u>ReaL</u>location*)
is a distributed system for efficient RLHF training with LLMs.
ReaL introduces a novel technique called *parameter reallocation*,
which dynamically moves model parameters and changes the parallel strategies of models
during training.
By tailoring optimized allocations and parallelism degrees for each computation workload,
ReaL can largely reduce the communication overhead caused by parallelization,
while minimizing the GPU idle time.
This can be done automatically by ReaL's customized search engine.

ReaL can achieve substantially higher PPO training throughput than the state-of-the-art
open-source systems:

![Throughput Comparison](docs/source/images/vws.svg)

## Installation and Usage


**We provide hands-on tutorials to use ReaL and reproduce the full procedure of RLHF**
**with 4x LLaMA-7B models within just half an hour!**

Please check our documentation site.

[Documentation](https://openpsi-project.github.io/ReaLRLHF/)


## Acknowledgement

## Citation

If you find our system useful for your research or production,
please cite our paper.
