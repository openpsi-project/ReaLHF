

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/source/images/real_logo_dark.svg">
    <img alt="ReaL" src="docs/source/images/real_logo.svg" width=55%>
  </picture>
</p>

<p align="center">
| <a href="https://openpsi-project.github.io/ReaLRLHF/"><b>Documentation</b></a> | <a href="https://openpsi-project.github.io/ReaLRLHF/"><b>Paper</b></a> |

</p>


<h1 align="center">
<em>ReaL</em>: Efficient RLHF Training for LLMs <br>with Parameter Reallocation
</h1>

***ReaL*** (the abbreviation for *<ins>ReaL</ins>location*)
is a distributed system for efficient RLHF training with LLMs.

ReaL proposes a novel approach
named *parameter reallocation*, which dynamically redistributes LLM parameters
in the cluster and adapts parallelization strategies during training.
By tailoring optimized allocations and parallelism degrees for each computation workload,
ReaL can largely eliminate redundant communication while maximizing GPU utilization.

ReaL can achieve substantially higher PPO training throughput than the state-of-the-art
open-source systems.

(In the following figure, with the increased number of GPUs,
we increase the model size from
LLaMA 7B, LLaMA 13B, CodeLLaMA 34B, to the largest LLaMA 70B.)

![Throughput Comparison](docs/source/images/vws.svg)

## Hightlights

### Efficiency

- The state-of-the-art training throughput for RLHF using **parameter reallocation**.

- Large-scale training with 3D parallelism, ZeRO optimization, and sequence parallelism.

- Memory-efficient training with parameter and optimizer offloading.

### Easy-to-use

- Seamless integration with HuggingFace checkpoints and inference frameworks like vLLM.

- A single command to launch local or distributed experiments.

Check our [tutorial](https://openpsi-project.github.io/ReaLHF/quickstart.html)
on reproducing the full RLHF procedure (SFT/RW/PPO) with 4$\times$LLaMA-7B
in **merely 30 minutes**.

### Flexibility

- Highly customization configuration options with Hydra structured config.

- ReaL supports commonly used RLHF algorithms, including DPO, PPO, RAFT, etc.

- ReaL supports customized algorithm interface within 100 lines of additional code.

Check our [customization guide](https://openpsi-project.github.io/ReaLHF/customization.html)
for hands-on examples.

## Getting Started

We provide pre-built [Docker images](https://openpsi-project.github.io/ReaLHF/install.html#docker-images)
and [PyPI packages](https://openpsi-project.github.io/ReaLHF/install.html#install-from-pypi-or-source).

```bash
pip3 install realhf --no-build-isolation
```

Please check our [documentation site](https://openpsi-project.github.io/ReaLRLHF/)
for detailed information:

- [Quickstart](https://openpsi-project.github.io/ReaLHF/quickstart.html)

- [Experiment Configurations](https://openpsi-project.github.io/ReaLHF/expconfig.html)

- [Code Architecture](https://openpsi-project.github.io/ReaLHF/arch.html)

- [Contributing](https://openpsi-project.github.io/ReaLHF/contributing.html)

## Acknowledgement

Apart from the authors of our paper,
we thank Shusheng Xu and Jiaxuan Gao from Tsinghua University,
Weilin Liu, Wenjie Ye, and Chuyi He from OpenPsi Inc
for thoroughly testing and using ReaL in their research, 
and for providing valuable suggestions 
that have greatly improved the system during this process.

## Citation

If you find our system useful for your research or production,
please cite our paper.
