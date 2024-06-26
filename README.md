

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/source/images/real_logo_dark.svg">
    <img alt="ReaL" src="docs/source/images/real_logo.svg" width="55%">
  </picture>
</p>

<p align="center">
| <a href="https://openpsi-project.github.io/ReaLHF/"><b>Documentation</b></a> | <a href="https://arxiv.org/abs/2406.14088"><b>Paper</b></a> |
</p>

<h1 align="center">
<em>ReaL</em>: Efficient RLHF Training for LLMs <br>with Parameter Reallocation
</h1>

***ReaL*** (short for *<ins>ReaL</ins>location*) is a distributed system designed for efficient RLHF training with LLMs.

ReaL introduces a novel approach called *parameter reallocation*, which dynamically redistributes LLM parameters across the cluster and adapts parallelization strategies during training. By optimizing allocations and parallelism for each computation workload, ReaL minimizes redundant communication while maximizing GPU utilization.

ReaL achieves significantly higher PPO training throughput compared to state-of-the-art open-source systems.

(In the following figure, as the number of GPUs increases, the model size scales up from LLaMA 7B, LLaMA 13B, and CodeLLaMA 34B, to the largest LLaMA 70B.)

![Throughput Comparison](docs/source/images/vws.svg)

## Highlights

### Efficiency

- Achieves state-of-the-art training throughput for RLHF using **parameter reallocation**.
- Supports large-scale training with 3D parallelism, ZeRO optimization, and sequence parallelism.
- Enables memory-efficient training with parameter and optimizer offloading.

### Ease of Use

- Seamlessly integrates with HuggingFace checkpoints and inference frameworks like vLLM.
- Allows launching local or distributed experiments with a single command.

Check out our [tutorial](https://openpsi-project.github.io/ReaLHF/quickstart.html) to reproduce the full RLHF procedure (SFT/RW/PPO) with 4×LLaMA-7B in just **30 minutes**.

### Flexibility

- Offers versatile configuration customization with Hydra structured config.
- Supports many commonly used RLHF algorithms, including DPO, PPO, RAFT, and more.
- Allows the addition of custom algorithms with fewer than 100 lines of code.

Refer to our [customization guide](https://openpsi-project.github.io/ReaLHF/customization.html) for hands-on examples.

## Getting Started

We provide pre-built [Docker images](https://openpsi-project.github.io/ReaLHF/install.html#docker-images) and [PyPI packages](https://openpsi-project.github.io/ReaLHF/install.html#install-from-pypi-or-source).

```bash
pip3 install realhf --no-build-isolation
```

For detailed information, please visit our [documentation site](https://openpsi-project.github.io/ReaLHF/).

- [Quickstart](https://openpsi-project.github.io/ReaLHF/quickstart.html)

- [Experiment Configurations](https://openpsi-project.github.io/ReaLHF/expconfig.html)

- [Code Architecture](https://openpsi-project.github.io/ReaLHF/arch.html)

- [Contributing](https://openpsi-project.github.io/ReaLHF/contributing.html)

## Acknowledgement

We would like to thank the authors of our paper and the following individuals for their contributions: Shusheng Xu and Jiaxuan Gao from Tsinghua University, and Weilin Liu, Wenjie Ye, and Chuyi He from OpenPsi Inc, for thoroughly testing and using ReaL in their research, and for providing valuable suggestions that greatly improved the system.

## Citation

If you find our system useful for your research or production, please cite our [paper](https://arxiv.org/abs/2406.14088).

```
@misc{mei2024realhf,
      title={ReaLHF: Optimized RLHF Training for Large Language Models through Parameter Reallocation}, 
      author={Zhiyu Mei and Wei Fu and Kaiwei Li and Guangju Wang and Huanchen Zhang and Yi Wu},
      year={2024},
      eprint={2406.14088},
      archivePrefix={arXiv},
}
```
