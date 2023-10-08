---
pipeline_tag: text-generation
inference: true
widget:
- text: 'def print_hello_world():'
  example_title: Hello world
  group: Python
license: bigcode-openrail-m
datasets:
- bigcode/the-stack-dedup
metrics:
- code_eval
library_name: transformers
tags:
- code
model-index:
- name: StarCoder
  results:
  - task:
      type: text-generation
    dataset:
      type: openai_humaneval
      name: HumanEval (Prompted)
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.408
      verified: false
  - task:
      type: text-generation
    dataset:
      type: openai_humaneval
      name: HumanEval
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.336
      verified: false
  - task:
      type: text-generation
    dataset:
      type: mbpp
      name: MBPP
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.527
      verified: false
  - task:
      type: text-generation
    dataset:
      type: ds1000
      name: DS-1000 (Overall Completion)
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.26
      verified: false
  - task:
      type: text-generation
    dataset:
      type: nuprl/MultiPL-E
      name: MultiPL-HumanEval (C++)
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.3155
      verified: false
  - task:
      type: text-generation
    dataset:
      type: nuprl/MultiPL-E
      name: MultiPL-HumanEval (C#)
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.2101
      verified: false
  - task:
      type: text-generation
    dataset:
      type: nuprl/MultiPL-E
      name: MultiPL-HumanEval (D)
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.1357
      verified: false
  - task:
      type: text-generation
    dataset:
      type: nuprl/MultiPL-E
      name: MultiPL-HumanEval (Go)
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.1761
      verified: false
  - task:
      type: text-generation
    dataset:
      type: nuprl/MultiPL-E
      name: MultiPL-HumanEval (Java)
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.3022
      verified: false
  - task:
      type: text-generation
    dataset:
      type: nuprl/MultiPL-E
      name: MultiPL-HumanEval (Julia)
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.2302
      verified: false
  - task:
      type: text-generation
    dataset:
      type: nuprl/MultiPL-E
      name: MultiPL-HumanEval (JavaScript)
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.3079
      verified: false
  - task:
      type: text-generation
    dataset:
      type: nuprl/MultiPL-E
      name: MultiPL-HumanEval (Lua)
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.2389
      verified: false
  - task:
      type: text-generation
    dataset:
      type: nuprl/MultiPL-E
      name: MultiPL-HumanEval (PHP)
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.2608
      verified: false
  - task:
      type: text-generation
    dataset:
      type: nuprl/MultiPL-E
      name: MultiPL-HumanEval (Perl)
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.1734
      verified: false
  - task:
      type: text-generation
    dataset:
      type: nuprl/MultiPL-E
      name: MultiPL-HumanEval (Python)
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.3357
      verified: false
  - task:
      type: text-generation
    dataset:
      type: nuprl/MultiPL-E
      name: MultiPL-HumanEval (R)
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.155
      verified: false
  - task:
      type: text-generation
    dataset:
      type: nuprl/MultiPL-E
      name: MultiPL-HumanEval (Ruby)
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.0124
      verified: false
  - task:
      type: text-generation
    dataset:
      type: nuprl/MultiPL-E
      name: MultiPL-HumanEval (Racket)
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.0007
      verified: false
  - task:
      type: text-generation
    dataset:
      type: nuprl/MultiPL-E
      name: MultiPL-HumanEval (Rust)
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.2184
      verified: false
  - task:
      type: text-generation
    dataset:
      type: nuprl/MultiPL-E
      name: MultiPL-HumanEval (Scala)
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.2761
      verified: false
  - task:
      type: text-generation
    dataset:
      type: nuprl/MultiPL-E
      name: MultiPL-HumanEval (Bash)
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.1046
      verified: false
  - task:
      type: text-generation
    dataset:
      type: nuprl/MultiPL-E
      name: MultiPL-HumanEval (Swift)
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.2274
      verified: false
  - task:
      type: text-generation
    dataset:
      type: nuprl/MultiPL-E
      name: MultiPL-HumanEval (TypeScript)
    metrics:
    - name: pass@1
      type: pass@1
      value: 0.3229
      verified: false
extra_gated_prompt: >-
  ## Model License Agreement

  Please read the BigCode [OpenRAIL-M
  license](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement)
  agreement before accepting it.
    
extra_gated_fields:
  I accept the above license agreement, and will use the Model complying with the set of use restrictions and sharing requirements: checkbox
---


# StarCoder

![banner](https://huggingface.co/datasets/bigcode/admin/resolve/main/StarCoderBanner.png)

Play with the model on the [StarCoder Playground](https://huggingface.co/spaces/bigcode/bigcode-playground).

##  Table of Contents

1. [Model Summary](##model-summary)
2. [Use](##use)
3. [Limitations](##limitations)
4. [Training](##training)
5. [License](##license)
6. [Citation](##citation)

## Model Summary

The StarCoder models are 15.5B parameter models trained on 80+ programming languages from [The Stack (v1.2)](https://huggingface.co/datasets/bigcode/the-stack), with opt-out requests excluded. The model uses [Multi Query Attention](https://arxiv.org/abs/1911.02150), [a context window of 8192 tokens](https://arxiv.org/abs/2205.14135),  and was trained using the [Fill-in-the-Middle objective](https://arxiv.org/abs/2207.14255) on 1 trillion tokens. 

- **Repository:** [bigcode/Megatron-LM](https://github.com/bigcode-project/Megatron-LM)
- **Project Website:** [bigcode-project.org](https://www.bigcode-project.org)
- **Paper:** [💫StarCoder: May the source be with you!](https://arxiv.org/abs/2305.06161)
- **Point of Contact:** [contact@bigcode-project.org](mailto:contact@bigcode-project.org)
- **Languages:** 80+ Programming languages


## Use

### Intended use

The model was trained on GitHub code. As such it is _not_ an instruction model and commands like "Write a function that computes the square root." do not work well. However, by using the [Tech Assistant prompt](https://huggingface.co/datasets/bigcode/ta-prompt) you can turn it into a capable technical assistant.

**Feel free to share your generations in the Community tab!**

### Generation
```python
# pip install -q transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigcode/starcoder"
device = "cuda" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
```

### Fill-in-the-middle
Fill-in-the-middle uses special tokens to identify the prefix/middle/suffix part of the input and output:

```python
input_text = "<fim_prefix>def print_hello_world():\n    <fim_suffix>\n    print('Hello world!')<fim_middle>"
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
outputs = model.generate(inputs)
print(tokenizer.decode(outputs[0]))
```

### Attribution & Other Requirements

The pretraining dataset of the model was filtered for permissive licenses only. Nevertheless, the model can generate source code verbatim from the dataset. The code's license might require attribution and/or other specific requirements that must be respected. We provide a [search index](https://huggingface.co/spaces/bigcode/starcoder-search) that let's you search through the pretraining data to identify where generated code came from and apply the proper attribution to your code.

# Limitations

The model has been trained on source code from 80+ programming languages. The predominant natural language in source code is English although other languages are also present. As such the model is capable of generating code snippets provided some context but the generated code is not guaranteed to work as intended. It can be inefficient, contain bugs or exploits. See [the paper](https://drive.google.com/file/d/1cN-b9GnWtHzQRoE7M7gAEyivY0kl4BYs/view) for an in-depth discussion of the model limitations. 

# Training

## Model

- **Architecture:** GPT-2 model with multi-query attention and Fill-in-the-Middle objective
- **Pretraining steps:** 250k
- **Pretraining tokens:** 1 trillion
- **Precision:** bfloat16

## Hardware

- **GPUs:** 512 Tesla A100
- **Training time:** 24 days

## Software

- **Orchestration:** [Megatron-LM](https://github.com/bigcode-project/Megatron-LM)
- **Neural networks:** [PyTorch](https://github.com/pytorch/pytorch)
- **BP16 if applicable:** [apex](https://github.com/NVIDIA/apex)

# License
The model is licensed under the BigCode OpenRAIL-M v1 license agreement. You can find the full agreement [here](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement).
# Citation
```
@article{li2023starcoder,
      title={StarCoder: may the source be with you!}, 
      author={Raymond Li and Loubna Ben Allal and Yangtian Zi and Niklas Muennighoff and Denis Kocetkov and Chenghao Mou and Marc Marone and Christopher Akiki and Jia Li and Jenny Chim and Qian Liu and Evgenii Zheltonozhskii and Terry Yue Zhuo and Thomas Wang and Olivier Dehaene and Mishig Davaadorj and Joel Lamy-Poirier and João Monteiro and Oleh Shliazhko and Nicolas Gontier and Nicholas Meade and Armel Zebaze and Ming-Ho Yee and Logesh Kumar Umapathi and Jian Zhu and Benjamin Lipkin and Muhtasham Oblokulov and Zhiruo Wang and Rudra Murthy and Jason Stillerman and Siva Sankalp Patel and Dmitry Abulkhanov and Marco Zocca and Manan Dey and Zhihan Zhang and Nour Fahmy and Urvashi Bhattacharyya and Wenhao Yu and Swayam Singh and Sasha Luccioni and Paulo Villegas and Maxim Kunakov and Fedor Zhdanov and Manuel Romero and Tony Lee and Nadav Timor and Jennifer Ding and Claire Schlesinger and Hailey Schoelkopf and Jan Ebert and Tri Dao and Mayank Mishra and Alex Gu and Jennifer Robinson and Carolyn Jane Anderson and Brendan Dolan-Gavitt and Danish Contractor and Siva Reddy and Daniel Fried and Dzmitry Bahdanau and Yacine Jernite and Carlos Muñoz Ferrandis and Sean Hughes and Thomas Wolf and Arjun Guha and Leandro von Werra and Harm de Vries},
      year={2023},
      eprint={2305.06161},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```