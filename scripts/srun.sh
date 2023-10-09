#!/bin/bash

# PACKAGE_DIR=/home/meizy/randomstuff/rlhf

srun --gpus=tesla:1 --cpus-per-gpu=10 --mem=100G --container-image=llm/llm-gpu \
     --container-mounts=$PWD:/workspace,/data:/data \
     --pty bash 

     # transformers 4.32.0, deepspeed 0.10.1