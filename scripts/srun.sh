#!/bin/bash

# PACKAGE_DIR=/home/meizy/randomstuff/rlhf

srun --gpus=tesla:8 --cpus-per-gpu=10 --mem=400G --nodelist=frl8a138 --container-image=llm/llm-gpu \
     --container-mounts=$PWD:/workspace,/data:/data,/lustre:/lustre \
     --pty bash 

     # transformers 4.32.0, deepspeed 0.10.1