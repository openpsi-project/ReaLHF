#!/bin/bash

srun --gpus=tesla:1 --cpus-per-gpu=15 --mem=50G --nodelist=frl8a138 \
     --container-image=llm/llm-gpu --container-mounts=$PWD:/workspace,/data:/data,/lustre:/lustre \
     --pty bash