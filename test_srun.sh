#!/bin/bash

srun --gpus=tesla:6 --cpus-per-gpu=15 --mem=300G --nodelist=frl8a140 \
     --container-image=llm/llm-gpu --container-mounts=$PWD:/workspace,/data:/data,/lustre:/lustre \
     --pty bash