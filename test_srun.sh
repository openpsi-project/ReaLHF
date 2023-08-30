#!/bin/bash

srun --gpus=tesla:1 --cpus-per-gpu=32 --mem=100G \
     --container-image=llm/llm-gpu --container-mounts=$PWD:/workspace,/data:/data,/hddlustre:/hddlustre,/lustre:/lustre \
     --pty bash