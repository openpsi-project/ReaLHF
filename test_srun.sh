#!/bin/bash

srun --gpus=tesla:1 --cpus-per-gpu=16 --mem=100G --nodelist=frl8a141 \
     --container-image=llm/llm-gpu --container-mounts=$PWD:/workspace,/data:/data,/hddlustre:/hddlustre,/lustre:/lustre \
     --pty bash