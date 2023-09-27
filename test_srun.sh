#!/bin/bash

srun --gpus=tesla:4 --cpus-per-gpu=15 --mem-per-gpu=60G --nodelist=frl8a138 \
     --container-image=llm/llm-gpu \
     --container-mounts=/data:/data,/hddlustre:/hddlustre,/lustre:/lustre --container-mount-home \
     --pty bash