#!/bin/bash

srun --nodes=1 --gpus=tesla:2 --cpus-per-gpu=8 --mem=200G --container-image=llm/llm-gpu --nodelist=QH-com49 \
     --container-mounts=$PWD:/workspace,/lustre:/lustre --container-mount-home --export=PYTHONUSERBASE=/nonsense \
     --pty bash 
