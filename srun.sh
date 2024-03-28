#!/bin/bash

srun --nodes=1 --gpus=tesla:2 --cpus-per-gpu=8 --mem=100G --container-image=llm/llm-gpu --nodelist=QH-com48 \
     --container-mounts=$PWD:/workspace,/lustre:/lustre --container-mount-home --export=PYTHONUSERBASE=/nonsense \
     --pty bash 
