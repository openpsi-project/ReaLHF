#!/bin/bash

srun --nodes=1 --gpus=tesla:$2 --cpus-per-gpu=16 --mem=$(($2*300))G --container-image=llm/llm-gpu --nodelist=QH-com$1 \
     --container-mounts=$PWD:/workspace,/lustre:/lustre --container-mount-home --export=PYTHONUSERBASE=/nonsense \
     --pty bash 
