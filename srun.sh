#!/bin/bash

srun --nodes=1 --gpus=tesla:8 --cpus-per-gpu=8 --mem=700G --container-image=llm/llm-gpu --nodelist=QH-com46 \
     --container-mounts=$PWD:/workspace,/lustre:/lustre --container-mount-home --export=PYTHONUSERBASE=/nonsense \
     --pty bash 
