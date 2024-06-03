#!/bin/bash

srun --nodes=1 --gpus=tesla:$2 --cpus-per-gpu=8 --mem=$(($2*100))G --container-image=llm/llm-gpu:reallm-20240527 --nodelist=QH-com$1 \
     --container-mounts=$PWD:/distributed_llm,/lustre:/lustre --container-mount-home --export=PYTHONUSERBASE=/nonsense \
     --pty bash


# srun --nodes=1 --cpus-per-task=16 --mem=200G --container-image=llm/llm-cpu:reallm-20240527 --nodelist=QH-com$1 \
#      --container-mounts=$PWD:/distributed_llm,/lustre:/lustre --container-mount-home --export=PYTHONUSERBASE=/nonsense \
#      --pty bash