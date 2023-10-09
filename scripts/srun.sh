#!/bin/bash

PACKAGE_DIR=/home/meizy/randomstuff/rlhf

srun --gpus=tesla:1 --cpus-per-gpu=10 --mem=100G --nodelist=frl8a141 --container-image=meizy/llm-gpu:editable-installs \
     --container-mounts=$PWD:/workspace,/lustre:/lustre,$PACKAGE_DIR/transformers:/transformers,$PACKAGE_DIR/deepspeed:/deepspeed,$PACKAGE_DIR/deepspeedchat/logs:/datafiles \
     --pty bash 

     # transformers 4.32.0, deepspeed 0.10.1