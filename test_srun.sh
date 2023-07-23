#!/bin/bash

srun --gpus=tesla:1 --cpus-per-gpu=32 --mem=100G \
     --container-image=meizy/deepspeed:0.9.2 --container-mounts=$PWD:/workspace,/data:/data,/hddlustre:/hddlustre \
     --pty bash