#!/bin/sh

python3 -m apps.remote reset_name_resolve -e test -f test
CUDA_DEVICE_MAX_CONNECTIONS=1 \
OMP_NUM_THREADS=8 \
MASTER_ADDR=localhost \
MASTER_PORT=7777 \
torchrun --standalone --nnodes=1 --nproc-per-node=4 --module \
    tests.torch_profile_example