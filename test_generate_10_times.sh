#!/bin/bash

for i in {1..10}
do
    python3 -m tests.test_generate
done


# nccl path: /usr/lib/x86_64-linux-gnu/libnccl.so.2, /usr/lib/x86_64-linux-gnu/libnccl.so