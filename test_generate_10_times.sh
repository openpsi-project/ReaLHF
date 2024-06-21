#!/bin/bash

for i in {1..10}
do
    python3 -m tests.test_generate
done


# nccl path: /usr/lib/x86_64-linux-gnu/libnccl.so.2, /usr/lib/x86_64-linux-gnu/libnccl.so

# torch.Size([2369])
# torch.Size([16, 1])
# torch.Size([16])
# torch.Size([16, 513, 8, 128])
# rank 2: Capturing CUDA graph for decoding
# torch.Size([16, 513, 8, 128])
# 256
# torch.Size([17])
# None

# torch.Size([2369])
# torch.Size([16, 1])
# torch.Size([16])
# torch.Size([16, 513, 8, 128])
# torch.Size([16, 513, 8, 128])
# 256
# torch.Size([17])
# None