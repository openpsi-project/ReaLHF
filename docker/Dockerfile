ARG base_image

FROM ${base_image}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install -y ca-certificates
RUN sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
RUN sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
RUN apt update
RUN apt install -y libpgm-dev net-tools \
    libibverbs-dev librdmacm-dev ibverbs-utils \
    rdmacm-utils python3-pyverbs opensm ibutils perftest

RUN pip3 install -U pip
# RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install the latest NCCL
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt-get update
RUN apt install libnccl2=2.20.5-1+cuda12.2 libnccl-dev=2.20.5-1+cuda12.2

# build pytorch with the latest NCCL version
ENV NCCL_INCLUDE_DIR="/usr/include/" NCCL_LIB_DIR="/usr/lib/" USE_SYSTEM_NCCL=1

# build pytorch from source, such that we can use MPI backend
RUN apt-get update && apt-get -y install ninja-build
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
ENV _GLIBCXX_USE_CXX11_ABI=1
ENV TORCH_CUDA_ARCH_LIST="8.0;9.0"
RUN pip3 uninstall -y torch
WORKDIR /tmp
RUN git clone --depth 1 --branch v2.1.0 https://github.com/pytorch/pytorch
WORKDIR /tmp/pytorch
RUN git submodule sync && git submodule update --init --recursive
RUN pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN python3 setup.py install

# install other utility pip packages
RUN pip3 install -U ray[default,serve,data] -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install -U transformers huggingface_hub datasets accelerate bitsandbytes -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install -U deepspeed -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip3 install -U blosc cmake prometheus_client wandb redis scipy h5py nltk \
    ninja packaging tensorboardx sentencepiece numba viztracer[full] pyzmq colorlog colorama hydra-core \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

ENV PATH="${PATH}:/opt/hpcx/ompi/bin:/opt/hpcx/ucx/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ompi/lib:/opt/hpcx/ucx/lib/"

# # Add an ARG to force a rebuild of the image from this point.
# ARG INCUBATOR_VER=unknown

# set environment variables for building transformer engine
ENV NVTE_WITH_USERBUFFERS=1
ENV NVTE_FRAMEWORK=pytorch
ENV MPI_HOME=/usr/local/mpi
ENV MAX_JOBS=64

# install transformer engine
# note that we should in stall transformer_engine before flash attention,
# because the former forces to install a specific flash attn version
COPY thirdparty/TransformerEngine /tmp/TransformerEngine
WORKDIR /tmp/TransformerEngine
RUN pip3 uninstall -y transformer_engine && pip3 install /tmp/TransformerEngine

# install flash attention and cugae
# compile from scratch will cause OOM, install a prebuilt wheel
COPY thirdparty/flash-attention/dist/flash_attn-2.3.6-cp310-cp310-linux_x86_64.whl /tmp/flash_attn-2.3.6-cp310-cp310-linux_x86_64.whl
RUN pip3 uninstall -y flash_attn && pip3 install /tmp/flash_attn-2.3.6-cp310-cp310-linux_x86_64.whl
COPY thirdparty/cugae/dist/cugae-0.1.0-cp310-cp310-linux_x86_64.whl /tmp/cugae-0.1.0-cp310-cp310-linux_x86_64.whl
RUN pip3 install /tmp/cugae-0.1.0-cp310-cp310-linux_x86_64.whl

# final clean up
WORKDIR /app
RUN rm -rf /tmp/TransformerEngine /tmp/flash_attn-2.3.6-cp310-cp310-linux_x86_64.whl /tmp/cugae-0.1.0-cp310-cp310-linux_x86_64.whl /tmp/pytorch

COPY dist/dllm_cuda-0.0.0-cp310-cp310-linux_x86_64.whl /dllm_cuda-0.0.0-cp310-cp310-linux_x86_64.whl
RUN pip3 install /dllm_cuda-0.0.0-cp310-cp310-linux_x86_64.whl && rm /dllm_cuda-0.0.0-cp310-cp310-linux_x86_64.whl

COPY dist/dllm_cuda-0.0.0-cp310-cp310-linux_x86_64.whl /tmp/dllm_cuda-0.0.0-cp310-cp310-linux_x86_64.whl
RUN pip3 install /tmp/dllm_cuda-0.0.0-cp310-cp310-linux_x86_64.whl && rm /tmp/dllm_cuda-0.0.0-cp310-cp310-linux_x86_64.whl
# install zmq with pgm
# COPY ./zeromq-4.3.4.tar.gz /
# WORKDIR /
# RUN tar zxvf zeromq-4.3.4.tar.gz
# WORKDIR /zeromq-4.3.4
# RUN ./configure --with-pgm && make && make install
# WORKDIR /
# RUN pip3 install -I --no-binary=:all: pyzmq
