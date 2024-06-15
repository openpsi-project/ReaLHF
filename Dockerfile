# >>>>>> CPU image
FROM ubuntu:22.04 as cpu

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install -y ca-certificates
RUN sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
RUN sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
RUN apt update
RUN apt install -y net-tools python3-pip pkg-config libopenblas-base libopenmpi-dev git

RUN pip3 install -U pip
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

COPY ./requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt && rm /requirements.txt

COPY . /reallm
RUN pip3 install -e /reallm --no-build-isolation
WORKDIR /reallm

# >>>>>> Documentation images
# FROM cpu AS docs-builder
# RUN pip install -U sphinx sphinx-nefertiti -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN sphinx-build -M html /reallm/docs/source/ /reallm/docs/build/
FROM nginx:alpine AS docs
COPY ./docs/build/html /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]

# >>>>>> GPU image
FROM nvcr.io/nvidia/pytorch:23.10-py3 AS base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install -y ca-certificates
RUN sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
RUN sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
RUN apt update
RUN apt install -y net-tools \
    libibverbs-dev librdmacm-dev ibverbs-utils \
    rdmacm-utils python3-pyverbs opensm ibutils perftest

RUN pip3 install -U pip
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

COPY . /reallm
RUN pip3 install -e /reallm --no-build-isolation
WORKDIR /reallm

ENV PATH="${PATH}:/opt/hpcx/ompi/bin:/opt/hpcx/ucx/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ompi/lib:/opt/hpcx/ucx/lib/"

# >>>>>> GPU image with TransformerEngine
FROM base AS te

# set environment variables for building transformer engine
ENV NVTE_WITH_USERBUFFERS=1
ENV NVTE_FRAMEWORK=pytorch
ENV MPI_HOME=/usr/local/mpi
ENV MAX_JOBS=64

WORKDIR /tmp
RUN git clone --depth 1 --branch v1.6 https://github.com/NVIDIA/TransformerEngine.git
WORKDIR /tmp/TransformerEngine
RUN pip3 uninstall -y transformer_engine && pip3 install /tmp/TransformerEngine --no-build-isolation
RUN rm -rf /tmp/TransformerEngine /tmp/flash_attn-2.3.6-cp310-cp310-linux_x86_64.whl /tmp/cugae-0.1.0-cp310-cp310-linux_x86_64.whl /tmp/pytorch

RUN pip install flash-attn --no-build-isolation

WORKDIR /reallm

