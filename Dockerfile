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
# Install PyTorch in advance to prevent rebuilding this large Docker layer.
RUN pip3 install torch

COPY ./requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt && rm /requirements.txt

COPY . /realhf
RUN REAL_CUDA=0 pip3 install -e /realhf --no-build-isolation
WORKDIR /realhf

# >>>>>> Documentation images
# FROM cpu AS docs-builder
# RUN pip install -U sphinx sphinx-nefertiti -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN sphinx-build -M html /realhf/docs/source/ /realhf/docs/build/
FROM nginx:alpine AS docs
COPY ./docs/build/html /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]

# >>>>>> GPU image
FROM nvcr.io/nvidia/pytorch:23.10-py3 AS gpu

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

# set environment variables for building transformer engine
ENV NVTE_WITH_USERBUFFERS=1 NVTE_FRAMEWORK=pytorch MAX_JOBS=8 MPI_HOME=/usr/local/mpi
ENV PATH="${PATH}:/opt/hpcx/ompi/bin:/opt/hpcx/ucx/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ompi/lib:/opt/hpcx/ucx/lib/"

# NOTE: we should also install flash_attn and transformer_engine in the image
# However, using `pip install flash_attn -no-build-isolation` will cause the
# building process to get stuck forever, so we have to pre-compile the wheel
# and install it locally. If these wheels do not exist in your docker build
# environment, please build them first.
# If you can't build these packages, use our provided docker images.

ENV TE_WHL_NAME=transformer_engine-1.7.0+4e7caa1-cp310-cp310-linux_x86_64.whl
ENV FLA_WHL_NAME=flash_attn-2.5.9-cp310-cp310-linux_x86_64.whl
COPY ./dist/$TE_WHL_NAME /$TE_WHL_NAME
# We don't use TransformerEngine's flash-attn integration, so it's okay to disrespect dependencies
RUN pip3 install /$TE_WHL_NAME --no-dependencies && rm /$TE_WHL_NAME
COPY ./dist/$FLA_WHL_NAME /$FLA_WHL_NAME
RUN pip3 install /$FLA_WHL_NAME && rm /$FLA_WHL_NAME

COPY . /realhf
RUN REAL_CUDA=1 pip3 install -e /realhf --no-build-isolation
WORKDIR /realhf
