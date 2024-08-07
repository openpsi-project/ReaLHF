ARG REAL_CPU_BASE_IMAGE
ARG REAL_GPU_BASE_IMAGE

# >>>>>> CPU image
FROM ${REAL_CPU_BASE_IMAGE} as cpu

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
RUN pip3 install torch==2.3.1

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
FROM ${REAL_GPU_BASE_IMAGE} AS gpu

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

COPY ./requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt && rm /requirements.txt

# We don't use TransformerEngine's flash-attn integration, so it's okay to disrespect dependencies
RUN pip3 install git+https://github.com/NVIDIA/TransformerEngine.git@v1.8 --no-deps --no-build-isolation
RUN pip3 install flash-attn==2.4.2 --no-build-isolation
# Install grouped_gemm for MoE acceleration
RUN pip3 install grouped_gemm

COPY . /realhf
RUN REAL_CUDA=1 pip3 install -e /realhf --no-build-isolation
WORKDIR /realhf