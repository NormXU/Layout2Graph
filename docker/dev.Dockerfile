# syntax = docker/dockerfile:experimental
#
# This file can build images for cpu and gpu env. By default it builds image for CPU.
# Use following option to build image for cuda/GPU: --build-arg BASE_IMAGE=nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
# Here is complete command for GPU/cuda -
# $ DOCKER_BUILDKIT=1 docker build --file Dockerfile --build-arg BASE_IMAGE=nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04 -t torchserve:latest .
#
# Following comments have been shamelessly copied from https://github.com/pytorch/pytorch/blob/master/Dockerfile
#
# NOTE: To build this you will need a docker version > 18.06 with
#       experimental enabled and DOCKER_BUILDKIT=1
#
#       If you do not use buildkit you are not going to have a good time
#
#       For reference:
#           https://docs.docker.com/develop/develop-images/build_enhancements/


ARG BASE_IMAGE=ubuntu:18.04

FROM ${BASE_IMAGE} AS base

# This is useful for set this env
ARG BASE_IMAGE=ubuntu:18.04
RUN rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list

FROM dockerhub.datagrand.com/ysocr/file_image:release as file-server

FROM base AS compile-image
ENV PYTHONUNBUFFERED TRUE
RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y ca-certificates g++ \
#    python3.8-dev python3.8-distutils python3.8-venv \
    ca-certificates curl git g++ build-essential openssl libssl-dev libffi-dev \
    cmake openssh-client libsm6 libxext6 libjpeg-dev libpng-dev \
    && rm -rf /var/lib/apt/lists/*
#    && cd /tmp \
#    && curl -O https://bootstrap.pypa.io/get-pip.py \
#    && python3.8 get-pip.py

COPY --from=file-server /data/download_packages/python/Python-3.8.12.tgz /home/Python-3.8.12.tgz
RUN cd /home && tar zxvf Python-3.8.12.tgz && rm -rf Python-3.8.12.tgz && cd Python-3.8.12 && \
    ./configure --without-doc-strings --prefix=/usr/local/lib/python3.8.12 && make -j 8 && make install && rm -rf /home/Python-3.8.12

RUN update-alternatives --install /usr/bin/python python /usr/local/lib/python3.8.12/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/local/lib/python3.8.12/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python3m python3m /usr/local/lib/python3.8.12/bin/python3.8 1 \
    && update-alternatives --install /usr/local/bin/pip pip /usr/local/lib/python3.8.12/bin/pip3 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/local/lib/python3.8.12/bin/pip3 1

RUN python -m venv /home/venv

ENV PATH="/home/venv/bin:$PATH"

RUN pip install --no-cache-dir -U pip setuptools \
    -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com

# install opencv

# This is only useful for cuda env
RUN export USE_CUDA=1

ARG CUDA_VERSION=""

RUN TORCH_VER=$(curl --silent --location https://pypi.org/pypi/torch/json | python -c "import sys, json, pkg_resources; releases = json.load(sys.stdin)['releases']; print(sorted(releases, key=pkg_resources.parse_version)[-1])") && \
    TORCH_VISION_VER=$(curl --silent --location https://pypi.org/pypi/torchvision/json | python -c "import sys, json, pkg_resources; releases = json.load(sys.stdin)['releases']; print(sorted(releases, key=pkg_resources.parse_version)[-1])") && \
    # Specify TORCH_VER and TORCH_VISION_VER
    if echo "$BASE_IMAGE" | grep -q "cuda:"; then \
        # Install CUDA version specific binary when CUDA version is specified as a build arg
        if [ "$CUDA_VERSION" ]; then \
            pip install --no-cache-dir torch==$TORCH_VER+$CUDA_VERSION torchvision==$TORCH_VISION_VER+$CUDA_VERSION -f https://download.pytorch.org/whl/torch_stable.html \
            -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com; \
        # Install the binary with the latest CUDA version support
        else \
            pip install --no-cache-dir torch torchvision -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com; \
        fi \
    # Install the CPU binary
    else \
        pip install --no-cache-dir torch==$TORCH_VER+cpu torchvision==$TORCH_VISION_VER+cpu -f https://download.pytorch.org/whl/torch_stable.html \
         -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com; \
    fi
RUN pip install -i https://mirrors.aliyun.com/pypi/simple --no-cache-dir Cython>=0.29

COPY .ssh /root/.ssh
COPY all_requirements.txt all_requirements.txt
COPY requirements requirements
RUN chmod 600 /root/.ssh/id_rsa
RUN pip install --no-cache-dir -r all_requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com && \
    rm -rf /root/.ssh && rm -rf all_requirements.txt  && rm -rf requirements


# Final image for production
FROM base AS dev-image
MAINTAINER dockerhub.datagrand.com
ARG BASE_IMAGE=ubuntu:18.04
ENV PYTHONUNBUFFERED TRUE

RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    python3.8 python3.8-dev python3.8-distutils openssh-server git vim openjdk-11-jdk  \
    build-essential  ca-certificates  dpkg-dev fakeroot sudo git curl wget \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python3m python3m /usr/bin/python3.8 1

RUN mkdir -p /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
EXPOSE 22 8080 8081 8082 7070 7071
ENV TZ Asia/Shanghai

COPY --from=compile-image /home/venv /home/venv

ENV PATH="/home/venv/bin:$PATH"
WORKDIR /home/ysocr/
ENV WORK_DIR /home/ysocr

COPY . ${WORK_DIR}
COPY --from=file-server /data/models/yslm_models/models ${WORK_DIR}/data/pretrain/layoutlm


RUN mkdir /data && \
    ln -s $WORK_DIR/log /data/log && \
    ln -s $WORK_DIR/data /data/data

CMD /usr/sbin/sshd -D



