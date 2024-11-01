FROM tensorflow/tensorflow:2.6.0-gpu-jupyter AS base

########################################################################################################################
# Manually install cmake-3.12
# tensorflow-model-optimization installation requires 3.12 or later

# Get the public key for the CUDA repository
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    gpg \
    wget

RUN wget https://github.com/Kitware/CMake/releases/download/v3.12.0/cmake-3.12.0-Linux-x86_64.sh \
    && chmod +x cmake-3.12.0-Linux-x86_64.sh \
    && ./cmake-3.12.0-Linux-x86_64.sh --skip-license --prefix=/usr/local \
    && rm cmake-3.12.0-Linux-x86_64.sh

########################################################################################################################
# Python dependencies

# Install python dependencies
RUN apt-get update \
    && apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    python3-dev \
    unzip \
    vim

# Install python packages
RUN pip3 install --upgrade pip \
    && pip3 install \
    numpy \
    matplotlib \
    scikit-learn \
    pandas \
    dm-tree==0.1.6

# Install tensorflow-model-optimization
# Keep this in a separate RUN command because
# dm-tree needs to be installed before tensorflow-model-optimization
RUN pip3 install \
    tensorflow-model-optimization

# Install linting tools
RUN pip3 install \
    autoflake \
    black \
    docformatter \
    isort

########################################################################################################################
# User setup

ARG USERID
ARG GROUPID
ARG USER
ARG GROUP

RUN apt-get install -y \
    sudo

RUN addgroup --gid $GROUPID $GROUP \
    && adduser --uid $USERID --ingroup $GROUP --home /home/$USER --shell /bin/sh --disabled-password --gecos "" $USER \
    && adduser $USER sudo \
    && echo "$USER ALL=NOPASSWD: ALL" >> /etc/sudoers.d/$USER \
    && chown -R $USER:$GROUP /home/$USER

USER $USER:$GROUP

# Source .bashrc by default
COPY .bashrc /home/$USER/.bashrc

WORKDIR /home/$USER/qtensor

ENV PYTHONPATH="${PYTHONPATH}:/home/$USER/qtensor/src"


CMD ["/bin/bash"]
