FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04

ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update &&\
    apt-get -y install build-essential yasm nasm \
    cmake unzip git wget tmux nano \
    sysstat libtcmalloc-minimal4 pkgconf autoconf libtool \
    python3 python3-pip python3-dev python3-setuptools \
    libsm6 libxext6 libxrender-dev \
    python3-tk libasound-dev libportaudio2 &&\
    ln -s /usr/bin/python3 /usr/bin/python &&\
    # ln -s /usr/bin/pip3 /usr/bin/pip &&\
    apt-get -y install libgl1-mesa-glx &&\
    apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 &&\
    apt-get install -y libsndfile1-dev &&\

    apt-get clean &&\
    apt-get autoremove &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf /var/cache/apt/archives/*

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir numpy
RUN pip3 install --no-cache-dir packaging

# Install PyTorch
# RUN pip3 install --no-cache-dir \
#     torch \
#     torchvision
RUN pip3 install --no-cache-dir torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

RUN git clone https://github.com/NVIDIA/apex &&\
    cd apex &&\
    # git checkout 3bae8c83494184673f01f3867fa051518e930895 &&\
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . &&\
    cd .. && rm -rf apex

# Install python packages
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH $PYTHONPATH:/workdir
ENV TORCH_HOME=/workdir/data/.torch

WORKDIR /workdir
