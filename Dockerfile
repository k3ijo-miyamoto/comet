FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# 基本ツールのインストール
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    python3.8 \
    python3-pip \
    python3.8-dev \
    && apt-get clean

# シンボリックリンク
RUN ln -s /usr/bin/python3.8 /usr/bin/python \
    && ln -s /usr/bin/pip3 /usr/bin/pip

# 作業ディレクトリ
WORKDIR /workspace

# UniADのクローン
RUN git clone https://github.com/OpenDriveLab/UniAD.git

WORKDIR /workspace/UniAD

# torch, torchvision, mmcv などのバージョンは調整要
RUN pip install --upgrade pip && \
    pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118

# UniADの依存ライブラリインストール
RUN pip install -r requirements.txt

# MMDetection3D + MMEngine系も必要（多くはここでつまずく）
RUN pip install \
    mmcv-full==1.7.1 \
    mmdet==2.28.2 \
    mmdet3d==1.0.0rc6 \
    openmim && \
    mim install mmsegmentation

# Detectron2系（MapFormerなどが利用）
RUN pip install git+https://github.com/facebookresearch/detectron2.git

# pythonpathを設定
ENV PYTHONPATH=/workspace/UniAD:$PYTHONPATH
