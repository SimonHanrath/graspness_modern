# PyTorch 2.3.1 + CUDA 12.1 (nvcc + nvrtc present)
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=Europe/Berlin
ENV TZ=${TZ}

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build python3-opencv tzdata \
 && ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime \
 && dpkg-reconfigure -f noninteractive tzdata \
 && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
      numpy==1.26.* scipy==1.11.* tqdm==4.* \
      tensorboard==2.14.* protobuf==4.25.* \
 && pip install --no-cache-dir "pytorch3d==0.7.8" \
      -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt231/download.html \
 && pip install --no-cache-dir "spconv-cu121>=2.3,<2.5"

# Compile for the SMs you actually target (drop +PTX to reduce runtime JIT)
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"

# Make kernel caches persistent/central
ENV XDG_CACHE_HOME=/opt/cache
RUN mkdir -p /opt/cache && chmod -R 777 /opt/cache

WORKDIR /workspace
