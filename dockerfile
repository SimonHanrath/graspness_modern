# PyTorch 2.7.1 + CUDA 12.6 (nvcc + nvrtc present)
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive
ARG TZ=Europe/Berlin
ENV TZ=${TZ}

# Ensure CUDA-based extensions are compiled even if no GPUs are visible during docker build
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
# Architectures to compile CUDA extensions for (Turing/Ampere/Ada/Hopper)
ENV TORCH_CUDA_ARCH_LIST="6.1;7.5;8.0;8.6;8.9;9.0"
# Cache dir (rw for everyone to avoid permission issues with mounted volumes/CI)
ENV XDG_CACHE_HOME=/opt/cache

# System deps (toolchain, OpenCV, and common headers some builds require)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake ninja-build python3-opencv tzdata \
    pkg-config libjpeg-dev zlib1g-dev \
 && ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime \
 && dpkg-reconfigure -f noninteractive tzdata \
 && rm -rf /var/lib/apt/lists/*

# Python build tooling for native/CUDA extensions (needed by PyTorch3D, etc.)
RUN python -m pip install --no-cache-dir --upgrade \
      pip \
      "setuptools<75" \
      wheel \
      cmake \
      ninja \
      packaging \
      "scikit-build-core>=0.10"

# Core Python deps (modern, broadly compatible)
RUN pip install --no-cache-dir \
      "numpy>=2.3,<3" \
      "scipy>=1.16,<2" \
      "tqdm>=4.67,<5" \
      "tensorboard>=2.20,<3" \
      "protobuf>=6,<7" \
      "fvcore>=0.1.5,<0.2" \
      "iopath>=0.1.10,<0.2"

# spconv: use cu121 wheel (works on CUDA 12.x runtimes)
RUN pip install --no-cache-dir "spconv-cu121>=2.3,<2.5"

# PyTorch3D: build from source against installed torch/CUDA (no build isolation)
# Use the latest tag that exists (v0.7.7). If this fails against Torch 2.7.1,
# switch to cloning 'main' or use an older Torch (e.g., 2.4.1 + cu12.1).
RUN git clone --depth=1 --branch v0.7.7 https://github.com/facebookresearch/pytorch3d.git /tmp/pytorch3d \
 && python -m pip install --no-cache-dir --no-build-isolation -v /tmp/pytorch3d \
 && rm -rf /tmp/pytorch3d

# Prepare cache dir with open permissions
RUN mkdir -p /opt/cache && chmod -R 777 /opt/cache

WORKDIR /workspace
