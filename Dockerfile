
# Dockerfile
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_EXTENSIONS_DIR=/opt/torch_extensions \
    TORCH_CUDA_ARCH_LIST="8.6;8.9+PTX"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3-venv \
    git curl ca-certificates \
    build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Convenience symlinks
RUN ln -sf /usr/bin/python3.10 /usr/local/bin/python && ln -sf /usr/bin/pip3 /usr/local/bin/pip

RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch 2.4 + CUDA 12.4 wheels :contentReference[oaicite:0]{index=0}
RUN pip install \
    torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip install ninja numpy jaxtyping rich tqdm wandb pillow

RUN pip install tyro

RUN pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu124

WORKDIR /workspace
