# syntax=docker/dockerfile:1.14-labs
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# Set environment variables for non-interactive installs and CUDA compilation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for building CUDA extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install ninja numpy jaxtyping rich tqdm wandb pillow tyro
RUN pip install gsplat

ENV TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"


# Create a non-root user
ARG USERNAME=devuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && mkdir -p /workspace && chown $USERNAME:$USERNAME /workspace

# Prefer node-local temp if Slurm provides it; fallback to /tmp.
ENV JOBTMP="/tmp"
ENV TORCH_EXTENSIONS_DIR="$JOBTMP/torch_extensions"
ENV XDG_CACHE_HOME="$JOBTMP/.cache"
ENV TMPDIR="$JOBTMP/tmp"
RUN mkdir -p "$TORCH_EXTENSIONS_DIR" "$XDG_CACHE_HOME" "$TMPDIR"

# reduce parallelism (optional but helps on shared FS/quotas)
ENV MAX_JOBS=32
# RUN mkdir /arf #truba icin
USER $USERNAME
#make dir

WORKDIR /workspace
