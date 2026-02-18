# syntax=docker/dockerfile:1.14-labs
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# Set environment variables for non-interactive installs and CUDA compilation
ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX" \
    FORCE_CUDA=1 \
    PIP_NO_CACHE_DIR=1

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

# Trigger JIT compilation of gsplat CUDA kernels during the build process.
# This requires BuildKit and the --device flag enabled by the labs syntax directive.
ENV MAX_JOBS=12
RUN --device=nvidia.com/gpu=all python -c "import torch; from gsplat import rasterization; \
    dev = torch.device('cuda'); \
    N = 100; \
    means = torch.randn((N, 3), device=dev); \
    quats = torch.nn.functional.normalize(torch.randn((N, 4), device=dev), dim=-1); \
    scales = torch.exp(torch.randn((N, 3), device=dev)); \
    opacities = torch.sigmoid(torch.randn((N,), device=dev)); \
    colors = torch.randn((N, 1, 3), device=dev); \
    viewmats = torch.eye(4, device=dev)[None]; \
    Ks = torch.tensor([[[32, 0, 16], [0, 32, 16], [0, 0, 1]]], dtype=torch.float32, device=dev); \
    out, _, _ = rasterization( \
    means=means, \
    quats=quats, \
    scales=scales, \
    opacities=opacities, \
    colors=colors, \
    viewmats=viewmats, \
    Ks=Ks, \
    width=32, \
    height=32, \
    sh_degree=0, \
    packed=False \
    ); \
    print(f'gsplat JIT compilation successful. Output shape: {out.shape}')"

WORKDIR /workspace
