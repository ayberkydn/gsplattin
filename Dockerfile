# Dockerfile for Gaussian Splatting Project with CUDA Support
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    python3-packaging \
    curl \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager for fast Python dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Create virtual environment and install dependencies
RUN uv venv --python 3.11
ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Install PyTorch with CUDA support
RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
RUN uv pip install gsplat pillow numpy wandb

# Install project in development mode
RUN uv pip install -e .

# Verify CUDA availability
RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
