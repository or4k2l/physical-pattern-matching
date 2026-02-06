# Robust Vision Production Image - Multi-Stage Build
# Repository: https://github.com/or4k2l/robust-vision

# ============================================
# Stage 1: Builder (with dev dependencies)
# ============================================
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS builder

LABEL maintainer="or4k2l"
LABEL repository="https://github.com/or4k2l/robust-vision"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python and build dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . /app/

# Install the package
RUN pip install --no-cache-dir -e .

# ============================================
# Stage 2: Runtime (minimal, production-ready)
# ============================================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

LABEL maintainer="or4k2l"
LABEL repository="https://github.com/or4k2l/robust-vision"
LABEL description="Production-ready robust vision training framework"

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV JAX_PLATFORM_NAME=cuda
ENV PYTHONDONTWRITEBYTECODE=1

# Install Python (runtime only)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /app /app

# Create directories for data, checkpoints, and logs
RUN mkdir -p /app/data /app/checkpoints /app/logs /app/results

# Expose TensorBoard port
EXPOSE 6006

# Create volume mount points
VOLUME ["/app/data", "/app/checkpoints", "/app/results"]

# Set the default command
CMD ["python", "scripts/train.py", "--config", "configs/baseline.yaml"]
