FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Basic dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Set environment
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Default command
CMD ["bash"] 