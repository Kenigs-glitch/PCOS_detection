FROM facefusion-docker-facefusion-cuda:latest

# Basic dependencies
RUN apt-get update && apt-get install -y \
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