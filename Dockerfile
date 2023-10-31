# Use the base Ubuntu 20.04 image with NVIDIA CUDA support
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        software-properties-common \
        git \
        python3-pip \
        python3.9 \
        python3.9-dev \
        python3-opencv \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*  

# Create symbolic links for Python and pip
RUN ln -sf /usr/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/bin/python3.9 /usr/bin/python3 && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install any python packages you need
COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt

# Install PyTorch and torchvision
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Set the working directory
WORKDIR /app

# Set the entrypoint
ENTRYPOINT [ "python3" ]
