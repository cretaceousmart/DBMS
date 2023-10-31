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

# Force Install attrs==23.1.0 and pyrdf2vec==0.2.3
# Because the library restrition of pyrdf2vec is wrong
RUN pip install attrs==23.1.0
RUN pip install pyrdf2vec==0.2.3

# Install Jupyter Notebook and IPython kernel
RUN pip3 install jupyter ipykernel

# Set the working directory
WORKDIR /app

# Start Jupyter Notebook Server
CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]