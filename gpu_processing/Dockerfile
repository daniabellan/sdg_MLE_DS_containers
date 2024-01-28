FROM nvidia/cuda:12.1.0-base-ubuntu20.04

# Disable debian frontend to avoid user input
ARG DEBIAN_FRONTEND=noninteractive

# dont write pyc files
# dont buffer to stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install python3.9
RUN apt-get update && apt-get install -y python3.9 python3.9-dev python3-pip

# Install java
RUN apt-get install openjdk-8-jdk-headless -qq    

# Install git
RUN apt-get install -y git

# Install pytorch with GPU (CUDA 12.1)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install git+https://github.com/daniabellan/sdg_MLE_DS.git