FROM nvidia/cuda:12.1.0-base-ubuntu20.04

WORKDIR /app

# Disable debian frontend to avoid user input
ARG DEBIAN_FRONTEND=noninteractive

# dont write pyc files
# dont buffer to stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install python3.9
RUN apt-get update && apt-get install -y python3.9 python3.9-dev python3-pip

# Install requirements
COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip \
    && rm requirements.txt

# Install pytorch with GPU (CUDA 12.1)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY ./ /app